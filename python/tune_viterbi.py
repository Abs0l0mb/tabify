"""
tune_viterbi.py — Per-file automatic parameter tuning for viterbi_predict.py

The goal: given a single JSONL (from a user-submitted MIDI), find viterbi parameters
that produce the most "guitar-like" output — no ground truth required.

Reference-free objective: match output statistics (span, jump, fret position, string
center) to a "target profile" learned from the ground-truth dataset.

WORKFLOW
--------
# 1. Build the target profile once from the GT dataset:
python tune_viterbi.py --phase profile \\
    --dataset_dir ../dev_folder/jsonl_dataset \\
    --profile_out tune_profile.json

# 2. Tune for a specific file (called automatically from complete_workflow):
python tune_viterbi.py --phase tune_file \\
    --input_jsonl path/to/piece.jsonl \\
    --profile tune_profile.json \\
    --out_params best_params.json \\
    --n_trials 60
"""

import argparse
import json
import math
import os
import random
import sys
import time
from copy import deepcopy
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple

import commentjson

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import viterbi_predict as vp


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_base_config(path: str) -> Dict[str, Any]:
    return vp.load_config(path)


def apply_overrides(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Apply flat dotted-key overrides onto a deep copy of base config."""
    cfg = deepcopy(base)
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        node = cfg
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = value
    return cfg


def fast_cfg(base: Dict[str, Any], beam_size: int = 20) -> Dict[str, Any]:
    """Return config stripped down for fast tuning runs."""
    cfg = deepcopy(base)
    cfg["search"]["beam_size"] = beam_size
    return cfg


# ---------------------------------------------------------------------------
# Output statistics (reference-free quality metrics)
# ---------------------------------------------------------------------------

def compute_output_stats(preds: List[vp.Voicing]) -> Optional[Dict[str, float]]:
    """Compute ergonomic statistics of a Viterbi output sequence."""
    active = [p for p in preds if p.pitches]  # skip rests
    if not active:
        return None

    spans = [float(p.span) for p in active]
    frets = [p.avgpos for p in active if p.avgpos > 0]
    string_centers = [
        float(sum(p.strings) / len(p.strings)) for p in active if p.strings
    ]

    jumps = []
    prev_anchor = None
    for p in preds:
        if p.pitches:
            if prev_anchor is not None:
                jumps.append(abs(p.anchor - prev_anchor))
            prev_anchor = p.anchor

    # String discontinuity rate: fraction of events with holes in string coverage
    disc_count = 0
    for p in active:
        if len(p.strings) > 1:
            s = sorted(p.strings)
            holes = (s[-1] - s[0] + 1) - len(s)
            if holes > 0:
                disc_count += 1
    disc_rate = disc_count / len(active)

    return {
        "span_mean":          mean(spans),
        "span_std":           stdev(spans) if len(spans) > 1 else 0.0,
        "jump_mean":          mean(jumps) if jumps else 0.0,
        "jump_p75":           sorted(jumps)[int(0.75 * len(jumps))] if jumps else 0.0,
        "fret_mean":          mean(frets) if frets else 0.0,
        "string_center_mean": mean(string_centers) if string_centers else 3.5,
        "disc_rate":          disc_rate,
    }


# ---------------------------------------------------------------------------
# Phase: profile  (one-time, runs on GT dataset)
# ---------------------------------------------------------------------------

def build_profile(dataset_dir: str, n_files: int, seed: int) -> Dict[str, Any]:
    """
    Sample n_files from the GT dataset and compute target statistics.
    These become the reference for the per-file tuning objective.
    """
    import glob as _glob
    all_files = sorted(_glob.glob(os.path.join(dataset_dir, "*.jsonl")))
    rng = random.Random(seed)
    rng.shuffle(all_files)
    sample = all_files[:n_files]

    print(f"Building profile from {len(sample)} files...")
    all_stats: Dict[str, List[float]] = {}

    for fp in sample:
        try:
            events = vp.load_events_from_jsonl(fp)
            if not events or not any(ev.get("notes") for ev in events):
                continue
            # Extract stats from GROUND TRUTH, not predictions
            gt_preds = _gt_as_voicings(events)
            if gt_preds is None:
                continue
            s = compute_output_stats(gt_preds)
            if s is None:
                continue
            for k, v in s.items():
                all_stats.setdefault(k, []).append(v)
        except Exception:
            continue

    profile = {}
    for k, vals in all_stats.items():
        vals.sort()
        profile[k] = {
            "mean": mean(vals),
            "std":  stdev(vals) if len(vals) > 1 else 1.0,
            "p25":  vals[len(vals) // 4],
            "p50":  vals[len(vals) // 2],
            "p75":  vals[3 * len(vals) // 4],
        }

    profile["_n_files"] = len(all_stats.get("span_mean", []))
    print(f"Profile built from {profile['_n_files']} valid files.")
    return profile


def _gt_as_voicings(events: List[Dict[str, Any]]) -> Optional[List[vp.Voicing]]:
    """Convert ground-truth notes in events to Voicing objects."""
    voicings = []
    for ev in events:
        notes = ev.get("notes", [])
        if not notes:
            # rest
            voicings.append(vp.Voicing(
                pitches=tuple(), strings=tuple(), frets=tuple(),
                local_cost=0.0, span=0, anchor=0.0, avgpos=0.0,
            ))
        else:
            pitches = tuple(int(n["pitch"]) for n in notes)
            strings = tuple(int(n["string"]) for n in notes)
            frets = tuple(int(n["fret"]) for n in notes)
            voicings.append(vp.Voicing(
                pitches=pitches,
                strings=strings,
                frets=frets,
                local_cost=0.0,
                span=vp.chord_span_ignore_open(list(frets)),
                anchor=vp.anchor_pos(list(frets)),
                avgpos=vp.avg_fretted_pos(list(frets)),
            ))
    return voicings if voicings else None


# ---------------------------------------------------------------------------
# Reference-free objective
# ---------------------------------------------------------------------------

def score_against_profile(stats: Dict[str, float], profile: Dict[str, Any]) -> float:
    """
    Returns a score (higher = better) measuring how closely the output stats
    match the target profile (human guitar playing).

    Each stat is z-scored against the profile mean/std, then we penalize
    deviation from zero. We weight jump more heavily than span since it has
    the most impact on playability.
    """
    weights = {
        "jump_mean":          2.0,
        "jump_p75":           1.5,
        "span_mean":          1.5,
        "disc_rate":          2.0,
        "fret_mean":          0.5,
        "string_center_mean": 0.5,
        "span_std":           0.3,
    }
    total = 0.0
    for k, w in weights.items():
        if k not in stats or k not in profile:
            continue
        mu = profile[k]["mean"]
        sigma = max(profile[k]["std"], 1e-6)
        z = (stats[k] - mu) / sigma
        total += w * (z ** 2)  # penalize deviation from profile mean
    return -total  # higher is better


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

# (low, high) for each tunable parameter
PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "local_cost.w_span":                    (0.0, 3.0),
    "local_cost.w_high":                    (0.0, 1.0),
    "local_cost.w_string_range":            (0.0, 1.0),
    "local_cost.w_preferred_zone":          (-3.0, 0.0),
    "local_cost.w_high_string":             (0.0, 5.0),
    "string_discontinuity.w_holes":         (0.0, 8.0),
    "string_discontinuity.w_gap":           (0.0, 2.0),
    "string_discontinuity.w_blocks":        (0.0, 8.0),
    "transition_cost.w_jump":               (0.0, 3.0),
    "transition_cost.jump_power":           (1.0, 2.5),
    "transition_cost.jump_threshold_penalty": (0.0, 10.0),
    "transition_cost.w_avg_jump":           (0.0, 2.0),
    "transition_cost.w_string_center":      (0.0, 6.0),
    "transition_cost.close_jump_bonus":     (-3.0, 0.0),
    "transition_cost.w_span_change":        (0.0, 1.0),
    "transition_cost.w_streak":             (0.0, 8.0),
}


# ---------------------------------------------------------------------------
# Phase: tune_file  (per-file, fast)
# ---------------------------------------------------------------------------

def tune_file(
    input_jsonl: str,
    base_cfg: Dict[str, Any],
    profile: Dict[str, Any],
    n_trials: int,
    beam_size: int,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run Optuna on a single JSONL file. Returns best param overrides dict.
    No ground truth needed.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    events = vp.load_events_from_jsonl(input_jsonl)
    if not events:
        raise ValueError(f"No events in {input_jsonl}")

    base_fast = fast_cfg(base_cfg, beam_size=beam_size)

    def objective(trial: "optuna.Trial") -> float:
        overrides = {
            k: trial.suggest_float(k, lo, hi)
            for k, (lo, hi) in PARAM_BOUNDS.items()
        }
        cfg_i = apply_overrides(base_fast, overrides)
        try:
            preds = vp.viterbi_decode(events, cfg_i)
            stats = compute_output_stats(preds)
            if stats is None:
                return -1e9
            return score_against_profile(stats, profile)
        except Exception:
            return -1e9

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    if verbose:
        print(f"  Best score: {best.value:.4f}  (after {n_trials} trials)")
        # Show how much the output stats shifted
        best_cfg = apply_overrides(base_fast, best.params)
        preds = vp.viterbi_decode(events, best_cfg)
        stats = compute_output_stats(preds)
        if stats:
            print(f"  Output stats: span={stats['span_mean']:.2f}  jump={stats['jump_mean']:.2f}  "
                  f"fret={stats['fret_mean']:.2f}  disc={stats['disc_rate']:.2%}")

    return best.params


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Viterbi per-file parameter tuner")
    ap.add_argument("--phase", required=True, choices=["profile", "tune_file"],
                    help="'profile': build GT stats from dataset. 'tune_file': tune for one JSONL.")
    ap.add_argument("--config", default="viterbi_config.jsonc")

    # profile phase
    ap.add_argument("--dataset_dir", default="../dev_folder/jsonl_dataset")
    ap.add_argument("--n_profile_files", type=int, default=2000,
                    help="How many GT files to use for the profile")
    ap.add_argument("--profile_out", default="tune_profile.json")

    # tune_file phase
    ap.add_argument("--input_jsonl", help="JSONL file to tune for")
    ap.add_argument("--profile", default="tune_profile.json")
    ap.add_argument("--out_params", default="best_params.json",
                    help="Where to write the best overrides (JSON)")
    ap.add_argument("--n_trials", type=int, default=60)
    ap.add_argument("--beam_size", type=int, default=20,
                    help="Beam size during tuning (keep low for speed)")

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    if args.phase == "profile":
        profile = build_profile(args.dataset_dir, args.n_profile_files, seed=args.seed)
        with open(args.profile_out, "w") as f:
            json.dump(profile, f, indent=2)
        print(f"Profile saved to {args.profile_out}")
        print("\nTarget stats (mean):")
        for k, v in profile.items():
            if isinstance(v, dict):
                print(f"  {k:<25s} mean={v['mean']:.3f}  std={v['std']:.3f}")

    elif args.phase == "tune_file":
        if not args.input_jsonl:
            ap.error("--input_jsonl required for tune_file phase")
        if not os.path.exists(args.profile):
            ap.error(f"Profile not found: {args.profile}. Run --phase profile first.")

        with open(args.profile) as f:
            profile = json.load(f)

        base_cfg = load_base_config(args.config)

        print(f"Tuning for: {args.input_jsonl}")
        t0 = time.time()
        best_params = tune_file(
            input_jsonl=args.input_jsonl,
            base_cfg=base_cfg,
            profile=profile,
            n_trials=args.n_trials,
            beam_size=args.beam_size,
            verbose=True,
        )
        dt = time.time() - t0
        print(f"  Done in {dt:.1f}s")

        with open(args.out_params, "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"Best params saved to {args.out_params}")


if __name__ == "__main__":
    main()

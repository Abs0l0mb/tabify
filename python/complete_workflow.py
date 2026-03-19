import os
import argparse
import tempfile

from midi_to_jsonl import midi_to_jsonl_active_compressed
from viterbi_predict import load_config, load_events_from_jsonl, viterbi_decode, save_predicted_jsonl
from jsonl_to_tab import load_pred_events, build_gp5_from_pred_rows, E_STD_TUNING

DEFAULT_PROFILE = os.path.join(os.path.dirname(__file__), "tune_profile.json")


def run(midi_path: str, out_gp5_path: str, config_path: str, step: int, gpq: int, tempo: int,
        auto_tune: bool = False, profile_path: str = DEFAULT_PROFILE,
        tune_trials: int = 60, tune_beam: int = 20) -> None:
    cfg = load_config(config_path) if os.path.exists(config_path) else {}
    tuning = cfg.get("guitar", {}).get("tuning", E_STD_TUNING)

    with tempfile.TemporaryDirectory() as tmp:
        raw_jsonl = os.path.join(tmp, "raw.jsonl")
        pred_jsonl = os.path.join(tmp, "pred.jsonl")

        # Step 1: MIDI -> JSONL
        midi_to_jsonl_active_compressed(
            midi_path=midi_path,
            out_path=raw_jsonl,
            gp_quarter_ticks=gpq,
            step=step,
        )

        # Step 1b (optional): auto-tune parameters for this file
        if auto_tune:
            if not os.path.exists(profile_path):
                print(f"[tune] Profile not found at {profile_path} — skipping auto-tune.")
                print(f"       Run: python tune_viterbi.py --phase profile --profile_out {profile_path}")
            else:
                import json
                from tune_viterbi import tune_file, load_base_config
                print(f"[tune] Auto-tuning parameters for this file ({tune_trials} trials)...")
                with open(profile_path) as f:
                    profile = json.load(f)
                best_params = tune_file(
                    input_jsonl=raw_jsonl,
                    base_cfg=cfg,
                    profile=profile,
                    n_trials=tune_trials,
                    beam_size=tune_beam,
                    verbose=True,
                )
                # Apply best params on top of base config
                from tune_viterbi import apply_overrides
                cfg = apply_overrides(cfg, best_params)

        # Step 2: Viterbi prediction
        events = load_events_from_jsonl(raw_jsonl)
        preds = viterbi_decode(events, cfg)
        save_predicted_jsonl(events, preds, pred_jsonl)

        # Step 3: JSONL -> GP5
        rows = load_pred_events(pred_jsonl)
        build_gp5_from_pred_rows(
            pred_rows=rows,
            out_gp5_path=out_gp5_path,
            title=os.path.splitext(os.path.basename(midi_path))[0],
            tempo=tempo,
            time_sig=(4, 4),
            quarter_ticks=gpq,
            tuning=tuning,
        )

    print(f"[Done] {midi_path} -> {out_gp5_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MIDI -> Guitar Pro 5 tab via Viterbi")
    ap.add_argument("--in", dest="midi", required=True, help="Input MIDI file")
    ap.add_argument("--out", required=True, help="Output GP5 file")
    ap.add_argument("--config", default="viterbi_config.jsonc", help="Config file (default: viterbi_config.jsonc)")
    ap.add_argument("--step", type=int, default=60, help="Quantization step in GP ticks (default: 60 = 1/64 note)")
    ap.add_argument("--gpq", type=int, default=960, help="GP quarter note ticks (default: 960)")
    ap.add_argument("--tempo", type=int, default=120, help="Tempo in BPM (default: 120)")
    ap.add_argument("--auto_tune", action="store_true",
                    help="Auto-tune Viterbi parameters for this file (~30s extra)")
    ap.add_argument("--profile", default=DEFAULT_PROFILE,
                    help="Path to tune_profile.json (built with tune_viterbi.py --phase profile)")
    ap.add_argument("--tune_trials", type=int, default=60,
                    help="Optuna trials for auto-tuning (default: 60)")
    ap.add_argument("--tune_beam", type=int, default=20,
                    help="Beam size during tuning (smaller = faster, default: 20)")
    args = ap.parse_args()

    run(
        midi_path=args.midi,
        out_gp5_path=args.out,
        config_path=args.config,
        step=args.step,
        gpq=args.gpq,
        tempo=args.tempo,
        auto_tune=args.auto_tune,
        profile_path=args.profile,
        tune_trials=args.tune_trials,
        tune_beam=args.tune_beam,
    )

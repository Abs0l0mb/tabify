# step2_5_make_manifest.py
# Génère un manifest (JSON) + un rapport console pour détecter des outliers.
# Ajouts demandés:
# - métriques de "fret span" (écartement) en ignorant les cordes à vide pour le min
# - métriques de position (anchor/avg) ignorent les open strings (fret=0) quand pertinent
# - jumps (sauts) basés sur une position "anchor" robuste (min fretté par event, fallback 0)
# - flags/outliers adaptés

import os
import glob
import json
import math
from typing import Dict, Any, List, Tuple


def safe_mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def safe_min(xs: List[float]) -> float:
    return float(min(xs)) if xs else 0.0


def safe_max(xs: List[float]) -> float:
    return float(max(xs)) if xs else 0.0


def percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    xs_sorted = sorted(xs)
    k = (len(xs_sorted) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(xs_sorted[int(k)])
    d0 = xs_sorted[f] * (c - k)
    d1 = xs_sorted[c] * (k - f)
    return float(d0 + d1)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def chord_fret_span(frets: List[float]) -> float:
    """
    Écartement main: max(fretted) - min(fretted), en ignorant les cordes à vide pour le min.
    - si aucune note frettée (tout open), span=0
    """
    if not frets:
        return 0.0
    fretted = [f for f in frets if f > 0]
    if not fretted:
        return 0.0
    return float(max(fretted) - min(fretted))


def chord_anchor_position(frets: List[float]) -> float:
    """
    Position d'ancrage de la main pour un event:
    - min fret fretté (ignore open strings)
    - fallback 0 si aucun fret > 0
    C'est une approximation robuste pour mesurer les jumps.
    """
    fretted = [f for f in frets if f > 0]
    return float(min(fretted)) if fretted else 0.0


def chord_avg_fretted_position(frets: List[float]) -> float:
    """
    Position moyenne (frettée seulement). Fallback 0 si tout open.
    Utile pour stats, moins robuste que l'ancre pour les jumps.
    """
    fretted = [f for f in frets if f > 0]
    return safe_mean([float(f) for f in fretted]) if fretted else 0.0


def summarize_piece(rows: List[Dict[str, Any]], file_path: str) -> Dict[str, Any]:
    if not rows:
        return {
            "file": os.path.basename(file_path),
            "path": file_path,
            "piece_id": None,
            "n_events": 0,
        }

    piece_id = rows[0].get("piece_id")
    durs = [float(r.get("dur", 0.0)) for r in rows if "dur" in r]
    chord_sizes = [len(r.get("notes", [])) for r in rows]

    # pitches/frets sur tout le morceau
    all_pitches: List[float] = []
    all_frets: List[float] = []
    all_frets_fretted: List[float] = []  # uniquement >0

    # métriques par event
    spans: List[float] = []
    anchor_positions: List[float] = []
    avg_positions: List[float] = []

    for r in rows:
        notes = r.get("notes", [])
        frets = []
        for n in notes:
            if "pitch" in n:
                all_pitches.append(float(n["pitch"]))
            if "fret" in n:
                f = float(n["fret"])
                all_frets.append(f)
                frets.append(f)
                if f > 0:
                    all_frets_fretted.append(f)

        # métriques ergonomiques par event
        if frets:
            spans.append(chord_fret_span(frets))
            anchor_positions.append(chord_anchor_position(frets))
            avg_positions.append(chord_avg_fretted_position(frets))
        else:
            spans.append(0.0)
            anchor_positions.append(0.0)
            avg_positions.append(0.0)

    # sauts de position successifs (basés sur ancre frettée)
    jumps = [abs(anchor_positions[i] - anchor_positions[i - 1]) for i in range(1, len(anchor_positions))]

    total_dur = float(sum(durs))

    summary = {
        "file": os.path.basename(file_path),
        "path": file_path,
        "piece_id": piece_id,
        "n_events": len(rows),
        "total_dur": total_dur,

        # Durées
        "dur_min": safe_min(durs),
        "dur_p50": percentile(durs, 0.50),
        "dur_p95": percentile(durs, 0.95),
        "dur_max": safe_max(durs),
        "dur_mean": safe_mean(durs),

        # Pitches
        "min_pitch": safe_min(all_pitches),
        "max_pitch": safe_max(all_pitches),

        # Frets (incluant open)
        "min_fret": safe_min(all_frets),
        "max_fret": safe_max(all_frets),
        "mean_fret": safe_mean(all_frets),

        # Frets frettés uniquement (ignore open)
        "min_fret_fretted": safe_min(all_frets_fretted),
        "max_fret_fretted": safe_max(all_frets_fretted),
        "mean_fret_fretted": safe_mean(all_frets_fretted),

        # Accords
        "max_chord_size": int(max(chord_sizes) if chord_sizes else 0),
        "mean_chord_size": safe_mean([float(x) for x in chord_sizes]),

        # Ergonomie: span (écartement) et position
        "span_mean": safe_mean(spans),
        "span_p95": percentile(spans, 0.95),
        "span_max": safe_max(spans),

        "anchor_pos_mean": safe_mean(anchor_positions),
        "anchor_pos_p95": percentile(anchor_positions, 0.95),
        "anchor_pos_max": safe_max(anchor_positions),

        "avg_pos_mean": safe_mean(avg_positions),
        "avg_pos_p95": percentile(avg_positions, 0.95),
        "avg_pos_max": safe_max(avg_positions),

        # Ergonomie: jumps (sauts de position) sur ancre frettée
        "jump_mean": safe_mean([float(x) for x in jumps]),
        "jump_p95": percentile(jumps, 0.95),
        "jump_max": safe_max([float(x) for x in jumps]),
    }

    # Flags/outliers (seuils à ajuster selon ton dataset)
    summary["flag_fret_gt_24"] = summary["max_fret"] > 24
    summary["flag_fret_fretted_gt_24"] = summary["max_fret_fretted"] > 24

    # span (écartement)
    summary["flag_span_gt_10"] = summary["span_max"] > 10
    summary["flag_span_gt_12"] = summary["span_max"] > 12

    # jumps de position
    summary["flag_jump_gt_7"] = summary["jump_max"] > 7
    summary["flag_jump_gt_12"] = summary["jump_max"] > 12

    # accords impossibles
    summary["flag_big_chords"] = summary["max_chord_size"] > 6  # guitare 6 cordes
    summary["flag_pitch_out_of_guitar"] = (summary["min_pitch"] < 40) or (summary["max_pitch"] > 88)

    return summary


def write_manifest(jsonl_dir: str, out_manifest_path: str, topk: int = 30) -> None:
    jsonl_files = sorted(glob.glob(os.path.join(jsonl_dir, "*.jsonl")))
    if not jsonl_files:
        raise FileNotFoundError(f"Aucun .jsonl trouvé dans: {jsonl_dir}")

    manifest = []
    for fp in jsonl_files:
        rows = load_jsonl(fp)
        manifest.append(summarize_piece(rows, fp))

    os.makedirs(os.path.dirname(out_manifest_path) or ".", exist_ok=True)

    with open(out_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Rapport console (top outliers)
    def top_by(key: str) -> List[Tuple[str, float]]:
        vals = [(m["file"], float(m.get(key, 0.0))) for m in manifest]
        vals.sort(key=lambda x: x[1], reverse=True)
        return vals[:topk]

    print(f"[OK] Manifest écrit: {out_manifest_path}")
    print(f"Nb morceaux: {len(manifest)}")

    print("\n=== Top par max_fret ===")
    for name, v in top_by("max_fret"):
        print(f"{name:40s}  {v}")

    print("\n=== Top par max_fret_fretted ===")
    for name, v in top_by("max_fret_fretted"):
        print(f"{name:40s}  {v}")

    print("\n=== Top par span_max (écartement) ===")
    for name, v in top_by("span_max"):
        print(f"{name:40s}  {v}")

    print("\n=== Top par jump_max (saut position, ancre frettée) ===")
    for name, v in top_by("jump_max"):
        print(f"{name:40s}  {v}")

    print("\n=== Top par max_chord_size ===")
    for name, v in top_by("max_chord_size"):
        print(f"{name:40s}  {v}")

    print("\n=== Flags ===")
    flags = [
        "flag_fret_gt_24",
        "flag_fret_fretted_gt_24",
        "flag_span_gt_10",
        "flag_span_gt_12",
        "flag_jump_gt_7",
        "flag_jump_gt_12",
        "flag_big_chords",
        "flag_pitch_out_of_guitar",
    ]
    for fl in flags:
        c = sum(1 for m in manifest if m.get(fl))
        print(f"{fl:28s}: {c}")


if __name__ == "__main__":
    # Ajuste chemins
    JSONL_DIR = "./jsonl_dataset"
    OUT_MANIFEST = "./jsonl_dataset_manifest.json"
    write_manifest(JSONL_DIR, OUT_MANIFEST, topk=25)

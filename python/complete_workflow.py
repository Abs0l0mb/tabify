import os
import argparse
import tempfile

from midi_to_jsonl import midi_to_jsonl_active_compressed
from viterbi_predict import load_config, load_events_from_jsonl, viterbi_decode, save_predicted_jsonl
from jsonl_to_tab import load_pred_events, build_gp5_from_pred_rows, E_STD_TUNING


def run(midi_path: str, out_gp5_path: str, config_path: str, step: int, gpq: int, tempo: int) -> None:
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
    args = ap.parse_args()

    run(
        midi_path=args.midi,
        out_gp5_path=args.out,
        config_path=args.config,
        step=args.step,
        gpq=args.gpq,
        tempo=args.tempo,
    )

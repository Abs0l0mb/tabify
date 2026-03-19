"""
server.py — Production FastAPI backend for Tabify

Serves:
  POST /api/tabify          → MIDI (base64) + viterbi params → GP5 binary
  POST /api/suggest-params  → MIDI (base64) → best viterbi params as JSON
  GET  /api/me              → returns not-connected (triggers TabifyPage in the frontend)
  GET  /*                   → static frontend SPA (index.html fallback)

Usage:
  uvicorn server:app --host 0.0.0.0 --port 8000
  uvicorn server:app --host 0.0.0.0 --port 8000 --workers 2
"""

import asyncio
import base64
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Optional

from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from midi_to_jsonl import midi_to_jsonl_active_compressed
from viterbi_predict import load_config, load_events_from_jsonl, viterbi_decode, save_predicted_jsonl
from jsonl_to_tab import load_pred_events, build_gp5_from_pred_rows, E_STD_TUNING
from tune_viterbi import tune_file

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG_PATH  = os.path.join(os.path.dirname(__file__), "viterbi_config.jsonc")
PROFILE_PATH = os.path.join(os.path.dirname(__file__), "tune_profile.json")
STATIC_DIR   = os.environ.get(
    "STATIC_DIR",
    str(Path(__file__).parent.parent / "frontend" / "runtime" / "dist" / "build")
)
MAX_MIDI_BYTES = 10 * 1024 * 1024  # 10 MB

_executor = ThreadPoolExecutor(max_workers=int(os.environ.get("VITERBI_WORKERS", "2")))

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Tabify", docs_url=None, redoc_url=None)


# ---------------------------------------------------------------------------
# /api/me — Auth stub: always returns not-connected so the frontend shows TabifyPage
# ---------------------------------------------------------------------------

@app.get("/api/me")
async def me():
    return JSONResponse({"error": True, "content": "not-connected"})


# ---------------------------------------------------------------------------
# /api/suggest-params — Auto-tune viterbi params for a specific MIDI file
# ---------------------------------------------------------------------------

# dotted keys from tune_viterbi → flat keys used by the frontend
_DOTTED_TO_FLAT = {
    "local_cost.w_span":                    "w_span",
    "local_cost.w_high":                    "w_high",
    "local_cost.w_string_range":            "w_string_range",
    "local_cost.w_preferred_zone":          "w_preferred_zone",
    "local_cost.w_high_string":             "w_high_string",
    "string_discontinuity.w_holes":         "w_holes",
    "string_discontinuity.w_gap":           "w_gap",
    "string_discontinuity.w_blocks":        "w_blocks",
    "transition_cost.w_jump":               "w_jump",
    "transition_cost.jump_power":           "jump_power",
    "transition_cost.jump_threshold_penalty": "jump_threshold_penalty",
    "transition_cost.w_avg_jump":           "w_avg_jump",
    "transition_cost.w_string_center":      "w_string_center",
    "transition_cost.close_jump_bonus":     "close_jump_bonus",
    "transition_cost.w_span_change":        "w_span_change",
    "transition_cost.w_streak":             "w_streak",
}


@app.post("/api/suggest-params")
async def suggest_params(
    midi_base64: Annotated[str,           Form()],
    midi_name:   Annotated[Optional[str], Form()] = "input.mid",
    n_trials:    Annotated[Optional[int], Form()] = 60,
    beam_size:   Annotated[Optional[int], Form()] = 20,
):
    if not os.path.exists(PROFILE_PATH):
        return JSONResponse({"error": True, "content": "tune-profile-not-built"})

    b64 = midi_base64.split(",", 1)[-1] if "," in midi_base64 else midi_base64
    try:
        midi_bytes = base64.b64decode(b64)
    except Exception:
        return JSONResponse({"error": True, "content": "invalid-midi-file"})

    if len(midi_bytes) > MAX_MIDI_BYTES:
        return JSONResponse({"error": True, "content": "midi-file-too-large"})

    import json as _json
    with open(PROFILE_PATH) as f:
        profile = _json.load(f)

    base_cfg = load_config(CONFIG_PATH)

    def suggest() -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            midi_path = os.path.join(tmp, "input.mid")
            raw_jsonl  = os.path.join(tmp, "raw.jsonl")

            with open(midi_path, "wb") as f:
                f.write(midi_bytes)

            midi_to_jsonl_active_compressed(
                midi_path=midi_path,
                out_path=raw_jsonl,
                gp_quarter_ticks=960,
                step=60,
            )

            best_params = tune_file(
                input_jsonl=raw_jsonl,
                base_cfg=base_cfg,
                profile=profile,
                n_trials=n_trials or 60,
                beam_size=beam_size or 20,
                verbose=False,
            )

        # Convert dotted keys → flat keys, round to 4 decimal places
        return {
            _DOTTED_TO_FLAT[k]: round(v, 4)
            for k, v in best_params.items()
            if k in _DOTTED_TO_FLAT
        }

    try:
        loop = asyncio.get_event_loop()
        flat_params = await loop.run_in_executor(_executor, suggest)
    except Exception as e:
        return JSONResponse({"error": True, "content": str(e)})

    return JSONResponse({"error": False, "content": flat_params})


# ---------------------------------------------------------------------------
# /api/tabify — Main endpoint
# ---------------------------------------------------------------------------

@app.post("/api/tabify")
async def tabify(
    midi_base64:  Annotated[str,           Form()],
    midi_name:    Annotated[Optional[str], Form()] = "input.mid",
    # General
    step:         Annotated[Optional[int], Form()] = 60,
    gpq:          Annotated[Optional[int], Form()] = 960,
    tempo:        Annotated[Optional[int], Form()] = 120,
    # Search
    max_fret:     Annotated[Optional[int], Form()] = None,
    per_pitch_k:  Annotated[Optional[int], Form()] = None,
    chord_k:      Annotated[Optional[int], Form()] = None,
    beam_size:    Annotated[Optional[int], Form()] = None,
    # Local cost
    w_span:                Annotated[Optional[float], Form()] = None,
    w_high:                Annotated[Optional[float], Form()] = None,
    high_fret_threshold:   Annotated[Optional[int],   Form()] = None,
    w_open_bonus:          Annotated[Optional[float], Form()] = None,
    w_string_range:        Annotated[Optional[float], Form()] = None,
    preferred_min_fret:    Annotated[Optional[int],   Form()] = None,
    preferred_max_fret:    Annotated[Optional[int],   Form()] = None,
    w_preferred_zone:      Annotated[Optional[float], Form()] = None,
    high_string_threshold: Annotated[Optional[int],   Form()] = None,
    w_high_string:         Annotated[Optional[float], Form()] = None,
    # String discontinuity
    w_holes:  Annotated[Optional[float], Form()] = None,
    w_gap:    Annotated[Optional[float], Form()] = None,
    w_blocks: Annotated[Optional[float], Form()] = None,
    # Transition cost
    w_jump:                 Annotated[Optional[float], Form()] = None,
    jump_power:             Annotated[Optional[float], Form()] = None,
    jump_threshold:         Annotated[Optional[float], Form()] = None,
    jump_threshold_penalty: Annotated[Optional[float], Form()] = None,
    w_avg_jump:             Annotated[Optional[float], Form()] = None,
    avg_jump_power:         Annotated[Optional[float], Form()] = None,
    w_span_change:          Annotated[Optional[float], Form()] = None,
    w_string_center:        Annotated[Optional[float], Form()] = None,
    close_jump_threshold:   Annotated[Optional[float], Form()] = None,
    close_jump_bonus:       Annotated[Optional[float], Form()] = None,
    rest_enter_penalty:     Annotated[Optional[float], Form()] = None,
    rest_exit_penalty:      Annotated[Optional[float], Form()] = None,
    w_streak:               Annotated[Optional[float], Form()] = None,
    streak_min_len:         Annotated[Optional[int],   Form()] = None,
    streak_speed_threshold: Annotated[Optional[int],   Form()] = None,
):
    # ── Decode MIDI ──────────────────────────────────────────────────────
    b64 = midi_base64.split(",", 1)[-1] if "," in midi_base64 else midi_base64
    try:
        midi_bytes = base64.b64decode(b64)
    except Exception:
        return JSONResponse({"error": True, "content": "invalid-midi-file"})

    if len(midi_bytes) > MAX_MIDI_BYTES:
        return JSONResponse({"error": True, "content": "midi-file-too-large"})

    # ── Build config with overrides ───────────────────────────────────────
    cfg = load_config(CONFIG_PATH)

    def set_param(section: str, key: str, val, cast=float):
        if val is not None:
            cfg[section][key] = cast(val)

    set_param("search", "max_fret",    max_fret,    int)
    set_param("search", "per_pitch_k", per_pitch_k, int)
    set_param("search", "chord_k",     chord_k,     int)
    set_param("search", "beam_size",   beam_size,   int)

    set_param("local_cost", "w_span",                w_span)
    set_param("local_cost", "w_high",                w_high)
    set_param("local_cost", "high_fret_threshold",   high_fret_threshold,   int)
    set_param("local_cost", "w_open_bonus",          w_open_bonus)
    set_param("local_cost", "w_string_range",        w_string_range)
    set_param("local_cost", "preferred_min_fret",    preferred_min_fret,    int)
    set_param("local_cost", "preferred_max_fret",    preferred_max_fret,    int)
    set_param("local_cost", "w_preferred_zone",      w_preferred_zone)
    set_param("local_cost", "high_string_threshold", high_string_threshold, int)
    set_param("local_cost", "w_high_string",         w_high_string)

    set_param("string_discontinuity", "w_holes",  w_holes)
    set_param("string_discontinuity", "w_gap",    w_gap)
    set_param("string_discontinuity", "w_blocks", w_blocks)

    set_param("transition_cost", "w_jump",                 w_jump)
    set_param("transition_cost", "jump_power",             jump_power)
    set_param("transition_cost", "jump_threshold",         jump_threshold)
    set_param("transition_cost", "jump_threshold_penalty", jump_threshold_penalty)
    set_param("transition_cost", "w_avg_jump",             w_avg_jump)
    set_param("transition_cost", "avg_jump_power",         avg_jump_power)
    set_param("transition_cost", "w_span_change",          w_span_change)
    set_param("transition_cost", "w_string_center",        w_string_center)
    set_param("transition_cost", "close_jump_threshold",   close_jump_threshold)
    set_param("transition_cost", "close_jump_bonus",       close_jump_bonus)
    set_param("transition_cost", "rest_enter_penalty",     rest_enter_penalty)
    set_param("transition_cost", "rest_exit_penalty",      rest_exit_penalty)
    set_param("transition_cost", "w_streak",               w_streak)
    set_param("transition_cost", "streak_min_len",         streak_min_len,         int)
    set_param("transition_cost", "streak_speed_threshold", streak_speed_threshold, int)

    # ── Run Viterbi in thread pool (CPU-bound) ────────────────────────────
    step_  = step  or 60
    gpq_   = gpq   or 960
    tempo_ = tempo or 120

    def process() -> bytes:
        tuning = cfg.get("guitar", {}).get("tuning", E_STD_TUNING)
        with tempfile.TemporaryDirectory() as tmp:
            midi_path  = os.path.join(tmp, "input.mid")
            raw_jsonl  = os.path.join(tmp, "raw.jsonl")
            pred_jsonl = os.path.join(tmp, "pred.jsonl")
            gp5_path   = os.path.join(tmp, "output.gp5")

            with open(midi_path, "wb") as f:
                f.write(midi_bytes)

            midi_to_jsonl_active_compressed(
                midi_path=midi_path,
                out_path=raw_jsonl,
                gp_quarter_ticks=gpq_,
                step=step_,
            )

            events = load_events_from_jsonl(raw_jsonl)
            preds  = viterbi_decode(events, cfg)
            save_predicted_jsonl(events, preds, pred_jsonl)

            rows = load_pred_events(pred_jsonl)
            build_gp5_from_pred_rows(
                pred_rows=rows,
                out_gp5_path=gp5_path,
                title=os.path.splitext(os.path.basename(midi_name or "output"))[0],
                tempo=tempo_,
                time_sig=(4, 4),
                quarter_ticks=gpq_,
                tuning=tuning,
            )

            with open(gp5_path, "rb") as f:
                return f.read()

    try:
        loop = asyncio.get_event_loop()
        gp5_bytes = await loop.run_in_executor(_executor, process)
    except Exception as e:
        return JSONResponse({"error": True, "content": str(e)})

    stem = (midi_name or "output").rsplit(".", 1)[0]
    return Response(
        content=gp5_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{stem}.gp5"'},
    )


# ---------------------------------------------------------------------------
# Static frontend (must come last)
# ---------------------------------------------------------------------------

if os.path.isdir(STATIC_DIR):
    for sub in ("assets", "css", "js"):
        sub_dir = os.path.join(STATIC_DIR, sub)
        if os.path.isdir(sub_dir):
            app.mount(f"/{sub}", StaticFiles(directory=sub_dir), name=sub)

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        path = os.path.join(STATIC_DIR, "favicon.ico")
        return FileResponse(path) if os.path.exists(path) else Response(status_code=204)

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str):
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

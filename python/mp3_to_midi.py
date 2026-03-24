"""
mp3_to_midi.py — Audio source separation + note transcription pipeline

Pipeline:
  Audio (MP3/WAV/FLAC) → Demucs → per-stem WAV files
                        → Basic Pitch → MIDI per stem

Demucs model options:
  "htdemucs_6s"  6 stems: drums, bass, other, vocals, guitar, piano  (recommended)
  "htdemucs"     4 stems: drums, bass, other, vocals  (guitar ends up in "other")

Usage (standalone):
  python mp3_to_midi.py song.mp3 --stems guitar bass --out ./output
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional

# Stems that carry pitched notes and can be meaningfully tabified
TABBABLE_STEMS = {"bass", "guitar", "other", "piano", "vocals"}
DEFAULT_STEMS   = {"guitar", "bass"}

# Basic Pitch inference defaults
DEFAULT_ONSET_THRESHOLD  = 0.5
DEFAULT_FRAME_THRESHOLD  = 0.3
DEFAULT_MIN_NOTE_LENGTH  = 127.70   # ms  (~1/16 note at 120 bpm)
DEFAULT_MIN_FREQ: Optional[float] = None
DEFAULT_MAX_FREQ: Optional[float] = None


# ---------------------------------------------------------------------------
# Step 1 — Source separation
# ---------------------------------------------------------------------------

def separate_stems(
    audio_path: str,
    output_dir: str,
    model: str = "htdemucs_6s",
) -> dict[str, str]:
    """
    Separate an audio file into stems using the Demucs Python API.

    Uses librosa to load audio (avoids torchaudio MP3/torchcodec issues),
    then applies the Demucs model and saves each stem as a WAV file.

    Args:
        audio_path: Path to the input audio file (MP3, WAV, FLAC, …).
        output_dir: Directory where per-stem WAV files will be written.
        model:      Demucs model name (e.g. "htdemucs_6s", "htdemucs").

    Returns:
        dict mapping stem name (e.g. "guitar") → absolute path to WAV file.
    """
    import torch
    import librosa
    import soundfile as sf
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    demucs_model = get_model(model)
    demucs_model.eval()
    target_sr = demucs_model.samplerate
    n_channels = demucs_model.audio_channels  # typically 2 (stereo)

    # Load audio with librosa (bypasses torchaudio/torchcodec entirely)
    print(f"Loading audio: {audio_path}")
    y, _ = librosa.load(audio_path, sr=target_sr, mono=False)
    if y.ndim == 1:                        # mono → stereo
        y = np.stack([y, y], axis=0)
    if y.shape[0] > n_channels:            # trim extra channels
        y = y[:n_channels]

    # shape: (batch=1, channels, samples)
    wav = torch.tensor(y, dtype=torch.float32).unsqueeze(0)

    # Separate
    print(f"Separating stems with {model}…")
    with torch.no_grad():
        sources = apply_model(demucs_model, wav, progress=True)
    # sources shape: (batch=1, num_sources, channels, samples)

    # Save each stem
    stem_paths: dict[str, str] = {}
    for i, stem_name in enumerate(demucs_model.sources):
        stem_audio = sources[0, i].cpu().numpy()   # (channels, samples)
        out_path = os.path.join(output_dir, f"{stem_name}.wav")
        sf.write(out_path, stem_audio.T, target_sr)
        stem_paths[stem_name] = out_path
        print(f"  saved {stem_name} -> {out_path}")

    return stem_paths


# ---------------------------------------------------------------------------
# Step 2 — Audio → MIDI transcription
# ---------------------------------------------------------------------------

def audio_to_midi(
    wav_path: str,
    out_midi_path: str,
    onset_threshold:     float           = DEFAULT_ONSET_THRESHOLD,
    frame_threshold:     float           = DEFAULT_FRAME_THRESHOLD,
    minimum_note_length: float           = DEFAULT_MIN_NOTE_LENGTH,
    minimum_frequency:   Optional[float] = DEFAULT_MIN_FREQ,
    maximum_frequency:   Optional[float] = DEFAULT_MAX_FREQ,
) -> str:
    """
    Transcribe a WAV file to MIDI using Spotify's Basic Pitch model.

    Args:
        wav_path:            Path to the input WAV file.
        out_midi_path:       Destination path for the output MIDI file.
        onset_threshold:     Note onset confidence threshold (0–1).
                             Lower → more notes detected, including false positives.
        frame_threshold:     Frame-level pitch confidence threshold (0–1).
        minimum_note_length: Shortest note to keep, in milliseconds.
        minimum_frequency:   Ignore pitches below this frequency (Hz), or None.
        maximum_frequency:   Ignore pitches above this frequency (Hz), or None.

    Returns:
        out_midi_path (for chaining).
    """
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH

    _, midi_data, _ = predict(
        wav_path,
        ICASSP_2022_MODEL_PATH,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_length,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        multiple_pitch_bends=False,
        melodia_trick=True,
    )

    midi_data.write(out_midi_path)
    return out_midi_path


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------

def mp3_to_midis(
    audio_path: str,
    output_dir: str,
    stems:               Optional[set[str]] = None,
    model:               str                = "htdemucs_6s",
    onset_threshold:     float              = DEFAULT_ONSET_THRESHOLD,
    frame_threshold:     float              = DEFAULT_FRAME_THRESHOLD,
    minimum_note_length: float              = DEFAULT_MIN_NOTE_LENGTH,
    minimum_frequency:   Optional[float]    = DEFAULT_MIN_FREQ,
    maximum_frequency:   Optional[float]    = DEFAULT_MAX_FREQ,
) -> dict[str, str]:
    """
    Full pipeline: audio file → per-stem MIDI files.

    Args:
        audio_path:  Path to the input audio file.
        output_dir:  Working directory for Demucs output and generated MIDIs.
        stems:       Which stems to transcribe (e.g. {"guitar", "bass"}).
                     Drums are always skipped regardless of this setting.
                     Defaults to DEFAULT_STEMS = {"guitar", "bass"}.
        model:       Demucs model (see separate_stems docstring).
        onset_threshold, frame_threshold, minimum_note_length,
        minimum_frequency, maximum_frequency: Forwarded to audio_to_midi.

    Returns:
        dict mapping stem name → path to the generated MIDI file.
        Only stems that were both requested and produced by Demucs are included.
    """
    if stems is None:
        stems = DEFAULT_STEMS
    stems = stems & TABBABLE_STEMS  # silently drop drums / non-pitched stems

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: separate
    all_wavs = separate_stems(audio_path, output_dir, model=model)

    # Step 2: transcribe each requested stem
    midi_paths: dict[str, str] = {}
    for stem_name, wav_path in all_wavs.items():
        if stem_name not in stems:
            continue
        out_midi = os.path.join(output_dir, f"{stem_name}.mid")
        audio_to_midi(
            wav_path=wav_path,
            out_midi_path=out_midi,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=minimum_note_length,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
        )
        midi_paths[stem_name] = out_midi

    return midi_paths


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Separate an audio file into stems and transcribe each to MIDI."
    )
    parser.add_argument("audio",  help="Input audio file (MP3, WAV, FLAC, …)")
    parser.add_argument("--out",  default="./mp3_to_midi_output", help="Output directory")
    parser.add_argument("--stems", nargs="+", default=list(DEFAULT_STEMS),
                        help=f"Stems to transcribe (default: {sorted(DEFAULT_STEMS)})")
    parser.add_argument("--model", default="htdemucs_6s",
                        help="Demucs model (htdemucs_6s or htdemucs)")
    parser.add_argument("--onset",  type=float, default=DEFAULT_ONSET_THRESHOLD,
                        help="Basic Pitch onset threshold (0–1)")
    parser.add_argument("--frame",  type=float, default=DEFAULT_FRAME_THRESHOLD,
                        help="Basic Pitch frame threshold (0–1)")
    parser.add_argument("--min-note-ms", type=float, default=DEFAULT_MIN_NOTE_LENGTH,
                        help="Minimum note length in milliseconds")
    parser.add_argument("--min-freq", type=float, default=None,
                        help="Minimum frequency for pitch detection (Hz)")
    parser.add_argument("--max-freq", type=float, default=None,
                        help="Maximum frequency for pitch detection (Hz)")
    args = parser.parse_args()

    midi_files = mp3_to_midis(
        audio_path=args.audio,
        output_dir=args.out,
        stems=set(args.stems),
        model=args.model,
        onset_threshold=args.onset,
        frame_threshold=args.frame,
        minimum_note_length=args.min_note_ms,
        minimum_frequency=args.min_freq,
        maximum_frequency=args.max_freq,
    )

    print("Generated MIDI files:")
    for stem, path in midi_files.items():
        print(f"  {stem:12s} -> {path}")

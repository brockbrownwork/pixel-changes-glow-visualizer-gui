# CLAUDE.md

## Project Overview

Motion visualization tool that detects per-pixel movement between consecutive video frames and generates a "motion ghosting" heatmap. Moving pixels appear red; static pixels age over time with hue rotation and brightness decay.

## Tech Stack

- **Python 3.9+** — core language
- **numpy** — vectorized pixel-level computation
- **Pillow (PIL)** — image loading/saving
- **tkinter** — GUI (standard library)
- **ffmpeg / ffprobe** — video I/O (external, must be on PATH)

## Project Structure

```
motion_ghosting.py      # Core engine: motion detection, heatmap rendering, frame processing
motion_ghosting_gui.py  # Tkinter GUI wrapping the core engine
```

## Running

```bash
# Install deps
pip install numpy pillow

# GUI
python motion_ghosting_gui.py

# CLI
python motion_ghosting.py --input frames --output output --fps 60 --fade-seconds 1
```

## Key Processing Parameters

- `--threshold` (0-255): movement detection sensitivity
- `--fps`: frames per second for timing
- `--fade-seconds`: time for static pixels to fully age
- `--io-workers`: parallel image decode workers
- `--input-mode` (pil|ffmpeg): frame decoding backend

## Architecture Notes

- Two processing modes: PIL-based frame loading (flexible) and direct ffmpeg pipe (fast)
- GUI runs processing on a worker thread; uses `widget.after()` for thread-safe UI updates
- Callback architecture (`on_frame_done`) for live preview during processing
- Temp directories are cleaned up even on errors

## Commit Style

Lowercase, action-oriented, concise messages. No emojis or conventional commit prefixes.

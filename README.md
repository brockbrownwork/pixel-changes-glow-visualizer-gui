# Motion Ghosting Heatmap (Frame-to-Frame)

This tool takes a folder of sequential PNG frames (e.g. extracted from a video), detects per-pixel movement between consecutive frames, and renders a “motion ghosting” visualization as output frames you can stitch back into a video.

- **Movement pixels** reset their “age” to 0.
- **Non-moving pixels** accumulate age each frame.
- Output color is based on age (a heatmap-like HSV mapping), producing a ghosting trail effect over time.

<img width="1440" height="1080" alt="output_005188" src="https://github.com/user-attachments/assets/aada61cb-8542-4886-8da9-1820fb75f3bb" />


---

## Requirements

- Python **3.9+** recommended
- Dependencies:
  - `numpy`
  - `Pillow` (PIL)

Optional (recommended for video workflows):
- `ffmpeg` available on your PATH

Install Python deps:

```bash
python -m pip install numpy pillow
```

## Useful commands for the average workflow

Create a folder

```bash
mkdir frames
```

Split your input video into frames

```bash
ffmpeg -i input.mp4 -vsync 0 frames/frame_%06d.png
```

Process the frames

```bash
python motion_ghosting.py --input frames --output output --fps 60 --fade-seconds 1 --pad-digits 6
```

Merge processed frames back into a video

```bash
ffmpeg -framerate 60 -i output/output_%06d.png -c:v libx264 -pix_fmt yuv420p output.mp4
```

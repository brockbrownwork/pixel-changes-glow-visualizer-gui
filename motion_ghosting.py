import argparse
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor
import subprocess

import numpy as np
from PIL import Image


def detect_movement(curr, prev, threshold):
    diff = np.abs(curr.astype(np.int16) - prev.astype(np.int16))
    return diff >= threshold


def update_idle_time(idle_time, moved):
    idle_time[moved] = 0.0
    idle_time[~moved] += 1.0
    return idle_time


def hsv_to_rgb(h, s, v):
    c = v * s
    h6 = h * 6.0
    x = c * (1.0 - np.abs((h6 % 2.0) - 1.0))
    m = v - c

    z = np.zeros_like(h)
    r = np.where(h6 < 1, c, np.where(h6 < 2, x, np.where(h6 < 4, z, np.where(h6 < 5, x, c))))
    g = np.where(h6 < 1, x, np.where(h6 < 3, c, np.where(h6 < 4, x, z)))
    b = np.where(h6 < 2, z, np.where(h6 < 3, x, np.where(h6 < 5, c, x)))

    r = (r + m) * 255.0
    g = (g + m) * 255.0
    b = (b + m) * 255.0
    return np.stack([r, g, b], axis=2).astype(np.uint8)


def to_heatmap(idle_time, fade_frames):
    # Hue rotates across the color wheel as pixels age.
    normalized = np.clip(idle_time / float(fade_frames), 0.0, 1.0)
    hue = normalized
    value = 1.0 - normalized
    return hsv_to_rgb(hue, 1.0, value)


def frame_key(path):
    match = re.search(r"(\d+)", path.stem)
    if match:
        return int(match.group(1))
    return path.stem


def output_name(path, digits):
    match = re.search(r"(\d+)", path.stem)
    if match:
        num = int(match.group(1))
        return f"output_{num:0{digits}d}.png"
    return path.name


def infer_sequence_pattern(path):
    match = re.match(r"(.*?)(\d+)(\.\w+)$", path.name)
    if not match:
        raise SystemExit("Cannot infer numeric pattern from filenames.")
    prefix, number, suffix = match.groups()
    digits = len(number)
    pattern = f"{prefix}%0{digits}d{suffix}"
    return pattern, digits


def process_frames(
    input_dir,
    output_dir,
    threshold,
    fps,
    fade_seconds,
    start,
    end,
    step,
    pad_digits,
    io_workers,
    input_mode,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(input_dir.glob("*.png"), key=frame_key)
    if end is None:
        end = len(frames)
    frames = frames[start:end:step]
    if len(frames) < 2:
        raise SystemExit("Need at least two frames to detect movement.")

    fade_frames = max(int(round(fade_seconds * fps)), 1)
    def load_frame(path):
        return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)

    if input_mode == "ffmpeg" and step != 1:
        raise SystemExit("--input-mode ffmpeg requires --step 1.")

    if io_workers < 1:
        raise SystemExit("--io-workers must be >= 1.")

    if input_mode == "ffmpeg":
        sample = Image.open(frames[0]).convert("L")
        w, h = sample.size
        pattern, _ = infer_sequence_pattern(frames[0])
        start_number = frame_key(frames[0])
        count = len(frames)
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-start_number",
            str(start_number),
            "-i",
            str(input_dir / pattern),
            "-vframes",
            str(count),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        try:
            frame_bytes = w * h
            prev = np.frombuffer(proc.stdout.read(frame_bytes), dtype=np.uint8).reshape(
                (h, w)
            )
            idle_time = np.zeros_like(prev, dtype=np.float32) + float(fade_frames)
            for frame_path in frames[1:]:
                data = proc.stdout.read(frame_bytes)
                if len(data) != frame_bytes:
                    raise SystemExit("ffmpeg stream ended early.")
                curr = np.frombuffer(data, dtype=np.uint8).reshape((h, w))
                moved = detect_movement(curr, prev, threshold)
                idle_time = update_idle_time(idle_time, moved)
                out = to_heatmap(idle_time, fade_frames)
                out_name = output_name(frame_path, pad_digits)
                Image.fromarray(out, mode="RGB").save(output_dir / out_name)
                prev = curr
        finally:
            if proc.stdout:
                proc.stdout.close()
            proc.wait()
    else:
        if io_workers == 1:
            iterator = (load_frame(p) for p in frames)
        else:
            executor = ThreadPoolExecutor(max_workers=io_workers)
            iterator = executor.map(load_frame, frames)

        try:
            prev = next(iterator)
            idle_time = np.zeros_like(prev, dtype=np.float32) + float(fade_frames)
            for frame_path, curr in zip(frames[1:], iterator):
                moved = detect_movement(curr, prev, threshold)
                idle_time = update_idle_time(idle_time, moved)
                out = to_heatmap(idle_time, fade_frames)
                out_name = output_name(frame_path, pad_digits)
                Image.fromarray(out, mode="RGB").save(output_dir / out_name)
                prev = curr
        finally:
            if io_workers != 1:
                executor.shutdown(wait=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render motion ghosting: red for movement"
    )
    parser.add_argument("--input", default="frames", help="Input frames folder.")
    parser.add_argument("--output", default="output", help="Output frames folder.")
    parser.add_argument(
        "--threshold", type=int, default=12, help="Movement threshold in grayscale."
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second.")
    parser.add_argument(
        "--fade-seconds",
        type=float,
        default=3.0,
        help="Seconds before a stable pixel ages.",
    )
    parser.add_argument(
        "--pad-digits",
        type=int,
        default=6,
        help="Zero-pad output frame numbers to this width.",
    )
    parser.add_argument(
        "--io-workers",
        type=int,
        default=4,
        help="Parallel image decode workers (1 disables parallelism).",
    )
    parser.add_argument(
        "--input-mode",
        choices=["pil", "ffmpeg"],
        default="pil",
        help="Frame decode backend (ffmpeg is faster; requires step=1).",
    )
    parser.add_argument("--start", type=int, default=0, help="Start frame index.")
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End frame index (exclusive).",
    )
    parser.add_argument("--step", type=int, default=1, help="Process every Nth frame.")
    return parser.parse_args()


def main():
    args = parse_args()
    process_frames(
        args.input,
        args.output,
        args.threshold,
        args.fps,
        args.fade_seconds,
        args.start,
        args.end,
        args.step,
        args.pad_digits,
        args.io_workers,
        args.input_mode,
    )


if __name__ == "__main__":
    main()

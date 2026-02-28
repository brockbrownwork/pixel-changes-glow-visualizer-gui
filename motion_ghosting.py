import argparse
import json
from collections import deque
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor
import subprocess

import numpy as np
from PIL import Image

try:
    import cupy
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def _xp(gpu=True):
    """Return cupy when available and requested, else numpy."""
    if gpu and HAS_CUPY:
        return cupy
    return np


def _asnumpy(arr):
    """Ensure array is on CPU (no-op if already numpy)."""
    if HAS_CUPY and isinstance(arr, cupy.ndarray):
        return cupy.asnumpy(arr)
    return np.asarray(arr)


def _get_xp(*arrays):
    """Infer the array module from input arrays."""
    if HAS_CUPY:
        for a in arrays:
            if isinstance(a, cupy.ndarray):
                return cupy
    return np


# ── GPU / hwaccel helpers ────────────────────────────────────────────

_hwaccel_cache = None


def detect_hwaccel():
    """Return the best available ffmpeg hwaccel method, or None."""
    global _hwaccel_cache
    if _hwaccel_cache is not None:
        return _hwaccel_cache if _hwaccel_cache != "" else None
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-hwaccels"],
            capture_output=True, text=True, timeout=10,
        )
        methods = result.stdout.strip().splitlines()[1:]  # skip header line
        methods = [m.strip() for m in methods if m.strip()]
        # prefer cuda > d3d11va > dxva2 > vaapi > videotoolbox > auto
        for preferred in ("cuda", "d3d11va", "dxva2", "vaapi", "videotoolbox"):
            if preferred in methods:
                _hwaccel_cache = preferred
                return preferred
        _hwaccel_cache = ""
        return None
    except Exception:
        _hwaccel_cache = ""
        return None


_gpu_encoder_cache = None


def detect_gpu_encoder():
    """Return the best available h264 GPU encoder, or 'libx264' as fallback."""
    global _gpu_encoder_cache
    if _gpu_encoder_cache is not None:
        return _gpu_encoder_cache
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        text = result.stdout
        for enc in ("h264_nvenc", "h264_amf", "h264_qsv"):
            if enc in text:
                _gpu_encoder_cache = enc
                return enc
    except Exception:
        pass
    _gpu_encoder_cache = "libx264"
    return "libx264"


def _hwaccel_decode_flags():
    """Return ffmpeg input flags for hardware-accelerated decoding."""
    method = detect_hwaccel()
    if method:
        return ["-hwaccel", method]
    return []


def detect_movement(curr, prev, threshold):
    xp = _get_xp(curr, prev)
    diff = xp.abs(curr.astype(xp.int16) - prev.astype(xp.int16))
    return diff >= threshold


def update_idle_time(idle_time, moved):
    idle_time[moved] = 0.0
    idle_time[~moved] += 1.0
    return idle_time


def update_fatigue(fire_freq, threshold, moved, fps,
                   target_freq, tau, adjust_rate,
                   min_thresh, max_thresh):
    xp = _get_xp(fire_freq)
    alpha = 1.0 - xp.exp(xp.float32(-1.0 / (tau * fps)))
    fired = moved.astype(xp.float32)
    fire_freq *= (1.0 - alpha)
    fire_freq += fired * fps * alpha
    error = fire_freq - target_freq
    threshold += adjust_rate * error
    xp.clip(threshold, min_thresh, max_thresh, out=threshold)
    return fire_freq, threshold


def hsv_to_rgb(h, s, v):
    xp = _get_xp(h)
    c = v * s
    h6 = h * 6.0
    x = c * (1.0 - xp.abs((h6 % 2.0) - 1.0))
    m = v - c

    z = xp.zeros_like(h)
    r = xp.where(h6 < 1, c, xp.where(h6 < 2, x, xp.where(h6 < 4, z, xp.where(h6 < 5, x, c))))
    g = xp.where(h6 < 1, x, xp.where(h6 < 3, c, xp.where(h6 < 4, x, z)))
    b = xp.where(h6 < 2, z, xp.where(h6 < 3, x, xp.where(h6 < 5, c, x)))

    r = (r + m) * 255.0
    g = (g + m) * 255.0
    b = (b + m) * 255.0
    return xp.stack([r, g, b], axis=2).astype(xp.uint8)


def to_heatmap(idle_time, fade_frames):
    # Hue rotates across the color wheel as pixels age.
    xp = _get_xp(idle_time)
    normalized = xp.clip(idle_time / float(fade_frames), 0.0, 1.0)
    hue = normalized
    value = 1.0 - normalized
    return hsv_to_rgb(hue, 1.0, value)


def threshold_to_heatmap(thresh_arr, min_thresh, max_thresh):
    """Visualize per-pixel threshold as a heatmap (blue=sensitive → red=desensitized)."""
    xp = _get_xp(thresh_arr)
    normalized = xp.clip(
        (thresh_arr - min_thresh) / (max_thresh - min_thresh), 0.0, 1.0
    )
    # hue 0.66 (blue, most sensitive) → 0.0 (red, most desensitized)
    hue = (1.0 - normalized) * 0.66
    ones = xp.ones_like(normalized)
    return hsv_to_rgb(hue, ones, ones)


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


def _probe_video_dimensions(video_path):
    """Return (width, height) of the first video stream using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json",
            str(video_path),
        ],
        capture_output=True, text=True, timeout=15,
    )
    info = json.loads(result.stdout)
    s = info["streams"][0]
    return int(s["width"]), int(s["height"])


def process_frames_from_video(
    video_path,
    output_dir,
    threshold,
    fps,
    fade_seconds,
    pad_digits,
    on_frame_done=None,
    gpu=True,
    avg_window=1,
    fatigue=True,
    target_freq=1.0,
    fatigue_tau=2.0,
    adjust_rate=0.5,
    min_thresh=2.0,
    max_thresh=200.0,
    stop_event=None,
):
    """Process a video file directly — no intermediate PNG extraction."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    w, h = _probe_video_dimensions(video_path)
    fade_frames = max(int(round(fade_seconds * fps)), 1)
    frame_bytes = w * h

    hwaccel = _hwaccel_decode_flags() if gpu else []
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        *hwaccel,
        "-i", str(video_path),
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "-",
    ]
    xp = _xp(gpu)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    try:
        raw = proc.stdout.read(frame_bytes)
        if len(raw) != frame_bytes:
            raise SystemExit("Could not read first frame from video.")
        prev_raw = xp.asarray(np.frombuffer(raw, dtype=np.uint8).reshape((h, w)))

        if avg_window > 1:
            buf = deque(maxlen=avg_window)
            rsum = xp.zeros((h, w), dtype=xp.float32)
            f = prev_raw.astype(xp.float32)
            buf.append(f)
            rsum += f
            prev = (rsum / len(buf)).astype(xp.uint8)
        else:
            prev = prev_raw

        idle_time = xp.zeros((h, w), dtype=xp.float32) + float(fade_frames)

        if fatigue:
            thresh_arr = xp.full((h, w), float(threshold), dtype=xp.float32)
            fire_freq = xp.zeros((h, w), dtype=xp.float32)
        else:
            thresh_arr = threshold

        frame_idx = 1
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            data = proc.stdout.read(frame_bytes)
            if len(data) != frame_bytes:
                break
            curr_raw = xp.asarray(np.frombuffer(data, dtype=np.uint8).reshape((h, w)))

            if avg_window > 1:
                f = curr_raw.astype(xp.float32)
                if len(buf) == buf.maxlen:
                    rsum -= buf[0]
                buf.append(f)
                rsum += f
                curr = (rsum / len(buf)).astype(xp.uint8)
            else:
                curr = curr_raw

            moved = detect_movement(curr, prev, thresh_arr)
            if fatigue:
                update_fatigue(fire_freq, thresh_arr, moved, fps,
                               target_freq, fatigue_tau, adjust_rate,
                               min_thresh, max_thresh)
            idle_time = update_idle_time(idle_time, moved)
            out = to_heatmap(idle_time, fade_frames)
            out_name = f"output_{frame_idx:0{pad_digits}d}.png"
            out_path = output_dir / out_name
            Image.fromarray(_asnumpy(out), mode="RGB").save(out_path)
            if on_frame_done is not None:
                orig_img = Image.fromarray(_asnumpy(curr_raw), mode="L")
                sens_img = None
                if fatigue:
                    sens_data = threshold_to_heatmap(thresh_arr, min_thresh, max_thresh)
                    sens_img = Image.fromarray(_asnumpy(sens_data), mode="RGB")
                avg_img = None
                if avg_window > 1:
                    avg_img = Image.fromarray(_asnumpy(curr), mode="L")
                on_frame_done(orig_img, out_path, sens_img, avg_img)
            prev = curr
            frame_idx += 1
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.wait()


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
    on_frame_done=None,
    gpu=True,
    avg_window=1,
    fatigue=True,
    target_freq=1.0,
    fatigue_tau=2.0,
    adjust_rate=0.5,
    min_thresh=2.0,
    max_thresh=200.0,
    stop_event=None,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(input_dir.glob("*.bmp"), key=frame_key)
    if end is None:
        end = len(frames)
    frames = frames[start:end:step]
    if len(frames) < 2:
        raise SystemExit("Need at least two frames to detect movement.")

    fade_frames = max(int(round(fade_seconds * fps)), 1)
    def load_frame(path):
        data = path.read_bytes()
        offset = int.from_bytes(data[10:14], 'little')
        w = int.from_bytes(data[18:22], 'little')
        h = int.from_bytes(data[22:26], 'little')
        stride = (w + 3) & ~3
        pixels = np.frombuffer(data, dtype=np.uint8, offset=offset)
        return pixels.reshape((h, stride))[:, :w][::-1]

    if input_mode == "ffmpeg" and step != 1:
        raise SystemExit("--input-mode ffmpeg requires --step 1.")

    if io_workers < 1:
        raise SystemExit("--io-workers must be >= 1.")

    xp = _xp(gpu)

    if input_mode == "ffmpeg":
        hdr = frames[0].read_bytes()[:26]
        w = int.from_bytes(hdr[18:22], 'little')
        h = int.from_bytes(hdr[22:26], 'little')
        pattern, _ = infer_sequence_pattern(frames[0])
        start_number = frame_key(frames[0])
        count = len(frames)
        hwaccel = _hwaccel_decode_flags() if gpu else []
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            *hwaccel,
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
            prev_raw = xp.asarray(
                np.frombuffer(proc.stdout.read(frame_bytes), dtype=np.uint8).reshape((h, w))
            )
            if avg_window > 1:
                buf = deque(maxlen=avg_window)
                rsum = xp.zeros((h, w), dtype=xp.float32)
                f = prev_raw.astype(xp.float32)
                buf.append(f)
                rsum += f
                prev = (rsum / len(buf)).astype(xp.uint8)
            else:
                prev = prev_raw
            idle_time = xp.zeros((h, w), dtype=xp.float32) + float(fade_frames)
            if fatigue:
                thresh_arr = xp.full((h, w), float(threshold), dtype=xp.float32)
                fire_freq = xp.zeros((h, w), dtype=xp.float32)
            else:
                thresh_arr = threshold
            for frame_path in frames[1:]:
                if stop_event is not None and stop_event.is_set():
                    break
                data = proc.stdout.read(frame_bytes)
                if len(data) != frame_bytes:
                    raise SystemExit("ffmpeg stream ended early.")
                curr_raw = xp.asarray(np.frombuffer(data, dtype=np.uint8).reshape((h, w)))
                if avg_window > 1:
                    f = curr_raw.astype(xp.float32)
                    if len(buf) == buf.maxlen:
                        rsum -= buf[0]
                    buf.append(f)
                    rsum += f
                    curr = (rsum / len(buf)).astype(xp.uint8)
                else:
                    curr = curr_raw
                moved = detect_movement(curr, prev, thresh_arr)
                if fatigue:
                    update_fatigue(fire_freq, thresh_arr, moved, fps,
                                   target_freq, fatigue_tau, adjust_rate,
                                   min_thresh, max_thresh)
                idle_time = update_idle_time(idle_time, moved)
                out = to_heatmap(idle_time, fade_frames)
                out_name = output_name(frame_path, pad_digits)
                out_path = output_dir / out_name
                Image.fromarray(_asnumpy(out), mode="RGB").save(out_path)
                if on_frame_done is not None:
                    sens_img = None
                    if fatigue:
                        sens_data = threshold_to_heatmap(thresh_arr, min_thresh, max_thresh)
                        sens_img = Image.fromarray(_asnumpy(sens_data), mode="RGB")
                    avg_img = None
                    if avg_window > 1:
                        avg_img = Image.fromarray(_asnumpy(curr), mode="L")
                    on_frame_done(frame_path, out_path, sens_img, avg_img)
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
            prev_raw = xp.asarray(next(iterator))
            if avg_window > 1:
                buf = deque(maxlen=avg_window)
                rsum = xp.zeros_like(prev_raw, dtype=xp.float32)
                f = prev_raw.astype(xp.float32)
                buf.append(f)
                rsum += f
                prev = (rsum / len(buf)).astype(xp.uint8)
            else:
                prev = prev_raw
            idle_time = xp.zeros_like(prev_raw, dtype=xp.float32) + float(fade_frames)
            if fatigue:
                thresh_arr = xp.full_like(prev_raw, float(threshold), dtype=xp.float32)
                fire_freq = xp.zeros_like(prev_raw, dtype=xp.float32)
            else:
                thresh_arr = threshold
            for frame_path, curr_np in zip(frames[1:], iterator):
                if stop_event is not None and stop_event.is_set():
                    break
                curr_raw = xp.asarray(curr_np)
                if avg_window > 1:
                    f = curr_raw.astype(xp.float32)
                    if len(buf) == buf.maxlen:
                        rsum -= buf[0]
                    buf.append(f)
                    rsum += f
                    curr = (rsum / len(buf)).astype(xp.uint8)
                else:
                    curr = curr_raw
                moved = detect_movement(curr, prev, thresh_arr)
                if fatigue:
                    update_fatigue(fire_freq, thresh_arr, moved, fps,
                                   target_freq, fatigue_tau, adjust_rate,
                                   min_thresh, max_thresh)
                idle_time = update_idle_time(idle_time, moved)
                out = to_heatmap(idle_time, fade_frames)
                out_name = output_name(frame_path, pad_digits)
                out_path = output_dir / out_name
                Image.fromarray(_asnumpy(out), mode="RGB").save(out_path)
                if on_frame_done is not None:
                    sens_img = None
                    if fatigue:
                        sens_data = threshold_to_heatmap(thresh_arr, min_thresh, max_thresh)
                        sens_img = Image.fromarray(_asnumpy(sens_data), mode="RGB")
                    avg_img = None
                    if avg_window > 1:
                        avg_img = Image.fromarray(_asnumpy(curr), mode="L")
                    on_frame_done(frame_path, out_path, sens_img, avg_img)
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
    parser.add_argument(
        "--avg-window",
        type=int,
        default=1,
        help="Running average window size (1 = off).",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU-accelerated ffmpeg decode/encode.",
    )
    parser.add_argument(
        "--no-fatigue",
        action="store_true",
        help="Disable per-pixel adaptive sensitivity.",
    )
    parser.add_argument(
        "--target-freq",
        type=float,
        default=1.0,
        help="Target firing rate per pixel in Hz (default 1.0).",
    )
    parser.add_argument(
        "--fatigue-tau",
        type=float,
        default=2.0,
        help="EMA time constant in seconds for frequency tracking (default 2.0).",
    )
    parser.add_argument(
        "--adjust-rate",
        type=float,
        default=0.5,
        help="How aggressively thresholds shift per frame (default 0.5).",
    )
    parser.add_argument(
        "--min-thresh",
        type=float,
        default=2.0,
        help="Minimum per-pixel threshold (default 2).",
    )
    parser.add_argument(
        "--max-thresh",
        type=float,
        default=200.0,
        help="Maximum per-pixel threshold (default 200).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    gpu = not args.no_gpu
    use_fatigue = not args.no_fatigue
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
        gpu=gpu,
        avg_window=args.avg_window,
        fatigue=use_fatigue,
        target_freq=args.target_freq,
        fatigue_tau=args.fatigue_tau,
        adjust_rate=args.adjust_rate,
        min_thresh=args.min_thresh,
        max_thresh=args.max_thresh,
    )


if __name__ == "__main__":
    main()

"""GUI for Pixel Changes Glow Visualizer – full video-to-video pipeline."""

import json
import shutil
import subprocess
import tempfile
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

from motion_ghosting import process_frames


# ── ffmpeg helpers ────────────────────────────────────────────────────

def _ffmpeg_available():
    return shutil.which("ffmpeg") is not None


def _ffprobe_fps(video_path):
    """Detect the frame rate of a video file using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "json",
                str(video_path),
            ],
            capture_output=True, text=True, timeout=15,
        )
        info = json.loads(result.stdout)
        rate = info["streams"][0]["r_frame_rate"]
        num, den = rate.split("/")
        return round(int(num) / int(den), 3)
    except Exception:
        return None


def _extract_frames(video_path, frames_dir, log):
    """Extract video → PNG frames."""
    log("Extracting frames from video…")
    frames_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-vsync", "0",
        str(frames_dir / "frame_%06d.png"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg extract failed:\n{proc.stderr}")


def _stitch_video(processed_dir, output_video, fps, log):
    """Stitch processed PNG frames → output video."""
    log("Stitching output video…")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y",
        "-framerate", str(fps),
        "-i", str(processed_dir / "output_%06d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(output_video),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg stitch failed:\n{proc.stderr}")


# ── GUI ───────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pixel Changes Glow Visualizer")
        self.resizable(False, False)

        self._build_ui()
        self._center_window()

    # ── UI construction ──────────────────────────────────────────────

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}
        row = 0

        # ── Input / Output ───────────────────────────────────────────
        io_frame = ttk.LabelFrame(self, text="Input / Output", padding=8)
        io_frame.grid(row=row, column=0, sticky="ew", **pad)
        io_frame.columnconfigure(1, weight=1)
        row += 1

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()

        ttk.Label(io_frame, text="Input video:").grid(row=0, column=0, sticky="w")
        ttk.Entry(io_frame, textvariable=self.input_var, width=50).grid(row=0, column=1, padx=4, sticky="ew")
        ttk.Button(io_frame, text="Browse…", command=self._browse_input).grid(row=0, column=2)

        ttk.Label(io_frame, text="Output video:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(io_frame, textvariable=self.output_var, width=50).grid(row=1, column=1, padx=4, sticky="ew", pady=(4, 0))
        ttk.Button(io_frame, text="Browse…", command=self._browse_output).grid(row=1, column=2, pady=(4, 0))

        # ── Parameters ───────────────────────────────────────────────
        param_frame = ttk.LabelFrame(self, text="Parameters", padding=8)
        param_frame.grid(row=row, column=0, sticky="ew", **pad)
        row += 1

        self.threshold_var = tk.IntVar(value=12)
        self.fps_var = tk.DoubleVar(value=30.0)
        self.fade_var = tk.DoubleVar(value=3.0)
        self.workers_var = tk.IntVar(value=4)
        self.mode_var = tk.StringVar(value="ffmpeg")
        self.start_var = tk.IntVar(value=0)
        self.end_var = tk.StringVar(value="")
        self.step_var = tk.IntVar(value=1)

        params = [
            ("Threshold:", self.threshold_var, "Movement detection sensitivity (0-255)"),
            ("FPS:", self.fps_var, "Auto-detected from video, or set manually"),
            ("Fade seconds:", self.fade_var, "Time for static pixels to fully fade"),
            ("IO workers:", self.workers_var, "Parallel image decode workers"),
            ("Start frame:", self.start_var, "Start frame index (inclusive)"),
            ("End frame:", self.end_var, "End frame index (exclusive, blank = all)"),
            ("Step:", self.step_var, "Process every Nth frame"),
        ]

        for i, (label, var, tip) in enumerate(params):
            ttk.Label(param_frame, text=label).grid(row=i, column=0, sticky="w")
            entry = ttk.Entry(param_frame, textvariable=var, width=12)
            entry.grid(row=i, column=1, padx=4)
            ttk.Label(param_frame, text=tip, foreground="gray").grid(
                row=i, column=2, sticky="w", padx=(8, 0)
            )

        mode_row = len(params)
        ttk.Label(param_frame, text="Input mode:").grid(row=mode_row, column=0, sticky="w")
        ttk.Combobox(
            param_frame, textvariable=self.mode_var,
            values=["pil", "ffmpeg"], state="readonly", width=10,
        ).grid(row=mode_row, column=1, padx=4)
        ttk.Label(
            param_frame, text="Frame decode backend (ffmpeg requires step=1)",
            foreground="gray",
        ).grid(row=mode_row, column=2, sticky="w", padx=(8, 0))

        # ── Progress ─────────────────────────────────────────────────
        prog_frame = ttk.Frame(self, padding=(8, 0, 8, 4))
        prog_frame.grid(row=row, column=0, sticky="ew")
        row += 1

        self.progress = ttk.Progressbar(prog_frame, mode="indeterminate", length=300)
        self.progress.pack(fill="x")

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(prog_frame, textvariable=self.status_var).pack(anchor="w", pady=(2, 0))

        # ── Buttons ──────────────────────────────────────────────────
        btn_frame = ttk.Frame(self, padding=8)
        btn_frame.grid(row=row, column=0, sticky="e")

        self.run_btn = ttk.Button(btn_frame, text="Run", command=self._on_run)
        self.run_btn.pack(side="right", padx=(4, 0))
        ttk.Button(btn_frame, text="Quit", command=self.destroy).pack(side="right")

    # ── Helpers ───────────────────────────────────────────────────────

    def _center_window(self):
        self.update_idletasks()
        w, h = self.winfo_width(), self.winfo_height()
        x = (self.winfo_screenwidth() - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f"+{x}+{y}")

    def _browse_input(self):
        path = filedialog.askopenfilename(
            title="Select input video",
            filetypes=[
                ("Video files", "*.mp4 *.mkv *.avi *.mov *.webm *.flv *.wmv"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.input_var.set(path)
            # Auto-set output next to input
            p = Path(path)
            self.output_var.set(str(p.with_name(p.stem + "_glow.mp4")))
            # Auto-detect FPS
            fps = _ffprobe_fps(path)
            if fps:
                self.fps_var.set(fps)

    def _browse_output(self):
        path = filedialog.asksaveasfilename(
            title="Save output video as",
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4"), ("All files", "*.*")],
        )
        if path:
            self.output_var.set(path)

    def _log(self, msg):
        self.after(0, self.status_var.set, msg)

    # ── Pipeline ──────────────────────────────────────────────────────

    def _on_run(self):
        input_video = self.input_var.get().strip()
        output_video = self.output_var.get().strip()

        if not input_video or not Path(input_video).is_file():
            messagebox.showerror("Error", "Input video not found.")
            return
        if not output_video:
            messagebox.showerror("Error", "Please specify an output video path.")
            return
        if not _ffmpeg_available():
            messagebox.showerror(
                "Error",
                "ffmpeg was not found on your PATH.\n"
                "Install ffmpeg and make sure it is accessible from the command line.",
            )
            return

        end_text = self.end_var.get().strip()
        end_val = int(end_text) if end_text else None

        try:
            threshold = self.threshold_var.get()
            fps = self.fps_var.get()
            fade = self.fade_var.get()
            workers = self.workers_var.get()
            start = self.start_var.get()
            step = self.step_var.get()
            mode = self.mode_var.get()
        except tk.TclError:
            messagebox.showerror("Error", "Invalid parameter value — check your inputs.")
            return

        self.run_btn.config(state="disabled")
        self.status_var.set("Starting…")
        self.progress.start(10)

        def worker():
            tmp = Path(tempfile.mkdtemp(prefix="glow_"))
            frames_dir = tmp / "frames"
            processed_dir = tmp / "processed"
            try:
                # Step 1 — extract frames
                _extract_frames(input_video, frames_dir, self._log)

                # Step 2 — process frames
                self._log("Processing frames…")
                process_frames(
                    input_dir=str(frames_dir),
                    output_dir=str(processed_dir),
                    threshold=threshold,
                    fps=fps,
                    fade_seconds=fade,
                    start=start,
                    end=end_val,
                    step=step,
                    pad_digits=6,
                    io_workers=workers,
                    input_mode=mode,
                )

                # Step 3 — stitch output video
                _stitch_video(processed_dir, output_video, fps, self._log)

                self.after(0, self._done, None)
            except Exception as exc:
                self.after(0, self._done, exc)
            finally:
                # Clean up temp directory
                shutil.rmtree(tmp, ignore_errors=True)

        threading.Thread(target=worker, daemon=True).start()

    def _done(self, error):
        self.progress.stop()
        self.run_btn.config(state="normal")
        if error:
            self.status_var.set("Error!")
            messagebox.showerror("Processing failed", str(error))
        else:
            self.status_var.set("Done!")
            messagebox.showinfo(
                "Success",
                f"Output video saved to:\n{self.output_var.get()}",
            )


if __name__ == "__main__":
    App().mainloop()

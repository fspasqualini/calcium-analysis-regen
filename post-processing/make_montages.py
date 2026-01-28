#!/usr/bin/env python3
"""Generate montage videos and calcium transient plots for long-batch movies."""

import argparse
import csv
import json
import math
import os
import sys
import zipfile
from glob import glob

import numpy as np
import tifffile
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter
import matplotlib.cm as cm
import matplotlib.pyplot as plt


COLOR_RAW = "#000000"
COLOR_FAST = "#1f77b4"
COLOR_DEEPCAD = "#ff7f0e"
COLOR_TED = "#2ca02c"


def iter_tiff_frames(path):
    with tifffile.TiffFile(path) as tif:
        if len(tif.pages) > 1:
            for page in tif.pages:
                yield page.asarray()
            return

        arr = tif.pages[0].asarray()
        if arr.ndim != 3:
            raise ValueError(f"Unexpected TIFF shape for {path}: {arr.shape}")

        # Choose time axis by largest dimension (frames ~1000 vs 680 spatial)
        t_axis = int(np.argmax(arr.shape))
        if t_axis == 0:
            for i in range(arr.shape[0]):
                yield arr[i]
        elif t_axis == 2:
            for i in range(arr.shape[2]):
                yield arr[:, :, i]
        else:
            for i in range(arr.shape[1]):
                yield arr[:, i, :]


def get_frame_count_and_shape(path):
    with tifffile.TiffFile(path) as tif:
        if len(tif.pages) > 1:
            first = tif.pages[0].asarray()
            return len(tif.pages), first.shape
        arr = tif.pages[0].asarray()
        if arr.ndim != 3:
            raise ValueError(f"Unexpected TIFF shape for {path}: {arr.shape}")
        t_axis = int(np.argmax(arr.shape))
        if t_axis == 0:
            return arr.shape[0], arr.shape[1:]
        if t_axis == 2:
            return arr.shape[2], arr.shape[:2]
        return arr.shape[1], (arr.shape[0], arr.shape[2])


def min_max_tiff(path, bg_sigma=None):
    min_val = None
    max_val = None
    for frame in iter_tiff_frames(path):
        frame = frame.astype(np.float32)
        if bg_sigma is not None:
            bg = gaussian_filter(frame, sigma=bg_sigma)
            frame = frame - bg
            frame[frame < 0] = 0
        frame_min = float(np.min(frame))
        frame_max = float(np.max(frame))
        if min_val is None or frame_min < min_val:
            min_val = frame_min
        if max_val is None or frame_max > max_val:
            max_val = frame_max
    if min_val is None or max_val is None:
        raise ValueError(f"No frames found in {path}")
    return min_val, max_val


def build_colormap_lut(name="plasma"):
    cmap = cm.get_cmap(name, 256)
    lut = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    return lut


def normalize_frame(frame, p_low, p_high, lut):
    if p_high <= p_low:
        return np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    scaled = (frame.astype(np.float32) - p_low) / (p_high - p_low)
    scaled = np.clip(scaled, 0.0, 1.0)
    idx = (scaled * 255.0).astype(np.uint8)
    return lut[idx]


def choose_rois(height, width, roi_size, count, seed=0):
    rng = np.random.default_rng(seed)
    x0 = int(width * 0.25)
    x1 = int(width * 0.75) - roi_size
    y0 = int(height * 0.25)
    y1 = int(height * 0.75) - roi_size
    if x1 <= x0 or y1 <= y0:
        raise ValueError("ROI selection bounds are invalid for the given size")

    rois = []
    for _ in range(count):
        x = int(rng.integers(x0, x1 + 1))
        y = int(rng.integers(y0, y1 + 1))
        rois.append({"x": x, "y": y, "w": roi_size, "h": roi_size})
    return rois


def compute_traces_and_write_mp4(path, out_mp4, rois, p_low, p_high, fps, lut, bg_sigma=None):
    global_trace = []
    roi_traces = [[] for _ in rois]

    writer = imageio.get_writer(
        out_mp4,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        ffmpeg_params=["-crf", "18", "-preset", "slow"],
        macro_block_size=1,
    )

    for frame in iter_tiff_frames(path):
        frame = frame.astype(np.float32)
        if bg_sigma is not None:
            bg = gaussian_filter(frame, sigma=bg_sigma)
            frame = frame - bg
            frame[frame < 0] = 0
        global_trace.append(frame.mean())
        for idx, roi in enumerate(rois):
            y0, y1 = roi["y"], roi["y"] + roi["h"]
            x0, x1 = roi["x"], roi["x"] + roi["w"]
            roi_traces[idx].append(frame[y0:y1, x0:x1].mean())

        vis_rgb = normalize_frame(frame, p_low, p_high, lut)
        writer.append_data(vis_rgb)

    writer.close()
    return np.asarray(global_trace), np.asarray(roi_traces)


def dff(trace, eps=1e-6):
    f0 = np.percentile(trace, 10)
    denom = f0 if abs(f0) > eps else eps
    return (trace - f0) / denom


def write_global_csv(path, frames_ms, traces):
    fieldnames = ["time_ms"] + list(traces.keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, t in enumerate(frames_ms):
            row = {"time_ms": t}
            for key, values in traces.items():
                row[key] = float(values[i])
            writer.writerow(row)


def write_roi_csv(path, frames_ms, roi_traces):
    fieldnames = ["time_ms", "roi_id"] + list(roi_traces.keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        roi_count = next(iter(roi_traces.values())).shape[0]
        for roi_id in range(roi_count):
            for i, t in enumerate(frames_ms):
                row = {"time_ms": t, "roi_id": roi_id}
                for key, values in roi_traces.items():
                    row[key] = float(values[roi_id, i])
                writer.writerow(row)


def plot_global_dff(out_path, frames_ms, traces, title):
    plt.figure(figsize=(10, 4))
    for label, trace, color in traces:
        plt.plot(frames_ms, dff(trace), label=label, color=color, linewidth=1.5)
    plt.xlabel("Time (ms)")
    plt.ylabel("ΔF/F0")
    plt.title(title)
    plt.legend(loc="upper right", ncol=2)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_roi_dff(out_path, frames_ms, roi_traces, title):
    plt.figure(figsize=(10, 4))
    for label, traces, color in roi_traces:
        dff_traces = np.asarray([dff(t) for t in traces])
        mean = dff_traces.mean(axis=0)
        std = dff_traces.std(axis=0)
        plt.plot(frames_ms, mean, label=label, color=color, linewidth=1.5)
        plt.fill_between(frames_ms, mean - std, mean + std, color=color, alpha=0.2)
    plt.xlabel("Time (ms)")
    plt.ylabel("ΔF/F0")
    plt.title(title)
    plt.legend(loc="upper right", ncol=2)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def baseline_std(trace, quantile=20):
    cutoff = np.percentile(trace, quantile)
    low = trace[trace <= cutoff]
    if low.size == 0:
        return float(np.std(trace))
    return float(np.std(low))


def snr_proxy(trace):
    low = np.percentile(trace, 10)
    high = np.percentile(trace, 95)
    sigma = baseline_std(trace)
    if sigma == 0:
        return 0.0
    return float((high - low) / sigma)


def peak_timing_shift(ref, target):
    ref = ref - np.mean(ref)
    target = target - np.mean(target)
    corr = np.correlate(target, ref, mode="full")
    lag = int(np.argmax(corr) - (len(ref) - 1))
    return lag


def shape_correlation(ref, target):
    if np.std(ref) == 0 or np.std(target) == 0:
        return 0.0
    return float(np.corrcoef(ref, target)[0, 1])


def plot_metrics(out_path, metrics):
    labels = ["FAST", "DeepCAD-RT", "TeD"]
    noise = [metrics[k]["baseline_std"] for k in labels]
    snr = [metrics[k]["snr"] for k in labels]
    lag = [metrics[k]["timing_shift_ms"] for k in labels]
    corr = [metrics[k]["shape_corr"] for k in labels]

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes[0, 0].bar(labels, noise, color=[COLOR_FAST, COLOR_DEEPCAD, COLOR_TED])
    axes[0, 0].set_title("Baseline noise (lower is better)")
    axes[0, 0].set_ylabel("Std of low-activity ΔF/F0")

    axes[0, 1].bar(labels, snr, color=[COLOR_FAST, COLOR_DEEPCAD, COLOR_TED])
    axes[0, 1].set_title("SNR proxy (higher is better)")
    axes[0, 1].set_ylabel("(P95-P10)/σ")

    axes[1, 0].bar(labels, lag, color=[COLOR_FAST, COLOR_DEEPCAD, COLOR_TED])
    axes[1, 0].set_title("Timing shift vs raw")
    axes[1, 0].set_ylabel("Lag (ms)")

    axes[1, 1].bar(labels, corr, color=[COLOR_FAST, COLOR_DEEPCAD, COLOR_TED])
    axes[1, 1].set_title("Shape correlation vs raw")
    axes[1, 1].set_ylabel("Pearson r")

    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def draw_text_with_outline(draw, position, text, font, fill=(255, 255, 255), outline=(0, 0, 0)):
    x, y = position
    for dx in (-1, 1, 0, 0):
        for dy in (-1, 1, 0, 0):
            draw.text((x + dx, y + dy), text, font=font, fill=outline)
    draw.text(position, text, font=font, fill=fill)


def load_font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def add_timestamp(frame, ms):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    font = load_font(48)
    text = f"t = {ms:.0f} ms"
    draw_text_with_outline(draw, (8, 8), text, font)
    return np.asarray(img)


def add_scalebar(frame, microns=250, fov_microns=1500):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    h, w = frame.shape[:2]
    bar_len = int(round(w * (microns / fov_microns) * 4))
    bar_height = max(4, h // 200) * 4
    pad = 12

    bar_len = min(bar_len, w - 2 * pad)
    x1 = w - pad
    x0 = x1 - bar_len
    y1 = h - pad
    y0 = y1 - bar_height

    draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255))
    draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 0))
    return np.asarray(img)


def make_montage(raw_mp4, fast_mp4, deepcad_mp4, ted_mp4, out_mp4, fps):
    readers = [
        imageio.get_reader(raw_mp4),
        imageio.get_reader(fast_mp4),
        imageio.get_reader(deepcad_mp4),
        imageio.get_reader(ted_mp4),
    ]
    writer = imageio.get_writer(
        out_mp4,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        ffmpeg_params=["-crf", "18", "-preset", "slow"],
        macro_block_size=1,
    )

    frame_idx = 0
    while True:
        try:
            raw = readers[0].get_next_data()
            fast = readers[1].get_next_data()
            deepcad = readers[2].get_next_data()
            ted = readers[3].get_next_data()
        except Exception:
            break

        raw = add_timestamp(raw, frame_idx * 10)
        ted = add_scalebar(ted)

        h, w = raw.shape[:2]
        montage = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        montage[0:h, 0:w] = raw
        montage[0:h, w:2 * w] = fast
        montage[h:2 * h, 0:w] = deepcad
        montage[h:2 * h, w:2 * w] = ted

        writer.append_data(montage)
        frame_idx += 1

    for reader in readers:
        reader.close()
    writer.close()


def make_grid_montage(rows, out_mp4, fps):
    readers = []
    for row in rows:
        readers.append([imageio.get_reader(p) for p in row])

    writer = imageio.get_writer(
        out_mp4,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        ffmpeg_params=["-crf", "18", "-preset", "slow"],
        macro_block_size=1,
    )

    while True:
        try:
            row_frames = []
            for row in readers:
                frames = [r.get_next_data() for r in row]
                row_frames.append(np.concatenate(frames, axis=1))
            grid = np.concatenate(row_frames, axis=0)
        except Exception:
            break

        writer.append_data(grid)

    for row in readers:
        for r in row:
            r.close()
    writer.close()


def extract_fast_file(raw_name, zip_paths, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, raw_name)
    if os.path.exists(out_path):
        return out_path

    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                if member.endswith(f"/{raw_name}") or member.endswith(f"_{raw_name}"):
                    zf.extract(member, cache_dir)
                    extracted = os.path.join(cache_dir, member)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    os.replace(extracted, out_path)
                    return out_path

    return None


def find_deepcad_file(raw_name, deepcad_dir):
    stem = os.path.splitext(raw_name)[0]
    pattern = os.path.join(deepcad_dir, f"{stem}_E_10_Iter_6350_output.tif")
    matches = glob(pattern)
    return matches[0] if matches else None


def main():
    parser = argparse.ArgumentParser(description="Generate montage videos and calcium transient plots.")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--fast-zip-glob", required=True)
    parser.add_argument("--deepcad-dir", required=True)
    parser.add_argument("--ted-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--roi-size", type=int, default=32)
    parser.add_argument("--roi-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", help="Process a single raw filename (e.g., FGF2-G3_9.5110.tif)")
    parser.add_argument("--bg-sigma", type=float, default=30, help="Gaussian sigma for background subtraction")
    args = parser.parse_args()

    raw_files = sorted(glob(os.path.join(args.raw_dir, "*.tif")))
    if not raw_files:
        print(f"No raw files found in {args.raw_dir}", file=sys.stderr)
        return 1
    if args.only:
        raw_files = [p for p in raw_files if os.path.basename(p) == args.only]
        if not raw_files:
            print(f"No raw files match {args.only}", file=sys.stderr)
            return 1

    fast_zips = sorted(glob(args.fast_zip_glob))
    if not fast_zips:
        print(f"No FAST zip files found for {args.fast_zip_glob}", file=sys.stderr)
        return 1

    deepcad_dirs = glob(args.deepcad_dir)
    if not deepcad_dirs:
        print(f"No DeepCAD-RT folder found for {args.deepcad_dir}", file=sys.stderr)
        return 1
    deepcad_dir = sorted(deepcad_dirs)[0]

    os.makedirs(args.out_dir, exist_ok=True)
    fast_cache = os.path.join(args.out_dir, "_cache", "fast_phase1_default")
    aggregate_dir = os.path.join(args.out_dir, "aggregate")
    os.makedirs(aggregate_dir, exist_ok=True)

    aggregate_global = []
    aggregate_roi = []
    per_movie_metrics = []
    aggregate_montage_rows = []

    for raw_path in raw_files:
        raw_name = os.path.basename(raw_path)
        stem = os.path.splitext(raw_name)[0]
        print(f"Processing {raw_name}...")

        fast_path = extract_fast_file(raw_name, fast_zips, fast_cache)
        deepcad_path = find_deepcad_file(raw_name, deepcad_dir)
        ted_path = os.path.join(args.ted_dir, raw_name)

        missing = [
            name
            for name, path in [
                ("FAST", fast_path),
                ("DeepCAD-RT", deepcad_path),
                ("TeD", ted_path),
            ]
            if path is None or not os.path.exists(path)
        ]
        if missing:
            print(f"Skipping {raw_name}: missing {', '.join(missing)}", file=sys.stderr)
            continue

        out_movie_dir = os.path.join(args.out_dir, stem)
        os.makedirs(out_movie_dir, exist_ok=True)

        frames, shape = get_frame_count_and_shape(raw_path)
        height, width = shape
        rois = choose_rois(height, width, args.roi_size, args.roi_count, seed=args.seed)
        with open(os.path.join(out_movie_dir, "rois.json"), "w") as f:
            json.dump({"rois": rois}, f, indent=2)

        version_paths = {
            "Raw": raw_path,
            "FAST": fast_path,
            "DeepCAD-RT": deepcad_path,
            "TeD": ted_path,
        }

        traces = {}
        roi_traces = {}
        mp4_paths = {}

        lut = build_colormap_lut("plasma")

        min_vals = {}
        max_vals = {}
        for label, path in version_paths.items():
            print(f"  Scanning {label} for min/max...")
            min_vals[label], max_vals[label] = min_max_tiff(path)

        global_min = min(min_vals.values())
        global_max = max(max_vals.values())

        for label, path in version_paths.items():
            print(f"  Encoding {label}...")
            mp4_path = os.path.join(out_movie_dir, f"{label.lower().replace('-', '')}.mp4")
            global_trace, roi_trace = compute_traces_and_write_mp4(
                path, mp4_path, rois, global_min, global_max, args.fps, lut
            )
            traces[label] = global_trace
            roi_traces[label] = roi_trace
            mp4_paths[label] = mp4_path

        bg_min_vals = {}
        bg_max_vals = {}
        for label, path in version_paths.items():
            if label == "Raw":
                continue
            print(f"  Scanning {label} (BG) for min/max...")
            bg_min_vals[label], bg_max_vals[label] = min_max_tiff(path, bg_sigma=args.bg_sigma)

        bg_global_min = min(bg_min_vals.values())
        bg_global_max = max(bg_max_vals.values())

        bg_traces = {}
        bg_roi_traces = {}
        bg_mp4_paths = {}
        for label, path in version_paths.items():
            if label == "Raw":
                continue
            print(f"  Encoding {label} (BG)...")
            mp4_path = os.path.join(out_movie_dir, f"{label.lower().replace('-', '')}_bg.mp4")
            global_trace, roi_trace = compute_traces_and_write_mp4(
                path, mp4_path, rois, bg_global_min, bg_global_max, args.fps, lut, bg_sigma=args.bg_sigma
            )
            bg_traces[label] = global_trace
            bg_roi_traces[label] = roi_trace
            bg_mp4_paths[label] = mp4_path

        montage_path = os.path.join(out_movie_dir, "montage.mp4")
        print("  Building montage...")
        make_montage(
            mp4_paths["Raw"],
            mp4_paths["FAST"],
            mp4_paths["DeepCAD-RT"],
            mp4_paths["TeD"],
            montage_path,
            args.fps,
        )

        montage_bg_path = os.path.join(out_movie_dir, "montage_bg.mp4")
        print("  Building montage (BG)...")
        make_montage(
            mp4_paths["Raw"],
            bg_mp4_paths["FAST"],
            bg_mp4_paths["DeepCAD-RT"],
            bg_mp4_paths["TeD"],
            montage_bg_path,
            args.fps,
        )

        # 10 ms per frame is fixed by requirement, not fps (fps controls playback rate)
        min_len = min(len(t) for t in traces.values())
        frames_ms = np.arange(min_len) * 10

        plot_global_dff(
            os.path.join(out_movie_dir, "global_dff.pdf"),
            frames_ms,
            [
                ("Raw", traces["Raw"][:min_len], COLOR_RAW),
                ("FAST", traces["FAST"][:min_len], COLOR_FAST),
                ("DeepCAD-RT", traces["DeepCAD-RT"][:min_len], COLOR_DEEPCAD),
                ("TeD", traces["TeD"][:min_len], COLOR_TED),
            ],
            f"Global ΔF/F0 - {raw_name}",
        )

        plot_roi_dff(
            os.path.join(out_movie_dir, "roi_dff.pdf"),
            frames_ms,
            [
                ("Raw", roi_traces["Raw"][:, :min_len], COLOR_RAW),
                ("FAST", roi_traces["FAST"][:, :min_len], COLOR_FAST),
                ("DeepCAD-RT", roi_traces["DeepCAD-RT"][:, :min_len], COLOR_DEEPCAD),
                ("TeD", roi_traces["TeD"][:, :min_len], COLOR_TED),
            ],
            f"ROI ΔF/F0 (mean ± std) - {raw_name}",
        )

        global_csv = os.path.join(out_movie_dir, "global_dff.csv")
        roi_csv = os.path.join(out_movie_dir, "roi_dff.csv")
        trace_dict = {
            "raw": dff(traces["Raw"][:min_len]),
            "fast": dff(traces["FAST"][:min_len]),
            "deepcadrt": dff(traces["DeepCAD-RT"][:min_len]),
            "ted": dff(traces["TeD"][:min_len]),
            "fast_bg": dff(bg_traces["FAST"][:min_len]),
            "deepcadrt_bg": dff(bg_traces["DeepCAD-RT"][:min_len]),
            "ted_bg": dff(bg_traces["TeD"][:min_len]),
        }
        write_global_csv(global_csv, frames_ms, trace_dict)

        roi_trace_dict = {
            "raw": np.asarray([dff(t) for t in roi_traces["Raw"][:, :min_len]]),
            "fast": np.asarray([dff(t) for t in roi_traces["FAST"][:, :min_len]]),
            "deepcadrt": np.asarray([dff(t) for t in roi_traces["DeepCAD-RT"][:, :min_len]]),
            "ted": np.asarray([dff(t) for t in roi_traces["TeD"][:, :min_len]]),
            "fast_bg": np.asarray([dff(t) for t in bg_roi_traces["FAST"][:, :min_len]]),
            "deepcadrt_bg": np.asarray([dff(t) for t in bg_roi_traces["DeepCAD-RT"][:, :min_len]]),
            "ted_bg": np.asarray([dff(t) for t in bg_roi_traces["TeD"][:, :min_len]]),
        }
        write_roi_csv(roi_csv, frames_ms, roi_trace_dict)

        aggregate_global.append(
            {
                "movie": raw_name,
                "frames_ms": frames_ms,
                "traces": trace_dict,
            }
        )
        aggregate_roi.append(
            {
                "movie": raw_name,
                "frames_ms": frames_ms,
                "roi_traces": roi_trace_dict,
            }
        )

        ref = trace_dict["raw"]
        metrics = {}
        for label, key in [("FAST", "fast_bg"), ("DeepCAD-RT", "deepcadrt_bg"), ("TeD", "ted_bg")]:
            target = trace_dict[key]
            metrics[label] = {
                "baseline_std": baseline_std(target),
                "snr": snr_proxy(target),
                "timing_shift_ms": peak_timing_shift(ref, target) * 10,
                "shape_corr": shape_correlation(ref, target),
            }
        per_movie_metrics.append(metrics)

        aggregate_montage_rows.append(
            [mp4_paths["Raw"], bg_mp4_paths["FAST"], bg_mp4_paths["DeepCAD-RT"], bg_mp4_paths["TeD"]]
        )

    if aggregate_global:
        global_csv = os.path.join(aggregate_dir, "global_dff.csv")
        fieldnames = [
            "movie",
            "time_ms",
            "raw",
            "fast",
            "deepcadrt",
            "ted",
            "fast_bg",
            "deepcadrt_bg",
            "ted_bg",
        ]
        with open(global_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in aggregate_global:
                frames_ms = entry["frames_ms"]
                traces = entry["traces"]
                for i, t in enumerate(frames_ms):
                    row = {"movie": entry["movie"], "time_ms": t}
                    for key in fieldnames[2:]:
                        row[key] = float(traces[key][i])
                    writer.writerow(row)

    if aggregate_roi:
        roi_csv = os.path.join(aggregate_dir, "roi_dff.csv")
        fieldnames = [
            "movie",
            "roi_id",
            "time_ms",
            "raw",
            "fast",
            "deepcadrt",
            "ted",
            "fast_bg",
            "deepcadrt_bg",
            "ted_bg",
        ]
        with open(roi_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in aggregate_roi:
                frames_ms = entry["frames_ms"]
                roi_traces = entry["roi_traces"]
                roi_count = roi_traces["raw"].shape[0]
                for roi_id in range(roi_count):
                    for i, t in enumerate(frames_ms):
                        row = {
                            "movie": entry["movie"],
                            "roi_id": roi_id,
                            "time_ms": t,
                        }
                        for key in fieldnames[3:]:
                            row[key] = float(roi_traces[key][roi_id, i])
                        writer.writerow(row)

    if per_movie_metrics:
        metrics_avg = {"FAST": {}, "DeepCAD-RT": {}, "TeD": {}}
        for label in metrics_avg:
            for key in ["baseline_std", "snr", "timing_shift_ms", "shape_corr"]:
                metrics_avg[label][key] = float(
                    np.mean([m[label][key] for m in per_movie_metrics])
                )
        plot_metrics(os.path.join(aggregate_dir, "aggregate_metrics.pdf"), metrics_avg)

    if aggregate_montage_rows:
        make_grid_montage(
            aggregate_montage_rows,
            os.path.join(aggregate_dir, "montage_5x4.mp4"),
            args.fps,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate montage videos and calcium transient plots for long-batch movies."""

import argparse
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


def sample_percentiles(path, seed=0, samples=50, p_low=1, p_high=99):
    frames, _ = get_frame_count_and_shape(path)
    rng = np.random.default_rng(seed)
    if frames <= samples:
        indices = list(range(frames))
    else:
        indices = rng.choice(frames, size=samples, replace=False)
        indices.sort()

    values = []
    with tifffile.TiffFile(path) as tif:
        if len(tif.pages) > 1:
            for idx in indices:
                values.append(tif.pages[idx].asarray().ravel())
        else:
            arr = tif.pages[0].asarray()
            t_axis = int(np.argmax(arr.shape))
            if t_axis == 0:
                for idx in indices:
                    values.append(arr[idx].ravel())
            elif t_axis == 2:
                for idx in indices:
                    values.append(arr[:, :, idx].ravel())
            else:
                for idx in indices:
                    values.append(arr[:, idx, :].ravel())

    sample = np.concatenate(values)
    return np.percentile(sample, p_low), np.percentile(sample, p_high)


def normalize_frame(frame, p_low, p_high):
    if p_high <= p_low:
        return np.zeros_like(frame, dtype=np.uint8)
    scaled = (frame.astype(np.float32) - p_low) / (p_high - p_low)
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


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


def compute_traces_and_write_mp4(path, out_mp4, rois, p_low, p_high, fps):
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
        global_trace.append(frame.mean())
        for idx, roi in enumerate(rois):
            y0, y1 = roi["y"], roi["y"] + roi["h"]
            x0, x1 = roi["x"], roi["x"] + roi["w"]
            roi_traces[idx].append(frame[y0:y1, x0:x1].mean())

        vis = normalize_frame(frame, p_low, p_high)
        vis_rgb = np.stack([vis, vis, vis], axis=-1)
        writer.append_data(vis_rgb)

    writer.close()
    return np.asarray(global_trace), np.asarray(roi_traces)


def dff(trace, eps=1e-6):
    f0 = np.percentile(trace, 10)
    denom = f0 if abs(f0) > eps else eps
    return (trace - f0) / denom


def plot_global_dff(out_png, frames_ms, traces, title):
    plt.figure(figsize=(10, 4))
    for label, trace, color in traces:
        plt.plot(frames_ms, dff(trace), label=label, color=color, linewidth=1.5)
    plt.xlabel("Time (ms)")
    plt.ylabel("ΔF/F0")
    plt.title(title)
    plt.legend(loc="upper right", ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_roi_dff(out_png, frames_ms, roi_traces, title):
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
    plt.savefig(out_png, dpi=200)
    plt.close()


def draw_text_with_outline(draw, position, text, font, fill=(255, 255, 255), outline=(0, 0, 0)):
    x, y = position
    for dx in (-1, 1, 0, 0):
        for dy in (-1, 1, 0, 0):
            draw.text((x + dx, y + dy), text, font=font, fill=outline)
    draw.text(position, text, font=font, fill=fill)


def add_timestamp(frame, ms):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    text = f"t = {ms:.0f} ms"
    draw_text_with_outline(draw, (8, 8), text, font)
    return np.asarray(img)


def add_scalebar(frame, microns=250, fov_microns=1500):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    h, w = frame.shape[:2]
    bar_len = int(round(w * (microns / fov_microns)))
    bar_height = max(4, h // 200)
    pad = 12

    x1 = w - pad
    x0 = x1 - bar_len
    y1 = h - pad
    y0 = y1 - bar_height

    draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255))
    draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 0))
    label = f"{microns} um"
    draw_text_with_outline(draw, (x0, y0 - 14), label, font)
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

        for label, path in version_paths.items():
            print(f"  Encoding {label}...")
            p_low, p_high = sample_percentiles(path, seed=args.seed)
            mp4_path = os.path.join(out_movie_dir, f"{label.lower().replace('-', '')}.mp4")
            global_trace, roi_trace = compute_traces_and_write_mp4(
                path, mp4_path, rois, p_low, p_high, args.fps
            )
            traces[label] = global_trace
            roi_traces[label] = roi_trace
            mp4_paths[label] = mp4_path

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

        # 10 ms per frame is fixed by requirement, not fps (fps controls playback rate)
        min_len = min(len(t) for t in traces.values())
        frames_ms = np.arange(min_len) * 10

        plot_global_dff(
            os.path.join(out_movie_dir, "global_dff.png"),
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
            os.path.join(out_movie_dir, "roi_dff.png"),
            frames_ms,
            [
                ("Raw", roi_traces["Raw"][:, :min_len], COLOR_RAW),
                ("FAST", roi_traces["FAST"][:, :min_len], COLOR_FAST),
                ("DeepCAD-RT", roi_traces["DeepCAD-RT"][:, :min_len], COLOR_DEEPCAD),
                ("TeD", roi_traces["TeD"][:, :min_len], COLOR_TED),
            ],
            f"ROI ΔF/F0 (mean ± std) - {raw_name}",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

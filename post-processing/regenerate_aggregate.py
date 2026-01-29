#!/usr/bin/env python3
"""Rebuild the 5x4 aggregate montage with timestamp/scalebar overlays."""

import argparse
import os
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont


def load_font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_text_with_outline(draw, position, text, font, fill=(255, 255, 255), outline=(0, 0, 0)):
    x, y = position
    for dx in (-1, 1, 0, 0):
        for dy in (-1, 1, 0, 0):
            draw.text((x + dx, y + dy), text, font=font, fill=outline)
    draw.text(position, text, font=font, fill=fill)


def add_timestamp(frame, ms):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    font = load_font(48)
    draw_text_with_outline(draw, (8, 8), f"t = {ms:.0f} ms", font)
    return np.asarray(img)


def add_scalebar(frame, microns=250, fov_microns=1500):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    h, w = frame.shape[:2]
    bar_len = int(round(w * (microns / fov_microns)))
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, help="post-processing/output")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    movie_dirs = sorted(
        d for d in os.listdir(args.output_root)
        if d.startswith("FGF2-G3_") and os.path.isdir(os.path.join(args.output_root, d))
    )
    if not movie_dirs:
        raise SystemExit("No movie folders found")

    rows = []
    for d in movie_dirs:
        base = os.path.join(args.output_root, d)
        row = [
            os.path.join(base, "raw.mp4"),
            os.path.join(base, "fast_bg.mp4"),
            os.path.join(base, "deepcadrt_bg.mp4"),
            os.path.join(base, "ted_bg.mp4"),
        ]
        rows.append(row)

    readers = []
    for row in rows:
        readers.append([imageio.get_reader(p) for p in row])

    writer = imageio.get_writer(
        args.out,
        fps=args.fps,
        codec="libx264",
        pixelformat="yuv420p",
        ffmpeg_params=["-crf", "18", "-preset", "slow"],
        macro_block_size=1,
    )

    frame_idx = 0
    while True:
        try:
            row_frames = []
            for r, row in enumerate(readers):
                frames = [reader.get_next_data() for reader in row]
                if r == 0:
                    frames[0] = add_timestamp(frames[0], frame_idx * 10)
                if r == len(readers) - 1:
                    frames[-1] = add_scalebar(frames[-1])
                row_frames.append(np.concatenate(frames, axis=1))
            grid = np.concatenate(row_frames, axis=0)
        except Exception:
            break

        writer.append_data(grid)
        frame_idx += 1

    for row in readers:
        for r in row:
            r.close()
    writer.close()


if __name__ == "__main__":
    main()

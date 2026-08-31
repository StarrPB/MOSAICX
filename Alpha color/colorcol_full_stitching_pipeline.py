from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "Colorcol"
PAIR_SPECS = (
    ("column000.png", "column001.png", 34, 72),
    ("column003.png", "column004.png", 20, 118),
    ("column005.png", "column007.png", 52, 170),
    ("column008.png", "column009.png", 12, 118),
    ("column010.png", "column011.png", 26, 125),
    ("column012.png", "column013.png", 31, 108),
)
GROUP_SPECS = (
    ("column000_column001", "column003_column004", 81, 300),
    ("column005_column007", "column008_column009", 77, 283),
    ("column010_column011", "column012_column013", 46, 230),
)


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read {path}")
    return image


def read_rgba(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f"Could not read transparent PNG {path}")
    return image


def specimen_mask(image: np.ndarray, threshold: int = 10, border_crop: int = 5, left_extra_crop: int = 30) -> np.ndarray:
    """Segment the full specimen so its dark internal structures are retained."""
    seed = (image.max(axis=2) > threshold).astype(np.uint8)
    closed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if count <= 1:
        raise ValueError("Could not isolate a specimen from the black background")
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == component).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    if border_crop:
        size = 2 * border_crop + 1
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    for row in mask:
        active = np.flatnonzero(row)
        if active.size:
            row[active[0] : min(active[0] + left_extra_crop, row.size)] = 0
    return mask.astype(np.float32)


def alpha_stitch(first: np.ndarray, second: np.ndarray, dx: int, dy: int, fade_width: float = 26.0) -> tuple[np.ndarray, np.ndarray]:
    """Place second at (dx, dy); alpha blending happens only where both are valid."""
    h1, w1 = first.shape[:2]
    h2, w2 = second.shape[:2]
    min_x, min_y = min(0, dx), min(0, dy)
    max_x, max_y = max(w1, dx + w2), max(h1, dy + h2)
    width, height = max_x - min_x, max_y - min_y
    first_canvas = np.zeros((height, width, 4), dtype=np.uint8)
    second_canvas = np.zeros((height, width, 4), dtype=np.uint8)
    first_canvas[-min_y : -min_y + h1, -min_x : -min_x + w1] = first
    second_canvas[dy - min_y : dy - min_y + h2, dx - min_x : dx - min_x + w2] = second
    first_present = first_canvas[:, :, 3] > 0
    second_present = second_canvas[:, :, 3] > 0
    shared = first_present & second_present
    overlap_left = max(0, dx) - min_x
    overlap_right = min(w1, dx + w2) - min_x
    if overlap_right <= overlap_left:
        raise ValueError("Images have no horizontal overlap")
    x = np.arange(width, dtype=np.float32)[None, :]
    seam = (overlap_left + overlap_right) / 2.0
    first_alpha = np.clip(0.5 - (x - seam) / fade_width, 0.0, 1.0)
    second_alpha = 1.0 - first_alpha
    first_weight = np.where(shared, first_alpha, first_present.astype(np.float32))
    second_weight = np.where(shared, second_alpha, second_present.astype(np.float32))
    total_weight = first_weight + second_weight
    valid = total_weight > 0
    output = np.zeros((height, width, 4), dtype=np.uint8)
    rgb_sum = first_canvas[:, :, :3].astype(np.float32) * first_weight[:, :, None]
    rgb_sum += second_canvas[:, :, :3].astype(np.float32) * second_weight[:, :, None]
    output[:, :, :3][valid] = np.clip(
        np.rint(rgb_sum[valid] / total_weight[valid, None]), 0, 255
    ).astype(np.uint8)
    output[:, :, 3] = np.where(valid, 255, 0).astype(np.uint8)
    source_map = np.zeros((height, width, 3), dtype=np.uint8)
    source_map[valid] = (85, 180, 80)
    source_map[shared] = (40, 190, 250)
    return output, source_map


def rgba_from_rgb_and_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = np.where(mask > 0, 255, 0).astype(np.uint8)
    return rgba


def write_outputs(output_dir: Path, stem: str, rgba: np.ndarray, source_map: np.ndarray) -> Path:
    output = output_dir / f"{stem}.png"
    preview = output_dir / f"{stem}_preview_white.png"
    map_path = output_dir / f"{stem}_sourcemap.png"
    cv2.imwrite(str(output), rgba)
    cv2.imwrite(str(map_path), source_map)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    white = np.full(rgba.shape[:2] + (3,), 255, dtype=np.uint8)
    cv2.imwrite(str(preview), (rgba[:, :, :3] * alpha + white * (1.0 - alpha)).astype(np.uint8))
    return output


def stage_one(output_dir: Path) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    for first_name, second_name, dy, dx in PAIR_SPECS:
        first = read_rgb(INPUT_DIR / first_name)
        second = read_rgb(INPUT_DIR / second_name)
        first_rgba = rgba_from_rgb_and_mask(first, specimen_mask(first))
        second_rgba = rgba_from_rgb_and_mask(second, specimen_mask(second))
        rgba, source_map = alpha_stitch(first_rgba, second_rgba, dx, dy)
        key = f"{Path(first_name).stem}_{Path(second_name).stem}"
        outputs[key] = write_outputs(output_dir, f"colorcol_stitched_{key}_leftcrop30_bordercrop5_straight_alpha", rgba, source_map)
        print(f"stage 1: {key} at (x={dx}, y={dy})")
    return outputs


def stage_two(output_dir: Path, stage_one_outputs: dict[str, Path]) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    for first_key, second_key, dy, dx in GROUP_SPECS:
        rgba, source_map = alpha_stitch(read_rgba(stage_one_outputs[first_key]), read_rgba(stage_one_outputs[second_key]), dx, dy)
        key = f"{first_key}_{second_key}"
        outputs[key] = write_outputs(output_dir, f"colorcol_group_{key}_straight_alpha", rgba, source_map)
        print(f"stage 2: {key} at (x={dx}, y={dy})")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "colorcol_full_stitch_outputs")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = stage_one(args.output_dir)
    groups = stage_two(args.output_dir, pairs)
    first_key = "column000_column001_column003_column004"
    second_key = "column005_column007_column008_column009"
    eight_rgba, eight_map = alpha_stitch(read_rgba(groups[first_key]), read_rgba(groups[second_key]), dx=533, dy=133)
    eight = write_outputs(args.output_dir, "colorcol_group_column000_to_column009_straight_alpha", eight_rgba, eight_map)
    final_key = "column010_column011_column012_column013"
    final_rgba, final_map = alpha_stitch(read_rgba(eight), read_rgba(groups[final_key]), dx=1032, dy=258)
    final = write_outputs(args.output_dir, "colorcol_stitched_column000_to_column013_straight_alpha", final_rgba, final_map)
    print(f"final: {final}")


if __name__ == "__main__":
    main()

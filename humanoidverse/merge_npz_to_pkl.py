from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
from rich.progress import track

EXPECTED_FIELDS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
EXPECTED_FIELD_SET = frozenset(EXPECTED_FIELDS)


def format_bytes(num_bytes: int) -> str:
    size = float(num_bytes)
    units = ("B", "KB", "MB", "GB", "TB")
    unit_idx = 0
    while size >= 1024.0 and unit_idx < len(units) - 1:
        size /= 1024.0
        unit_idx += 1
    return f"{size:.2f} {units[unit_idx]}"


def discover_npz_files(input_root: Path, limit: int | None = None) -> tuple[list[Path], dict[Path, int]]:
    input_root = Path(input_root).expanduser()
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {input_root}")
    if limit is not None and limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")

    npz_paths: list[Path] = []
    directory_npz_counts: dict[Path, int] = {}

    for dirpath, dirnames, filenames in os.walk(input_root):
        dirnames.sort()
        npz_filenames = sorted(filename for filename in filenames if filename.endswith(".npz"))
        dir_path = Path(dirpath)
        directory_npz_counts[dir_path] = len(npz_filenames)

        for filename in npz_filenames:
            npz_paths.append(dir_path / filename)
            if limit is not None and len(npz_paths) >= limit:
                return npz_paths, directory_npz_counts

    return npz_paths, directory_npz_counts


def build_key(npz_path: Path, input_root: Path, directory_npz_count: int | None = None) -> str:
    relative_path = npz_path.relative_to(input_root)
    if directory_npz_count is None:
        directory_npz_count = sum(1 for child in npz_path.parent.iterdir() if child.suffix == ".npz")
    if npz_path.name == "motion.npz" and directory_npz_count == 1:
        return relative_path.parent.as_posix()
    return relative_path.with_suffix("").as_posix()


def load_npz_as_dict(npz_path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            actual_fields = frozenset(data.files)
            if actual_fields != EXPECTED_FIELD_SET:
                raise ValueError(
                    f"Unexpected fields in {npz_path}: "
                    f"expected {sorted(EXPECTED_FIELD_SET)}, got {sorted(actual_fields)}"
                )
            return {field: np.array(data[field], copy=True) for field in EXPECTED_FIELDS}
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to read npz file: {npz_path}") from exc


def collect_keys(
    npz_paths: list[Path],
    input_root: Path,
    directory_npz_counts: dict[Path, int],
) -> dict[Path, str]:
    keys_by_path: dict[Path, str] = {}
    source_by_key: dict[str, Path] = {}

    for npz_path in npz_paths:
        directory_npz_count = directory_npz_counts.get(npz_path.parent)
        key = build_key(npz_path=npz_path, input_root=input_root, directory_npz_count=directory_npz_count)
        if key in source_by_key:
            raise ValueError(
                f"Duplicate generated key '{key}' for files '{source_by_key[key]}' and '{npz_path}'"
            )
        source_by_key[key] = npz_path
        keys_by_path[npz_path] = key

    return keys_by_path


def merge_npz_files(npz_paths: list[Path], keys_by_path: dict[Path, str]) -> dict[str, dict[str, np.ndarray]]:
    merged: dict[str, dict[str, np.ndarray]] = {}

    for npz_path in track(npz_paths, description="Loading .npz files", disable=not npz_paths):
        key = keys_by_path[npz_path]
        if key in merged:
            raise ValueError(f"Duplicate generated key encountered while merging: {key}")
        merged[key] = load_npz_as_dict(npz_path)

    return merged


def main(
    input_root: Path,
    output_path: Path,
    compression: int = 0,
    dry_run: bool = False,
    overwrite: bool = False,
    limit: int | None = None,
) -> Path | None:
    input_root = Path(input_root).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()

    if not 0 <= compression <= 9:
        raise ValueError(f"compression must be between 0 and 9, got {compression}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output path already exists: {output_path}")

    npz_paths, directory_npz_counts = discover_npz_files(input_root=input_root, limit=limit)
    if not npz_paths:
        raise ValueError(f"No .npz files found under {input_root}")

    total_source_bytes = sum(npz_path.stat().st_size for npz_path in npz_paths)
    keys_by_path = collect_keys(npz_paths=npz_paths, input_root=input_root, directory_npz_counts=directory_npz_counts)

    print(f"Discovered {len(npz_paths)} .npz files under {input_root}")
    print(f"Total source size: {format_bytes(total_source_bytes)}")
    if limit is not None:
        print(f"Applied file limit: {limit}")
    print("Warning: writing one monolithic .pkl materializes the merged dataset in memory before dump.")

    if dry_run:
        print("Dry run enabled; skipping .pkl write.")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged = merge_npz_files(npz_paths=npz_paths, keys_by_path=keys_by_path)
    joblib.dump(merged, output_path, compress=compression)
    print(f"Saved merged dataset to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recursively merge Isaac Lab motion .npz files into one joblib .pkl.")
    parser.add_argument("input_root", type=Path, help="Root directory to traverse for .npz files.")
    parser.add_argument("output_path", type=Path, help="Output .pkl path.")
    parser.add_argument("--compression", type=int, default=0, help="joblib compression level between 0 and 9.")
    parser.add_argument("--dry-run", action="store_true", help="Scan and validate keys without writing the .pkl.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it already exists.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of .npz files to process.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        input_root=args.input_root,
        output_path=args.output_path,
        compression=args.compression,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        limit=args.limit,
    )

"""Augment valve point clouds stored as .las files in ./Data.

For every .las file in Data/, this script creates N_AUG geometrically
augmented point clouds in ./dataset_output (PCD + PLY), a manifest CSV with
the exact transform used for each sample, and a preview PNG.

Note: this is direct 3D geometric augmentation (rotations, scaling, jitter,
small shifts and point dropout). It is NOT SRN novel-view synthesis, which
would require rendered images with camera poses and a trained SRN.
"""

from pathlib import Path
import csv

import numpy as np
import laspy
import open3d as o3d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "Data"
OUT_DIR = BASE / "dataset_output"
N_AUG = 12
SEED = 2026


def read_las(path: Path) -> np.ndarray:
    las = laspy.read(path)
    return np.vstack((las.x, las.y, las.z)).T.astype(np.float64)


def rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rx_m = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry_m = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz_m = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz_m @ ry_m @ rx_m


def augment(points: np.ndarray, rng: np.random.Generator, index: int):
    """Apply one random geometric transform; keep the original point count."""
    center = points.mean(axis=0)
    p = points - center

    yaw = rng.uniform(-np.pi, np.pi)          # full 360-degree spin around Z
    pitch = rng.uniform(-0.25, 0.25)          # small tilt around X
    roll = rng.uniform(-0.25, 0.25)           # small tilt around Y
    scale = rng.uniform(0.90, 1.10)
    jitter = rng.uniform(0.001, 0.004)
    shift = rng.uniform(-0.02, 0.02, size=3)
    drop_frac = rng.uniform(0.0, 0.15)

    q = p @ rotation_matrix(pitch, yaw, roll).T * scale
    q = q + rng.normal(0.0, jitter, q.shape)

    if drop_frac > 0:
        n = len(q)
        n_drop = int(n * drop_frac)
        dropped = rng.choice(n, size=n_drop, replace=False)
        pool = np.delete(np.arange(n), dropped)
        donors = rng.choice(pool, size=n_drop, replace=True)
        q[dropped] = q[donors] + rng.normal(0.0, jitter * 0.5, (n_drop, 3))

    q = q + center + shift

    params = dict(
        yaw_deg=float(np.degrees(yaw)),
        pitch_deg=float(np.degrees(pitch)),
        roll_deg=float(np.degrees(roll)),
        scale=float(scale),
        jitter_sigma=float(jitter),
        shift_x=float(shift[0]),
        shift_y=float(shift[1]),
        shift_z=float(shift[2]),
        drop_frac=float(drop_frac),
    )
    return q.astype(np.float32), params


def write_point_cloud(points: np.ndarray, path_pcd: Path, path_ply: Path) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(str(path_pcd), pcd)
    o3d.io.write_point_cloud(str(path_ply), pcd)


def make_preview(entries) -> None:
    n = len(entries)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(14, 3.6 * rows))
    for i, (title, points) in enumerate(entries):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        show = points if len(points) <= 2000 else points[::len(points) // 2000]
        ax.scatter(show[:, 0], show[:, 1], show[:, 2], s=1.2, c="#1f77b4")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
    fig.suptitle("Valve point cloud: original + augmentations", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "preview.png", dpi=110)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(DATA_DIR.glob("*.las"))
    if not files:
        raise SystemExit(f"No .las files found in {DATA_DIR}")

    manifest_path = OUT_DIR / "manifest.csv"
    preview_entries = []
    total = 0
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "source", "output_pcd", "output_ply", "aug_index", "seed",
            "num_points", "yaw_deg", "pitch_deg", "roll_deg", "scale",
            "jitter_sigma", "shift_x", "shift_y", "shift_z", "drop_frac",
        ])

        for file_index, src in enumerate(files):
            points = read_las(src)
            rng = np.random.default_rng(SEED + file_index * 1000)
            preview_entries.append((f"{src.stem} original", points))
            print(f"{src.name}: {len(points)} points")

            for aug_index in range(N_AUG):
                augmented, params = augment(points, rng, aug_index)
                stem = f"{src.stem}_aug_{aug_index:02d}"
                pcd_path = OUT_DIR / f"{stem}.pcd"
                ply_path = OUT_DIR / f"{stem}.ply"
                write_point_cloud(augmented, pcd_path, ply_path)
                writer.writerow([
                    src.name, pcd_path.name, ply_path.name, aug_index,
                    SEED + file_index * 1000, len(augmented),
                    params["yaw_deg"], params["pitch_deg"], params["roll_deg"],
                    params["scale"], params["jitter_sigma"],
                    params["shift_x"], params["shift_y"], params["shift_z"],
                    params["drop_frac"],
                ])
                preview_entries.append((stem, augmented))
                total += 1

    make_preview(preview_entries)
    print(f"Created {total} augmented point clouds in {OUT_DIR}")
    print(f"Preview: {OUT_DIR / 'preview.png'}")


if __name__ == "__main__":
    main()

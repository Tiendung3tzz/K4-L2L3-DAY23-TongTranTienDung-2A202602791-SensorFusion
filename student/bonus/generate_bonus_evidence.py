"""Generate reproducible evidence for the three optional lab bonuses.

The official CP5 artifacts are left untouched.  This script runs a short,
deterministic prefix of the same Waymo pipeline into temporary suffixed
artifacts, records track states around the camera update, exports them through
the supplied CVAT hook, renders annotated plots, and repeats the run with
three camera-extrinsic perturbations.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STUDENT = ROOT / "student"
PLATFORM = ROOT / "platform"
BONUS = STUDENT / "bonus"
CONFIG = STUDENT / "config" / "paths.yaml"
BONUS_FRAMES = 40

os.environ.setdefault("DAY23_STUDENT_ROOT", str(STUDENT))
os.environ.setdefault("FUSION_LAB_PLATFORM", str(PLATFORM))
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))
if str(STUDENT) not in sys.path:
    sys.path.insert(0, str(STUDENT))

from fusion_lab import tracking_params
from fusion_lab.export_cvat import export_tracks_json
from fusion_lab.scripts import run_lab
from fusion_lab.tracking.manager import TrackManager as PlatformTrackManager
from fusion_lab.tracking.sensors import Sensor as PlatformSensor


def _cleanup_suffix(suffix: str) -> None:
    for name in (f"metrics{suffix}.json", f"grade_run{suffix}.log"):
        (STUDENT / "artifacts" / name).unlink(missing_ok=True)


def _copy_calibration(calib: Any, matrix: np.ndarray) -> Any:
    cloned = type(calib)()
    cloned.CopyFrom(calib)
    del cloned.extrinsic.transform[:]
    cloned.extrinsic.transform.extend(np.asarray(matrix, dtype=float).reshape(-1).tolist())
    return cloned


class SnapshotManager(PlatformTrackManager):
    """TrackManager that records the state after each sensor pass."""

    snapshots: list[dict[str, Any]] = []

    def manage_tracks(self, unassigned_tracks, unassigned_meas, sensor):
        super().manage_tracks(unassigned_tracks, unassigned_meas, sensor)
        frame = int(round(self.track_list[0].t / tracking_params.dt)) if self.track_list else None
        if frame is None:
            return
        self.snapshots.append(
            {
                "frame": frame,
                "sensor": sensor.name,
                "tracks": [
                    {
                        "id": int(track.id),
                        "x": [float(value) for value in np.asarray(track.x).reshape(-1)[:6]],
                        "score": float(track.score),
                        "state": str(track.state),
                    }
                    for track in self.track_list
                ],
            }
        )


def _association_wrapper(
    original: Callable[..., Any], stats: dict[str, Any], association_module: Any
) -> Callable[..., Any]:
    """Collect camera innovation/gating evidence without changing association."""

    def wrapped(manager, meas_list, filter_obj, sensor):
        if sensor.name == "camera":
            for track in manager.track_list:
                for meas in meas_list:
                    if not sensor.in_fov(track.x):
                        continue
                    gamma = association_module.kalman.innovation(track.x, meas)
                    distance = association_module.mahalanobis_distance(track, meas)
                    stats["innovations_px"].append(float(np.linalg.norm(gamma)))
                    stats["candidate_pairs"] += 1
                    if association_module.chi2_gate(distance, sensor):
                        stats["gate_pass"] += 1
        return original(manager, meas_list, filter_obj, sensor)

    return wrapped


def _run_mode(
    *,
    suffix: str,
    manager_cls: Any,
    sensor_cls: Any,
    stats: dict[str, Any],
) -> dict[str, Any]:
    original_manager = run_lab.TrackManager
    original_sensor = run_lab.Sensor
    original_load_workspace = run_lab.load_workspace
    workspace = original_load_workspace()
    association_module = workspace["association"]
    original_assoc = association_module.associate_and_update
    run_lab.TrackManager = manager_cls
    run_lab.Sensor = sensor_cls
    association_module.associate_and_update = _association_wrapper(
        original_assoc, stats, association_module
    )
    run_lab.load_workspace = lambda: workspace
    try:
        return run_lab.run(
            CONFIG,
            fusion_mode="fused",
            max_frames=BONUS_FRAMES,
            artifact_suffix=suffix,
            seed=0,
        )
    finally:
        run_lab.TrackManager = original_manager
        run_lab.Sensor = original_sensor
        association_module.associate_and_update = original_assoc
        run_lab.load_workspace = original_load_workspace
        _cleanup_suffix(suffix)


def _new_stats() -> dict[str, Any]:
    return {"innovations_px": [], "candidate_pairs": 0, "gate_pass": 0}


def _load_front_sensor() -> tuple[Any, Any]:
    run_lab._setup_import_paths()
    cfg = run_lab._load_paths_config(CONFIG)
    from simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2
    from simple_waymo_open_dataset_reader import utils as waymo_utils

    tfrecord = Path(cfg["waymo_dir"]) / cfg["segment"]
    frame = next(iter(WaymoDataFileReader(str(tfrecord))))
    calib = waymo_utils.get(frame.context.camera_calibrations, dataset_pb2.CameraName.FRONT)
    camera = PlatformSensor("camera", calib, run_lab.load_workspace()["camera_fusion"])
    return camera, calib


def _stats_summary(stats: dict[str, Any]) -> dict[str, Any]:
    values = np.asarray(stats["innovations_px"], dtype=float)
    return {
        "candidate_pairs": int(stats["candidate_pairs"]),
        "gate_pass": int(stats["gate_pass"]),
        "gate_accept_rate": float(stats["gate_pass"] / stats["candidate_pairs"])
        if stats["candidate_pairs"]
        else 0.0,
        "innovation_mean_px": float(values.mean()) if values.size else None,
        "innovation_p95_px": float(np.percentile(values, 95)) if values.size else None,
    }


def _snapshot_map(snapshots: list[dict[str, Any]], sensor: str) -> dict[int, dict[int, dict[str, Any]]]:
    result: dict[int, dict[int, dict[str, Any]]] = {}
    for item in snapshots:
        if item["sensor"] != sensor:
            continue
        result[item["frame"]] = {track["id"]: track for track in item["tracks"]}
    return result


def _select_frame(snapshots: list[dict[str, Any]]) -> tuple[int, int]:
    lidar = _snapshot_map(snapshots, "lidar")
    camera = _snapshot_map(snapshots, "camera")
    best: tuple[float, int, int] | None = None
    for frame in sorted(set(lidar) & set(camera)):
        for track_id in set(lidar[frame]) & set(camera[frame]):
            before = np.asarray(lidar[frame][track_id]["x"][:3])
            after = np.asarray(camera[frame][track_id]["x"][:3])
            delta = float(np.linalg.norm(after - before))
            candidate = (delta, frame, track_id)
            if best is None or candidate > best:
                best = candidate
    if best is None:
        raise RuntimeError("No track was present in both lidar and camera snapshots")
    return best[1], best[2]


def _track_rows(snapshots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in snapshots:
        rows.append(
            {
                "frame": item["frame"],
                "sensor_pass": item["sensor"],
                "tracks": item["tracks"],
            }
        )
    return rows


def _plot_bev(
    path: Path,
    frame: int,
    lidar_tracks: dict[int, dict[str, Any]],
    camera_tracks: dict[int, dict[str, Any]],
    title: str,
    use_camera: bool,
) -> None:
    tracks = camera_tracks if use_camera else lidar_tracks
    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
    ax.axhline(0, color="0.8", linewidth=0.8)
    ax.axvline(0, color="0.8", linewidth=0.8)
    for track_id, track in sorted(tracks.items()):
        x, y = np.asarray(track["x"][:2])
        ax.scatter(x, y, s=90, label=f"track {track_id}")
        ax.annotate(f"ID {track_id}", (x, y), xytext=(6, 6), textcoords="offset points")
    for track_id in sorted(set(lidar_tracks) & set(camera_tracks)):
        before = np.asarray(lidar_tracks[track_id]["x"][:2])
        after = np.asarray(camera_tracks[track_id]["x"][:2])
        ax.annotate("", xy=after, xytext=before, arrowprops={"arrowstyle": "->", "color": "tab:red"})
    ax.scatter([0], [0], marker="x", color="black", s=80, label="vehicle")
    ax.set_xlabel("vehicle x / m (forward)")
    ax.set_ylabel("vehicle y / m (left)")
    ax.set_title(f"{title}\nframe {frame}; red arrow = camera correction")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_camera(
    path: Path,
    frame: int,
    camera_sensor: Any,
    lidar_tracks: dict[int, dict[str, Any]],
    camera_tracks: dict[int, dict[str, Any]],
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
    for track_id in sorted(set(lidar_tracks) & set(camera_tracks)):
        before = np.asarray(camera_sensor.get_hx(np.asmatrix(lidar_tracks[track_id]["x"]).T)).reshape(-1)
        after = np.asarray(camera_sensor.get_hx(np.asmatrix(camera_tracks[track_id]["x"]).T)).reshape(-1)
        ax.scatter(before[0], before[1], marker="o", s=90, color="tab:orange")
        ax.scatter(after[0], after[1], marker="x", s=100, color="tab:blue")
        ax.annotate(f"ID {track_id}", (after[0], after[1]), xytext=(6, 6), textcoords="offset points")
        ax.annotate("", xy=after, xytext=before, arrowprops={"arrowstyle": "->", "color": "tab:red"})
    ax.set_xlim(0, camera_sensor.c_i * 2)
    ax.set_ylim(camera_sensor.c_j * 2, 0)
    ax.set_xlabel("u / pixel")
    ax.set_ylabel("v / pixel")
    ax.set_title(f"FRONT camera projection, frame {frame}\norange = after LiDAR, blue = after camera update")
    ax.grid(True, alpha=0.25)
    ax.legend(["LiDAR state", "fused state", "camera correction"], loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_calibration(path: Path, rows: list[dict[str, Any]]) -> None:
    x = np.asarray([row["dy_m"] for row in rows])
    rmse = np.asarray([row["rmse"] for row in rows])
    innovation = np.asarray([row["innovation_mean_px"] for row in rows])
    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=140)
    ax1.plot(x, rmse, marker="o", color="tab:blue", label="fused RMSE")
    ax1.set_xlabel("camera extrinsic lateral offset / m")
    ax1.set_ylabel("fused RMSE / m", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(x, innovation, marker="s", color="tab:red", label="mean innovation")
    ax2.set_ylabel("mean camera innovation / px", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_title("Calibration perturbation study (frames 0–39, seed 0)")
    ax1.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    BONUS.mkdir(parents=True, exist_ok=True)
    SnapshotManager.snapshots = []
    baseline_stats = _new_stats()
    baseline_metrics = _run_mode(
        suffix="_bonus_baseline",
        manager_cls=SnapshotManager,
        sensor_cls=PlatformSensor,
        stats=baseline_stats,
    )
    snapshots = list(SnapshotManager.snapshots)
    selected_frame, selected_track = _select_frame(snapshots)
    lidar_map = _snapshot_map(snapshots, "lidar")
    camera_map = _snapshot_map(snapshots, "camera")
    lidar_tracks = lidar_map[selected_frame]
    camera_tracks = camera_map[selected_frame]

    camera_sensor, _ = _load_front_sensor()
    export_tracks_json(_track_rows(snapshots), BONUS / "cvat_tracks.json")
    (BONUS / "track_snapshots.json").write_text(json.dumps(_track_rows(snapshots), indent=2))
    _plot_bev(
        BONUS / "track_bev_before_camera.png",
        selected_frame,
        lidar_tracks,
        camera_tracks,
        "BEV track state after LiDAR association",
        use_camera=False,
    )
    _plot_bev(
        BONUS / "track_bev_after_camera.png",
        selected_frame,
        lidar_tracks,
        camera_tracks,
        "BEV track state after camera EKF update",
        use_camera=True,
    )
    _plot_camera(BONUS / "track_camera_projection.png", selected_frame, camera_sensor, lidar_tracks, camera_tracks)

    calibration_rows: list[dict[str, Any]] = []
    perturbations = [
        ("baseline", 0.00, 0.00),
        ("mild", 0.15, 0.50),
        ("strong", 0.30, 1.00),
    ]
    baseline_summary = _stats_summary(baseline_stats)
    calibration_rows.append(
        {
            "level": "baseline",
            "dy_m": 0.0,
            "yaw_deg": 0.0,
            **baseline_summary,
            "rmse": float(baseline_metrics["tracking"]["fused"]["rmse"]),
            "matches": int(baseline_metrics["tracking"]["fused"]["matches"]),
            "ghost_track_frames": int(baseline_metrics["tracking"]["fused"]["ghost_track_frames"]),
            "missed_gt_frames": int(baseline_metrics["tracking"]["fused"]["missed_gt_frames"]),
        }
    )

    original_sensor = PlatformSensor
    for level, dy, yaw_deg in perturbations[1:]:
        stats = _new_stats()
        yaw = math.radians(yaw_deg)
        rotation = np.array(
            [[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
        )

        class PerturbedSensor(original_sensor):
            def __init__(self, name, calib, camera_fusion):
                if name == "camera":
                    matrix = np.asarray(calib.extrinsic.transform, dtype=float).reshape(4, 4)
                    perturb = np.eye(4)
                    perturb[:3, :3] = rotation
                    perturb[:3, 3] = [0.0, dy, 0.0]
                    calib = _copy_calibration(calib, perturb @ matrix)
                super().__init__(name, calib, camera_fusion)

        metrics = _run_mode(
            suffix=f"_bonus_calib_{level}",
            manager_cls=PlatformTrackManager,
            sensor_cls=PerturbedSensor,
            stats=stats,
        )
        calibration_rows.append(
            {
                "level": level,
                "dy_m": dy,
                "yaw_deg": yaw_deg,
                **_stats_summary(stats),
                "rmse": float(metrics["tracking"]["fused"]["rmse"]),
                "matches": int(metrics["tracking"]["fused"]["matches"]),
                "ghost_track_frames": int(metrics["tracking"]["fused"]["ghost_track_frames"]),
                "missed_gt_frames": int(metrics["tracking"]["fused"]["missed_gt_frames"]),
            }
        )

    (BONUS / "calibration_analysis.json").write_text(json.dumps(calibration_rows, indent=2))
    with (BONUS / "calibration_analysis.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(calibration_rows[0]))
        writer.writeheader()
        writer.writerows(calibration_rows)
    _plot_calibration(BONUS / "calibration_effect.png", calibration_rows)

    summary = {
        "frames": [0, BONUS_FRAMES - 1],
        "selected_frame": selected_frame,
        "selected_track": selected_track,
        "camera_update_delta_m": float(
            np.linalg.norm(
                np.asarray(camera_tracks[selected_track]["x"][:3])
                - np.asarray(lidar_tracks[selected_track]["x"][:3])
            )
        ),
        "baseline_fused_rmse": calibration_rows[0]["rmse"],
        "calibration_levels": [row["level"] for row in calibration_rows],
    }
    (BONUS / "bonus_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

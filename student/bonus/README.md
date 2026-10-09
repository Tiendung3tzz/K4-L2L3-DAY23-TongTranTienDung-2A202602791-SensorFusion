# Bonus evidence

`generate_bonus_evidence.py` runs the first 40 frames with seed 0 using
temporary suffixed artifacts, then removes those temporary logs/metrics. The
official CP5 compare artifacts are not modified.

- `cvat_tracks.json`: output of `fusion_lab.export_cvat.export_tracks_json`.
- `track_snapshots.json`: source snapshots for the BEV and camera plots.
- `track_bev_before_camera.png` / `track_bev_after_camera.png`: selected frame
  before and after the camera EKF pass.
- `track_camera_projection.png`: the same correction shown in FRONT pixel
  coordinates.
- `calibration_analysis.csv` / `.json`: baseline, mild, and strong extrinsic
  perturbation measurements.
- `calibration_effect.png`: innovation and fused-RMSE comparison.

The CVAT JSON is ready for import. The actual CVAT import screenshot is kept
separate because the available CVAT web instance requires the student's own
account login before a local file can be uploaded.

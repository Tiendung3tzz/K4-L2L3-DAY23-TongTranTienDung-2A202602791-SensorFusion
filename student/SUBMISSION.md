# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Tống Trần Tiến Dũng
- MSSV: 2A202602791
- Email: tiendung3t@gmail.com
- Link repo (fork): https://github.com/Tiendung3tzz/K4-L2L3-DAY23-TongTranTienDung-2A202602791-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): xem giá trị HEAD sau khi push commit CP6 cuối.

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`: `compare`, `[0, 198]`, `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`, `0`.
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`: `0.970093`, `0.700405`, `519/16/222`.
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: `0.150323`, `502`, `11.343651`, `0`, `239`, `2.522613`.
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: `0.135867`, `502`, `9.266811`, `0`, `239`, `2.522613`.
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss: fused giảm RMSE khoảng `9.6%` và giảm `sum_sq_err` từ `11.343651` xuống `9.266811`, trong khi số ghép, ghost, miss và số track confirmed trung bình giữ nguyên. Vì camera ở đây cung cấp tâm hộp FRONT có nhiễu seeded để cập nhật EKF, nó cải thiện state của các track đã ghép nhưng không tự tạo/làm mất track.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. Lidar đo vị trí 3D trong hệ sensor nên `z` có ba phần tử và `R` có đơn vị mét bình phương. Camera đo pixel 2D nên `z` là `[u, v]`, `R` là covariance theo pixel bình phương; EKF dùng Jacobian tương ứng với từng sensor.
2. Gating Mahalanobis chuẩn hóa sai số theo covariance của track và measurement. Nó loại các cặp không hợp lý trước greedy matching, tránh ghép theo khoảng cách Euclid khi độ bất định giữa các track khác nhau.
3. Pipeline là track-then-fuse: mỗi frame predict track trước, AssocL update bằng LiDAR, sau đó AssocC update camera trên cùng track. Artifact compare xác nhận fused giữ nguyên `502` matches và giảm RMSE từ `0.150323` xuống `0.135867`.
4. Calibration lệch làm innovation/residual camera có bias có hệ thống, thường tăng độ lớn hoặc đẩy nhiều phép chiếu ra ngoài FOV; association camera sẽ bị reject nhiều hơn.
5. Vì frame rỗng không có measurement để suy ra modality. `sensor` tường minh giúp `associate_and_update` vẫn gọi đúng lifecycle pass; LiDAR được phép score/init/delete, còn camera chỉ EKF update và không thay đổi vòng đời track.
   Giải thích vì sao lidar quyết định score/init/delete còn camera chỉ EKF update.
6. Track được xác nhận khi score vượt `confirmed_threshold`; track confirmed vẫn giữ state sau một miss. Track bị xóa khi phương sai ngang vượt `max_P`, confirmed có score dưới `delete_threshold`, hoặc track chưa confirmed có score không dương.

## Bonus (không bắt buộc)

Liệt kê phần bonus đã làm, file bằng chứng trong `student/bonus/` và kết quả chính
(xem [RUBRIC.md](../RUBRIC.md) mục 2). Không làm thì ghi "Không".

- **Trực quan hóa track và đo trên BEV/ảnh camera (+3):** Đã tạo
  `student/bonus/track_bev_before_camera.png`,
  `student/bonus/track_bev_after_camera.png` và
  `student/bonus/track_camera_projection.png`. Ở frame 31, track ID 4 thay
  đổi `3.1365 m` sau camera EKF update; mũi tên đỏ trên ảnh chỉ hướng hiệu
  chỉnh từ state sau LiDAR sang state fused.
- **Phân tích calibration (+4):** Bảng đầy đủ nằm trong
  `student/bonus/calibration_analysis.csv` và `.json`, biểu đồ ở
  `student/bonus/calibration_effect.png`. Trên frame 0–39, seed 0, ba mức là
  baseline `(dy=0 m, yaw=0°)`, mild `(0.15 m, 0.5°)` và strong `(0.30 m, 1°)`.
  Innovation trung bình lần lượt là `262.48`, `266.94`, `268.20 px`, p95 là
  `481.00`, `502.45`, `530.92 px`; tỉ lệ cặp qua gate giảm từ `11.83%`
  xuống `2.09%`. Đây là triệu chứng calibration lệch: residual tăng và
  Mahalanobis gate chặn hầu hết camera update. RMSE fused lần lượt là
  `0.1456`, `0.1271`, `0.1271 m`; RMSE không tăng ở hai mức lệch vì gate đã
  loại phần lớn update sai, nên phải đọc cùng innovation và gate rate.
- **Export track sang CVAT (+3):** Đã xuất qua
  `fusion_lab.export_cvat.export_tracks_json` tại
  `student/bonus/cvat_tracks.json`, gồm snapshot track cho frame 0–39 và
  sensor pass LiDAR/camera. CVAT web hiện yêu cầu đăng nhập để import và chụp
  ảnh xác nhận; ảnh import/CVAT proof sẽ được bổ sung ngay sau bước đăng nhập
  tài khoản CVAT.

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): ChatGPT.
- Dùng cho phần nào (hàm, câu hỏi, debug): Hỗ trợ triển khai và rà công thức Parts E–H, giải thích lỗi môi trường Windows/NumPy/PyTorch, và kiểm tra cấu trúc báo cáo.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): Đối chiếu với test contract/regression, `py_compile`, chạy toàn bộ `128 passed`, chạy Waymo compare frame `0–198` với seed `0`, và chạy `python tools/check_submission.py`.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; self-check xác nhận không còn TODO.
- [x] Part A–D: không sửa.
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ phần báo cáo hiện có, gồm khai báo AI.
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key.
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`.
- [x] Đã push các checkpoint CP2–CP5; sau commit CP6 cuối sẽ nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md)).

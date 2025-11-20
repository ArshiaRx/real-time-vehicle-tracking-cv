<!-- .github/copilot-instructions.md
Guidance tailored for AI coding agents (Copilot/assistant) working on this repository.
Keep this concise and actionable — reference real files and patterns discovered in the codebase.
-->

# Assistant Guidance — real-time-vehicle-tracking-cv

Summary

- Purpose: Real-time vehicle tracking + counting using Optical Flow (Lucas-Kanade) and optional YOLOv8 detection.
- Key entry points: `main.py` (CLI), `app.py` (Streamlit UI), and `src/VideoProcessor` pipeline in `src/video_processor.py`.

What to change and why

- Preserve the separation: detection (YOLO) vs tracking (optical flow), association, track management, and counting.
- Small, focused edits are preferred; don't rewrite entire pipeline unless adding a well-scoped feature (e.g., new ROI type).

Important files & their responsibilities

- `main.py`: CLI runner and webcam support. Uses `src/video_processor.VideoProcessor` and `src.utils.ROISelector` for interactive ROI selection.
- `app.py`: Streamlit interface (launch with `streamlit run app.py`). Use this for UI-related changes.
- `src/video_processor.py`: Orchestrates detection/tracking, Kalman logic, display modes, and counting. Reference this for display flags and pipeline flow.
- `src/yolo_detector.py`: Loads YOLO (ultralytics) when available. If `ultralytics` not installed, code falls back to optical flow. Keep YOLO usage guarded by availability checks.
- `src/optical_flow_tracker.py`: Implements Lucas-Kanade sparse tracking. Tracks are lists of points and output format is a list of dicts with `id`, `position`, `history`.
- `src/vehicle_counter.py`: Expects tracks as dicts with `id` and `position` (numpy array or tuple). ROI modes: `'line'` (2 points) or `'polygon'` (>=3 points).
- `src/kalman_filter.py`: Kalman prediction/update used by `VideoProcessor` when `use_kalman` is True.
- `src/utils.py`: Contains drawing helpers (e.g., `draw_tracks`, `draw_roi`) and `ROISelector` used in interactive ROI selection.

Key conventions & patterns

- Display modes: `VideoProcessor.display_mode` accepts `'clean'`, `'verbose'`, `'minimal'`. Respect these in any visualization changes.
- Toggle flags: `show_tracks`, `show_help`, `show_legend` are toggled at runtime by `main.py` keybindings — preserve keys if you modify UI behavior.
- YOLO vs Optical Flow:
  - `VideoProcessor.use_yolo` enables YOLO internal tracker. When YOLO is enabled, Kalman filtering is disabled (`use_kalman` is set False).
  - `yolo_detector.YOLODetector` uses `ultralytics.YOLO(...)` and `yolov8<n>.pt` naming convention. Guard import and fallback behavior.
- Track format:
  - Optical flow returns: { 'id': int, 'position': np.array([x,y]), 'history': [...], 'length': int }
  - YOLO track returns: { 'id': int, 'bbox': [x1,y1,x2,y2], 'position': np.array([x,y]), 'class_name': str }
  - `vehicle_counter.update()` reads `id` and `position` fields; include those when adding new trackers.

Run / debug commands (Windows PowerShell)

- Run CLI (video):
  ```powershell
  python .\main.py --video .\data\sample_traffic_test.mp4 --yolo
  ```
- Run CLI (webcam):
  ```powershell
  python .\main.py --webcam --yolo
  ```
- Run Streamlit UI:
  ```powershell
  streamlit run .\app.py
  ```
- Quick test to verify deps:
  ```powershell
  python -c "import cv2, ultralytics, streamlit; print('deps ok')"
  ```

Project-specific gotchas

- ROI selection: `main.py` uses `ROISelector` to interactively select a line or polygon on the first frame. Automated changes must preserve the interactive flow or provide a CLI flag for pre-supplied ROI.
- Video writer uses `mp4v` fourcc and saves with OpenCV `VideoWriter`. Keep frame size and fps consistent — `main.py` reads them from capture before creating writer.
- Display errors: `main.py` contains try/except around `cv2.namedWindow` and `cv2.imshow` to handle environments without GUI. Use the `--no-display` flag for headless CI runs.
- YOLO model path/name: `src/yolo_detector.py` expects local file names like `yolov8n.pt` or uses ultralytics model loading. Don't hardcode absolute paths.

Testing & validation

- Unit tests are not present; validate by running `python main.py --video data/sample_traffic_test.mp4 --no-display --no-save` for a smoke run.
- For changes to drawing helpers in `src/utils.py`, run with `--no-save` and `--video` while observing the Streamlit UI or local display.

Examples to copy/paste

- Creating a new tracker output that integrates with counter:
  ```py
  # produce track in expected format
  track = {'id': new_id, 'position': np.array([cx, cy]), 'history': history}
  vehicle_counter.update([track])
  ```

When merging PRs

- Keep diffs small and focused — pipeline behavior is stateful and easy to regress.
- Add a short example command in the PR description showing how to reproduce the behavior (exact CLI flags).

If unsure, check these files first

- `main.py`, `src/video_processor.py`, `src/yolo_detector.py`, `src/optical_flow_tracker.py`, `src/vehicle_counter.py`, `src/utils.py`.

Questions to ask maintainers (when uncertain)

- Should new detectors integrate with YOLODetector API or be wrapped by VideoProcessor?
- Any preferred maximum frame rate / performance target for real-time on CPU vs GPU?

End of file

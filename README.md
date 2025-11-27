# Mind-Power Telekinesis UI

Created and developed by @tubakhxn.

Inspired by Stranger Things; this project is a real-time telekinesis simulator that uses MediaPipe hand tracking and OpenCV visual effects to let you control floating objects with gestures.

---

## Features
- Real-time hand tracking (MediaPipe Hands)
- Gestures: Pinch (grab/move), Open Palm (levitate), Fist (crush), Palm Push (throw)
- Physics-like object movement (velocity, damping, impulses)
- Visual effects: neon glow, particle wisps, shockwaves, distortion, chromatic freeze
- On-screen sliders and controls to tune gesture sensitivity and visual intensity

## Requirements
- Python 3.8+
- Packages: `mediapipe`, `opencv-python`, `numpy`

Install dependencies:
```powershell
python -m pip install --upgrade pip
python -m pip install mediapipe opencv-python numpy
```

## Run
Open PowerShell and run:
```powershell
python c:/Users/Tuba Khan/Downloads/eleven/project/main.py
```

Press `Esc` to quit the app.

## Controls & UI
- The window `Mind-Power Telekinesis UI` shows real-time sliders at the top for tuning detection thresholds: `pinch_px`, `fist_count`, `push_z_thresh`, `levitate_spread`, and `push_power_trigger`.
- Top-right clickable buttons: `Vol+`, `Vol-`, `Int+`, `Int-`, `Spawn`, `Reset`, `Sound` (use the mouse to click).

## Gestures (short)
- Pinch: touch thumb + index → grab and move nearest object
- Open Palm (slow, steady): levitate nearby objects gently
- Fist: compress/crush nearby object (shrink + glitch)
- Palm Push: quick forward palm motion → shockwave + throw; strong pushes trigger a chromatic freeze effect

## Forking / Contributing
1. Fork this repository on GitHub.
2. Clone your fork:
```bash
git clone https://github.com/<your-username>/code5.git
```
3. Create a branch, implement changes, and open a pull request.

## Notes
- This project is provided without a license file. No license is included; use at your own risk.
- Assets are minimal and included in the `assets/` folder (placeholder).

If you want a license, documentation, or packaged installer, tell me which license or packaging target and I can add it.

# BEAR: Basketball Experience using Augmented Reality

## How to run this repo?
- Install packages in ``requirements.txt``
- Calibrate your camera
  - Record a video including the chessboard file located in ``markers/camera-calibration-checker-board_9x7.pdf``
  - Run ``camera_calibration.py`` in ``calibration/`` and press ``space`` several times to generate images for calibration, which saves intrinsic parameters to ``camera_parameters.npy``
- You're all set, simply run ``main.py`` under ``src/`` to start!

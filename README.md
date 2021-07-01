# BEAR: Basketball Experience using Augmented Reality

We developed an **AR basketball system** by which we can play basketball without a basket and a ball. 

<!-- ![](https://github.com/jeffrey1227/BEAR/blob/main/README.assets/demo.gif) -->
<img src="https://github.com/jeffrey1227/BEAR/blob/main/README.assets/demo.gif" width="480" height="270"/>

### How to run this repo?

- Install packages in ``requirements.txt``
- Calibrate your camera
  - Record a video including the chessboard file located in ``markers/camera-calibration-checker-board_9x7.pdf``
  - Run ``camera_calibration.py`` in ``calibration/`` and press ``space`` several times to generate images for calibration, which saves intrinsic parameters to ``camera_parameters.npy``
- You're all set, simply run ``main.py`` under ``src/`` to start!



**See demo video [here](https://drive.google.com/file/d/1WukbbCArZPf4oBcswMDyZRPqQv3nMbdz/view?usp=sharing), slides [here](https://docs.google.com/presentation/d/1mpYlLV4sO4_4FU2N52p17GadXN28YNB9rH8DJHv4uko/edit#slide=id.p), and report [here](https://github.com/jeffrey1227/BEAR/blob/main/report.pdf)!**


# Intro
Two examples:
1. 3D convolution neural network. (C3D)
2. two stream network. (vgg16)

PS: tested through UCF-101 dataset.

# Package Used
  * cson
  * OpenCV
  * tensorflow (v1.0)

# Util
  * You can use `./tools/_list_to_json.py` to generate lists of videos with labels.

# Configuration
  Tuning cson files in `./config` directory to modify your model.

# Train
  Call the `train.py` script with json filename(without `".json"`).

  ## train from scratch

  e.g.
  ```
    python train.py c3d --clear
  ```

  ## fine-tuning

  e.g.
  ```
    python train.py c3d
  ```

# Validation
  ```
    python eval.py c3d
  ```

# Test
```
  python -m tools.video_test c3d
```

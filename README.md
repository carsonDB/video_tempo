# Intro
Two examples:
1. 3D convolution neural network. ([C3D](https://arxiv.org/abs/1412.0767))
2. Two stream network. ([vgg16](http://yjxiong.me/others/action_recog/))

PS: tested through UCF-101 dataset.

# Package Used
  * cson
  * OpenCV (v3)
  * tensorflow (v1.0)

# Pretrained Model Download
  Download pretrained model, e.g. vgg16 into `ckpts` directory.
  ```
  mkdir ckpts
  cd ckpts
  aria2c 'http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz'
  tar xvzf vgg_16_2016_08_28.tar.gz && rm vgg_16_2016_08_28.tar.gz
  ```

# Utils
  * You can use `./tools/_list_to_json.py` to generate lists of videos with labels.

# Configuration
  Tuning cson files in `./config` directory to modify your model.

# Train
  Call the `multi_gpu_train.py` script with json filename(without `".json"`).

  ## train from scratch

  e.g.
  ```
    python multi_gpu_train.py vgg16_rgb --clear
  ```

  ## fine-tuning

  e.g.
  ```
    python multi_gpu_train.py vgg16_rgb
  ```

# Validation
  ```
    python eval.py vgg16_rgb
  ```

# Test
```
  python -m tools.video_test vgg16_rgb
```

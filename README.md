# Intro
made up of two parts:
1. 3D convolution neural network.
2. Recurrent neural network. (beta)

PS: I've tested codes through UCF-11 dataset.

# Package Used
  * commentjson
  * OpenCV (v3.1.0)
  * tensorflow (v0.10)

# Util
  * You can use **gen_lst.py** to generate lists of videos with labels.
  
  e.g.
  ```
    python util/gen_lst.py  /your/video/directory  /your/destination/path
  ```

# Configuration
  Tuning json files in **config** directory to modify your model.

# Preprocess support:
  * resize clips
  * random crop
  * flip horizontally
  * whitening per frame
  * cnn pretrained model

# Train
  Call the `train.py` script with json filename(without `".json"`).
  
  ## train from scratch

  e.g.
  ```
    python train.py C3D --clear
  ```

  ## fine-tuning
  
  e.g.
  ```
    python train.py C3D
  ```
 

# Evaluation (beta)
  Call the `eval.py` script with json filename(without `".json"`).
  
  e.g.
  ```
    python eval.py C3D
  ```

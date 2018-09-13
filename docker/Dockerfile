FROM tensorflow/tensorflow:1.0.1-gpu
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/' /etc/apt/sources.list
RUN apt-get update && apt install -y libsm6 libxrender1 libxext6
RUN pip install cson opencv-python ipdb h5py progressbar
WORKDIR /video_tempo

# build: docker build -t carsondb/video_tempo:c2-tensorflow-1.0 .
# sudo nvidia-docker run --rm --name $NAME -v $HOME/proj/video_tempo:/video_tempo -it carsondb/video_tempo:c2-tensorflow-1.0 /bin/bash

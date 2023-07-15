# Demonstrates how to compile CSVA from clean ubuntu
ARG CUDA_VER=11.8.0
ARG UBUNTU_VER=22.04

FROM nvidia/cuda:${CUDA_VER}-cudnn8-runtime-ubuntu${UBUNTU_VER}

RUN apt update && apt install -y cmake g++ wget unzip libboost-all-dev
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
RUN wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip
RUN unzip opencv.zip
RUN unzip opencv_contrib.zip
RUN mkdir -p build && cd build
RUN cmake -DOPENCV_ENABLE_NONFREE:BOOL=ON -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.x/modules ../opencv-4.x
RUN make -j8
RUN make install
RUN apt install -y git
RUN git clone https://github.com/malashinroman/CSVA
RUN cd CSVA && git checkout opencv4 && mkdir build && cd build && \
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON .. && make -j8

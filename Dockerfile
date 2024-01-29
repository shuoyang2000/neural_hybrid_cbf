# adapted from https://github.com/f1tenth/f1tenth_gym

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND="noninteractive"
ENV LIBGL_ALWAYS_INDIRECT=1
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}

ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

RUN apt-get update --fix-missing && \
    apt-get install -y \
                    python3-dev \
                    python3-pip \
                    git \
                    build-essential \
                    libgl1-mesa-dev \
                    mesa-utils \
                    libglu1-mesa-dev \
                    fontconfig \
                    libfreetype6-dev

RUN pip3 install --upgrade pip
RUN pip3 install PyOpenGL \
                 PyOpenGL_accelerate

RUN mkdir /learn_cbf
COPY . /learn_cbf

RUN cd /learn_cbf && \
    pip3 install -e .

WORKDIR /learn_cbf

ENTRYPOINT ["/bin/bash"]

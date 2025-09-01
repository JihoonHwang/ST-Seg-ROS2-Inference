FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirror.kakao.com/ubuntu/|g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        apt-utils \
        build-essential \
        git \
        wget \
        vim \
        libgl1-mesa-glx \
        libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Python3
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip && \
    python3.10 -m pip install --upgrade pip setuptools wheel && \
    rm -rf /var/lib/apt/lists/*

# mmsegmentation
RUN python3.10 -m pip install \
        torch==2.0.1+cu117 \
        torchvision==0.15.2+cu117 \
        --extra-index-url https://download.pytorch.org/whl/cu117 && \
    python3.10 -m pip install \
        'numpy<2' \
        openmim \
        ftfy \
        regex && \
    mim install mmengine && \
    mim install 'mmcv>=2.0.0'

# ROS2 Humble
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        curl && \
    add-apt-repository universe && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
        http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
        > /etc/apt/sources.list.d/ros2.list

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        ros-humble-ros-base \
        ros-humble-image-transport-plugins \
        ros-humble-foxglove-bridge \
        python3-colcon-common-extensions && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

WORKDIR /workspace

CMD ["/bin/bash"]

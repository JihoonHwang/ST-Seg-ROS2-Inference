FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirror.kakao.com/ubuntu/|g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        apt-utils \
        build-essential \
        git \
        wget \
        vim \
        openssh-client \
        libgl1-mesa-glx \
        libglib2.0-0 \
        cmake \
        ninja-build && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Python3
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    python3.10 -m pip install --upgrade pip setuptools wheel && \
    python3.10 -m pip cache purge

# MIC
COPY MIC_humble_requirements.txt /root/requirements.txt

RUN python3.10 -m pip install \
        --no-cache-dir \
        -r /root/requirements.txt \
        -f https://download.pytorch.org/whl/torch_stable.html && \
    python3.10 -m pip install mmcv-full==1.3.7 && \
    python3.10 -m pip install -U openmim && \
    python3.10 -m pip cache purge && \
    mim install mmengine

# ROS2 Humble
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
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

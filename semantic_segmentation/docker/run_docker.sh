#!/bin/bash

########## For ST-Seg Humble ##########
docker run -it \
    --rm \
    -v /home/jiwon/workspace:/workspace \
    --gpus all \
    --shm-size=64G \
    --network host \
    --ipc host \
    -e ROS_DOMAIN_ID=77 \
    -e TZ=Asia/Seoul \
    -e TERM=xterm-256color \
    --name tmp_seg \
    st-seg:latest \
    /bin/bash
# install ROS2 humble
# install mmcv using pip install -e .
# install mmseg using pip install -e .


# ########## For MIC Humble ##########
# docker run -it \
#     --rm \
#     -v /home/jiwon/workspace:/workspace \
#     --gpus all \
#     --shm-size=64G \
#     --network host \
#     --ipc host \
#     -e ROS_DOMAIN_ID=77 \
#     -e TZ=Asia/Seoul \
#     -e TERM=xterm-256color \
#     --name VI_DEMO_SEG_INDOOR_UDA \
#     mic_humble:cuda117_torch113 \
#     /bin/bash


########## For MMSEG Humble ##########
# docker run -it \
#     --rm \
#     -v /home/jiwon/workspace:/workspace \
#     --gpus all \
#     --shm-size=64G \
#     --network host \
#     --ipc host \
#     -e ROS_DOMAIN_ID=77 \
#     -e TZ=Asia/Seoul \
#     -e TERM=xterm-256color \
#     --name VI_DEMO_SEG_INDOOR \
#     mmseg_humble:cuda117_torch201 \
#     /bin/bash

# cd mmsegmentation/
# pip install -v -e .

# chown -R 1015:1015 /workspace/demo/ros2_ws

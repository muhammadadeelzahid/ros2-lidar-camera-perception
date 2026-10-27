FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# ROS / simulation / perception toolchain
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-pip \
    ros-humble-vision-msgs \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    ros-humble-robot-state-publisher \
    ros-humble-xacro \
    ros-humble-rviz2 \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-teleop-twist-keyboard \
    ros-humble-velodyne \
    ros-humble-usb-cam \
    ros-humble-realsense2-camera \
    python3-opencv \
    python3-numpy \
    python3-scipy \
    python3-matplotlib \
    python3-yaml \
    && rm -rf /var/lib/apt/lists/*

# CPU torch is enough for inference with the clustering detector and YOLO.
# To train the 3D detector, use a CUDA base image with a matching torch build
# and install OpenPCDet and spconv.
RUN pip3 install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && pip3 install --no-cache-dir ultralytics

WORKDIR /ws
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]

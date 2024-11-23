FROM nvcr.io/nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04

# Install Tools
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y ssh vim \
    git iputils-ping net-tools tar unzip wget curl \
    gcc make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl libncurses5-dev \
    libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev \
    gcc libtinfo-dev zlib1g-dev build-essential libedit-dev libxml2-dev \
    build-essential  ninja-build \
    libosmesa6-dev libxrandr-dev libxinerama-dev libxcursor-dev libegl1-mesa-dev \
    pkg-config
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && echo "root:turing+621615" | chpasswd

# Install Python Libraries
RUN apt update && apt install python3 python3-pip -y && \
    pip3 install --upgrade pip && \
    pip3 install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/ -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 install mmengine==0.10.5 -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 install openmim -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html && \
    pip3 install mmdet==3.0.0 mmyolo==0.6.0 mmsegmentation==1.2.2 mmpretrain==1.2.0 -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 install torch-pruning==1.4.3 onnx==1.17.0 onnxruntime==1.16.2 onnxsim==0.4.36 -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 install ftfy==6.2.3 regex==2024.9.11 thop==0.1.1.post2209072238 -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 install numpy==1.23.0 supervision==0.22.0 transformers==4.36.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 启动命令
WORKDIR /root
RUN mkdir /var/run/sshd
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
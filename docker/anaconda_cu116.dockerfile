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

# Install Anaconda2024.6
WORKDIR /root
RUN wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh && \
    chmod +x Anaconda3-2024.10-1-Linux-x86_64.sh && \
    ./Anaconda3-2024.10-1-Linux-x86_64.sh -b -p /root/anaconda3 && \
    rm Anaconda3-2024.10-1-Linux-x86_64.sh
COPY ./conda_init.sh /root
RUN cat /root/conda_init.sh >> /root/.bashrc
RUN rm /root/conda_init.sh

# 启动命令
WORKDIR /root
RUN mkdir /var/run/sshd
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
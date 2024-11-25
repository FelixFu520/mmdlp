## 开发环境
将oe和mm系列合并, 用于开发环境, 基于`hub.hobot.cc/aitools/ai_toolchain_ubuntu_20_x5_gpu:v1.2.8`镜像

```
# 代理
```

```
docker run --name tt -itd hub.hobot.cc/aitools/ai_toolchain_ubuntu_20_x5_gpu:v1.2.8
docker exec -it tt bash


# 设置环境变量，定义新用户的用户名和密码
NEW_USER=fa.fu
NEW_USER_PASSWORD=fa.fu
# 更新软件包列表并安装必要的软件（如sudo）
apt -y update && apt -y install sudo
# 创建新用户
useradd -m -s /bin/bash $NEW_USER
# 设置新用户密码
echo "$NEW_USER:$NEW_USER_PASSWORD" | chpasswd
# 将新用户添加到sudo组，使其具有sudo权限（可选）
usermod -aG sudo $NEW_USER


# 代理, 将下述内容写到/etc/profile中
export https_proxy=http://10.112.12.231:7890 http_proxy=http://10.112.12.231:7890 all_proxy=socks5://10.112.12.231:7890


# 安装mm系列
pip install mmengine==0.10.3 
pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    mmdet==3.0.0 mmyolo==0.6.0 mmsegmentation==1.2.2 mmpretrain==1.2.0 \
    numpy==1.23.0 supervision==0.22.0 \
    transformers==4.36.2 \
    torch-pruning==1.4.3 ftfy==6.2.3 regex==2024.9.11 thop==0.1.1.post2209072238 \
    onnxsim==0.4.36

docker commit tt hub.hobot.cc/aitools/ai_toolchain_ubuntu_20_x5_gpu:v1.2.8_fa.fu_v3
docker run -itd \
        -v /horizon-bucket/AIoT-data-bucket/:/horizon-bucket/AIoT-data-bucket \
        -v /horizon-bucket/hat_data:/horizon-bucket/hat_data  \
        -v /horizon-bucket/SD_Algorithm:/horizon-bucket/SD_Algorithm    \
        -v /horizon-bucket/aidi_public_data:/horizon-bucket/aidi_public_data    \
        -v /horizon-bucket/d-robotics-bucket:/horizon-bucket/d-robotics-bucket \
        -v /home/users/fa.fu/work:/home/fa.fu/work \
        --name fa.fu_1.2.8 \
        --shm-size 128G \
        --gpus all \
        --privileged=true  \
        -u fa.fu \
        -w /home/fa.fu/work \
        hub.hobot.cc/aitools/ai_toolchain_ubuntu_20_x5_gpu:v1.2.8_fa.fu_v3

```
# Docker

## cuda116 + torch1.13 + openmmlab
```
docker build -f ./mmdlp_cu116.dockerfile -t fusimeng/mmdlp:cu116 .

docker run -itd \
    --name 10105 \
    -v /data/u10104:/root/data \
    -v /home/felix/work/mmdlp:/root/mmdlp \
    -p 10105:22 \
    --shm-size 128G \
    --gpus all \
    --privileged=true \
    -w /root/mmdlp \
    fusimeng/mmdlp:cu116
```

## anaconda cuda116
```
docker build -f ./anaconda_cu116.dockerfile -t fusimeng/anaconda:cu116 .

docker run -itd \
    --name 10104 \
    -v /data/u10104:/root/data \
    -v /home/felix/work/mmdlp:/root/mmdlp \
    -p 10104:22 \
    --shm-size 128G \
    --gpus all \
    --privileged=true \
    -w /root/mmdlp \
    fusimeng/anaconda:cu116
```
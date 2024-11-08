# DeepLearning Projects based on openmmlab
## 介绍
这个库是深度学习项目汇总库, 是基于[OpenMMLab](https://github.com/open-mmlab)的, 会把工作中遇到的内容往里面添加
## 依赖介绍
1. torch
2. openmmlab系列: 
3. onnx系列: onnx, onnxruntime, onnxsim
4. 模型剪枝: 
## 安装
### 开发环境
#### 基于CUDA11.6
```
由于torch和mmcv的安装需要指定pip源, 所以可以先手动安装torch, mmengine, mmcv, 然后在安装requirements/develop_cuda116.txt
pip3 install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip3 install mmengine==0.10.5 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html
pip3 install -r requirements/develop_cuda116.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

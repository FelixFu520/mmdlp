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
```
由于torch和mmcv的安装需要指定pip源, 所以可以先手动安装torch, mmengine, mmcv, 然后在安装requirements/develop.txt
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip3 install mmengine==0.10.5
pip3 install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.4/index.html
pip3 install -r requirements/develop.txt --index-url https://download.pytorch.org/whl/cu118
```

# 把这部分代码添加到train.py中, Build runner 后, 用于临时导出onnx
import torch
class Net(torch.nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        self.model = model 
    def forward(self, x):
        x = self.model.extract_feat(x)
        return self.model.bbox_head.deploy_x_series(x)

net = Net(runner.model)
torch.onnx.export(net, torch.rand((1,3,384,2048)).cuda(), "/home/users/fa.fu/work/yolov5s.onnx", opset_version=11)

import onnx
import onnxsim
model_opt, check_ok =  onnxsim.simplify("/home/users/fa.fu/work/yolov5s.onnx")
onnx.save(model_opt, "/home/users/fa.fu/work/yolov5s-simple.onnx")
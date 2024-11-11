class Net(torch.nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        self.model = model 
    def forward(self, x):
        x = self.model.extract_feat(x)
        return self.model.bbox_head.deploy_x_series(x)

net = Net(self.model)
torch.onnx.export(net, torch.rand((1,3,384,2048)), "/home/users/fa.fu/work/yolov5-1.onnx", opset_version=11)

import onnx
import onnxsim
model_opt, check_ok =  onnxsim.simplify("/home/users/fa.fu/work/yolov5-1.onnx")
onnx.save(model_opt, "/home/users/fa.fu/work/yolov5-2.onnx")
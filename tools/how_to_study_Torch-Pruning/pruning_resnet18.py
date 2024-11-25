import os
from PIL import Image
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch_pruning as tp

WORK_DIR = "/home/users/fa.fu/work/work_dirs/work_dir/output/pruning"
NUM_CLASSES = 100
DEVICE_ID = 4
device = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")
DATA_DIR = "/home/users/fa.fu/work/datasets/imagenet100"
LABELS = os.path.join(DATA_DIR, 'Labels.json')
NUM_EPOCHS = 40
BATCH_SIZE = 512
LR = 0.001
STEPLR_STEP = 400
RESUME = True


# 模型
def get_model():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    if RESUME:
        model.load_state_dict(torch.load(os.path.join(WORK_DIR, "resnet18-best.pth")))
    return model

# 数据
def get_data():
    # 数据转换
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    # 数据加载
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), data_transforms['train']),
        'val': datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), data_transforms['val']),
    }
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=8, persistent_workers=True),
        'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=8, persistent_workers=True),
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    print(f"Classes: {class_names}")
    print(f"Train image size: {dataset_sizes['train']}")
    print(f"Val image size: {dataset_sizes['val']}")

    class_to_idx = image_datasets['train'].class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return dataloaders, dataset_sizes, idx_to_class

# 训练函数
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, dataset_sizes, best_acc=0.0):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}, and learning rate: {scheduler.get_last_lr()}')
        print('-' * 10)

        # 每个epoch都有一个训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()   # 设置模型为评估模式

            running_loss = 0.0
            running_corrects = 0

            # 迭代数据
            for idx, (inputs, labels) in enumerate(dataloaders[phase]):
                if idx % 40 == 0:
                    print(f"*********{phase}, {idx}/{len(dataloaders[phase])}")
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 零参数梯度
                optimizer.zero_grad()

                # 前向传播
                # 跟踪历史如果只需要验证损失
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 只有在训练阶段才反向传播和优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if epoch == 1:
                iterative_steps = 5 # progressive pruning
                imp = tp.importance.TaylorImportance()
                ignored_layers = []
                for m in model.modules():
                    if isinstance(m, torch.nn.Linear) and m.out_features == 100:
                        ignored_layers.append(m) # DO NOT prune the final classifier!
                example_inputs = torch.randn(1, 3, 224, 224).to(device)
                pruner = tp.pruner.MagnitudePruner(
                    model,
                    example_inputs,
                    importance=imp,
                    iterative_steps=iterative_steps,
                    ch_sparsity=0.2, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
                    ignored_layers=ignored_layers,
                )
                loss = model(example_inputs).sum() # a dummy loss for TaylorImportance
                loss.backward() # before pruner.step()
                pruner.step()
                
            if phase == 'train':
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            else:
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                # 保存模型
                _epoch_acc = epoch_acc.detach().cpu().numpy().item()
                if _epoch_acc > best_acc and epoch % 40 == 0:
                    best_acc = _epoch_acc
                    print(f"Save best model at epoch {epoch}, best acc: {best_acc}")
                    torch.save(model.state_dict(), os.path.join(WORK_DIR, f"resnet18-best-{best_acc}-epoch_{epoch}.pth"))

        scheduler.step()
        

# 图片预处理
def process_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # 添加批次维度
    return image

# 获取类别名称
def get_class_names(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    class_names = {k: v for k, v in data.items()}
    return class_names

# 推理图片
def predict(image_path, model, class_names):
    image = process_image(image_path)
    image = image.to(device if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        outputs = model(image)
        score, predicted = torch.max(outputs, 1)
    return predicted.cpu().numpy().item(), score.cpu().numpy().item()

# 推理单张图片
def infer_one_image(image_path, model, idx_to_class):
    class_names = get_class_names(os.path.join(DATA_DIR, "Labels.json"))  # 获取类别名称

    # 推理并打印结果
    class_index, score = predict(image_path, model, class_names)
    print(f'Predicted class no:{idx_to_class[class_index]}, class: {class_names[idx_to_class[class_index]]}, score: {score}')




if __name__ == "__main__":
    if not os.path.exists(WORK_DIR):
        os.makedirs(WORK_DIR)

    # 模型
    model = get_model()
    model = model.to(device)

    # 数据
    dataloaders, dataset_sizes, idx_to_class = get_data()

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEPLR_STEP, gamma=0.1)

    # 训练模型
    train_model(model=model, dataloaders=dataloaders, criterion=criterion, optimizer=optimizer, scheduler=scheduler, num_epochs=NUM_EPOCHS, dataset_sizes=dataset_sizes)

    # 推理单张图片
    infer_one_image(os.path.join(DATA_DIR, 'val', 'n01443537', 'ILSVRC2012_val_00000236.JPEG'), model, idx_to_class)
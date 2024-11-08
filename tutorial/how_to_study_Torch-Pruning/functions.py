import torch
from torchvision.models import resnet18
import torch_pruning as tp
import shutil
import os

# *********** Naive pruning
model = resnet18(pretrained=True).eval()
tp.prune_conv_out_channels(model.conv1, idxs=[0,1]) # remove channel 0 and channel 1
# output = model(torch.randn(1,3,224,224)) # error
print(model.conv1)
print(model.bn1)




# *********** A Minimal Example
model = resnet18(pretrained=True).eval()
# 1. build dependency graph for resnet18
DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1,3,224,224))

# 2. Specify the to-be-pruned channels. Here we prune those channels indexed by [2, 6, 9].
group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=[2, 6, 9] )

# 3. prune all grouped layers that are coupled with model.conv1 (included).
if DG.check_pruning_group(group): # avoid full pruning, i.e., channels=0.
    group.prune()

# 4. Save & Load
model.zero_grad() # We don't want to store gradient information
torch.save(model, 'model.pth') # without .state_dict
model = torch.load('model.pth') # load the model object
os.remove('model.pth') # remove the model file



# *********** Pruning with High-level Pruners
model = resnet18(pretrained=True)

# Importance criteria
example_inputs = torch.randn(1, 3, 224, 224)
imp = tp.importance.TaylorImportance()

ignored_layers = []
for m in model.modules():
    if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
        ignored_layers.append(m) # DO NOT prune the final classifier!

iterative_steps = 5 # progressive pruning
pruner = tp.pruner.MagnitudePruner(
    model,
    example_inputs,
    importance=imp,
    iterative_steps=iterative_steps,
    ch_sparsity=0.5, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
    ignored_layers=ignored_layers,
)

base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
for i in range(iterative_steps):
    if isinstance(imp, tp.importance.TaylorImportance):
        # Taylor expansion requires gradients for importance estimation
        loss = model(example_inputs).sum() # a dummy loss for TaylorImportance
        loss.backward() # before pruner.step()
    pruner.step()
    macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
    # finetune your model here
    # finetune(model)
    # ...



# *********** Sparse Training with High-level Pruners
# for epoch in range(epochs):
#     model.train()
#     for i, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         out = model(data)
#         loss = F.cross_entropy(out, target)
#         loss.backward()
#         pruner.regularize(model) # <== for sparse learning
#         optimizer.step()
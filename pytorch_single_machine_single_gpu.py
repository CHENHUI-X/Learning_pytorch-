'''
Use single-device training if the data and model can fit in one GPU,
and training speed is not a concern.
'''

import torch
torch.cuda.is_available() # if you have set cuda ,it will be True

# Actually , in this py , lets assume we just have one gpu
torch.cuda.current_device() # returns 0 in this case
torch.cuda.device_count() # returns 1 in this case
torch.cuda.get_device_name(0) # returns your gpu name

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Then define your model and dataloader
model = ...
train_loader , val_loader = ... , ...
model.to(device) # set your model to gpu
'''
Then these methods will recursively go over all modules 
and convert their parameters and buffers to CUDA tensors
'''

for i ,(train_x,label ) in enumerate(train_loader):
    train_x = train_x.to(device)
    label = label.to(device)
    '''
    Please note that just calling my_tensor.to(device) 
    returns a new copy of my_tensor on GPU instead of 
    rewriting my_tensor. You need to assign it to a new 
    tensor and use that tensor on the GPU.
    '''

    output = model(train_x)
    ...

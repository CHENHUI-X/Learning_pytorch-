'''
In this py , lets assume we have one machine but
have more than one gpu .
'''

# ----------------------------------------------------------------
# ----------------------------------------------------------------
'''
--  nn.DataParallel is single-process, multi-thread, 
    and only works on a single machine.
    
--  This container parallelizes the application of the given module 
    by splitting the input across the specified devices (GPU) 
    by chunking in the batch dimension (other objects will be 
    copied once per device). In the forward pass, the module is 
    replicated on each device, and each replica handles a portion 
    of the input. During the backwards pass, gradients from 
    each replica are summed into the original module.
    
--  Notice :

    *** 
        Because GIL of Python interpreter,although you have 8 gpus,
        but in fact you only have one GPU computing each moment, 
        and you don't get the full performance of 8 GPUs.
    ***
    ***
        It is recommended to use "DistributedDataParallel", 
        instead of "DataParallel" to do multi-GPU training, 
        even if there is only a single node. 
    ***
    ***
        But for learning, we will still write the DataParallel example.
        Actually , the DataParallel and DistributedDataParallel also 
        works with one gpu .
    ***

'''

import torch
import torch.nn as nn

torch.cuda.is_available() # if you have set cuda ,it will be True

# Actually , in this py , lets assume we just have one gpu
torch.cuda.current_device() # returns 0 in this case
torch.cuda.device_count() # returns 8 if you have 8 gpu
torch.cuda.get_device_name(0) # returns your gpu name

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Then define your model and dataloader
model = (... ,
         print("\tIn Model: input size", input.size(),
          "output size", output.size())
         )
train_loader , val_loader = ... , ...

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [64, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)

model.to(device)

for i ,(train_x,label ) in enumerate(train_loader):
    train_x = train_x.to(device)
    label = label.to(device)
    output = model(train_x)
    print("Outside: input size", input.size(),
          "output_size", output.size())
'''
Assume we input shape(-1,5) and output shape(-1,2) , batch size is 64
# ------------------------------------------------------
If you have 1 GPU ,(and total 100 elements ) the output should like :

        In Model: input size torch.Size([64, 5]) output size torch.Size([64, 2])
Outside: input size torch.Size([64, 5]) output_size torch.Size([64, 2])
        In Model: input size torch.Size([36, 5]) output size torch.Size([36, 2])
Outside: input size torch.Size([36, 5]) output_size torch.Size([36, 2])
        
# ------------------------------------------------------
If you have 8 GPU ,(and total 100 elements ) the output should like :

Let's use 8 GPUs!
    In Model: input size torch.Size([8, 5]) output size torch.Size([8, 2])
    In Model: input size torch.Size([8, 5]) output size torch.Size([8, 2])
    In Model: input size torch.Size([8, 5]) output size torch.Size([8, 2])
    In Model: input size torch.Size([8, 5]) output size torch.Size([8, 2])
    In Model: input size torch.Size([8, 5]) output size torch.Size([8, 2])
    In Model: input size torch.Size([8, 5]) output size torch.Size([8, 2])
    In Model: input size torch.Size([8, 5]) output size torch.Size([8, 2])
    In Model: input size torch.Size([8, 5]) output size torch.Size([8, 2])
Outside: input size torch.Size([64, 5]) output_size torch.Size([64, 2])
    In Model: input size torch.Size([8, 5]) output size torch.Size([8, 2])
    In Model: input size torch.Size([8, 5]) output size torch.Size([8, 2])
    In Model: input size torch.Size([8, 5]) output size torch.Size([8, 2])
    In Model: input size torch.Size([8, 5]) output size torch.Size([8, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
Outside: input size torch.Size([36, 5]) output_size torch.Size([36, 2])
---------------------------------------------------------------------------
The output of inside and outside is different , which means the input is 
is split into 8 pieces (if possible) and sent to each GPU separately.
---------------------------------------------------------------------------
DataParallel splits your data automatically and sends job orders to 
multiple models on several GPUs. After each model finishes their job, 
DataParallel collects and merges the results before returning it to you.

'''

# ****************************************************************************************
# ****************************************************************************************

'''
--  DistributedDataParallel is multi-process, can works on multi machine ,
    multi GPU . (surely it can works on single machine ) ,
    and it is recommended to use DistributedDataParallel, 
    instead of DataParallel to do multi-GPU training, 
    even if there is only a single node.

--  This container parallelizes the application of the given module by 
    splitting the input across the specified devices by chunking in the 
    batch dimension. The module is replicated on each machine and each device, 
    and each such replica handles a portion of the input. 
    During the backwards pass, gradients from each node are averaged.
    
---     
    The difference between DistributedDataParallel and 
    DataParallel is: "DistributedDataParallel" uses "multiprocessing" 
    where a process is created for each GPU, 
    while "DataParallel" uses "multithreading". 
    By using multiprocessing, each GPU has its dedicated 
    process, this avoids the performance overhead caused 
    by GIL of Python interpreter.
    
    DistributedDataParallel is proven to be significantly faster than 
    torch.nn.DataParallel for single-node multi-GPU data parallel training.
    
--  Notice :

    *** 
        To use DistributedDataParallel on a host(node or machine) with N GPUs, 
        you should spawn up N processes, ensuring that each process exclusively
        works on a single GPU from 0 to N-1. 
        This can be done by either setting CUDA_VISIBLE_DEVICES 
        for every process or by calling:
            torch.cuda.set_device(i)
        where i is from 0 to N-1.   
    ***
    ***
        Some useful recommends see 
            https://mrxiao.net/torch-DistributedDataParallel.html
            https://blog.csdn.net/daixiangzi/article/details/106162971
            https://gist.github.com/sgraaf/5b0caa3a320f28c27c12b5efeb35aa4c
    
    ***        
    
    
    
'''

'''
Let us start with a simple torch.nn.parallel.DistributedDataParallel example. 
This example uses a torch.nn.Linear as the local model, wraps it with DDP, 
and then runs one forward pass, one backward pass, and an optimizer step on 
the DDP model. After that, parameters on the local model will be updated, 
and all models on different processes should be exactly the same.

'''
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


def example(rank, world_size):
    # create default process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    '''
                |    Node1  |   Node2    |
    ____________| p1 |  p2  |  p3  |  p4 |
    local_rank  | 0  |   1  |  0   |   1 |
    rank        | 0  |   1  |  2   |   4 |
    
    here :
    world_size = 4
    rank = [ 0 , 1 , 2 , 3 ]
    there 2 gpu for every node (machine),total 4
    '''
    # create local model
    model = nn.Linear(10, 10).to(rank)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # forward pass
    outputs = ddp_model(torch.randn(20, 10).to(rank))
    labels = torch.randn(20, 10).to(rank)
    # backward pass
    loss_fn(outputs, labels).backward()
    # update parameters
    optimizer.step()

def main():
    world_size = 4
    mp.spawn(example,
        args=(world_size,),
        nprocs=world_size,
        join=True)

if __name__=="__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()












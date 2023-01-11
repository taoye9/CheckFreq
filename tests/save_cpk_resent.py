import torch
from torch import optim
from torch.multiprocessing import Pool, Process, set_start_method, Manager, Value, Lock

t = Manager()

import torchvision

import sys
sys.path.append('../')
sys.path.append('../src')

chk_prefix = "./chk"

from cf_checkpoint import CFCheckpoint
from cf_manager import CFManager, CFMode

model = torchvision.models.resnet50()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

print(optimizer)

chk = CFCheckpoint(model=model, optimizer=optimizer)

cf_manager = CFManager(chk_prefix, chk)

cf_manager.save(synchronous=True, persist=True)


import torch
import torch.optim as optim
import torch.nn as nn
from models import Net
from dataloader import train_loader1,train_loader2
from utils import OneHotEncoding2


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")



def train(model, device, train_loader1, train_loader2, optimizer, epoch):
  
  model.train()
  for batch_idx in range(len(train_loader1)):
    
    x1,y1 = next(iter(train_loader1))
    x2,y2 = next(iter(train_loader2))
    x1,x2 = x1.to(device),x2.to(device)
    y1 = y1.to(dtype = torch.long).to(device)
    y2 = y2.to(dtype = torch.long).to(device)
    target = OneHotEncoding2(y1,y2,32).to(device)

    optimizer.zero_grad()
    output = model(x1,x2)
    loss = nn.MSELoss()
    l_value = loss(output,target)

    l_value.backward()
    optimizer.step()
    if batch_idx % 100 ==0:
        print(l_value.item(),' ',batch_idx)



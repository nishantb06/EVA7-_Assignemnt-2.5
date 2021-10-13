import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from utils import OneHotEncoding

BATCH_SIZE = 32
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}



train_loader1 = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
                    batch_size=BATCH_SIZE, shuffle=True,**kwargs)

class RandomNumbers(Dataset):
  def __init__(self,length_of_dataset):

    self.data = torch.randint(
                            size = (length_of_dataset,),
                            low = 0, 
                            high = 1
                            )
  
  def __len__(self):
    return len(self.data)

  def __getitem__(self,idx):
    return OneHotEncoding(self.data[idx]), self.data[idx].to(dtype = torch.float32)

train_loader2 = torch.utils.data.DataLoader(RandomNumbers(60_000), batch_size = BATCH_SIZE, shuffle=True)


test_loader1 = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=32, shuffle=True, **kwargs)

test_loader2 = torch.utils.data.DataLoader(RandomNumbers(10_000), batch_size = 32, shuffle=True)
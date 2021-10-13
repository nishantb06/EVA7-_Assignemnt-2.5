import torch

def OneHotEncoding(n):
  t = torch.zeros(10)
  t[n] = 1
  return t

def OneHotEncoding2(y1,y2,batch_size):
    t = torch.zeros(batch_size,29)
    for i in range(batch_size):
        t[i,y1[i]] = 1
        t[i,10+y1[i]+y2[i]] = 1

    return t
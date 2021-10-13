from utils import OneHotEncoding2
import torch
import torch.nn as nn

def test(model,device,test_loader1,test_loader2):
  model.eval()
  test_loss = 0
  correct = 0
  both_correct = 0
  image_correct = 0
  number_correct = 0

  with torch.no_grad():
      for batch_idx in range(len(test_loader1)):
          x1,y1 = next(iter(test_loader1))
          x2,y2 = next(iter(test_loader2))
          x1,x2 = x1.to(device),x2.to(device)
          y1 = y1.to(dtype = torch.long).to(device)
          y2 = y2.to(dtype = torch.long).to(device)
          target = OneHotEncoding2(y1,y2,32).to(device)

          output = model(x1,x2)
          loss = nn.MSELoss()
          test_loss += loss(output,target)
          pred1,pred2 = output[:,:10].argmax(dim=1,keepdim = True), output[:,10:].argmax(dim=1,keepdim = True)
          image_correct += pred1.eq(y1.view_as(pred1)).sum().item()
          number_correct += pred2.eq(y2.view_as(pred2)).sum().item()
          both_correct += (pred1.eq(y1.view_as(pred1)) & pred2.eq(y2.view_as(pred2))).sum().item()
          
      test_loss /= len(test_loader1.dataset)

      print('Average  loss = ',test_loss)
      print('both correct = ',both_correct,'Accuracy = ',both_correct*100/len(test_loader1.dataset))
      print('image correct = ',image_correct,'Accuracy = ',image_correct*100/len(test_loader1.dataset))
      print('number correct = ',number_correct,'Accuracy = ',number_correct*100/len(test_loader1.dataset))
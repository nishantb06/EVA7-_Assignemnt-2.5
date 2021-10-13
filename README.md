# EVA7 - Assignement 2.5
### A neural network that can take two inputs - an image from the MNSIT dataset and a random number(between 0 and 9) and gives two outputs - digit on the image and the sum of the two numbers
---
### Nishant Bhansali
#### nishantbhansali80@gmail.com
 ----
## Metric
********
The metric used for this project was  _Accuracy_ .
I wanted to treat this problem as a **Multi class,Multi label Classification** problem instead of a regression problem. Therefore 3 accuracies have been used to evaluate the model(on a test set of 10,000 images)
1. **Both correct** - Number of instances where the model correctly predicted the MNIST digit as well as the sum
2. **Image Correct** - Number of instances where the model correctly predicted the MNIST digit
3. **Number Correct** - Number of instances where the model correctly predicted the sum of the two numberes


## Results
********
| Metric | Accuracy |
| ---- | :--- |
| Both Correct | 9.52 % |
| Image  Correct | 53.49 % |
| Both Correct | 13.68 % |

![Imgur](https://i.imgur.com/E8vH62r.png)

## Data Representation
********
1. **x1** : Tensor form of the images from the MNIST dataset given to the model - each image is 1x28x28 with a batch size of 32. Intitial part of the model(CNN) has been trained using only x1. After the last convolutions layer (which has 10 channels)), it is also connected to a FC layer with 20 neurons where x2 is concated with this to make it a 30 neuron layer.
2. **y1** : Class to which the corresponding x1 belongs to. Also a pytorch tensor 
3. **x2** : The random number which is OneHotEncoded and thus is 1x10 dimensional tensor. The RandomNumbers class defined in dataloaders.py is used to generate a dataset of random numbers. Over-writing __len__ and __getitem__ functions has been done here.
4. **y2** : The random number for which the OneHot Encoding was generated
5. **Output** - output of the model is a 29 dimensional vector. the first 10 neurons are used to predict the class of the MNIST image and the last 19 neurons are used to predict the sum of MNIST image and the random number. Log_Softmax has also been applied.

## File Structure 
********
The project contain 4 files
1. dataloader.py - Contains code for the train and test data loaders  for the MNIST digits as well as  custom data loader written for the random numbers which are passed as an input to the model for training
2. models.py  - model architecure is defined in this file.The model architecture is as follows.
X1 -> Conv(X1) -> Conv(X1) -> MaxPool(X1) -> Conv(X1) -> Conv(X1) ->  MaxPool(X1) -> Conv(X1) -> Conv(X1) ->  MaxPool(X1) -> Conv(X1) -> Conv(X1) ->  Conv(X1) -> FC(X1) -> X1+X2 -> FC(X1+X2) -> FC (X1+X2) -> output
3. utils.py contains two functions defined used for OneHotEncoding the data representations.
4. train.py - train loop has been defined over here
5. test.py - test loop has been defined over here
6. README.md - Readme file which contains the project description
7. EVA7-2.5.ipynb - Colab notebook shows how the model has been trained,logs can be found over here

![Imgur](https://imgur.com/LhuBsWB.png)

## Loss Function
********
Mean Squared error has been chosen as loss function for this task. I treat this problem as a multiclass.multilabel classification problem (I dont know any other loss functions suitable for this task). Therefore, I aim to reduce the L2 norm between the two 29 dimensional vectors(output of my model and the target).

[0.01 , 0.01 , 0.01 , 0.95 ..... , 0.01 ,0.01 ,0.01 ...... ,0.98 ,0.01 ,0.01 ] 

and

[0.00 , 0.00 , 0.00 , 1.00 ..... , 0.00 ,0.00 ,0.00 ...... ,1.00 , 0.0 , 0.0 ]

----


## Statoil/C-CORE Iceberg Classifier Challenge (Kaggle)
Ship or iceberg, can you decide from space?

This project is used to demonstrate the use of a convolutional neural network in tensorflow on the Statoil/C-CORE Iceberg Classifier Challenge dataset found at Kaggle’s [website](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/data)

The purpose of this is to demonstrate the use of tensorflow to create a convolutional neural net that classifies an iceberg from a boat.

## Getting Started
This script uses python 3.6 and tensorflow 1.1.0
Supporting libraries are:
* **pandas** for data structuring
* **numpy** for linear algebra (mostly dealing with matrices)
* **sklearn** for log loss metrics for this particular competition
* **keras** for configuring the CNN model and pipelining the data augmentation process 

## Data
Here the data is represented in flattened 75x75 pixels of radar bands.
### Ship Images:
![alt text](https://i.imgur.com/wmnljrR.png "ship")
### Iceberg Images:
![alt text](https://i.imgur.com/ZrqG4aL.png "iceberg")

## Usage
We recommend using a python virtual environment specifically for use of tensorflow. I am using anaconda with a conda virtual environment



## Model Improvements 

1. [Base CNN model](https://github.com/dbrownambi/iceberg-ship-classification/blob/master/Models/Base%20CNN%20model.py):
* Batch Size:   32 
* Epoch:  20
* Steps/Epoch: 1283
* Channels: 2
* Augmentation: No Augmentation
* Callbacks: No callbacks
* Dropout: 25% in the Convolution Layer 2 & 3 and 50 % dropout in Fully connected MLP
* Batch Normalization: Applied in only the Convolution Layers
* CNN Layers :  3 layers
* MLP hidden layers: 2 layers
* Activation function: RELU for all except output with sigmoid function

2. [CNN-1 model](https://github.com/dbrownambi/iceberg-ship-classification/blob/master/Models/CNN-1%20model.py):
* Batch Size:   32       
* Epoch:  20        
* Steps/Epoch:   42        
* Channels: 3      
* Augmentation: Datagenerator (shift,rotate, width and height adjustments)        
* Callbacks: Annealer learning rate scheduler       
* Dropout: 20 to 30 % dropout in last layer of CNN and Fully connected MLP.
* Batch Normalization: Not Used.
* CNN Layers :  4 layers              
* MLP hidden layers: 2 layers
* Activation function: RELU for all except output with sigmoid function.

3. [CNN-2 model](https://github.com/dbrownambi/iceberg-ship-classification/blob/master/Models/CNN-2%20model.py): 
* Batch Size:   32       
* Epoch:  20        
* Steps/Epoch:   42        
* Channels: 3      
* Augmentation: Datagenerator (shift,rotate, width and height adjustments)        
* Callbacks: Annealer learning rate scheduler       
* Dropout: 20 to 30 % dropout in all layers of CNN and Fully connected MLP.
* Batch Normalization: Applied to all layers of CNN and MLP.
* CNN Layers :  4 layers              
* MLP hidden layers: 2 layers
* Activation function: RELU for all except output with sigmoid function.

4. [CNN-3 model](https://github.com/dbrownambi/iceberg-ship-classification/blob/master/Models/CNN-3%20model.py): 
* Batch Size:   32       
* Epoch:   50        
* Steps/Epoch:  42        
* Channels: 3      
* Augmentation: Datagenerator (shift,rotate, width and height adjustments)        
* Callbacks: Annealer learning rate scheduler       
* Dropout: 20 to 30 % dropout in all layers of CNN and Fully connected MLP.
* Batch Normalization:  Not used
* CNN Layers :  4 layers              
* MLP hidden layers: 2 layers
* Activation function: RELU for all except output with sigmoid function.

5. [CNN-4 model](https://github.com/dbrownambi/iceberg-ship-classification/blob/master/Models/CNN-4%20model.py): 
* Batch Size:   32       
* Epoch:   30        
* Steps/Epoch:  42        
* Channels: 3      
* Augmentation: Datagenerator (shift,rotate, width and height adjustments)        
* Callbacks: Annealer learning rate scheduler       
* Dropout: 20 to 30 % dropout in all layers of CNN and Fully connected MLP.
* Batch Normalization:  Not used            
* CNN Layers :  4 layers              
* MLP hidden layers: 2 layers
* Activation function: RELU for all except output with sigmoid function.
            
6. [CNN-5 model](https://github.com/dbrownambi/iceberg-ship-classification/blob/master/Models/CNN-5%20model.py): 
* Batch Size:   32       
* Epoch:  10      
* Steps/Epoch:  42        
* Channels: 3      
* Augmentation: Datagenerator (shift,rotate, width and height adjustments)        
* Callbacks: Annealer learning rate scheduler       
* Dropout: 20 to 30 % dropout in all layers of CNN and Fully connected MLP.
* Batch Normalization:  Not used
* L2 Regularization:  alpha =0.01
* CNN Layers :  4 layers
* MLP hidden layers: 2 layers
* Activation function: RELU for all except output with sigmoid function.

7. [CNN-6 model](https://github.com/dbrownambi/iceberg-ship-classification/blob/master/Models/CNN-6%20model.py): 
* Batch Size:   32       
* Epoch:  30     
* Steps/Epoch:  42        
* Channels: 3      
* Augmentation: Datagenerator (shift,rotate, width and height adjustments)        
* Callbacks: Reduce Learning rate on Plateau     
* Dropout: 20 to 30 % dropout in all layers of CNN and Fully connected MLP.
* Batch Normalization:  Not used
* CNN Layers :  4 layers              
* MLP hidden layers: 2 layers
* Activation function: RELU for all except output with sigmoid function.

8. [CNN-7 model](https://github.com/dbrownambi/iceberg-ship-classification/blob/master/Models/CNN-7%20model.py):
* Batch Size:   32       
* Epoch:   20       
* Steps/Epoch:  128        
* Channels: 3      
* Augmentation: Datagenerator (shift,rotate, width and height adjustments)        
* Callbacks: Annealer learning rate scheduler       
* Dropout: 20 to 30 % dropout in all layers of CNN and Fully connected MLP.
* Batch Normalization:  Not used
* CNN Layers :  4 layers              
* MLP hidden layers: 2 layers
* Activation function: RELU for all except output with sigmoid function.
     

## Conclusion

* From my limited observations, this ConvNet was able to get a Log Loss score of around 0.194~0.205+ and has the potential to go deep by further regularization and optimizing such as to extend the training epochs, adjust early stopping, tweak optimizer, etc
* Finally, I would like to thank shivam207 for his [solution](https://github.com/shivam207/iceberg_challenge) implemented for the Iceberg Classifier Challenge. I learnt a lot for how to reshape and normalize the dataset features from his solution.



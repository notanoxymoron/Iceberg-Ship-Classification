# Iceberg or Ship Classification
### This project is used to demonstrate the use of a convolutional neural networks with Tensorflow on the dataset. 
The complete project report can be found [here](https://github.com/dbrownambi/iceberg-ship-classification/blob/master/Project%20Report.pdf). This is a Binary classification problem. The challenge is, given a list of images, predict whether an image contains a ship or an iceberg. The dataset used for this project is the Statoil/C-CORE Iceberg Classifier Challenge dataset found [here](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/data). The data (train.json, test.json) is presented in json format. The labels are provided by human experts and geographic knowledge on the target. All the images are 75x75 images with two bands.

The figure below shows the satellite captured images of both icebergs and ships:

![alt text](https://storage.googleapis.com/kaggle-media/competitions/statoil/NM5Eg0Q.png "satellite_imge")


The purpose of this project is to demonstrate the use of tensorflow/keras frameworks to create a convolutional neural net that classifies an iceberg from a boat:

![alt text](https://storage.googleapis.com/kaggle-media/competitions/statoil/8ZkRcp4.png "iceberg")

![alt text](https://storage.googleapis.com/kaggle-media/competitions/statoil/M8OP2F2.png "ship")


The project used [multiple iterations](https://github.com/dbrownambi/iceberg-ship-classification/tree/master/Models) of the basline CCN Model for training on the dataset. Based on the final accuracy result, it can be concluded that the **CNN-4 model** gave the best performance as shown in the table below:

![alt text](https://github.com/dbrownambi/iceberg-ship-classification/blob/master/Figures/final_table.JPG "final_table")

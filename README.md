#Perceptron vs VotedPerceptron

Luca Ciabini.

The code is written for python 3.x.

Into the dataSets folder there are 3 datasets one of which already divided. 
To instantiate a new Banknote or HTRU_2 dataset just pass in the constructor the name of the directory for Banknote and HTRU_2 dataset files(for example `dataset(filename='dataSets/data_banknote_authentication.txt')`). In order to create a Dorothea dataset, since it is already splitted, you need to call the dataSet class constructor with the value dorothea=True (`dataset(dorothea=True)`) that performes some operation to create a unique dataset from the two files. 
A dataset object (no matter what kind of dataset it represent) contain the examples matrix, the labeles vector, the dimension, and the number of features. Getter methods are provided.
Dataset class also has the function `minmaxScale()` that perform a minmax scalation over the dataset(useless for binary datasets).

In order to make an holdoutcross validation or a k-foldcross validation call the method `holdoutCrossValidation(dataset,scale)` where scale, if True, stands for "try even with the scaled version of the dataset" or `kFoldCrossValidation(k,dataset)` respectly.
The accuracy and the confusions matrix will be automatically printed by the test function which is called by the two validation algorithm.
To change the max number of iteration of the perceptron algorithm or the voted perceptron epochs just go to validation.py and search `perceptron.train(50)`,`votedPerceptron=vp.VotedPerceptron(trainingSet,5)` then change the numbers as written in comments in the code.

Perceptron and voted perceptron classes provide the method to train and to predict.

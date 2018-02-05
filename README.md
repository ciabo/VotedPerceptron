# VotedPerceptron
Into the dataSets folder there are 3 datasets one of which already divided. To instantiate a new dataset just pass in the constructor
the name of the directory for Banknote and HTRU_2 (for example dataset(filename='dataSets/data_banknote_authentication.txt')) or use
dorotheaTrain=True or dorotheaTest=True for the training set and the testset of dorothea respectly.
In order to make an holdoutcross validation or k-foldcross validation for banknote or htru dataset just call the method
holdoutCrossValidation or kFoldCrossValidation (that takes the number k) of Validation file.
For Dorothea dataset just call the test function in the "Validation" file passing the train and the test.
The accuracy and the confusions matrix will be automatically printed.

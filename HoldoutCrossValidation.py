import numpy as np;
import Perceptron as p
import VotedPerceptron as vp
import DataSet as d
import random

def splitAndTest(dataset):
    x=dataset.getx();
    y=dataset.gety();
    dim=dataset.getDimension();
    #70% train - 30% test
    traindim=dim//100*70;
    #create a random list of values to decide the examples that will be the train
    randomList=random.sample(range(0, dim), traindim);
    #sort reverse the list in order to delete the values form the originale dataset without errors
    randomList.sort(reverse=True);
    xtrain=np.empty(shape=(traindim,dataset.getFeaturesNumber()));
    ytrain=np.empty(traindim);

    for i in range(0, traindim):
        xtrain[i]=x[randomList[i]];
        x = np.delete(x, randomList[i],0);
        ytrain[i]=y[randomList[i]];
        y = np.delete(y, randomList[i]);

    xtest = x;
    ytest = y;

    trainingSet=d.DataSet(xtrain,ytrain,traindim,dataset.getFeaturesNumber());
    testSet=d.DataSet(xtest,ytest,dim-traindim,dataset.getFeaturesNumber());
    test(trainingSet,testSet);


def test(trainingSet, testSet):
    perceptron=p.Perceptron(trainingSet);
    perceptron.train(100); # <--- Change the number to change the max iterations of the perceptron
    votedPerceptron=vp.VotedPerceptron(trainingSet,5); # <--- Change the number to change the epochs of the voted Perceptron
    votedPerceptron.train()

    perceptronErrors = 0;
    votedPerceptronErrors = 0;
    pconfusionMatrix=np.zeros(shape=(2,2)); # (rows -> real) row 0 = +1 row 1 = -1 | (column -> predicted) column 0 = +1 column 1 = -1
    vpconfusionMatrix = np.zeros(shape=(2, 2));

    for i in range(0,testSet.getDimension()):

        perceptronPrediction=perceptron.predict(testSet.getx()[i]);
        if perceptronPrediction!=testSet.gety()[i] :
            perceptronErrors+=1;
            if perceptronPrediction==1:
                pconfusionMatrix[0][1]+=1;
            else:
                pconfusionMatrix[1][0] += 1;
        else:
            if perceptronPrediction==1:
                pconfusionMatrix[0][0]+=1;
            else:
                pconfusionMatrix[1][1] += 1;

        votedPerceptronPrediction=votedPerceptron.predict(testSet.getx()[i]);
        if votedPerceptronPrediction!=testSet.gety()[i] :
            votedPerceptronErrors+=1;
            if votedPerceptronPrediction==1:
                vpconfusionMatrix[0][1]+=1;
            else:
                vpconfusionMatrix[1][0] += 1;
        else:
            if votedPerceptronPrediction==1:
                vpconfusionMatrix[0][0]+=1;
            else:
                vpconfusionMatrix[1][1] += 1;

    perceptronAccuracy = round(((testSet.getDimension()-perceptronErrors)/testSet.getDimension())*100,2);
    votedPerceptronAccuracy = round(((testSet.getDimension() - votedPerceptronErrors) / testSet.getDimension()) * 100,2);

    print("Test set dimensions: ",testSet.getDimension());
    print("Voted perceptron errors: ",votedPerceptronErrors," | Accuracy: ",votedPerceptronAccuracy,"%");
    print(vpconfusionMatrix);
    print("Perceptron errors: ",perceptronErrors," | Accuracy: ",perceptronAccuracy,"%");
    print(pconfusionMatrix);
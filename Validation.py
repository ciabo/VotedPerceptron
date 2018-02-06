import numpy as np;
import Perceptron as p
import VotedPerceptron as vp
import DataSet as d
import random

def holdoutCrossValidation(dataset):
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

    #delete selected random values from x and put it into xtrain, same things for y
    for i in range(0, traindim):
        xtrain[i]=x[randomList[i]];
        x = np.delete(x, randomList[i],0);
        ytrain[i]=y[randomList[i]];
        y = np.delete(y, randomList[i]);

    xtest = x;
    ytest = y;

    trainingSet=d.DataSet(xtrain,ytrain,traindim,dataset.getFeaturesNumber());
    testSet=d.DataSet(xtest,ytest,dim-traindim,dataset.getFeaturesNumber());
    print("NOT MINMAXSCALED");
    test(trainingSet,testSet);
    #print("MINMAXSCALED");
    #trainingSet.minmaxScale();
    #testSet.minmaxScale();
    #list=test(trainingSet,testSet);
    return list;

def kFoldCrossValidation(k,dataset):
    x = dataset.getx();
    y = dataset.gety();
    dim = dataset.getDimension();
    subDataDim=dim//k;
    #mix the dataset
    randomx=x;
    randomy=y;
    randomList = random.sample(range(0, dim), dim);
    for i in range(0,dim):
        randomx[i]=x[randomList[i]];
        randomy[i]=y[randomList[i]];
    x=randomx;
    y=randomy;
    perceptronmatrix=np.zeros(shape=(2,2))
    list=[]
    for i in range(0,k):
        if i==k-1:
            xtest = x[i * subDataDim:];
            ytest = y[i * subDataDim:];
            xtrain = x[0:i * subDataDim];
            ytrain = y[0:i * subDataDim];
        else:
            xtest = x[i * subDataDim:i * subDataDim + subDataDim];
            ytest = y[i * subDataDim:i * subDataDim + subDataDim];
            if i==0:
                xtrain=x[i * subDataDim + subDataDim:];
                ytrain=y[i * subDataDim + subDataDim:];
            else:
                xtrain = x[:i * subDataDim];
                xtrain=np.concatenate((xtrain,x[i * subDataDim + subDataDim:]))
                ytrain = y[:i * subDataDim];
                ytrain=np.append(ytrain,y[i * subDataDim + subDataDim:])

        trainingSet = d.DataSet(xtrain, ytrain, len(ytrain), dataset.getFeaturesNumber());
        testSet = d.DataSet(xtest, ytest, len(ytest), dataset.getFeaturesNumber());
        test(trainingSet,testSet);


def test(trainingSet, testSet):
    perceptron=p.Perceptron(trainingSet);
    perceptron.train(50); # <--- Change the number to change the max iterations of the perceptron
    votedPerceptron=vp.VotedPerceptron(trainingSet,5); # <--- Change the number to change the epochs of the voted Perceptron
    votedPerceptron.train();

    perceptronErrors = 0;
    votedPerceptronErrors = 0;
    pconfusionMatrix=np.zeros(shape=(2,2)); # (rows -> real) row 0 = +1 row 1 = -1 | (column -> predicted) column 0 = +1 column 1 = -1
    vpconfusionMatrix = np.zeros(shape=(2, 2));

    #real test where predict each value in the test and check if correct. Create also the confusions matrix and take note of the errors
    for i in range(0,testSet.getDimension()):
        #for the perceptron
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
        #for the votedPerceptron
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
    list=[perceptronAccuracy,votedPerceptronAccuracy,pconfusionMatrix,vpconfusionMatrix]
    return list
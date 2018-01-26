import numpy as np

class DataSet:
    def __init__(self,x=0,y=0,dimension=0,featuresNumber=0,filename='',dorotheaTrain=False,dorotheaTest=False):
        self.x=x;
        self.y=y;
        self.dimension=dimension;
        self.featuresNumber=featuresNumber;
        if filename!='':
            self.populateByFilename(filename);
        if dorotheaTrain==True:
            self.populateWithDorotheaTrain();
        if dorotheaTest==True:
            self.populateWithDorotheaTest();

    def getx(self):
        return self.x;

    def gety(self):
        return self.y;

    def getDimension(self):
        return self.dimension;

    def getFeaturesNumber(self):
        return self.featuresNumber;

    def minmaxScale(self):
        for i in range(0,self.featuresNumber):
            min = self.x[0][i];
            max = self.x[0][i];
            for j in range(0,self.dimension):
                if self.x[j][i]>max:
                    max=self.x[j][i];
                elif self.x[j][i]<min:
                    min=self.x[j][i];
            for j in range(0, self.dimension):
                self.x[j][i]=(self.x[j][i]-min)/(max-min);

    def populateByFilename(self,filename):
        numLines = 0
        numColumns = 0;
        with open(filename) as f:
            for line in f:
                numLines += 1
            numColumns = len(line.split(","));
        self.x = np.empty(shape=(numLines, numColumns - 1));
        self.y = np.empty(numLines);
        self.dimension = numLines;
        self.featuresNumber = numColumns - 1;

        k = 0;
        with open(filename) as f:
            for line in f:
                l = line.split(",");
                for i in range(0, numColumns - 1):
                    self.x[k, i] = float(l[i]);
                if float(l[numColumns - 1]) == 0:
                    self.y[k] = -1;
                else:
                    self.y[k] = 1;
                k += 1;

    def populateWithDorotheaTrain(self):
        self.x = np.zeros(shape=(800, 100001));
        k = 0;
        with open('dataSets/dorothea_train.data') as f:
            for line in f:
                list = line.split(" ");
                list = list[:-1];
                for n in list:
                    n = int(n);
                    self.x[k][n] = 1;
                k += 1;

        self.y = np.zeros(800)
        k = 0;
        with open('dataSets/dorothea_train.labels') as f:
            for line in f:
                l = line.split("\n")
                self.y[k] = int(l[0]);
                k += 1;
        self.dimension=len(self.y);
        self.featuresNumber=len(self.x[0]);

    def populateWithDorotheaTest(self):
        self.x = np.zeros(shape=(350, 100001));
        k = 0;
        with open('dataSets/dorothea_valid.data') as f:
            for line in f:
                list = line.split(" ");
                list = list[:-1];
                for n in list:
                    n = int(n);
                    self.x[k][n] = 1;
                k += 1;

        self.y = np.zeros(350)
        k = 0;
        with open('dataSets/dorothea_valid.labels') as f:
            for line in f:
                l = line.split("\n")
                self.y[k] = int(l[0]);
                k += 1;
        self.dimension = len(self.y);
        self.featuresNumber = len(self.x[0]);
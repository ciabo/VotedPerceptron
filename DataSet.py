import numpy as np

class DataSet:
    def __init__(self,filename):
        numLines = 0
        numColumns = 0;
        with open(filename) as f:
            for line in f:
                numLines += 1
            numColumns = len(line.split(","));
        self.x = np.empty(shape=(numLines, numColumns - 1));
        self.y = np.empty(numLines);
        self.dimension=numLines;

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

    def getx(self):
        return self.x;

    def gety(self):
        return self.y;

    def getDimension(self):
        return self.dimension;
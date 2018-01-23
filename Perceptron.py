import numpy as np

class Perceptron:
    
    def __init__(self, dataSet):
        self.dataSet=dataSet;
        # w is the vector of weights
        self.w = np.zeros(len(self.dataSet.getx()[0]));
        self.b=0;

    def train(self,maxIterations):
        nerror=1;
        R=np.linalg.norm(self.dataSet.getx()[0]);
        for i in range(1, self.dataSet.getDimension()):
            if R<np.linalg.norm(self.dataSet.getx()[i]):
                R=np.linalg.norm(self.dataSet.getx()[i])
        iteration=0;
        while nerror!=0 and iteration!=maxIterations:
            nerror=0;
            for i in range(0,self.dataSet.getDimension()):
                if self.dataSet.gety()[i]*(np.dot(self.w,self.dataSet.getx()[i])+self.b)<=0:
                    self.w=self.w+self.dataSet.gety()[i]*self.dataSet.getx()[i];
                    self.b=self.b+self.dataSet.gety()[i]*np.square(R);
                    nerror+=1;
            iteration+=1;
        print(self.w,self.b);
    def predict(self,x):
        r=np.sign(np.dot(self.w,x)+self.b);
        if r==0:
            r=1;
        return r;

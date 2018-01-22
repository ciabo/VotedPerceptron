import numpy as np

class Perceptron:
    #x and y represent the dataset
    def __init__(self, x, y):
        self.x = x;
        self.y = y;
        # w is the vector of weights
        self.w = np.zeros(len(x[0]));
        self.b=0;

    def train(self):
        nerror=1;
        R=np.linalg.norm(self.x[0]);
        for i in range(1, len(self.x)):
            if R<np.linalg.norm(self.x[i]):
                R=np.linalg.norm(self.x[i])
        while nerror!=0:
            nerror=0;
            for i in range(0,len(self.x)):
                if self.y[i]*(np.dot(self.w,self.x[i])+self.b)<=0:
                    self.w=self.w+self.y[i]*self.x[i];
                    self.b=self.b+self.y[i]*np.square(R);
                    nerror+=1;
        print(self.w,self.b);
    def predict(self,x):
        return np.sign(np.dot(self.w,x));

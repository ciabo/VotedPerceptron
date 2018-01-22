import numpy as np

class VotedPerceptron:
    #x (vector of vectors) and y(vector of values) represent the dataset and t is the number of epoch
    def __init__(self , x , y , t):
        self.x = x;
        self.y = y;
        self.t = t;
        #w is the vector of weights
        self.w=np.zeros(len(x[0]));
        #v is the vector of all the different w found and c indicates the life of each w --to fix: they might not be long x
        #v and c are python list cause take only O(1) to append values and now they are only used for store
        self.v=[];
        self.v.append(self.w);
        self.c=[];
        self.c.append(1);

    def train(self):
        k=0;
        for epoch in range(0,self.t):
            dataSetDim=len(self.x)
            for i in range(0,dataSetDim):
                prediction=np.sign(np.dot(self.x[i],self.w));
                if prediction == self.y[i]:
                    self.c[k]+=1;
                else:
                    self.w=self.w+self.y[i]*self.x[i]; #update the value of w
                    self.v.append(self.w); #append to v the new value of w
                    self.c.append(1);
                    k+=1;

    def predict(self, x):
        s=0;
        #In order to make operations with the vector v i convert it to a numpy array. The op. costs O(n)
        historic=np.array(self.v);
        for i in range(0, len(self.v)):
            s+=self.c[i]*np.sign(np.dot(historic[i],x));
        return np.sign(s);
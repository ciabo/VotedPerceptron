import numpy as np

class VotedPerceptron:
    # t is the number of epoch
    def __init__(self ,dataSet , t):
        self.dataSet=dataSet;
        self.t = t;
        #w is the vector of weights
        self.w=np.zeros(len(self.dataSet.getx()[0]));
        #v is the vector of all the different w found and c indicates the life of each w --to fix: they might not be long x
        #v and c are python list cause take only O(1) to append values and now they are only used for store
        self.v=[];
        self.v.append(self.w);
        self.c=[];
        self.c.append(1);

    def train(self):
        k=0;
        for epoch in range(0,self.t):
            for i in range(0,self.dataSet.getDimension()):
                prediction=np.sign(np.dot(self.dataSet.getx()[i],self.w));
                if prediction==0:
                    prediction=1;
                if prediction == self.dataSet.gety()[i]:
                    self.c[k]+=1;
                else:
                    self.w=self.w+self.dataSet.gety()[i]*self.dataSet.getx()[i]; #update the value of w
                    self.v.append(self.w); #append to v the new value of w
                    self.c.append(1);
                    k+=1;


    def predict(self, x):
        s=0;
        #In order to make operations with the vector v it is converted to a numpy array. The op. costs O(n)
        historic=np.array(self.v);
        for i in range(0, len(self.v)):
            tmp=np.sign(np.dot(historic[i],x));
            if tmp==0:
                tmp=1;
            s+=self.c[i]*tmp;
        if np.sign(s)==0:
            return 1
        return np.sign(s);
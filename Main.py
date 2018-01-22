import numpy as np
import VotedPerceptron as vp
import Perceptron as p
x=np.array(([1,1],[1,3],[2,0],[2,2],[-1,-1],[3,2]));
y=np.array((-1, 1, -1, 1, -1, 1));
vp=vp.VotedPerceptron(x,y,2);
p=p.Perceptron(x,y);
vp.train();
p.train();
print(vp.predict([2,-1]));
print(p.predict([3,2]));
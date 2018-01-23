import numpy as np
import VotedPerceptron as vp
import Perceptron as p
import DataSet as dt

banknoteDataSet=dt.DataSet('data_banknote_authentication.txt')
htruDataSet=dt.DataSet('HTRU_2.txt');

vp=vp.VotedPerceptron(banknoteDataSet,2);
vp.train();
p=p.Perceptron(banknoteDataSet);
p.train(100);

import numpy as np
import VotedPerceptron as vp
import Perceptron as p
import DataSet as dt

banknoteDataSet=dt.DataSet('data_banknote_authentication.txt')

vp=vp.VotedPerceptron(banknoteDataSet.getx(),banknoteDataSet.gety(),2);
vp.train();

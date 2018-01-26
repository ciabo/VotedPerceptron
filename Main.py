import os
import DataSet as dt
import HoldoutCrossValidation as cv

banknoteDataSet=dt.DataSet(filename='dataSets/data_banknote_authentication.txt');
htryDataSet=dt.DataSet(filename='dataSets/HTRU_2.txt')
dorotheaTrain=dt.DataSet(dorotheaTrain=True);
dorotheaTest=dt.DataSet(dorotheaTest=True);

cv.splitAndTest(banknoteDataSet);
cv.splitAndTest(htryDataSet);
cv.test(dorotheaTrain,dorotheaTest);

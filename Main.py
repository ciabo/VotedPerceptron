import os
import DataSet as dt
import HoldoutCrossValidation as cv

banknoteDataSet=dt.DataSet(filename='dataSets/data_banknote_authentication.txt');
htryDataSet=dt.DataSet(filename='dataSets/HTRU_2.txt')
dorotheaTrain=dt.DataSet(dorotheaTrain=True);
dorotheaTest=dt.DataSet(dorotheaTest=True);

print("not minmaxscaled dataset")
cv.splitAndTest(banknoteDataSet);
cv.splitAndTest(htryDataSet);
print("")
print("")
print("minmaxscaled dataset")
banknoteDataSet.minmaxScale();
htryDataSet.minmaxScale();
cv.splitAndTest(banknoteDataSet);
cv.splitAndTest(htryDataSet);
print("------")
cv.test(dorotheaTrain,dorotheaTest);


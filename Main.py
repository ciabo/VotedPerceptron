import DataSet as dt
import Validation as cv
import numpy as np

for k in range(0,10):
    print("iterazione: ",k);
    banknoteDataSet=dt.DataSet(filename='dataSets/data_banknote_authentication.txt');
    htruDataSet=dt.DataSet(filename='dataSets/HTRU_2.txt');
    dorotheaDataSet = dt.DataSet(dorothea=True);
    print("-------");
    print("BANKNOTE DATASET");
    cv.holdoutCrossValidation(banknoteDataSet);
    print("-------");
    print("HTRU_2 DATASET");
    cv.holdoutCrossValidation(htruDataSet);
    print("-------");
    print("DOROTHEA");
    cv.holdoutCrossValidation(dorotheaDataSet);
    print("");
    print("");



print("");
print("");
print("------K-FOLD CROSS VALIDATION--------");
print("Banknote");
banknoteDataSet=dt.DataSet(filename='dataSets/data_banknote_authentication.txt');
cv.kFoldCrossValidation(5,banknoteDataSet);
print("HTRU");
htruDataSet=dt.DataSet(filename='dataSets/HTRU_2.txt');
cv.kFoldCrossValidation(5,htruDataSet);
print("DOROTHEA");
dorotheaDataSet = dt.DataSet(dorothea=True);
cv.kFoldCrossValidation(5,dorotheaDataSet);




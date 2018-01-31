import DataSet as dt
import Validation as cv

for k in range(0,5):
    print("iterazione: ",k);
    banknoteDataSet=dt.DataSet(filename='dataSets/data_banknote_authentication.txt');
    htruDataSet=dt.DataSet(filename='dataSets/HTRU_2.txt');
    dorotheaTrain=dt.DataSet(dorotheaTrain=True);
    dorotheaTest=dt.DataSet(dorotheaTest=True);
    print("-------");
    print("BANKNOTE DATASET");
    cv.holdoutCrossValidation(banknoteDataSet);
    print("-------");
    print("HTRU_2 DATASET");
    cv.holdoutCrossValidation(htruDataSet);
    print("------");
    print("DOROTHEA");
    cv.test(dorotheaTrain,dorotheaTest);
    print("");
    print("");


banknoteDataSet=dt.DataSet(filename='dataSets/data_banknote_authentication.txt');
cv.kFoldCrossValidation(5,banknoteDataSet)

import DataSet as dt
import Validation as cv

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
    cv.holdoutCrossValidation(dorotheaDataSet,False); #False to say at the holdoutCrossValidation not to try also with the scaled datasets
    print("");
    print("");


print("");
print("");
print("-------------------------------------")
print("------K-FOLD CROSS VALIDATION--------");
print("Banknote");
print("NOT MINMAXSCALED");
banknoteDataSet=dt.DataSet(filename='dataSets/data_banknote_authentication.txt');
cv.kFoldCrossValidation(5,banknoteDataSet);
print("MINMAXSCALED");
banknoteDataSet.minmaxScale();
cv.kFoldCrossValidation(5,banknoteDataSet);
print("HTRU");
print("NOT MINMAXSCALED");
htruDataSet=dt.DataSet(filename='dataSets/HTRU_2.txt');
cv.kFoldCrossValidation(5,htruDataSet);
print("MINMAXSCALED");
htruDataSet.minmaxScale();
cv.kFoldCrossValidation(5,htruDataSet);
print("DOROTHEA");
print("NOT MINMAXSCALED");
dorotheaDataSet = dt.DataSet(dorothea=True);
cv.kFoldCrossValidation(5,dorotheaDataSet);

import os

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from sklearn.datasets import load_svmlight_file

# Cria diretório se não existir
def createDirectory(path):
        if not os.path.isdir(path):
                os.mkdir(path)

def writeFile(text, file_name):

    createDirectory("results/")

    resultFile = open("results/"+file_name, "a+")
    resultFile.write(str(text)+"\n")
    resultFile.close


def dataLoad(trainFile, testFile, size):
    print("File size:"+str(size))

    # Loads data
    print("Loading data...")
    X_train, y_train = load_svmlight_file(trainFile)
    X_test, y_test = load_svmlight_file(testFile)

    X_train = X_train.toarray()
    X_test = X_test.toarray()

    # Normalizacao dos dados #######
    #print("Normalization data...")
    #scaler = preprocessing.MaxAbsScaler()
    #X_train = scaler.fit_transform(X_train[0:size])
    #X_test = scaler.fit_transform(X_test)

    print("Load data done")

    return X_train[0:size], y_train[0:size], X_test, y_test 

def writeResults(clf_name, clf, size, X_test, y_test, y_pred, fit_time, predict_time):
    # Writting results
    accuracy = str(clf.score(X_test, y_test))

    writeFile("accuracy:"+accuracy, clf_name+".txt")

    f1s = str(f1_score(y_test, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], average='weighted'))

    writeFile("f1s:"+f1s, clf_name+".txt")

    cm = confusion_matrix(y_test, y_pred)

    writeFile(cm, clf_name+".txt")

    writeFile(clf_name+","+str(size)+","+accuracy+","+f1s+","+str(fit_time)+","+str(predict_time), "geral.csv")
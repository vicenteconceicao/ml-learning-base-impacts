import classifiers
import sys

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Use: python classifiers.py <train file> <test file>")

    trainFile = sys.argv[1]
    testFile  = sys.argv[2]

    for i in range(1, 21):
        classifiers.writeResult("examples:"+str(i))
        # Data loading
        X_train, y_train, X_test, y_test  = classifiers.dataLoad(trainFile, testFile, (i*1000))
        # KNN fit
        #neigh, y_pred = classifiers.knn(3, "manhattan", X_train, y_train, X_test)

        gnb, y_pred = classifiers.naiveBayes(X_train, y_train, X_test)

        # Writting results
        accuracy = str(gnb.score(X_test, y_test))

        classifiers.writeResult("accuracy:"+accuracy)

        f1s = str(f1_score(y_test, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], average='weighted'))

        classifiers.writeResult("f1s:"+f1s)

        cm = confusion_matrix(y_test, y_pred)

        classifiers.writeResult(cm)
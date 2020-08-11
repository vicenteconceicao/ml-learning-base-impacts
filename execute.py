import classifiers
import sys
import utils

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Use: python classifiers.py <train file> <test file>")

    trainFile = sys.argv[1]
    testFile  = sys.argv[2]

    for i in range(1000, 21000, 1000):
        # Data loading
        X_train, y_train, X_test, y_test  = utils.dataLoad(trainFile, testFile, i)

        classif = ['knn','naive_bayes','lda','logistic_regression','perceptron']

        for classif_name in classif:
            utils.writeFile("example:"+str(i), classif_name+".txt")
            clf, y_pred, fit_time, predict_time = classifiers.classify(classif_name, X_train, y_train, X_test)
            utils.writeResults(classif_name, clf, i ,X_test, y_test, y_pred, fit_time, predict_time)
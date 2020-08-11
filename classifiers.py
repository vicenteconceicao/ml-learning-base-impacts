import time

# Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron

def classify(clf_name, X_train, y_train, X_test):
    if(clf_name == "knn"):
        clf = KNeighborsClassifier(n_neighbors=3, metric="manhattan")
    if(clf_name == "naive_bayes"):
        clf = GaussianNB()
    if(clf_name == "lda"):
        clf = LinearDiscriminantAnalysis()
    if(clf_name == "logistic_regression"):
        clf = LogisticRegression(random_state=0)
    if(clf_name == "perceptron"):
        clf = Perceptron(tol=1e-3, random_state=0)
    
    print('Fitting '+clf_name+'...')
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    fit_time = (end - start)

    print('Preditting '+clf_name+'...')
    start = time.time()
    y_pred = clf.predict(X_test)
    end = time.time()
    predict_time = (end - start)

    print(clf_name+" done")

    return clf, y_pred, fit_time, predict_time
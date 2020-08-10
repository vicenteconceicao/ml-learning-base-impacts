#from joblib import Memory
from sklearn import preprocessing
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

# Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def knn(k, metric, X_train, y_train, X_test):
    writeResult("Classifier: KNN")
    writeResult("K: "+str(k)+" metric:"+metric)

    neigh = KNeighborsClassifier(n_neighbors=k, metric=metric)
    
    print ('Fitting knn')
    neigh.fit(X_train, y_train)

    print ('Preditting knn')
    y_pred = neigh.predict(X_test)

    print("knn done")

    return neigh, y_pred

def naiveBayes(X_train, y_train, X_test):
    writeResult("Classifier: NAIVE BAYES")
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    return gnb, y_pred

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

    
def writeResult(text):
    resultFile = open("results/result.txt", "a+")
    resultFile.write(str(text)+"\n")
    resultFile.close
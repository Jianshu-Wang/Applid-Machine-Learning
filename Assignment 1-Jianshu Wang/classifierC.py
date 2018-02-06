path = 'pima-indians-diabete-data.txt'
with open(path) as f1: data = f1.read()


def getDataList(data):
    dataList = data.split('\n')
    return dataList


def getLinesCount(dataList):
    count = len(dataList)
    return count


def getTrainDataList(dataList, number, weight):
    trainNumber = number * weight
    trainList = [dataList[i] for i in range(0, int(trainNumber))]
    return trainList


def getTestDataList(dataList, number, weight):
    testNumber = number * weight
    testList = [dataList[i] for i in range(int(testNumber), len(dataList))]
    return testList


def extractItemMatrix(dataList):
    trainItems = []
    trainClusters = []

    for item in dataList:
        list = item.split(',')
        trainItems.append([float(list[i]) for i in range(0, 8)])
        trainClusters.append(int(list[len(list) - 1]))
    return trainItems


def extractClusterMatrix(dataList):
    trainItems = []
    trainClusters = []

    for item in dataList:
        list = item.split(',')
        trainItems.append([float(list[i]) for i in range(0, len(list) - 1)])
        trainClusters.append(int(list[len(list) - 1]))
    return trainClusters


def evaluateClassifier(testResults, testClusters):
    total = len(testResults)
    # print total
    counter = 0
    for i in range(0, total):
        list = testClusters[i]
        if int(testResults[i]) is int(list):
            counter += 1
    return float(counter) / total


def getMean(list):
    sum = 0
    counter = 0
    for number in list:
        sum += float(number)
        counter += 1


    mean = sum / counter
    # print sum
    # print 'counter: '+str(counter)
    return mean

dataList = getDataList(data)
number = getLinesCount(dataList)
weight = 0.8
trainDataList = getTrainDataList(dataList, number, weight)
testDataList = getTestDataList(dataList, number, weight)

trainItems = extractItemMatrix(trainDataList)
trainClusters = extractClusterMatrix(trainDataList)

testItems = extractItemMatrix(testDataList)
testClusters = extractClusterMatrix(testDataList)

# Do 10-fold-cross-validation
import numpy as np
from sklearn.model_selection import KFold

X = trainItems
Y = trainClusters
kf = KFold(n_splits=10)
kf.get_n_splits(X)

accuracy = []

for train_index, test_index in kf.split(X):
    # print("TRAIN:", train_index, "TEST:", test_index)
    # print train_index
    X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]

    y_train, y_test = np.array(Y)[train_index], np.array(Y)[test_index]

    from sklearn.naive_bayes import GaussianNB

    clf = GaussianNB()

    clf.fit(X_train, y_train)
    GaussianNB(priors=None)

    testResults = clf.predict(X_test)

    acc = evaluateClassifier(testResults, y_test)
    accuracy.append(acc)


print 'The mean value of 10-cross-validation is: '+ str(getMean(accuracy))



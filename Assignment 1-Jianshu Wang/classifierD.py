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


def calculateAccuracy(result,testClusters):
    counter = 0
    for i in range(0,len(result)):
        if int(result[i]) is int(testClusters[i]):
            counter += 1

    return float(counter)/len(result)


dataList = getDataList(data)
number = getLinesCount(dataList)
weight = 0.8
trainDataList = getTrainDataList(dataList, number, weight)
testDataList = getTestDataList(dataList, number, weight)

trainItems = extractItemMatrix(trainDataList)
trainClusters = extractClusterMatrix(trainDataList)

testItems = extractItemMatrix(testDataList)
testClusters = extractClusterMatrix(testDataList)
# print testItems[0]
# print testClusters[0]


from sklearn import svm
X = trainItems
y = trainClusters
clf = svm.SVC()
clf.fit(X, y)
result = clf.predict(testItems)
# print result

print calculateAccuracy(result,testClusters)




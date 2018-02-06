import numpy
import math
import scipy.stats

path = 'pima-indians-diabete-data.txt'
statisticPath = 'pima-indians-diabetes-statistics.txt'

with open(path) as f1: data = f1.read()
with open(statisticPath) as f2: statisticData = f2.read()


def getDataList(data):
    dataList = data.split('\n')
    return dataList


def getDataDic(dataList):
    result = {}
    for line in dataList:
        lineResultList = line.split(',')
        key = lineResultList[8]
        if key not in result:
            result[key] = 1
        else:
            result[key] += 1

    return result


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


def separateDataByKey(trainDataList, dataDic):
    dic = {}

    for key in dataDic:
        dic[key] = []

    for member in trainDataList:
        mList = member.split(',')
        mKey = mList[8]
        list = [mList[i] for i in range(0, 8)]
        dic[mKey].append(list)
    return dic


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


def getStd(list):
    mean = getMean(list)
    squareSum = 0
    counter = 0
    for number in list:
        squareSum += math.pow(float(number) - float(mean), 2)
        counter += 1
    std = math.sqrt(squareSum / counter)
    # print squareSum
    return std


def computeNormalDis(preparedTrainData, key):
    parameters = []
    matrix = preparedTrainData[key]
    rows = len(matrix)
    for position in range(0, len(matrix[0])):
        list = []
        for row in range(0, rows):
            list.append(float(matrix[row][position]))


        stdnum = getStd(list)
        # print 'std is: '+ str(stdnum)
        meannum = getMean(list)
        # print 'mean is: '+str(meannum)
        parameters.append([meannum, stdnum])

    return parameters


def computePro(attribute, parameters, position):
    num = float(attribute)
    mu = float(parameters[position][0])
    sigma = float(parameters[position][1])
    from scipy import stats
    prob = stats.norm.pdf(num, mu, sigma)
    # print attribute
    # print prob
    return prob


def computeAttributePro(attributeList, preparedTrainData, trainDataList, statisticDataList):
    propability = {}
    total = getLinesCount(trainDataList)
    # print total

    for key in preparedTrainData:
        parameters = computeNormalDis(preparedTrainData, key)
        value = float(len(preparedTrainData[key])) / total
        # print len(preparedTrainData[key])
        # print value
        logPY = math.log1p(value)
        sumLogPx = 0
        for position in range(0, len(attributeList) - 1):
            attributePro = computePro(attributeList[position], parameters, position)
            sumLogPx += attributePro
        propability[key] = sumLogPx + logPY

    return propability


def classifer(testDataList, preparedTrainData, trainDataList):
    outputList = []
    for item in testDataList:
        proResult = computeAttributePro(item.split(','), preparedTrainData, trainDataList, statisticDataList)
        if (proResult['0'] > proResult['1']):
            outputList.append(0)
        else:
            outputList.append(1)
    return outputList


def evaluateClassifier(resultList,testDataList):
    total = len(resultList)
    # print total
    counter = 0
    for i in range(0,total):
        list = testDataList[i].split(',')
        if resultList[i] is int(list[8]):
            counter += 1
            # print counter


    return float(counter)/total


dataList = getDataList(data)
number = getLinesCount(dataList)
weight = 0.8
dataDic = getDataDic(dataList)
statisticDataList = getDataList(statisticData)
# print statisticDataList
trainDataList = getTrainDataList(dataList, number, weight)
testDataList = getTestDataList(dataList, number, weight)
# print len(testDataList)
preparedTrainData = separateDataByKey(trainDataList, dataDic)
# value = computeAttributePro(testDataList[153].split(','), preparedTrainData, trainDataList, statisticDataList)
# print value


outputResult = classifer(testDataList, preparedTrainData, trainDataList)
accuracy = evaluateClassifier(outputResult,testDataList)

# print testDataList

# print outputResult
print accuracy


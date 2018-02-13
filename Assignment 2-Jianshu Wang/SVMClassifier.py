import random
import numpy as np
import math
import matplotlib.pyplot as plt

path = 'AdultIncome.txt'
with open(path) as f1: data = f1.read()


def readData(data):
    list = []
    for line in data.split('\n'):
        dataSet = []
        for feature in line.split(','):
            dataSet.append(str.strip(feature))
        list.append(dataSet)
    return list


def dropExample(dataList, position):
    cleanedDataList = []
    for dataSet in dataList:
        continuousData = []
        for i in position:
            continuousData.append(float(dataSet[i]))

        n = len(dataSet)
        # print dataSet[n-1]
        if str(dataSet[n - 1]) == '<=50K' or str(dataSet[n - 1]) == '<=50K.':
            continuousData.append(-1)
        elif str(dataSet[n - 1]) == '>50K' or str(dataSet[n - 1]) == '>50K.':
            continuousData.append(1)

        cleanedDataList.append(continuousData)
    return cleanedDataList


def normalizeData(dataList):
    data1 = []
    data2 = []
    data3 = []
    data4 = []
    data5 = []
    data6 = []
    for data in dataList:
        data1.append(data[0])
        data2.append(data[1])
        data3.append(data[2])
        data4.append(data[3])
        data5.append(data[4])
        data6.append(data[5])

    variance1 = np.var(data1)
    variance2 = np.var(data2)
    variance3 = np.var(data3)
    variance4 = np.var(data4)
    variance5 = np.var(data5)
    variance6 = np.var(data6)

    k1 = math.sqrt(1 / variance1)
    k2 = math.sqrt(1 / variance2)
    k3 = math.sqrt(1 / variance3)
    k4 = math.sqrt(1 / variance4)
    k5 = math.sqrt(1 / variance5)
    k6 = math.sqrt(1 / variance6)

    for data in dataList:
        data[0] = float(data[0]) * k1
        data[1] = float(data[1]) * k2
        data[2] = float(data[2]) * k3
        data[3] = float(data[3]) * k4
        data[4] = float(data[4]) * k5
        data[5] = float(data[5]) * k6

    return dataList


def reserveTest(dataList, test):
    total = len(dataList)
    testLen = int(total * test)
    pickTest = []
    while len(pickTest) < testLen:
        pick = random.choice(list(dataList))
        dataList.remove(pick)
        pickTest.append(pick)

    return (dataList, pickTest)


def randomlySplit(dataList, slices, pickFraction):
    total = len(dataList)
    print 'Training set(for cross validation) examples number is: ' + str(total)
    validationLen = int(float(pickFraction) / slices * total)
    validationList = []

    while len(validationList) < validationLen:
        pick = random.choice(list(dataList))
        dataList.remove(pick)
        validationList.append(pick)

    return (dataList, validationList)


def randomlyPick(dataList, pickNum):
    validationList = []
    copyList = list(dataList)
    # print len(dataList)

    while len(validationList) < pickNum:
        pick = random.choice(copyList)
        copyList.remove(pick)
        validationList.append(pick)

    return (copyList, validationList)


def updateLearningRate(s):
    return float(1) / (float(s) * 200 + float(50))


def evaluateAccuracy(a, b, dataList):
    total = len(dataList)
    counter = 0
    for item in dataList:
        gama = float(0)
        length = len(item)
        for i in range(0, length - 1):
            gama += a[i] * item[i]
        gama += b
        if gama * float(item[length - 1]) >= 0:
            counter += 1
    return float(counter) / total


def predict(a, b, vector):
    length = len(vector)
    gama = float(0)
    for i in range(0, length - 1):
        gama += float(vector[i]) * a[i]
    gama += b

    return gama


def updateEstimate1(current, original, lamb, stepLength):
    next = []
    for i in range(0, len(current)):
        next.append(float(current[i]) - float(lamb) * stepLength * current[i])
    return next


def updateEstimate2(current, original, lamb, stepLength, dataVector, y_k):
    next = []
    for i in range(0, len(current)):
        next.append(
            float(current[i]) - float(lamb) * stepLength * current[i] + float(stepLength) * y_k * dataVector[i])
    return next


def updateGradient(a_current, b_current, batchVector, a_original, b_original, lambdaValue, stepLength):
    length = len(batchVector)

    y_k = float(batchVector[length - 1])
    y_predict = predict(a_current, b_current, batchVector)

    if y_k * y_predict >= 1:
        a_next = updateEstimate1(a_current, a_original, lambdaValue, stepLength)
        b_next = float(b_current)
    else:
        a_next = updateEstimate2(a_current, a_original, lambdaValue, stepLength, batchVector, y_k)
        b_next = b_current + stepLength * y_k

    return (a_next, b_next)


# Prepare Data for training and testing
dataList = readData(data)
position = [0, 2, 4, 10, 11, 12]
cleanedData = dropExample(dataList, position)
normalizedData = normalizeData(cleanedData)
reserveResult = reserveTest(normalizedData, 0.1)
reservedTrainData = reserveResult[0]
print 'Reversed train data length is: ' + str(len(reservedTrainData))
reservedTestData = reserveResult[1]
print 'Reversed test data length is: ' + str(len(reservedTestData))

# Start cross validation calculation
lambdaList = [float(0.001), float(0.01), float(0.1), float(1)]
N_batch = 1
N_step = 300
k = 30
N_season = 100
evaluateNum = 50
crossNum = 1

accuracyMatrix = []
resultContentMatrix = []
splitResult = randomlySplit(reservedTrainData, 9, 1)
trainData = splitResult[0]
validationData = splitResult[1]

colorVector = ['r-','b-','g-','c-']


for i in range(0,len(lambdaList)):
    lambdaValue = lambdaList[i]
    color = colorVector[i]
    print color+'...'

    a_original = [float(0), float(0), float(0), float(0), float(0), float(0)]
    b_original = float(0)
    a_current = a_original
    b_current = b_original
    positionList = []
    resultContent = []
    accuracyVector = []
    validatedAccuracyVector = []
    magVector = []
    # print len(trainData)
    for season in range(0, N_season):

        stepLength = updateLearningRate(season)
        splitEvaluate = randomlyPick(trainData, evaluateNum)
        trainData_evaluate = splitEvaluate[0]
        evaluateData_evaluate = splitEvaluate[1]
        for step in range(0, N_step):
            pickVector = random.choice(list(trainData_evaluate))
            gradientResult = updateGradient(a_current, b_current, pickVector, a_original, b_original, lambdaValue,
                                            stepLength)
            a_current = gradientResult[0]
            b_current = gradientResult[1]
            if (int(step) + 1) % k == 0:
                e_accuracy = evaluateAccuracy(a_current, b_current, evaluateData_evaluate)
                u = (lambdaValue, a_current, b_current)
                position = float(int(step) + 1)/N_step + float(season)
                positionList.append(position)
                resultContent.append(u)
                accuracyVector.append(e_accuracy)
                x = np.array(a_current)
                magVector.append(np.sqrt(x.dot(x)))

    resultContentMatrix.append(resultContent)
    plt.figure(1)
    plt.plot(positionList, accuracyVector,color)
    plt.figure(2)
    plt.plot(positionList, magVector, color)


print 'Finished.'
plt.figure(1)
plt.ylabel('Accuracy')
plt.xlabel('Season')
plt.figure(2)
plt.ylabel('Magnitude')
plt.xlabel('Season')
plt.figure(1)
plt.axis([0, N_season, 0, 1])
plt.figure(2)
plt.axis([0, N_season, 0, 1.2])
plt.legend(lambdaList)
plt.figure(1)
plt.legend(lambdaList)

plt.show()


testAccuracyMatrix = []
for lam in resultContentMatrix:
    testAccuracy = []
    for i in range(100,len(lam)):
        element = lam[i]
        a = element[1]
        b = element[2]
        accuracy = evaluateAccuracy(a, b, validationData)
        testAccuracy.append(accuracy)
    meanValue = sum(testAccuracy) / float(len(testAccuracy))
    testAccuracyMatrix.append(meanValue)

print 'The average accuracy for validation data set corresponding to lambdas is: '
print testAccuracyMatrix
print 'So from the result we pick lambda of 0.001 for our training'


finalResultVector = []
maxAccuracyIndex = 0
for i in range(0,len(lambdaList)-3):
    lambdaValue = lambdaList[i]
    color = colorVector[i]
    print color+'...'

    a_original = [float(0), float(0), float(0), float(0), float(0), float(0)]
    b_original = float(0)
    a_current = a_original
    b_current = b_original
    positionList = []
    resultContent = []
    accuracyVector = []
    validatedAccuracyVector = []
    magVector = []
    # print len(trainData)
    for season in range(0, N_season):

        stepLength = updateLearningRate(season)
        splitEvaluate = randomlyPick(reservedTrainData, evaluateNum)
        trainData_evaluate = splitEvaluate[0]
        evaluateData_evaluate = splitEvaluate[1]
        for step in range(0, N_step):
            pickVector = random.choice(list(trainData_evaluate))
            gradientResult = updateGradient(a_current, b_current, pickVector, a_original, b_original, lambdaValue,
                                            stepLength)
            a_current = gradientResult[0]
            b_current = gradientResult[1]
            if (int(step) + 1) % k == 0:
                e_accuracy = evaluateAccuracy(a_current, b_current, evaluateData_evaluate)
                u = (lambdaValue, a_current, b_current)
                position = float(int(step) + 1)/N_step + float(season)
                positionList.append(position)
                resultContent.append(u)
                accuracyVector.append(e_accuracy)
                x = np.array(a_current)
                magVector.append(np.sqrt(x.dot(x)))

    finalResultVector.extend(resultContent)
    maxAccuracyIndex = accuracyVector.index(max(accuracyVector))
    plt.figure(3)
    plt.plot(positionList, accuracyVector,color)
    plt.figure(4)
    plt.plot(positionList, magVector, color)
print 'Finished.'
plt.figure(3)
plt.ylabel('Accuracy')
plt.xlabel('Season')
plt.figure(4)
plt.ylabel('Magnitude')
plt.xlabel('Season')
plt.figure(3)
plt.axis([0, N_season, 0, 1])
plt.figure(4)
plt.axis([0, N_season, 0, 1.2])

plt.show()

finalAccuracyVector = []

for item in finalResultVector:
    a = item[1]
    b = item[2]
    accuracy = evaluateAccuracy(a, b, reservedTestData)
    finalAccuracyVector.append(accuracy)


print 'The maximum accuracy for this lambda is:'
print max(finalAccuracyVector)









# a_original = [float(0.5),float(0.5),float(0.5),float(0.5),float(0.5),float(0.5)]
# b_original = float(0.5)
# a_current = a_original
# b_current = b_original
#
# for season in range(0,N_season):
#     stepLength = updateLearningRate(season)
#     splitEvaluate = randomlyPick(trainData,evaluateNum)
#     trainData_evaluate = splitEvaluate[0]
#     evaluateData_evaluate = splitEvaluate[1]
#     for step in range(0,N_step):
#         pickVector = random.choice(list(trainData_evaluate))
#         gradientResult = updateGradient(a_current, b_current, pickVector, a_original, b_original, lambdaValue, stepLength)
#         a_current = gradientResult[0]
#         b_current = gradientResult[1]
#         if (int(step) + 1)%k == 0:
#             e_accuracy = evaluateAccuracy(a_current, b_current, evaluateData_evaluate)
#             u = (a_current, b_current)
#             position = float(int(step)+1) + season
#             resultVector = [position, u]
#             resultContent.append(resultVector)
#             accuracyVector.append(e_accuracy)

# peak = np.amax(accuracyVector)
# print peak
# position = accuracyVector.index(peak)
# print position
# print a_current
# print b_current

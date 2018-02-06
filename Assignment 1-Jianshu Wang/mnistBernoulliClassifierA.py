import mnist as mnist


def refreshImage(imageMatrix, threshold):
    for row in range(0, len(imageMatrix)):
        for col in range(0, len(imageMatrix[row])):
            if imageMatrix[row][col] < threshold:
                imageMatrix[row][col] = 0
            else:
                imageMatrix[row][col] = 255

    return imageMatrix


def boundingBox(imageMatrix):
    newImage = []
    for row in range(4, len(imageMatrix) - 4):
        newImage.append([])
        for col in range(4, len(imageMatrix[row]) - 4):
            newImage[row - 4].append(imageMatrix[row][col])

    return newImage


def resizeMatrix(image):
    from resizeimage import resizeimage
    cover = resizeimage.resize_cover(image, [20, 20])
    return cover


def stretchBounding(imageMatrix):
    left = len(imageMatrix[0])
    right = 0
    up = len(imageMatrix)
    down = 0
    for row in range(0, len(imageMatrix)):
        for col in range(0, len(imageMatrix[row])):
            if imageMatrix[row][col] != 0:
                if row < up: up = row
                if row > down: down = row
                if col > right: right = col
                if col < left: left = col

    stretchBoundingMatrix = []
    for row in range(up, down + 1):
        list = []
        for col in range(left, right + 1):
            list.append(imageMatrix[row][col])

        stretchBoundingMatrix.append(list)

    return stretchBoundingMatrix


def evaluateClassifier(testResults, testClusters):
    total = len(testResults)
    # print total
    counter = 0
    for i in range(0, total):
        list = testClusters[i]
        if int(testResults[i]) is int(list):
            counter += 1
    return float(counter) / total


import numpy as np
import scipy.ndimage

training_data = list(mnist.read(dataset='training', path='./MnistImages'))
testing_data = list(mnist.read(dataset='testing', path='./MnistImages'))

label_list = []
data_list_bounding = []
data_list_stretched = []
for label, pixel in training_data:

    image = refreshImage(pixel, 123)
    boundingBoxImage = boundingBox(image)
    stretchedMatrix = stretchBounding(image)
    import scipy.misc as sci

    stretchBoundedImage = refreshImage(sci.imresize(stretchedMatrix, (20, 20)), 123)

    label_list.append(label)
    single_data_bounding = []
    single_data_stretch = []
    for row in boundingBoxImage:
        for item in row:
            single_data_bounding.append(item)
    data_list_bounding.append(single_data_bounding)

    for row in stretchBoundedImage:
        for item in row:
            single_data_stretch.append(item)
    data_list_stretched.append(single_data_stretch)

test_label_list = []
test_data_list_bounding = []
test_data_list_stretched = []
for label, pixel in testing_data:

    image = refreshImage(pixel, 123)
    boundingBoxImage = boundingBox(image)
    stretchedMatrix = stretchBounding(image)
    import scipy.misc as sci

    stretchBoundedImage = refreshImage(sci.imresize(stretchedMatrix, (20, 20)), 123)

    test_label_list.append(label)
    single_data_bounding = []
    single_data_stretch = []
    for row in boundingBoxImage:
        for item in row:
            single_data_bounding.append(item)
    test_data_list_bounding.append(single_data_bounding)

    for row in stretchBoundedImage:
        for item in row:
            single_data_stretch.append(item)
    test_data_list_stretched.append(single_data_stretch)

from sklearn.naive_bayes import BernoulliNB

clf_stretch = BernoulliNB()
clf_stretch.fit(data_list_stretched, label_list)
test_stretch = clf_stretch.predict(test_data_list_stretched)
acc_stretch = evaluateClassifier(test_stretch, test_label_list)
print 'Accuracy for Bernoulii & stretched bounding box is: '
print acc_stretch

from sklearn.naive_bayes import BernoulliNB

clf_bounding = BernoulliNB()
clf_bounding.fit(data_list_bounding, label_list)
test_bounding = clf_bounding.predict(test_data_list_bounding)
acc_bounding = evaluateClassifier(test_bounding, test_label_list)
print 'Accuracy for Bernoulii & untouched image is: '
print acc_bounding

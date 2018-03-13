import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import math
from sklearn import preprocessing
from operator import itemgetter


def random_split(X):
    number = int(len(X))
    train_num = int(number * 0.8)
    train_set = []
    test_set = []
    for i in range(0, number):
        if i < train_num:
            train_set.append(X[i])
        else:
            test_set.append(X[i])
    return (train_set, test_set)


def extract_pieces(multi_signal):
    signal_list = []
    for signal in multi_signal:
        length = len(signal)
        piece_number = int(length) / 16
        for i in range(0, piece_number - 1):
            start = int(i) * 16
            end = int(i) * 16 + 32
            data = signal[start:end][:]
            signal_list.append(data)
    return signal_list


def extract_pieces_eachfile(signal):
    signal_list = []
    length = len(signal)
    piece_number = int(length) / 16
    for i in range(0, piece_number-1):
        start = int(i) * 16
        end = int(i)*16 + 32
        data = signal[start:end][:]
        signal_list.append(data)
    return signal_list


def flatten_data(array):
    vector = []
    for row in array:
        for item in row:
            vector.append(item)
    return vector


# def k_means(data_array):

def distance_cal(X, Y):
    length = len(X)
    error = float(0)
    for i in range(0, length):
        error += math.pow(X[i] - Y[i], 2)
    return math.sqrt(error)


def generate_histogram(signal_data, cluster_centers, k):
    histogram = [int(0)] * k
    for data in signal_data:
        error = []
        for center in cluster_centers:
            dis = distance_cal(data, center)
            error.append(dis)
        error = np.array(error)
        i = np.argmin(error)
        histogram[i] = histogram[i] + 1
    return histogram


# def trainData():
#

# def evaluateAccuracy():


path1 = 'HMP_Dataset'
path2 = ['Brush_teeth', 'Climb_stairs', 'Climb_stairs_MODEL', 'Comb_hair', 'Descend_stairs', 'Drink_glass',
         'Drink_glass_MODEL', 'Eat_meat', 'Eat_soup', 'Getup_bed', 'Getup_bed_MODEL', 'Liedown_bed',
         'Pour_water', 'Pour_water_MODEL', 'Sitdown_chair', 'Sitdown_chair_MODEL', 'Standup_chair',
         'Standup_chair_MODEL',
         'Use_telephone', 'Walk', 'Walk_MODEL']
labels = ['Brush_teeth', 'Climb_stairs', 'Comb_hair', 'Descend_stairs', 'Drink_glass', 'Eat_meat', 'Eat_soup',
          'Getup_bed', 'Liedown_bed',
          'Pour_water', 'Sitdown_chair', 'Standup_chair',
          'Use_telephone', 'Walk']

path_labels = [0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 9, 10, 10, 11, 11, 12, 13, 13]
# set k = 480
k = 600

data_dic = {}
for item in range(0, len(path2)):
    label = labels[path_labels[item]]
    if label not in data_dic:  data_dic[label] = []
    path = path1 + '/' + path2[item]
    for filename in os.listdir(path):
        file_path = path + '/' + filename

        with open(file_path) as f1:
            data = f1.read().strip()
        arr = []
        for line in data.split('\r\n'):
            lst_int = [int(x) for x in line.split(' ')]
            arr.append(lst_int)
        data_dic[label].append(arr)

print len(data_dic['Brush_teeth'])
print len(data_dic['Use_telephone'])
print len(data_dic['Sitdown_chair'])
print 'data looks ok!'
train_signal_vector = []
train_label_vector = []

test_signal_vector = []
test_label_vector = []

train_flat_pieces = []
for label in data_dic:
    matrix = data_dic[label]
    split_matrix = random_split(matrix)

    train_matrix = split_matrix[0]
    for signal in train_matrix:
        signal_flat_pieces = []
        signal_split = extract_pieces_eachfile(signal)
        for matrix in signal_split:
            flat_subsignal = flatten_data(matrix)
            signal_flat_pieces.append(flat_subsignal)
            train_flat_pieces.append(flat_subsignal)
        train_label_vector.append(label)
        train_signal_vector.append(signal_flat_pieces)

    test_matrix = split_matrix[1]
    for signal in test_matrix:
        signal_flat_pieces = []
        signal_split = extract_pieces_eachfile(signal)
        for matrix in signal_split:
            flat_subsignal = flatten_data(matrix)
            signal_flat_pieces.append(flat_subsignal)
        test_label_vector.append(label)
        test_signal_vector.append(signal_flat_pieces)

train_kmeans = KMeans(n_clusters=k, random_state=0).fit(train_flat_pieces)
train_cluster_centers = train_kmeans.cluster_centers_
# print train_cluster_centers
print len(train_label_vector)
print len(test_label_vector)
train_histogram_vector = []
for item in train_signal_vector:
    histogram = generate_histogram(item, train_cluster_centers, k)
    train_histogram_vector.append(histogram)

test_histogram_vector = []
for item in test_signal_vector:
    histogram = generate_histogram(item, train_cluster_centers, k)
    test_histogram_vector.append(histogram)

le = preprocessing.LabelEncoder()
le.fit(train_label_vector)
train_encoder_vector = le.transform(train_label_vector)


current_tree = 10
current_depth = 24

clf = RandomForestClassifier(n_estimators=current_tree, max_depth=current_depth)
clf.fit(train_histogram_vector, train_encoder_vector)

predict_vector = clf.predict(test_histogram_vector)

test_encoder_vector = le.transform(test_label_vector)
count = 0
length = len(test_encoder_vector)
print 'predict result is: '
print predict_vector
print 'true label is: '
print test_encoder_vector
for i in range(0, length):
    if predict_vector[i] == test_encoder_vector[i]:
        count += 1

accuracy = float(count) / length

print accuracy

result_confusion_matrix = np.zeros((14, 14))
for i in range(0,length):
    x = test_encoder_vector[i]
    y = predict_vector[i]
    result_confusion_matrix[x][y] += 1
print result_confusion_matrix
print le.inverse_transform([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])




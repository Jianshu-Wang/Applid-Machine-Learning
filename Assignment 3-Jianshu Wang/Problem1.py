from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt;

plt.rcdefaults()
from sklearn.metrics.pairwise import euclidean_distances


def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


def showPicture(vector, label):
    from PIL import Image
    import numpy as np
    w, h = 32, 32
    data = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(0, w):
        for j in range(0, h):
            r = i * 32 + j
            g = i * 32 + j + 1024
            b = i * 32 + j + 2048
            data[i, j] = [vector[r], vector[g], vector[b]]
    img = Image.fromarray(data, 'RGB')
    img.save(label + '.png')
    img.show()


def classifierMatrix(myDic, data, labels):
    li = data['data']
    length = len(li)
    for i in range(0, length):
        imageMatrix = data1['data'][i]
        y = data1['labels'][i]
        label = labels['label_names'][y]
        vector = list(imageMatrix)
        if label not in myDic:
            myDic[label] = [vector]
        else:
            myDic[label].append(vector)
    return myDic


def similarityCal(s, xi_A, xi_mean, pc_B):
    xi = np.array(xi_A)
    xi_m = np.array(xi_mean)
    pc_b = np.array(pc_B)
    value = xi_m
    for i in range(0, s):
        u_j = np.transpose(pc_b[:,i])
        u_j_T = np.transpose(u_j)
        pro = np.dot(np.dot(u_j_T, np.subtract(xi, xi_m)), u_j)
        value = np.add(value, pro)
    errorMatrix = np.subtract(value, xi)
    error = np.sum(np.multiply(errorMatrix,errorMatrix))
    return error


labelFile = 'cifar-10-batches-py/batches.meta'
file1 = 'cifar-10-batches-py/data_batch_1'
file2 = 'cifar-10-batches-py/data_batch_2'
file3 = 'cifar-10-batches-py/data_batch_3'
file4 = 'cifar-10-batches-py/data_batch_4'
file5 = 'cifar-10-batches-py/data_batch_5'
file6 = 'cifar-10-batches-py/test_batch'

# labels = unpickle(labelFile)
# data1 = unpickle(file1)
# imageMatrix = data1['data'][12]
# y = data1['labels'][12]
# label = labels['label_names'][y]
# print imageMatrix
# print type(list(imageMatrix))
# li = [1,2,3,4,5]
# print type(li)
# showPicture(imageMatrix)


resultDic = {}
labels = unpickle(labelFile)
data1 = unpickle(file1)
resultDic = classifierMatrix(resultDic, data1, labels)
data2 = unpickle(file2)
resultDic = classifierMatrix(resultDic, data2, labels)
data3 = unpickle(file3)
resultDic = classifierMatrix(resultDic, data3, labels)
data4 = unpickle(file4)
resultDic = classifierMatrix(resultDic, data4, labels)
data5 = unpickle(file5)
resultDic = classifierMatrix(resultDic, data5, labels)
data6 = unpickle(file6)
resultDic = classifierMatrix(resultDic, data6, labels)



# Problem 1
errorVector = [[],[]]

for key in sorted(resultDic):
    X = np.array(resultDic[key])
    pca = PCA(n_components=20, svd_solver='full')
    pca.fit(X)
    error = pca.explained_variance_[0]/pca.explained_variance_ratio_[0]-np.sum(pca.explained_variance_)
    errorVector[0].append(key)
    errorVector[1].append(error)

y_pos = np.arange(len(errorVector[0]))
plt.bar(y_pos,errorVector[1],  align='center', alpha=0.5)
plt.xticks(y_pos, errorVector[0])
plt.ylabel('Error')
plt.title('Sum of unused variances as error for each category')

plt.show()


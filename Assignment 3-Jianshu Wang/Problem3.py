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
    # print xi.shape
    xi_m = np.array(xi_mean)
    # print xi_m.shape
    pc_b = np.array(pc_B)
    value = xi_m
    for i in range(0, s):
        u_j = np.transpose(pc_b[:,i])
        u_j_T = np.transpose(u_j)
        a = np.sum(np.multiply(u_j_T, np.subtract(xi, xi_m)))
        pro = np.multiply(a, u_j)
        value = np.add(value, pro)
    errorMatrix = np.subtract(xi,value)
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


#problem 3
pc_list =[]
xi_list = []
x_mean_list = []
for key in sorted(resultDic):
    print key
    X = np.array(resultDic[key])
    pca = PCA(n_components=20, svd_solver='full')
    pca.fit(X)
    component = pca.components_
    xi = resultDic[key]
    xi_mean = pca.mean_
    pc_list.append(component)
    xi_list.append(xi)
    x_mean_list.append(xi_mean)


size = (10,10)
resultMatrix = np.zeros(size)
print 'Here now!!'
for keyA in range(0,10):
    pc_A = pc_list[keyA]
    pc_A = np.transpose(pc_A)
    print 'pc: '
    print pc_A.shape
    for keyB in range(0,10):
        x_B_mean = x_mean_list[keyB]
        x_B_mean = np.transpose(x_B_mean)
        xi = xi_list[keyB]
        errorT = 0
        for row in xi:
            xi_row = np.transpose(row)
            s = 20
            errorT += similarityCal(s, xi_row, x_B_mean, pc_A)
        resultMatrix[keyA][keyB] = errorT/len(xi)


print resultMatrix
result = np.zeros(size)
for i in range(0,10):
    for j in range(0,10):
        result[i][j] = float(0.5)*(resultMatrix[i][j] + resultMatrix[j][i])
print 'Here is the result: '
print result

from sklearn import manifold
data = list(result)

dists = data

adist = np.array(dists)
amax = np.amax(adist)
adist /= amax

mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=None)
results = mds.fit(adist)

coords = results.embedding_
# print coords

plt.subplots_adjust(bottom=0.1)
plt.scatter(
    coords[:, 0], coords[:, 1], marker='o'
)
for label, x, y in zip(labels['label_names'], coords[:, 0], coords[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.show()
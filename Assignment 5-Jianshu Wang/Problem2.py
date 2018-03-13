import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import math

file_path = 'physical.txt'


def convertSt2Float(content):
    matrix = []
    length = len(content)
    for i in range(1, length):
        li = content[i].split('\t')
        row = []
        for item in li:
            row.append(float(item))
        matrix.append(row)
    return matrix


with open(file_path) as f1:
    data = f1.readlines()

content = [x.strip() for x in data]

matrix = convertSt2Float(content)
# print matrix

matrix_array = np.array(matrix)
label_list = []
content_list = []
for item in matrix:
    label_list.append(item[0])
    li = []
    for i in range(1, 11):
        li.append(item[i])
    content_list.append(li)

# print label_list
# print content_list

reg = linear_model.LinearRegression()
reg.fit(content_list, label_list)

y_predict = reg.predict(content_list)
residual = []
for i in range(0, 22):
    residual.append(label_list[i] - y_predict[i])

residual_plot = plt.figure(1)
plt.scatter(y_predict, residual, color='blue')
residual_plot.axes[0].set_title('Residual against Fitted Value')
residual_plot.axes[0].set_xlabel('Fitted Value')
residual_plot.axes[0].set_ylabel('Residual')

print label_list
label_cuberoot_list = [math.pow(x, 1.0 / 3) for x in label_list]
reg2 = linear_model.LinearRegression()
reg2.fit(content_list, label_cuberoot_list)

y_cuberoot_predict = reg2.predict(content_list)
# print y_cuberoot_predict
residual2 = []
for i in range(0, 22):
    residual2.append(label_cuberoot_list[i] - y_cuberoot_predict[i])

y_cuberoot_predict_inOriginal = [math.pow(x,3.0) for x in y_cuberoot_predict]
# print y_cuberoot_predict_inOriginal
residual2_original = []
for j in range(0,22):
    residual2_original.append(label_list[j] - y_cuberoot_predict_inOriginal[j])

# print residual2_original

cuberoot_residual_plot = plt.figure(2)
plt.subplot(1,2,1)
plt.scatter(y_cuberoot_predict, residual2, color='blue')
plt.title('Cube Root Coordinate')
plt.xlabel('Fitted Value')
plt.ylabel('Residual')
plt.subplot(1,2,2)
plt.scatter(y_cuberoot_predict_inOriginal, residual2_original, color='red')
plt.title('Original Coordinates')
plt.xlabel('Fitted Value')
plt.ylabel('Residual')

plt.show()


R_original = np.var(y_predict)/np.var(label_list)
print R_original
R_cuberoot = np.var(y_cuberoot_predict_inOriginal)/np.var(label_list)
print R_cuberoot
if R_original > R_cuberoot:
    print 'Original is better'
else: print 'Cuberoot is better'
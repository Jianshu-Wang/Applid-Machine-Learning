import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import math
import random
from sklearn.linear_model import ElasticNet

file_path = 'abalone.txt'


def convertSt2Float(content):
    matrix = []
    label_list = []
    age_list = []
    length = len(content)
    for i in range(0, length):
        li = content[i].split(',')
        row = []
        row_len = len(li)
        label_list.append(li[0])
        age_list.append(float(li[row_len - 1]) + 1.5)
        for i in range(1, row_len - 1):
            row.append(float(li[i]))
        matrix.append(row)

    return (label_list, matrix, age_list)


def convertSt2Float2(content):
    matrix = []
    age_list = []
    length = len(content)
    for i in range(0, length):
        li = content[i].split(',')
        row = []
        row_len = len(li)
        if li[0] is 'F':
            row.append(float(-1))
        elif li[0] is 'I':
            row.append(float(0))
        elif li[0] is 'M':
            row.append(float(1))

        age_list.append(float(li[row_len - 1]) + 1.5)
        for i in range(1, row_len - 1):
            row.append(float(li[i]))
        matrix.append(row)
    return (matrix, age_list)


def calResidual(actual, predict):
    residual = []
    length = len(actual)
    for i in range(0, length):
        residual.append(actual[i] - predict[i])
    return residual


with open(file_path) as f1:
    data = f1.readlines()

content = [x.strip() for x in data]

matrix = convertSt2Float(content)

# part a
content_list = matrix[1]
age_list = matrix[2]
reg = linear_model.LinearRegression()
reg.fit(content_list, age_list)
age_predict = reg.predict(content_list)
residual1 = calResidual(age_list, age_predict)
age_prediction_plot = plt.figure(1)
plt.scatter(age_predict, residual1, s=80, facecolors='none', edgecolors='r')
plt.title('Residual vs Fitted Value Ignoring Gender')
plt.xlabel('Fitted Value')
plt.ylabel('Residual')

# part b
matrix2 = convertSt2Float2(content)
content_list2 = matrix2[0]
reg2 = linear_model.LinearRegression()
reg2.fit(content_list2, age_list)
age_predict2 = reg2.predict(content_list2)
residual2 = calResidual(age_list, age_predict2)
age_prediction_gender_plot = plt.figure(2)
plt.scatter(age_predict2, residual2, s=80, facecolors='none', edgecolors='b')
plt.title('Residual vs Fitted Value Considering Gender')
plt.xlabel('Fitted Value')
plt.ylabel('Residual')

# part c
age_list_log = [math.log(x) for x in age_list]
reg3 = linear_model.LinearRegression()
reg3.fit(content_list, age_list_log)
age_predict3 = reg3.predict(content_list)
age_predict3_convert = [math.exp(x) for x in age_predict3]
residual3 = calResidual(age_list, age_predict3_convert)
age_prediction_plot = plt.figure(3)
plt.scatter(age_predict3_convert, residual3, s=80, facecolors='none', edgecolors='r')
plt.title('Residual vs Fitted Log Value Ignoring Gender')
plt.xlabel('Fitted Value')
plt.ylabel('Residual')

# part d
reg4 = linear_model.LinearRegression()
reg4.fit(content_list2, age_list_log)
age_predict4 = reg4.predict(content_list2)
age_predict4_convert = [math.exp(x) for x in age_predict4]
residual4 = calResidual(age_list, age_predict4_convert)
age_prediction_plot = plt.figure(4)
plt.scatter(age_predict4_convert, residual4, s=80, facecolors='none', edgecolors='b')
plt.title('Residual vs Fitted Log Value Considering Gender')
plt.xlabel('Fitted Value')
plt.ylabel('Residual')

# part e
R_original_nogender = np.var(age_predict) / np.var(age_list)
print R_original_nogender

R_original_withgender = np.var(age_predict2) / np.var(age_list)
print R_original_withgender

R_log_nogender = np.var(age_predict3_convert) / np.var(age_list)
print R_log_nogender

R_log_withgender = np.var(age_predict4_convert) / np.var(age_list)
print R_log_withgender


# part f
def calSquareError(actual, predict):
    length = len(actual)
    result = float(0)
    for i in range(0, length):
        result += math.pow(actual[i] - predict[i], 2)
    return result


def randomlySplit(contentlist1, contentlist2, agelist, fraction):
    length = len(agelist)
    picknum = int(fraction * length)
    my_randoms = random.sample(xrange(length), picknum)
    content_list_train = []
    content_list_test = []

    content_list2_train = []
    content_list2_test = []

    age_list_train = []
    age_list_test = []

    for i in range(0, length):
        if i not in my_randoms:
            content_list_train.append(contentlist1[i])
            content_list2_train.append(contentlist2[i])
            age_list_train.append(agelist[i])
        else:
            content_list_test.append(contentlist1[i])
            content_list2_test.append(contentlist2[i])
            age_list_test.append(agelist[i])
    return (
        content_list_train, content_list2_train, age_list_train, content_list_test, content_list2_test, age_list_test)


N = 100
lambda_list = np.logspace(-4.5, 3, N, endpoint=True)

content_list

content_list2

age_list

age_list_log

all_data = randomlySplit(content_list, content_list2, age_list, 0.2)

content_list_nogender_train = all_data[0]
content_list_withgender_train = all_data[1]
age_list_train = all_data[2]
age_list_log_train = [math.log(x) for x in age_list_train]

content_list_nogender_test = all_data[3]
content_list_withgender_test = all_data[4]
age_list_test = all_data[5]
age_list_log_test = [math.log(x) for x in age_list_test]

# cross validation for no gender
squared_mean_error_nogender = []
for lambda_ in lambda_list:
    regr_glemnet = ElasticNet(alpha=lambda_, l1_ratio=0, random_state=0, max_iter=1000, normalize=False, positive=False, precompute=False)
    regr_glemnet.fit(content_list_nogender_train, age_list_train)
    length_test = len(age_list_test)
    predict_ontest = regr_glemnet.predict(content_list_nogender_test)
    squared_error = []
    for i in range(0, length_test):
        squared_error.append(math.pow(age_list_test[i] - predict_ontest[i], 2))
    squared_mean_error_nogender.append(np.mean(squared_error))

plt.figure(5)
# plt.scatter(lambda_list, squared_mean_error_nogender, s=80, facecolors='none', edgecolors='r')
plt.plot(lambda_list, squared_mean_error_nogender, 'ro')
plt.xscale('log')
plt.title('Cross Validation Error Ignoring Gender')
plt.xlabel('Lambda')
plt.ylabel('Mean-Sqared Error')

# cross validation for with gender
squared_mean_error_nogender = []
for lambda_ in lambda_list:
    regr_glemnet = ElasticNet(alpha=lambda_, l1_ratio=0, random_state=0, max_iter=1000, normalize=False, positive=False, precompute=False)
    regr_glemnet.fit(content_list_withgender_train, age_list_train)
    length_test = len(age_list_test)
    predict_ontest = regr_glemnet.predict(content_list_withgender_test)
    squared_error = []
    for i in range(0, length_test):
        squared_error.append(math.pow(age_list_test[i] - predict_ontest[i], 2))
    squared_mean_error_nogender.append(np.mean(squared_error))

plt.figure(6)
# plt.scatter(lambda_list, squared_mean_error_nogender, s=80, facecolors='none', edgecolors='r')
plt.plot(lambda_list, squared_mean_error_nogender, 'bo')
plt.xscale('log')
plt.title('Cross Validation Error Considering Gender')
plt.xlabel('Lambda')
plt.ylabel('Mean-Sqared Error')

# cross validation for no gender log
squared_mean_error_nogender = []
for lambda_ in lambda_list:
    regr_glemnet = ElasticNet(alpha=lambda_, l1_ratio=0, random_state=0, max_iter=1000, normalize=False, positive=False, precompute=False)
    regr_glemnet.fit(content_list_nogender_train, age_list_log_train)
    length_test = len(age_list_log_test)
    predict_ontest_log = regr_glemnet.predict(content_list_nogender_test)
    predict_ontest = [math.exp(x) for x in predict_ontest_log]
    squared_error = []
    for i in range(0, length_test):
        squared_error.append(math.pow(age_list_test[i] - predict_ontest[i], 2))
    squared_mean_error_nogender.append(np.mean(squared_error))

# print lambda_list
# print squared_mean_error_nogender
plt.figure(7)
# plt.scatter(lambda_list, squared_mean_error_nogender, s=80, facecolors='none', edgecolors='r')
plt.plot(lambda_list, squared_mean_error_nogender, 'ro')
plt.xscale('log')
plt.title('Cross Validation Error Ignoring Gender log ')
plt.xlabel('Lambda')
plt.ylabel('Mean-Sqared Error')


# cross validation for with gender log
squared_mean_error_nogender = []
for lambda_ in lambda_list:
    regr_glemnet = ElasticNet(alpha=lambda_, l1_ratio=0, random_state=0, max_iter=1000, normalize=False, positive=False, precompute=False)
    regr_glemnet.fit(content_list_withgender_train, age_list_log_train)
    length_test = len(age_list_log_test)
    predict_ontest_log = regr_glemnet.predict(content_list_withgender_test)
    predict_ontest = [math.exp(x) for x in predict_ontest_log]
    squared_error = []
    for i in range(0, length_test):
        squared_error.append(math.pow(age_list_test[i] - predict_ontest[i], 2))
    squared_mean_error_nogender.append(np.mean(squared_error))
#
# print lambda_list
# print squared_mean_error_nogender
plt.figure(8)
# plt.scatter(lambda_list, squared_mean_error_nogender, s=80, facecolors='none', edgecolors='r')
plt.plot(lambda_list, squared_mean_error_nogender, 'bo')
plt.xscale('log')
plt.title('Cross Validation Error Considering Gender log ')
plt.xlabel('Lambda')
plt.ylabel('Mean-Sqared Error')


print 'R square for regular regression'
regr_re = linear_model.LinearRegression()
regr_re.fit(content_list_nogender_train, age_list_train)
print regr_re.score(content_list_nogender_train, age_list_train)

regr_re = linear_model.LinearRegression()
regr_re.fit(content_list_withgender_train, age_list_train)
print regr_re.score(content_list_withgender_train, age_list_train)

regr_re = linear_model.LinearRegression()
regr_re.fit(content_list_nogender_train, age_list_log_train)
result_log= regr_re.predict(content_list_nogender_train)
result = [math.exp(x) for x in result_log]
print np.var(result)/np.var(age_list_train)

regr_re = linear_model.LinearRegression()
regr_re.fit(content_list_withgender_train, age_list_log_train)
result_log= regr_re.predict(content_list_withgender_train)
result = [math.exp(x) for x in result_log]
print np.var(result)/np.var(age_list_train)

print '///////////////////////////////////////////'


print 'R square for regularization: '
alpha = 0
al = -8.1
lambda_select = 10**(al)
regr_glemnet = ElasticNet(alpha=lambda_select, l1_ratio=alpha, random_state=0, max_iter=2000, normalize=False, positive=False, precompute=False)
regr_glemnet.fit(content_list_nogender_train, age_list_train)
length_test = len(age_list_train)
predict_ontest = regr_glemnet.predict(content_list_nogender_train)
R = np.var(predict_ontest)/np.var(age_list_train)
print R





lambda_select = 10**(al)
regr_glemnet = ElasticNet(alpha=lambda_select, l1_ratio=alpha, random_state=0, max_iter=2000, normalize=False, positive=False, precompute=False)
regr_glemnet.fit(content_list_withgender_train, age_list_train)
length_test = len(age_list_train)
predict_ontest = regr_glemnet.predict(content_list_withgender_train)
R = np.var(predict_ontest)/np.var(age_list_train)
print R



lambda_select = 10**(al)
regr_glemnet = ElasticNet(alpha=lambda_select, l1_ratio=alpha, random_state=0, max_iter=2000, normalize=False, positive=False, precompute=False)
regr_glemnet.fit(content_list_nogender_train, age_list_log_train)
length_test = len(age_list_log_train)
predict_ontest_log = regr_glemnet.predict(content_list_nogender_train)
predict_ontest = [math.exp(x) for x in predict_ontest_log]
R = np.var(predict_ontest)/np.var(age_list_train)
print R



lambda_select = 10**(al)
regr_glemnet = ElasticNet(alpha=lambda_select, l1_ratio=alpha, random_state=0, max_iter=2000, normalize=False, positive=False, precompute=False)
regr_glemnet.fit(content_list_withgender_train, age_list_log_train)
length_test = len(age_list_log_train)
predict_ontest_log = regr_glemnet.predict(content_list_withgender_train)
predict_ontest = [math.exp(x) for x in predict_ontest_log]
R = np.var(predict_ontest)/np.var(age_list_train)
print R



plt.show()

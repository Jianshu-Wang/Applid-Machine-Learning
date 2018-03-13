import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


text_file = open("brunhild.txt", "r")
hours = []	#input variable
sulphate = [] #target variable
i=0
for line in text_file:
	the_line = line.strip().split('\t')
	if(i != 0):
		hours.append(float(the_line[0]))
		sulphate.append(float(the_line[1]))
	i = i + 1
# reshping to variables we can operate upon
hours = np.reshape(hours,(-1,1))
sulphate = np.reshape(sulphate,(-1,1))
# print(hours)
# print(sulphate)

#convert to log values
log_hours = np.log(hours)
log_sulphate = np.log(sulphate)
#print(log_hours)
#print(log_sulphate)

# training set
X_train = log_hours
X_test = log_hours

# testing set
y_train = log_sulphate
y_test = log_sulphate

#Here, training set and testing set is the same

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.7f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.7f' % r2_score(y_test, y_pred))

# Plot 1(log -log co-ordinates)
log_log_coordinates = plt.figure(1)
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=2.0)
log_log_coordinates.axes[0].set_title('Log-log coordinates')
log_log_coordinates.axes[0].set_xlabel('Log Hours')
log_log_coordinates.axes[0].set_ylabel('Log Sulphate')

plt.xticks((np.arange(0,6)))
plt.yticks((np.arange(1,3,0.5)))


#plot 2(normal co-ordinates)
normal_coordinates = plt.figure(2)
predicted_y = np.e**y_pred
plt.scatter(hours, sulphate,  color='red')
plt.plot(hours, predicted_y, color='cyan', linewidth=2.0)
normal_coordinates.axes[0].set_title('Normal coordinates')
normal_coordinates.axes[0].set_xlabel('Hours')
normal_coordinates.axes[0].set_ylabel('Sulphate')

plt.xticks((np.arange(0,180,12)))
plt.yticks((np.arange(4,16)))


#plot 3 (residual vs fitted : log-log)
res_fit_log_log = plt.figure(3)
residual = y_test - y_pred
fitted = y_pred - y_pred
plt.scatter(y_pred, residual,  color='green')
plt.plot(y_pred, fitted, color='magenta', linewidth=2.0)
res_fit_log_log.axes[0].set_title('Residual vs fitted : log-log coordinates')
res_fit_log_log.axes[0].set_xlabel('Fitted values')
res_fit_log_log.axes[0].set_ylabel('Residual')

plt.xticks(np.arange(1,3,0.5))
plt.yticks(np.arange(-0.1,0.125,0.025))



#plot 4 (residual vs fitted : normal)
res_fit_normal = plt.figure(4)
residual = np.e**y_test - np.e**y_pred
fitted = np.e**y_pred - np.e**y_pred
plt.scatter(predicted_y, residual,  color='orange')
plt.plot(predicted_y, fitted, color='yellow', linewidth=2.0)
res_fit_normal.axes[0].set_title('Residual vs fitted : normal coordinates')
res_fit_normal.axes[0].set_xlabel('Fitted values')
res_fit_normal.axes[0].set_ylabel('Residual')
#
# plt.xticks()
# plt.yticks()




#plot 5 (histogram)
res_fit_normal = plt.figure(5)
import numpy as np
# print residual
# fixed bin size
bins = np.arange(-2, 2, 0.05) # fixed bin size
# print bins

plt.xlim([-2, 2])

plt.hist(residual, bins=bins, alpha=0.5)
plt.title('histogram of residual (fixed bin size)')
plt.xlabel('Residual')
plt.ylabel('Count')

plt.show()




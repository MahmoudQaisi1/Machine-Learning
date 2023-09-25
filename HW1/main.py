import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression as lr


def ReplaceZeros(data):
    for row in data:
        for x in range(0, len(row)):
            if row[x] == 0:
                row[x] = int(sum(row) / 5)
    return data


def SplitList(data):
    data1 = []
    data2 = []
    data3 = []
    data4 = []
    y = []
    for row in data:
        data1.append(int(row[0]))
        data2.append(int(row[1]))
        data3.append(int(row[2]))
        data4.append(int(row[3]))
        y.append(int(row[4]))
    return np.array(data1), np.array(data2), np.array(data3), np.array(data4), np.array(y)


def linearRegression(x, y):
    n = len(y)
    w1 = (sum(np.multiply(x, y)) - (sum(y) * sum(x)) / n) / (sum(np.multiply(x, x)) - (sum(x) * sum(x)) / n)
    w0 = (sum(y) / n) - (w1 * sum(x) / n)
    return w0, w1


def mean_squared_error(y_true, y_predicted):
    cost = np.sum((y_true - y_predicted) ** 2) / len(y_true)
    return cost


def gradient_descent(x, y, iterations=1000, learning_rate=0.0001, stopping_threshold=1e-6):
    current_weight = 0.1
    current_bias = 0.01
    iterations = iterations
    learning_rate = learning_rate
    n = float(len(x))

    costs = []
    weights = []
    previous_cost = None

    for i in range(iterations):

        y_predicted = (current_weight * x) + current_bias

        current_cost = mean_squared_error(y, y_predicted)

        if previous_cost and abs(previous_cost - current_cost) <= stopping_threshold:
            break

        previous_cost = current_cost

        costs.append(current_cost)
        weights.append(current_weight)

        weight_derivative = -(2 / n) * sum(x * (y - y_predicted))
        bias_derivative = -(2 / n) * sum(y - y_predicted)

        current_weight = current_weight - (learning_rate * weight_derivative)
        current_bias = current_bias - (learning_rate * bias_derivative)

    return current_weight, current_bias


data = pd.read_csv("grades.csv", skiprows=0, index_col=False)
data = data.values.tolist()
data = ReplaceZeros(data)
data1, data2, data3, data4, y = SplitList(data)


w0, w1 = linearRegression(data3, y)


y_pred = w0 + [i * w1 for i in data3]

print(w0, w1)

figure, axis = plt.subplots(2, 2)

axis[0, 0].scatter(data1, y, s=3)
axis[0, 0].set_title("HW1")

axis[0, 1].scatter(data2, y, s=3)
axis[0, 1].set_title("HW2")

axis[1, 0].scatter(data3, y, s=3)
axis[1, 0].set_title("Midterm")

axis[1, 1].scatter(data4, y, s=3)
axis[1, 1].set_title("Project")

plt.show()

w2, w3 = gradient_descent(data3, y)

y_pred2 = w2 * data3 + w3

regressor = lr()

dataT = data3.reshape(-1,1)
yT = y.reshape(-1,1)

regressor.fit(dataT, yT)

y_pred3 = regressor.coef_ * data3 + regressor.intercept_

plt.scatter(data3, y, s=3)
plt.plot(data3, y_pred2, c="g")
plt.plot(data3, y_pred, c="r")
plt.plot(data3, y_pred3[0])

plt.show()

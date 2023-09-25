import tensorflow as tf
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

indexes = np.arange(x_train.shape[0])
np.random.shuffle(indexes)
x_train = x_train[indexes]
y_train = y_train[indexes]

x_train, x_val = x_train[:40000], x_train[40000:]
y_train, y_val = y_train[:40000], y_train[40000:]


knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
knn.fit(x_train, y_train)
print("Validation set accuracy and hyper parameter tuning:")
y_pred = knn.predict(x_val)
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy -k = 1, metric: euclidean- :", accuracy)

knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(x_train, y_train)

y_pred = knn.predict(x_val)
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy -k = 3, metric: euclidean- :", accuracy)

knn = KNeighborsClassifier(n_neighbors=1, metric='manhattan')
knn.fit(x_train, y_train)

y_pred = knn.predict(x_val)
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy -k = 1, metric: manhatten- :", accuracy)

knn = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
knn.fit(x_train, y_train)

y_pred = knn.predict(x_val)
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy -k = 3, metric: manhatten- :", accuracy)

print("Testing set accuracy:")

y_pred = knn.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy -k = 3, metric: manhatten- :", accuracy)
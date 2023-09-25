import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

indexes = np.arange(x_train.shape[0])
np.random.shuffle(indexes)
x_train = x_train[indexes]
y_train = y_train[indexes]

x_train, x_val = x_train[:40000], x_train[40000:]
y_train, y_val = y_train[:40000], y_train[40000:]

x_train = np.array(x_train)
y_train = np.array(y_train)

x_val = np.array(x_val)
y_val = np.array(y_val)

x_test = np.array(x_test)
y_test = np.array(y_test)


def extract_hog_features(images):
    hog_features = []
    for image in images:
        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(7, 7), cells_per_block=(2, 2), visualize=True)
        hog_features.append(hog_image)
    return np.array(hog_features)


x_train_hog = extract_hog_features(x_train)
x_val_hog = extract_hog_features(x_val)
x_test_hog = extract_hog_features(x_test)

x_train_hog = x_train_hog.reshape(-1, 784)
x_val_hog = x_val_hog.reshape(-1, 784)
x_test_hog = x_test_hog.reshape(-1, 784)

x_train_hog = x_train_hog / 255
x_val_hog = x_val_hog / 255
x_test_hog = x_test_hog / 255

knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
knn.fit(x_train_hog, y_train)
print("Validation set accuracy and hyper parameter tuning:")
y_pred = knn.predict(x_val_hog)
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy -k = 1, metric: euclidean- :", accuracy)

knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(x_train_hog, y_train)

y_pred = knn.predict(x_val_hog)
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy -k = 3, metric: euclidean- :", accuracy)

knn = KNeighborsClassifier(n_neighbors=1, metric='manhattan')
knn.fit(x_train_hog, y_train)

y_pred = knn.predict(x_val_hog)
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy -k = 1, metric: manhatten- :", accuracy)

knn = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
knn.fit(x_train_hog, y_train)

y_pred = knn.predict(x_val_hog)
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy -k = 3, metric: manhatten- :", accuracy)

print("Testing set accuracy:")

y_pred = knn.predict(x_test_hog)
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy -k = 3, metric: manhatten- :", accuracy)

svm = SVC(kernel='rbf', C=1)
svm.fit(x_train_hog, y_train)

y_pred = svm.predict(x_val_hog)
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy(C = 1):", accuracy)

svm = SVC(kernel='rbf', C=5)
svm.fit(x_train_hog, y_train)

y_pred = svm.predict(x_val_hog)
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy(C = 5):", accuracy)

svm = SVC(kernel='rbf', C=10)
svm.fit(x_train_hog, y_train)

y_pred = svm.predict(x_val_hog)
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy(C = 10):", accuracy)

pSvmTest = svm.predict(x_test_hog)
accuracy = accuracy_score(y_test, pSvmTest)
print("SVM Testing Accuracy (C = 10):", accuracy)

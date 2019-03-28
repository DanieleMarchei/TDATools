import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from PIL import Image
import math

from os import listdir
from os.path import isfile, join

def fit_test(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train, Y_train)

    predicted = model.predict(X_test)

    print("Score:", end = " ")
    print(model.score(X_test, Y_test))

    conf_mat = confusion_matrix(Y_test, predicted, labels=[0,1])
    print("Confusion matrix:")
    print(conf_mat)

    precision = precision_score(Y_test, predicted)
    print("Precision :", end = " ")
    print(precision)

    recall = recall_score(Y_test, predicted)
    print("Recall :", end = " ")
    print(recall)

    return model.coef_

def plot_coeff(fig, coeff, location, title):
    pim = PersImage(pixels=[10,10], spread=1)
    ax = fig.add_subplot(location)
    inverse_image = np.copy(coeff).reshape((10,10))
    ax.set_title(title)
    pim.show(inverse_image, ax)


#load all the images
imgs = []
for f in listdir("dataset"):
    if isfile("dataset\\"+f):
        imgs.append(np.array([np.array(Image.open("dataset\\"+f)).flatten(), 1 if "circle" in f else 0]))

imgs = np.array(imgs)

print("Dataset size = " + str(len(imgs)))

X = imgs[:,0]

Y = imgs[:,1]

#divide test and training sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)
X_train = list(X_train)
Y_train = list(Y_train)
X_test = list(X_test)
Y_test = list(Y_test)

#----------------------- SVM
print("-------- SVM ---------")

svm = SVC(kernel = "linear")

coefSVM = fit_test(svm, X_train, Y_train, X_test, Y_test)

#-------------------------- Logistic Regression
print("-------- Logistic Regression ---------")

lr = LogisticRegression(solver = "liblinear")

coefLR = fit_test(lr, X_train, Y_train, X_test, Y_test)

#-------------------------- Perceptron
print("-------- Perceptron ---------")

perc = Perceptron(max_iter = 1000, tol = - math.inf)

coefPerc = fit_test(perc, X_train, Y_train, X_test, Y_test)


from persim import PersImage
import matplotlib.pyplot as plt

fig = plt.figure()

plot_coeff(fig, coefSVM, 131, "SVM")

plot_coeff(fig, coefLR, 132, "Logistic Regression")

plot_coeff(fig, coefPerc, 133, "Perceptron")



plt.show()
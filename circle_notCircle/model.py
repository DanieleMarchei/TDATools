import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from PIL import Image

#load all the images
filelist = []
for i in range(100):
    filelist.append("circle_" + str(i) + ".png")
    filelist.append("noise_" + str(i) + ".png")

imgs = np.array([np.array([np.array(Image.open("dataset\\"+fname)).flatten() , 1 if "circle" in fname else 0]) for fname in filelist])
X = imgs[:,0]

Y = imgs[:,1]
#divide test and training sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)
X_train = list(X_train)
Y_train = list(Y_train)
X_test = list(X_test)
Y_test = list(Y_test)


svm = SVC(kernel = "linear")

svm.fit(X_train, Y_train)

predicted = svm.predict(X_test)

print("Score:", end = " ")
print(svm.score(X_test, Y_test))

conf_mat = confusion_matrix(Y_test, predicted, labels=[0,1])
print("Confusion matrix:")
print(conf_mat)

precision = precision_score(Y_test, predicted)
print("Precision :", end = " ")
print(precision)

recall = recall_score(Y_test, predicted)
print("Recall :", end = " ")
print(recall)

coef = svm.coef_

from persim import PersImage
import matplotlib.pyplot as plt

pim = PersImage(pixels=[10,10], spread=1)

inverse_image = np.copy(coef).reshape((10,10))
plt.title("10x10 SVM coefficients")
pim.show(inverse_image)
plt.show()
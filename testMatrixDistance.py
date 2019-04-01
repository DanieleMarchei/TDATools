import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import linalg as LA

A = np.matrix([[1,2],[3,4]])
B = np.matrix([[42,-2],[34,5]])
M = A - B
iters = 100

X = np.zeros((iters, 3))

maxPoint = (0, 0, -1000)

for i in range(iters):
    s = np.sin(i)
    c = np.cos(i)
    arr = np.array([[s],[c]])
    X[i,0] = s
    X[i,1] = c
    X[i,2] = LA.norm(M * arr)
    if X[i,2] > maxPoint[2]:
        maxPoint = (s,c,X[i,2])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("sin")
ax.set_ylabel("cos")
ax.scatter(X[:,0],X[:,1],X[:,2])
print(maxPoint)
plt.show()
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sys import argv
import numpy as np

data = np.genfromtxt(argv[1], delimiter=",")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(data[:,0], data[:,1], data[:,2])
name = argv[1].replace(".csv","")

plt.title(name + "\nN = " + str(len(data)))

plt.show()
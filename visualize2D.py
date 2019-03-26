import matplotlib.pyplot as plt
from sys import argv
import numpy as np

data = np.genfromtxt(argv[1], delimiter=",")

if len(argv) == 3:
    plt.plot(data[:,0], data[:,1])
else:
    plt.scatter(data[:,0], data[:,1])

name = argv[1].replace(".csv","")

plt.title(name + "\nN = " + str(len(data)))

plt.show()
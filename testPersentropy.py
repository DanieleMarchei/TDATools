from ripser import ripser, plot_dgms
import matplotlib.pyplot as plt
import sys
import numpy as np
from persentropy import persentropy

script, fileName = sys.argv

data = np.genfromtxt(fileName, delimiter = ",")
dgms = ripser(data, maxdim = 2)["dgms"]
print(dgms[0])
entropies = persentropy(dgms)
print(entropies)

plot_dgms(dgms)
plt.show()
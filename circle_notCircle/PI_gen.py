import numpy as np
from sklearn import datasets

from ripser import ripser
from persim import PersImage, plot_diagrams
import scipy.misc

#generate circles
print("Generating circles")
data = None
pim = PersImage(spread=1, pixels=[10,10], verbose=False)

for i in range(100):
    if i % 10 != 0:
        print(str(i), end = " ")
    else:
        print(str(i))

    data = np.concatenate([150 * np.random.random((150,2)),
                np.random.randint(10,100) + 20 * datasets.make_circles(n_samples=150, factor=0.99)[0]])

    dgms = ripser(data)["dgms"]

    img = pim.transform(dgms[1])

    pImg = np.zeros((10, 10), dtype=np.uint8)

    for idxR, r in enumerate(img):
        for idxC, c in enumerate(r):
            pImg[idxR][idxC] = int(c * 255)

    scipy.misc.imsave("dataset\\circle_"+str(i)+".png", pImg)    

print("\nGenerating noise")
f = None
for i in range(100):
    if i != 0 and i % 10 == 0:
        print(str(i))
    else:
        print(str(i), end = " ")

    noise = 150 * np.random.random((300,2))

    data = noise
    dgms = ripser(data)["dgms"]

    img = pim.transform(dgms[1])

    pImg = np.zeros((10, 10), dtype=np.uint8)

    for idxR, r in enumerate(img):
        for idxC, c in enumerate(r):
            pImg[idxR][idxC] = int(c * 255)

    scipy.misc.imsave("dataset\\noise_"+str(i)+".png", pImg)    
    


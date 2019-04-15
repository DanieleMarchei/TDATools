import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
import PIL

from ripser import ripser, plot_dgms, lower_star_img


ts = np.linspace(-1, 1, 100)
x1 = np.exp(-ts**2/(0.1**2))
ts -= 0.4
x2 = np.exp(-ts**2/(0.1**2))
img = -x1[None, :]*x1[:, None] - 2*x1[None, :]*x2[:, None] - 3*x2[None, :]*x2[:, None]


dgm = lower_star_img(img)

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(img)
plt.colorbar()
plt.title("Test Image")
plt.subplot(122)
plot_dgms(dgm)
plt.title("0-D Persistence Diagram")
plt.tight_layout()
plt.show()
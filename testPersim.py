import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from ripser import ripser
from persim import PersImage, plot_diagrams
from sys import argv

dgms = None
first = True
sld_spread = None
sld_resolution = None
bgen = None
bgennoise = None
plt.figure()
old_spread = 1
old_resolution = 10

def calculatePI(val = None):
    global dgms, first
    if not first:
        r = int(sld_resolution.val)
        v = sld_spread.val
    else:
        r = 10
        v = 1
    pim = PersImage(spread=v, pixels=[r,r], verbose=False)
    img = pim.transform(dgms[1])
    ax_1 = plt.subplot(233)
    plt.title("PI for $H_1$\nspread = " + str(v)[0:4] + "\n" + str(r) + "x" + str(r))
    pim.show(img, ax_1)
    first = False

def generate(val = None):
    plt.clf()
    global dgms, sld_resolution, sld_spread, bgen, old_resolution, old_spread, bgennoise
    N_noise = 150
    N_circle1 = 200
    N_circle2 = 150
    N = N_noise + N_circle1 + N_circle2

    if val != None:
        data = np.random.random((N,2))
    else:
        data = np.concatenate([150 * np.random.random((N_noise,2)),
                        np.random.randint(10,100) + 10 * datasets.make_circles(n_samples=N_circle1, factor=0.99)[0],
                        np.random.randint(10,100) + 20 * datasets.make_circles(n_samples=N_circle2, factor=0.99)[0]])

    if argv[1] != None:
        data = np.genfromtxt(argv[1], delimiter = ",")

    dgms = ripser(data, maxdim = 2)["dgms"]
    plt.subplot(231)
    plt.scatter(data[:,0], data[:,1], s=4)
    plt.title("Scatter plot N = " + str(N))

    plt.subplot(232)
    plot_diagrams(dgms, legend=False, show=False, lifetime=True)

    plt.title("Persistence diagram\nof $H_0$ and $H_1$")

    #to remove the point at infinity
    dgms[0] = dgms[0][0:N-1]

    if not first:
        old_resolution = sld_resolution.val
        old_spread = sld_spread.val

    ax_spread = plt.axes([0.1, 0.2, 0.8, 0.05])
    sld_spread = Slider(ax_spread, "Spread", 0.1, 2, valinit=old_spread, valstep=0.1)

    ax_resolution = plt.axes([0.1, 0.1, 0.8, 0.05])
    sld_resolution = Slider(ax_resolution, "Resolution", 10, 100, valinit=old_resolution, valstep=10)

    ax_gen = plt.axes([0.8, 0.01, 0.1, 0.075])
    bgen = Button(ax_gen, 'Generate Circles')

    ax_gen_noise = plt.axes([0.6, 0.01, 0.1, 0.075])
    bgennoise = Button(ax_gen_noise, 'Generate Noise')

    sld_spread.on_changed(calculatePI)
    sld_resolution.on_changed(calculatePI)
    bgen.on_clicked(lambda x : generate(None))
    bgennoise.on_clicked(lambda x : generate(1))
    calculatePI()
    plt.show()

generate()
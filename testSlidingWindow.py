from TDA.slidingWindow import slidingWindow
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, TextBox, Button
import math

# Step 1: Setup the signal
T = 40 # The period in number of samples
NPeriods = 4 # How many periods to go through
N = T*NPeriods #The total number of samples
t = np.linspace(0, 2*np.pi*NPeriods, N+1)[:N] # Sampling indices in time

x = np.sin(t) + np.cos(3 * t) # The final signal
noise = np.random.rand(N)
#x += noise

sld_dim, sld_tau, sld_dt = None, None, None
txt_eig = None
first = True
ax1, ax2 = None, None

def bestDimension():
    f = open("SW-bestDimension3D.csv", "w+")
    f.write("dim,tau,score")
    for d in range(3, 100):
        for t in range(1, 40):
            X = slidingWindow(x, d, t * 0.05, sld_dt.val)
            pca = PCA(n_components = 3)
            pca.fit_transform(X)
            
            eig1, eig2 = pca.explained_variance_[0], pca.explained_variance_[2]
            if eig2 <= 0.0001:
                eig2 = pca.explained_variance_[1]
            dist = eig1 - eig2
            f.write(str(d) + "," + str(t) + "," + str(dist) + "\n")

def plot(val):

    global first, sld_dim, sld_tau, sld_dt, ax1, ax2, txt_eig
    if not first:
        ax1.clear()
        ax2.clear()
        
    #if first:
        #bestDimension()

    first = False

    ax1 = plt.subplot(221)
    ax1.set_title("Time Serie")
    ax1.plot(x)

    X = slidingWindow(x, int(sld_dim.val), sld_tau.val, sld_dt.val)
    print(X.shape)
    extent = int(sld_dim.val * sld_tau.val)
    yr = np.max(x)-np.min(x)
    yr = [np.min(x)-0.1*yr, np.max(x)+0.1*yr]
    ax1.plot([extent, extent], yr, 'r')
    ax1.plot([0, 0], yr, 'r')     
    ax1.plot([0, extent], [yr[0]]*2, 'r')
    ax1.plot([0, extent], [yr[1]]*2, 'r')

    pca = PCA(n_components = 3)
    Y = pca.fit_transform(X)
    eig = []
    for e in pca.explained_variance_:
        eig.append(round(e, 3))
    
    portion = round(sum(pca.explained_variance_ratio_) * 100,3)
    txt_eig.set_val(str(eig) + " ~ " + str(portion) + "%")

    ax2 = plt.subplot(222, projection='3d')
    ax2.set_title("3PCA sliding window")
    ax2.scatter(Y[:,0], Y[:,1], Y[:,2])

    plt.tight_layout()
    plt.show()

ax_dim = plt.axes([0.1, 0.3, 0.8, 0.05])
sld_dim = Slider(ax_dim, "Dimension", 5, 100, valinit=20, valstep=1)
sld_dim.on_changed(plot)

ax_tau = plt.axes([0.1, 0.2, 0.8, 0.05])
sld_tau = Slider(ax_tau, "Tau", 0, 2, valinit=1, valstep=0.05)
sld_tau.on_changed(plot)

ax_dt = plt.axes([0.1, 0.1, 0.8, 0.05])
sld_dt = Slider(ax_dt, "dT", 0.1, 2, valinit=0.5, valstep=0.05)
sld_dt.on_changed(plot)

ax_eig = plt.axes([0.4, 0.4, 0.5, 0.05])
txt_eig = TextBox(ax_eig, "Explained variances")

ax_bestDim = plt.axes([0.8, 0.01, 0.1, 0.075])
btn_bestDim = Button(ax_bestDim, "Best Dimension")
btn_bestDim.on_clicked(lambda x : bestDimension())

plot(None)
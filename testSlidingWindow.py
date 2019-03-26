import numpy as np
from ripser import ripser, plot_dgms

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import widgets
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA

from slidingWindow import slidingWindow

# Step 1: Setup the signal
T = 40 # The period in number of samples
NPeriods = 4 # How many periods to go through
N = T*NPeriods #The total number of samples
t = np.linspace(0, 2*np.pi*NPeriods, N+1)[:N] # Sampling indices in time
x = np.cos(t) # The final signal

def on_value_change(change):
    execute_computation()

ax = plt.axes([0.1, 0.2, 0.8, 0.05])

dimslider = widgets.Slider(ax, "Dimension", valmin=1, valmax=40, valinit=20)
dimslider.on_changed(on_value_change)

ax = plt.axes([0.1, 0.1, 0.8, 0.05])

Tauslider = widgets.Slider(ax,r'\(\tau :\)' ,valmin=0.1,valmax=5,valstep=0.1,valinit=1)
Tauslider.on_changed(on_value_change)

ax = plt.axes([0.1, 0, 0.8, 0.05])

dTslider = widgets.Slider(ax, "dt", valmin=0.1, valmax=5, valstep=0.1, valinit=0.5)
dTslider.on_changed(on_value_change)


plt.figure(figsize=(9.5, 3))

def execute_computation():
    plt.clf()
    # Step 1: Setup the signal again in case x was lost
    T = 40 # The period in number of samples
    NPeriods = 4 # How many periods to go through
    N = T*NPeriods # The total number of samples
    t = np.linspace(0, 2*np.pi*NPeriods, N+1)[0:N] # Sampling indices in time
    x = np.cos(t)  # The final signal
    
    # Get slider values
    dim = dimslider.val
    Tau = Tauslider.val
    dT = dTslider.val
    
    #Step 2: Do a sliding window embedding
    X = slidingWindow(x, dim, Tau, dT)
    extent = Tau*dim

    #Step 3: Perform PCA down to 2D for visualization
    pca = PCA(n_components = 2)
    Y = pca.fit_transform(X)
    eigs = pca.explained_variance_
    print("lambda1 = %g, lambda2 = %g"%(eigs[0], eigs[1]))

    #Step 4: Plot original signal and PCA of the embedding

    ax = plt.subplot(121)
    ax.plot(x)
    ax.set_ylim((-2*max(x), 2*max(x)))
    ax.set_title("Original Signal")
    ax.set_xlabel("Sample Number")
    yr = np.max(x)-np.min(x)
    yr = [np.min(x)-0.1*yr, np.max(x)+0.1*yr]
    ax.plot([extent, extent], yr, 'r')
    ax.plot([0, 0], yr, 'r')     
    ax.plot([0, extent], [yr[0]]*2, 'r')
    ax.plot([0, extent], [yr[1]]*2, 'r')
    ax2 = plt.subplot(122)
    ax2.set_title("PCA of Sliding Window Embedding")
    ax2.scatter(Y[:, 0], Y[:, 1])
    ax2.set_aspect('equal', 'datalim')
    plt.plot()
    
execute_computation()
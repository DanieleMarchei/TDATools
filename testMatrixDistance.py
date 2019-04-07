import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import linalg as LA
import TDA.matrixDistance as md

A = np.matrix([[1, 2], [3, 4]])

B = np.matrix([[4, 5], [5, 7]])

print("Frobenius:", end = " ")
print(md.frobenius(A,B))

print("Manhattan:", end = " ")
print(md.manhattan(A,B))

print("Euclidian:", end = " ")
print(md.euclidian(A,B))

print("Chebyshev:", end = " ")
print(md.chebyshev(A,B))

print("       dn:", end = " ")
print(md.dn(A,B))

print("       df:", end = " ")
print(md.df(A,B))

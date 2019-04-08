import numpy as np
from numpy import linalg as LA

def matrixTrace(A):
    assert A.shape[0] == A.shape[1] , "Matrix is not square"

    n = A.shape[0]
    sumDiagonal = 0
    for i in range(n):
        sumDiagonal += A[i,i]
    
    return sumDiagonal

def fnorm(A):
    assert A.shape[0] == A.shape[1] , "Matrix is not square"

    n = A.shape[0]
    totalSum = 0
    for i in range(n):
        for j in range(n):
            totalSum += A[i,j]**2
    return np.sqrt(totalSum)


def frobenius(A, B):
    assert A.shape[0] == A.shape[1] , "Matrix is not square"
    assert B.shape[0] == B.shape[1] , "Matrix is not square"
    assert A.shape == B.shape , "Matrices have not the same shape"

    inner = (A - B) * ((A - B).conj().T)
    trace = matrixTrace(inner)
    return np.sqrt(trace)

def manhattan(A, B):
    assert A.shape[0] == A.shape[1] , "Matrix is not square"
    assert B.shape[0] == B.shape[1] , "Matrix is not square"
    assert A.shape == B.shape
   
    return np.sum(np.abs(A-B))

def euclidian(A, B):
    assert A.shape[0] == A.shape[1] , "Matrix is not square"
    assert B.shape[0] == B.shape[1] , "Matrix is not square"
    assert A.shape == B.shape , "Matrices have not the same shape"

    n = A.shape[0]

    finalSum = 0

    for i in range(n):
        for j in range(n):
            finalSum += (A[i,j] - B[i,j])**2

    return np.sqrt(finalSum)

def chebyshev(A, B):
    assert A.shape[0] == A.shape[1] , "Matrix is not square"
    assert B.shape[0] == B.shape[1] , "Matrix is not square"
    assert A.shape == B.shape , "Matrices have not the same shape"

    n = A.shape[0]

    maxs = []

    for i in range(n):
        m = np.max(np.abs(A[:,i] - B[:,i]))
        maxs.append(m)
    
    return np.max(maxs)

def dn(A, B, iterations = 500):
    assert A.shape[0] == A.shape[1] , "Matrix is not square"
    assert B.shape[0] == B.shape[1] , "Matrix is not square"
    assert A.shape == B.shape , "Matrices have not the same shape"

    n = A.shape[0]
    M = A - B

    maxDist = -1
    for i in range(iterations):
        arr = []
        for j in range(n):
            arr.append([np.random.randint(-1000, 1000)])
        arr = np.array(arr)
        norm = LA.norm(arr)
        unit_arr = arr / norm
        d = LA.norm(M * unit_arr)
        if d > maxDist:
            maxDist = d

    return maxDist

def df(A, B):
    assert A.shape[0] == A.shape[1] , "Matrix is not square"
    assert B.shape[0] == B.shape[1] , "Matrix is not square"
    assert A.shape == B.shape , "Matrices have not the same shape"

    return np.abs(fnorm(A) - fnorm(B))

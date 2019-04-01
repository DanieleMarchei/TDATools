import numpy as np
from numpy import linalg as LA

def matrixTrace(A):
    assert(A.shape[0] == A.shape(1))

    n = A.shape[0]
    sumDiagonal = 0
    for i in range(n):
        sumDiagonal += A[i,i]
    
    return sumDiagonal

def fnorm(A):
    assert(A.shape[0] == A.shape(1))

    n = A.shape[0]
    totalSum = 0
    for i in range(n):
        for j in range(n):
            totalSum += A[i,j]**2
    return np.sqrt(totalSum)


def frobenius(A, B):
    assert(A.shape == B.shape)

    inner = (A - B) * ((A - B).conj().T)
    trace = matrixTrace(inner)
    return np.sqrt(trace)

def d1(A, B):
    assert(A.shape == B.shape)

    n = A.shape[0]

    finalSum = 0

    for i in range(n):
        for j in range(n):
            finalSum += np.abs(A[i,j] - B[i,j])

def d2(A, B):
    assert(A.shape == B.shape)

    n = A.shape[0]

    finalSum = 0

    for i in range(n):
        for j in range(n):
            finalSum += (A[i,j] - B[i,j])**2

    return np.sqrt(finalSum)

def dinf(A, B):
    assert(A.shape == B.shape)

    n = A.shape[0]

    maxs = []

    for i in range(n):
        m = np.max(np.abs(A[:,i] - B[:,i]))
        maxs.append(m)
    
    return np.max(maxs)

def dn(A, B, iterations = 50):
    raise NotImplementedError("Sill don't know how to do it in n dimensions")
    assert(A.shape == B.shape)

    n = A.shape[0]

    iters = []
    for i in range(iterations):
        s = np.sin(i)
        c = np.cos(i)
        arr = np.array([[s],[c]])
        iters.append(LA.norm(arr))
    
    return np.max(iters)

def df(A, B):
    assert(A.shape == B.shape)

    return np.abs(fnorm(A) - fnorm(B))

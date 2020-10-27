import numpy as np

def e(A, refNo):
    v = A[:, refNo]
    e = np.zeros(len(v))
    if (refNo > 0):
        for i in range(refNo):
            v[i] =0
    e[refNo] = 1
    e = np.array([e]).T 
    return np.linalg.norm(v) @ e 

A = np.array([[1,-5, 9], [-2, 6, -10], [3,-7,11], [-4, 8, 12]])

rows = A.shape[0]
cols = A.shape[1]

v = np.array([A[:,0]]).T + np.linalg.norm(A[:,0]) @ np.array


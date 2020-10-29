import numpy as np
from numpy.core.defchararray import index

def e(v, index, sign):
    base = np.zeros(len(v))
    base[index] = 1
    base = sign * np.linalg.norm(v) * base    
    return base 

def calculate_Q(v):
    rows = v.shape[0]
    k = (v.T @ v)
    q = np.eye(rows) - 2 * v @ v.T / k[0,0]
    return q

# A = np.array([[1,-5, 9], [-2, 6, -10], [3,-7,11], [-4, 8, 12]])
A = np.array([[1,-2, 13], [-6, 5, -4], [7,-8, 9], [-12, 11, -10]])

rows = A.shape[0]
cols = A.shape[1]
qList = []

for i in range(cols):
    v = A[:,i]
    sign = np.sign(v[i])
    if i>0:
        for j in range(i):
            v[j] = 0
    v = np.array([v]).T + np.array([e(v,i, sign)]).T
    print(v)
    QQ = calculate_Q(v)
    A = QQ @ A
    print(A)


Q = np.eye(rows)
# R = A
for qMatrix in qList:
    Q = Q @ qMatrix.T 
    # R = qMatrix @ A

# print()
# print(A)

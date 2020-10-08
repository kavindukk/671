  
from scipy.io import loadmat
import numpy as np

data = loadmat('mid1_p3_data.mat')
x = data['x']
t = data['t']
x = np.array(x)
t = np.array(t)

A = np.zeros((1001,4))
for i in range(1001):
    A[i,0] = np.sin(10*t[0,i])
    A[i,1] = t[0,i]**2
    A[i,2] = t[0,i]
    A[i,3] = 1

C = np.linalg.inv( np.transpose(A) @ A ) @ np.transpose(A) @ np.transpose(x)
# C = np.linalg.inv( np.transpose(A) @ A ) 
print(C)
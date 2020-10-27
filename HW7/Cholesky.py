import numpy as np

B = np.array([[1,2,4,1], [2,13,17,8], [4,17,29,16], [1,8,16,30]])

def calculate_chelesky(B:np.array):
    mIndex = B.shape[0]
    L = np.zeros((mIndex,mIndex))
    alpha = B[0,0]
    for i in range(mIndex-1):
        L[i,i] = np.sqrt(B[0,0])
        L[i,i+1:] = B[0,1:]/np.sqrt(alpha)
        vT = np.array([B[0,1:]])
        B = B[1:,1:] - vT.T @ vT / alpha
        alpha = B[0,0]        
    L[mIndex-1,mIndex-1] = np.sqrt(alpha)
    return L
    
    
L = calculate_chelesky(B)
print(L)


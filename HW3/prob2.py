from scipy.io import loadmat
import numpy as np

data = loadmat('prob2.mat')

A = data['A']
x = data['x']
xVector = x.flatten()

firstImage = A[:,:,0]
firstImageVector = firstImage.flatten()
imageVectorMatrix = np.array([firstImageVector])

for i in range(1,20):
    image = A[:,:,i]
    vector = image.flatten()
    vector = np.array([vector])
    imageVectorMatrix =  np.append(imageVectorMatrix, vector, axis=0)

innerProductofXwithP = np.zeros((20,1))
for i in range(20):
    innerProductofXwithP[i,0] = xVector @ imageVectorMatrix[i,:]

grammianMatrix = np.zeros((20,20))
for i in range(20):
    for j in range(20):
        grammianMatrix[i,j] = imageVectorMatrix[i,:] @ imageVectorMatrix[j,:]

coeffs = np.linalg.inv(grammianMatrix) @ innerProductofXwithP


print(coeffs)
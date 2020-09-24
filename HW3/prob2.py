from scipy.io import loadmat
import numpy as np

data = loadmat('prob2.mat')

A = data['A']
x = data['x']

firstImage = A[:,:,0]
firstImageVector = firstImage.flatten()
imageVectorMatrix = np.array([firstImageVector])

for i in range(1,20):
    image = A[:,:,i]
    vector = image.flatten()
    vector = np.array([vector])
    imageVectorMatrix =  np.append(imageVectorMatrix, vector, axis=0)

innerProductofXwithP = np.zeros((20,1))


print(v.shape)
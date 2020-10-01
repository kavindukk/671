import matplotlib.pyplot as plt
import numpy as np

x = np.array([2,2.5,3.5,9])
y = np.array([-4.2,-5.2, 1, 24.3])
plt.scatter(x,y)
plt.show()

y = np.array([-4.2,-5.2,1,24.3]).reshape(4,1)
A = np.transpose(np.array([[2,2.5,3.5,9],[1,1,1,1]]))
C = np.linalg.inv(np.transpose(A)@A)@np.transpose(A)@y

w = np.diag(np.array([10,1,1,10]))
wX = w@x
wY = w@y
wA = np.transpose(np.array([wX,[1,1,1,1]]))
wC = np.linalg.inv(np.transpose(wA)@wA)@np.transpose(wA)@wY
print(wC)
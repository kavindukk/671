import numpy as np
from matplotlib import pyplot as plt

polynomials = np.zeros((5,10000), dtype=np.float64)
step = np.linspace(-1,1, num=10000)
for i in range(5):
    for j in range(10000):
        if i==0:
            polynomials[i,j] = 1
        if i==1:
            polynomials[i,j] = step[j]
        if i==2:
            polynomials[i,j] = step[j]**2 - (1/3)
        if i==3:
            polynomials[i,j] = step[j]**3 - 3*step[j]/5
        if i==4:
            polynomials[i,j] = step[j]**4 + (step[j]**2)/2 -(11/30)

plt.plot(step, polynomials[0,:], label='first')
plt.plot(step, polynomials[1,:], label='second')
plt.plot(step, polynomials[2,:], label='third')
plt.plot(step, polynomials[3,:], label='fourth')
plt.plot(step, polynomials[4,:], label='fifth')
plt.show()


import numpy as np
from scipy.integrate import quad
from matplotlib import pyplot as plt

# part A
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
            polynomials[i,j] = step[j]**4 - (6/7)*(step[j]**2) +(2/15)
plt.figure('Legrande Polynomials by using Gram-Schmidt')
plt.plot(step, polynomials[0,:], label='first')
plt.plot(step, polynomials[1,:], label='second')
plt.plot(step, polynomials[2,:], label='third')
plt.plot(step, polynomials[3,:], label='fourth')
plt.plot(step, polynomials[4,:], label='fifth')
plt.xlabel('x value')
plt.ylabel('y value')
plt.legend()
plt.show()

# part B
# Calculating coefficients
def y_0(x):
    return 1
def y_1(x):
    return x
def y_2(x):
    return x**2 - (1/3)
def y_3(x):
    return x**3 - (3/5)*x
def y_4(x):
    return x**4 - (6/7)*(x**2) + (2/15)
def f_n(x):
    return np.exp(-x)
    
c0 = quad(lambda x: y_0(x)*f_n(x), -1, 1)[0]/quad(lambda x: y_0(x)*y_0(x), -1, 1)[0]
c1 = quad(lambda x: y_1(x)*f_n(x), -1, 1)[0]/quad(lambda x: y_1(x)*y_1(x), -1, 1)[0]
c2 = quad(lambda x: y_2(x)*f_n(x), -1, 1)[0]/quad(lambda x: y_2(x)*y_2(x), -1, 1)[0]
c3 = quad(lambda x: y_3(x)*f_n(x), -1, 1)[0]/quad(lambda x: y_3(x)*y_3(x), -1, 1)[0]
c4 = quad(lambda x: y_4(x)*f_n(x), -1, 1)[0]/quad(lambda x: y_4(x)*y_4(x), -1, 1)[0]
check = [c0, c1, c2, c3, c4]

# Plot f(t) and the approximation
eToThePowMinusT = [np.exp(-x) for x in step]
plt.figure('f(t)=np.exp(-t) approximation')
plt.plot(step, eToThePowMinusT, label='f(t)=e**(-t)')
plt.plot(step, polynomials[0,:]*c0, label='first approximation')
plt.plot(step, polynomials[1,:]*c1, label='second approximation')
plt.plot(step, polynomials[2,:]*c2, label='third approximation')
plt.plot(step, polynomials[3,:]*c3, label='fourth approximation')
plt.plot(step, polynomials[4,:]*c4, label='fifth approximation')
plt.xlabel('x value')
plt.ylabel('y value')
plt.legend()
plt.show()

# Calculate the error norm
approxError = [(f_n(x)-c0*y_0(x)-c1*y_1(x)-c2*y_2(x)-c3*y_3(x)-c4*y_4(x)) for x in step]
normE=0
for i in approxError:
    normE += i**2
normE = np.sqrt(normE)
print(normE)


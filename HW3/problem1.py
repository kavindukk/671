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
# plt.subplot(4,1,1)
plt.plot(step, polynomials[0,:], label='first')
plt.plot(step, polynomials[1,:], label='second')
plt.plot(step, polynomials[2,:], label='third')
plt.plot(step, polynomials[3,:], label='fourth')
plt.plot(step, polynomials[4,:], label='fifth')
# plt.title('Legrande Polynomials by using Gram-Schmidt')
plt.xlabel('x value')
plt.ylabel('y value')
plt.legend()
# plt.show()

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
# plt.subplot(4,1,2)
plt.plot(step, eToThePowMinusT, label='f(t)=e**(-t)')
plt.plot(step, polynomials[0,:]*c0, label='first approximation')
plt.plot(step, polynomials[1,:]*c1, label='second approximation')
plt.plot(step, polynomials[2,:]*c2, label='third approximation')
plt.plot(step, polynomials[3,:]*c3, label='fourth approximation')
plt.plot(step, polynomials[4,:]*c4, label='fifth approximation')
# plt.title('f(t)=np.exp(-t) approximation')
plt.xlabel('x value')
plt.ylabel('y value')
plt.legend()
# plt.show()

# Calculate the error norm
approxError = [(f_n(x)-c0*y_0(x)-c1*y_1(x)-c2*y_2(x)-c3*y_3(x)-c4*y_4(x)) for x in step]
normE=0
for i in approxError:
    normE += i**2
normE = np.sqrt(normE)
print(normE)


# Part c
# Chebyshev Polynomials
ChebyShevPolys = np.zeros((5,10000), dtype=np.float64)
for i in range(5):
    for j in range(10000):
        if i==0:
            ChebyShevPolys[i,j] = 1
        if i==1:
            ChebyShevPolys[i,j] = step[j]
        if i==2:
            ChebyShevPolys[i,j] = 2*step[j]**2 - 1
        if i==3:
            ChebyShevPolys[i,j] = 4*step[j]**3 - 3*step[j]
        if i==4:
            ChebyShevPolys[i,j] = 8*step[j]**4 - 8*(step[j]**2) + 1

plt.figure('ChebyShev Polynomials')
# plt.subplot(4,1,3)
plt.plot(step, ChebyShevPolys[0,:], label='first')
plt.plot(step, ChebyShevPolys[1,:], label='second')
plt.plot(step, ChebyShevPolys[2,:], label='third')
plt.plot(step, ChebyShevPolys[3,:], label='fourth')
plt.plot(step, ChebyShevPolys[4,:], label='fifth')
# plt.title('ChebyShev Polynomials')
plt.xlabel('x value')
plt.ylabel('y value')
plt.legend()
# plt.show()

# Part D
# Calculating coefficients for Chebyshev Polynomials
def ChSvY0(x):
    return 1
def ChSvY1(x):
    return x
def ChSvY2(x):
    return 2*(x**2) - 1
def ChSvY3(x):
    return 4*(x**3) - 3*x
def ChSvY4(x):
    return 8*(x**4) - 8*(x**2) + 1

ChSvc0 = quad(lambda x: ChSvY0(x)*f_n(x), -1, 1)[0]/quad(lambda x: ChSvY0(x)*ChSvY0(x), -1, 1)[0]
ChSvc1 = quad(lambda x: ChSvY1(x)*f_n(x), -1, 1)[0]/quad(lambda x: ChSvY1(x)*ChSvY1(x), -1, 1)[0]
ChSvc2 = quad(lambda x: ChSvY2(x)*f_n(x), -1, 1)[0]/quad(lambda x: ChSvY2(x)*ChSvY2(x), -1, 1)[0]
ChSvc3 = quad(lambda x: ChSvY3(x)*f_n(x), -1, 1)[0]/quad(lambda x: ChSvY3(x)*ChSvY3(x), -1, 1)[0]
ChSvc4 = quad(lambda x: ChSvY4(x)*f_n(x), -1, 1)[0]/quad(lambda x: ChSvY4(x)*ChSvY4(x), -1, 1)[0]
coeffChebyShev = [ChSvc0, ChSvc1, ChSvc2, ChSvc3, ChSvc4]

plt.figure('f(t)=np.exp(-t) approximation using Chebyshev equations')
# plt.subplot(4,1,4)
plt.plot(step, eToThePowMinusT, label='f(t)=e**(-t)')
plt.plot(step, ChebyShevPolys[0,:]*ChSvc0, label='first approximation')
plt.plot(step, ChebyShevPolys[1,:]*ChSvc1, label='second approximation')
plt.plot(step, ChebyShevPolys[2,:]*ChSvc2, label='third approximation')
plt.plot(step, ChebyShevPolys[3,:]*ChSvc3, label='fourth approximation')
plt.plot(step, ChebyShevPolys[4,:]*ChSvc4, label='fifth approximation')
plt.xlabel('x value')
plt.ylabel('y value')
# plt.title('f(t)=np.exp(-t) approximation using Chebyshev equations')
plt.legend()
# plt.tight_layout()
plt.show()

# Calculate the error norm Chebyshev method
ChSVApproxError = [(f_n(x)-ChSvc0*ChSvY0(x)-ChSvc1*ChSvY1(x)-ChSvc2*ChSvY2(x)-ChSvc3*ChSvY3(x)-ChSvc4*ChSvY4(x)) for x in step]
normE=0
for i in ChSVApproxError:
    normE += i**2
normE = np.sqrt(normE)
print(normE)

# Part E
# Error norm of ChebyShev equations seems to be low compared to Legrande Polynomials
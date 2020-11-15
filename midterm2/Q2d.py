import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class recursive_least_square:
    def __init__(self):
        self.P = np.diag(np.ones(3)*10**5)
        self.currCoeffT = np.array([[0., 0., 0.]])
        self.kalmanGain = np.array([[0,0,0]])
        self.QTranspose = np.array([[0,0,0]])
        self.Ts = 0.01
        self.steps = 100
        self.y0 = [0,0]
        self.t = [i*self.Ts for i in range(1,self.steps)]
        self.solution = odeint(self.system, self.y0, self.t)  
        self.YDoubleDot = np.zeros((self.steps))
        self.input = np.zeros((self.steps))
        self.Allcoeff = np.array([[0,0,0]])
        self.batchQ = np.array([[0,0,0]])
        self.batchH = np.array([0,0,0])

        self.calculate_values()  
        self.update_gains()
        self.calculate_batch_least_squares()

    def system(self, x, t):
        f, f_dot = x
        s = np.sin
        dydt = [f_dot, -f_dot-2*f + 3*(s(0.01*t) + s(0.1*t) + s(t) + s(10*t))]
        return dydt  

    def calculate_YDoubleDot(self,step):
        if step >=1:
            self.YDoubleDot[step] = -self.solution[step-1,1] -2*self.solution[step-1,0] + 3*self.input[step]

    def calculate_input(self, step):
        t = self.Ts*step
        u = np.sin(0.01*t) + np.sin(0.1*t) + np.sin(10*t)
        self.input[step] = u

    def calculate_QTranspose(self,step):
        if step >=1:
            i = -self.solution[step-1,1]
            j = -self.solution[step-1,0]
            k = self.input[step]
            self.QTranspose =  np.array([[ i, j, k ]])
            self.batchQ = np.append(self.batchQ, self.QTranspose, axis=0)

    def calculate_kalman_gain(self, step):        
        QT = self.QTranspose
        Q = QT.T 
        k= self.P @ Q
        scalar = Q.T @ self.P @ Q
        k = k/(1+scalar[0,0])
        self.kalmanGain = k.T

    def update_P(self):
        k = self.kalmanGain
        self.P = self.P - k.T @ self.QTranspose @ self.P

    def calculate_H(self, step):
        h = self.currCoeffT
        expectedY = self.QTranspose @ h.T 
        expectedY = expectedY[0,0]
        realY = self.YDoubleDot[step]
        nextCoeff = h + self.kalmanGain * (realY - expectedY)
        self.Allcoeff = np.append(self.Allcoeff, h, axis=0)
        self.currCoeffT = nextCoeff
    
    def calculate_batch_least_squares(self):
        A = self.batchQ
        y = np.array([self.YDoubleDot]).T
        h = np.linalg.inv( A.T @ A) @ A.T @ y
        h= h.T 
        self.batchH = h[0]

    def calculate_values(self):
        for i in range(self.steps):
            self.calculate_input(i)
            self.calculate_YDoubleDot(i)

    def update_gains(self):
        for i in range(self.steps):
            self.calculate_QTranspose(i)
            self.calculate_kalman_gain(i)
            self.update_P()
            self.calculate_H(i)


rls = recursive_least_square()

# print(rls.Y)
# print(rls.yDot)
# print(rls.YDoubleDot)

print(rls.Allcoeff)
print(rls.batchH)
# print(rls.input)
# print(rls.P)

    
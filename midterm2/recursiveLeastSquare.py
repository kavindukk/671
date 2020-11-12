import numpy as np

#Assume that this calculation is done using last 5 inputs
# Then Rt and Pt would be 5x5 diaganol matrices.

class recursive_least_square:
    def __init__(self):        
        self.P = np.diag(np.ones(5)*10**5)
        self.H = np.array([[0,0,0,0,0]]).T 
        self.QH = np.array([0,0,0,0,0])
        self.kalmanGain = np.zeros((5)).reshape(5,1)
        self.realGains = np.array([1,2,3]) # a0=1, a1=2, b0=3
        self.Y = np.array([0])
        self.U = np.array([0])
        self.Ts = 0.01
        self.timeStep = 1

    def calc_input(self):
        t = self.Ts * self.timeStep
        s = np.sin
        u = s(0.01*t) + s(0.1*t) + s(t) + s(10*t)
        np.append(self.U, u)

    def update_QH(self):
        u = self.U[self.timeStep]
        self.QH = self.QH[:-1]
        self.QH = np.insert(self.QH, 0, u, axis = 0)

    def calculate_kalman_gain(self):
        Q = np.array([self.QH]).T
        k= self.P @ Q
        scalar = Q.T @ self.P @ Q
        k = k/(1+scalar[0,0])
        self.kalmanGain = k

    def update_P(self):
        QH = np.array([self.QH])
        adjustment = self.kalmanGain @ QH @ self.P
        self.P = self.P - adjustment

    def calculate_Y(self):
        a0 = self.realGains[0]
        a1 = self.realGains[1]
        b0 = self.realGains[2]
        t = self.Ts
        if self.timeStep==1:
            y = b0*self.U[1]
            y = y / (a1/t  +  a0 - 1/t**2)
        if self.timeStep == 2:
            y =b0*self.U[2] - self.Y[1]*( a1/t - 2/t**2)
            y = y / (a1/t  +  a0 - 1/t**2)
        if self.timeStep >= 3:
            i = self.timeStep
            y =b0*self.U[i] - self.Y[i-1]*( a1/t - 2/t**2) -self.Y[i-2]/t**2
            y = y / (a1/t  +  a0 - 1/t**2)
        np.append(self.Y, y)
        self.timeStep = self.timeStep + 1


        


    def update_H(self):
        adjustment = self.kalmanGain 
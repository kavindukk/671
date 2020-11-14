import numpy as np

class recursive_least_square:
    def __init__(self):
        self.P = np.diag(np.ones(3)*10**5)
        self.currCoeffT = np.array([0., 0., 0.])
        self.Ts = 0.01
        self.steps = 100
        self.Y = np.zeros((self.steps))
        self.yDot = np.zeros((self.steps))
        self.YDoubleDot = np.zeros((self.steps))
        self.input = np.zeros((self.steps))
        self.Allcoeff = np.array([[0,0,0]])

        self.calculate_values()       

    def calculate_Y(self, step):
        if step >= 1:
            t = step*self.Ts
            f1 = -0.05143*np.exp(-2*t) + 0.13763*np.exp(t)
            f2 = 1.5*np.sin(0.01*t) - 0.0075*np.cos(0.01*t) \
                +1.5037*np.sin(0.1*t) - 0.0756*np.cos(0.1*t) \
                -0.0303*np.sin(10*t) - 0.0031*np.cos(10*t)
            self.Y[step] = f1 + f2

    def calculate_yDot(self, step):
        if step >= 1:
            self.yDot[step] = ( self.Y[step] - self.Y[step-1] )/self.Ts
        
    def calculate_YDoubleDot(self,step):
        if step >=1:
            self.YDoubleDot[step] = ( self.yDot[step] - self.yDot[step-1] )/self.Ts

    def calculate_input(self, step):
        t = self.Ts*step
        u = np.sin(0.01*t) + np.sin(0.1*t) + np.sin(10*t)
        self.input[step] = u

    def calculate_QTranspose(self,step):
        i = -self.yDot[step]
        j = -self.Y[step]
        k = self.input[step]
        return np.array([[ i, j, k ]])

    # def calculate_kalman_gain(self):
    #     Q = np.array([self.QH]).T
    #     k= self.P @ Q
    #     scalar = Q.T @ self.P @ Q
    #     k = k/(1+scalar[0,0])
    #     self.kalmanGain = k

    def calculate_values(self):
        for i in range(self.steps):
            self.calculate_input(i)
            self.calculate_Y(i)
            self.calculate_yDot(i)
            self.calculate_YDoubleDot(i)


    
import numpy as np

A = np.array([[1,-2,3], [-4,5,-6], [7,-8,9]])

class LU_calculation:
    def __init__(self, A):
        self.A = A
        self.sizeOfA = A.shape[0]
        self.P = np.eye(self.sizeOfA)
        self.E = np.eye(self.sizeOfA)
        self.L = np.eye(self.sizeOfA)

    def calculate_LU(self):
        pList = []
        eList = []
        for j in range(self.A.shape[0]-1):    
            index = np.argmax(np.absolute(self.A[j:,j])) + j
            if(self.A[j,j] != self.A[index,j]):
                self.P[[j,index]] = self.P[[index,j]]

            self.A = self.P @ self.A
            pList.append(self.P)
            self.P = np.eye(self.sizeOfA)

            for i in range(j+1, self.A.shape[0]):
                self.E[i,j] = -self.A[i,j]/self.A[j,j]

            self.A = self.E @ self.A
            eList.append(self.E)
            self.E = np.eye(self.sizeOfA)

        self.L = np.eye(self.sizeOfA)
        for i in range(len(eList)):
            for j in range(eList[i].shape[0]):
                for k in range(eList[i].shape[0]):
                    if j != k and eList[i][j,k] != 0:
                        eList[i][j,k] = -eList[i][j,k] 
        
        for i in range(len(eList)):
            self.L = self.L @ pList[i] @ eList[i]

        pList.reverse()
        for i in range(len(pList)):
            self.P = self.P @ pList[i]
        self.L = self.P @ self.L

        return self.L, self.A, self.P

mylu = LU_calculation(A)
L, U, P = mylu.calculate_LU()

# P = np.eye(3)
# E = np.eye(3)
# pList = []
# eList = []
# for j in range(A.shape[0]-1):    
#     index = np.argmax(np.absolute(A[j:,j])) + j
#     if(A[j,j] != A[index,j]):
#         P[[j,index]] = P[[index,j]]

#     A = P @ A
#     pList.append(P)
#     P = np.eye(3)

#     for i in range(j+1, A.shape[0]):
#             E[i,j] = -A[i,j]/A[j,j]

#     A = E @ A
#     eList.append(E)
#     E = np.eye(3)

# L = np.eye(3)
# for i in range(len(eList)):
#     for j in range(eList[i].shape[0]):
#         for k in range(eList[i].shape[0]):
#             if j != k and eList[i][j,k] != 0:
#                 eList[i][j,k] = -eList[i][j,k]  

# for i in range(len(eList)):
#     L = L @ pList[i] @ eList[i]

# pList.reverse()
# for i in range(len(pList)):
#     P = P @ pList[i]
# L = P @ L
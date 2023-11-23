import numpy as np
import torch

class Domain():
    def fc(self, l, u, W, b):
        pass 

    def relu(self, l, u):
        pass 

    def tanh(self, l, u):
        pass

    def fc_deriv(self, L, U, W, b):
        pass 

    def relu_deriv(self, l, u, L, U):
        pass 

    def tanh_deriv(self, l, u, L, U):
        pass

class Interval(Domain):
    def fc(self, l, u, W, b):
        S = (W >= 0) * 1
        W1 = S * W    
        W2 = (1 - S) * W

        b = b.reshape(-1, 1)

        l_new = W1 @ l + W2 @ u + b
        u_new = W2 @ l + W1 @ u + b

        return l_new, u_new

    def mult(self, L_1, U_1, L_2, U_2):
        return torch.min(torch.min(L_1*L_2, L_1*U_2), torch.min(U_1*L_2, U_1*U_2)), torch.max(torch.max(L_1*L_2, L_1*U_2), torch.max(U_1*L_2, U_1*U_2))
    
    def add(self, L_1, U_1, L_2, U_2):
        return L_1 + L_2, U_1 + U_2

    def fc(self, l, u, L, U, L2, U2, W, b):
        S = (W >= 0) * 1
        W1 = S * W    
        W2 = (1 - S) * W
        b = b.reshape(-1, 1)

        l_new = W1 @ l + W2 @ u + b
        u_new = W2 @ l + W1 @ u + b

        L_new = W1 @ L + W2 @ U
        U_new = W2 @ L + W1 @ U

        L2_new = W1 @ L2 + W2 @ U2
        U2_new = W2 @ L2 + W1 @ U2

        return l_new, u_new, L_new, U_new, L2_new, U2_new 
    
    def relu(self, l, u, L, U, L2, U2):
        l_new = torch.relu(l)
        u_new = torch.relu(u)

        L_temp = (l > 0) * 1
        U_temp = 1 - ((u < 0) * 1)
        L_new, U_new = self.mult(L,U, L_temp, U_temp)

        L2_new, U2_new = self.mult(W, W, L2, U2)
        return l_new, u_new, L_new, U_new, L2_new, U2_new 

    def tanh(self, l, u, L, U, L2, U2):
        l_new = torch.tanh(l)
        u_new = torch.tanh(u)

        deriv_l = 1 - torch.square(torch.tanh(l)) 
        deriv_u = 1 - torch.square(torch.tanh(u))

        L_temp = torch.min(deriv_l, deriv_u) 
        U_temp = torch.where((l<0)&(u>0), torch.ones(l.size()), torch.max(deriv_l, deriv_u))

        L_new, U_new = self.mult(L_temp, U_temp, L, U)

        temp1 = 2*l_new*l_new*l_new - 2*l_new 
        temp2 = 2*u_new*u_new*u_new - 2*u_new 
        
        L2_temp = torch.where((u > 0.65) & (l < 0.66), -0.77 * torch.ones(l.size()), torch.min(temp1, temp2))
        U2_temp = torch.where((u > -0.66) & (l < -0.65), 0.77 * torch.ones(u.size()), torch.max(temp1, temp2)) 

        L_sqr = torch.max(torch.min(L*L, U*U),torch.zeros(L.size()))
        U_sqr = torch.max(torch.max(L*L, U*U),torch.zeros(L.size()))

        L2_new, U2_new = self.add(*self.mult(L_temp, U_temp, L2, U2), *self.mult(L2_temp, U2_temp, L_sqr, U_sqr))

        return l_new, u_new, L_new, U_new, L2_new, U2_new 
import numpy as np
import torch

class Domain():
    def fc(l, u, W, b):
        pass 

    def relu(l, u):
        pass 

    def fc_deriv(L, U, W):
        pass 

    def relu_deriv(l, u, L, U):
        pass 

class Interval(Domain):
    def fc(l, u, W, b):
        S = (W >= 0) * 1
        W1 = S * W    
        W2 = (1 - S) * W

        b = b.reshape(-1, 1)

        l_new = W1 @ l + W2 @ u + b
        u_new = W2 @ l + W1 @ u + b

        return l_new, u_new

    def relu(l, u):
        l_new = torch.zeros(l.shape)
        u_new = u.clone()

        indices = (l>0)
        l_new[indices] = l[indices]

        indices = (u<0)
        u_new[indices] = 0

        return l_new, u_new
    
    def fc_deriv(L, U, W):
        S = (W >= 0) * 1
        W1 = S * W    
        W2 = (1 - S) * W

        L_new = W1 @ L + W2 @ U
        U_new = W2 @ L + W1 @ U

        return L_new, U_new
    
    def relu_deriv(l, u, L, U):
        L_relu = torch.zeros((l.shape))
        U_relu = torch.ones((l.shape))

        indices = (l>0)
        L_relu[indices] = 1

        indices = (u<0)
        U_relu[indices] = 0

        L_new = L * L_relu.view(-1,1)
        U_new = U * U_relu.view(-1,1)
        return L_new, U_new
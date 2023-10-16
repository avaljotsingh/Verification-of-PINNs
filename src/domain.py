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

    def relu(self, l, u):
        # l_new = torch.zeros(l.shape)
        # u_new = u.clone()

        # indices = (l>0)
        # l_new[indices] = l[indices]

        # indices = (u<0)
        # u_new[indices] = 0

        l_new = torch.relu(l)
        u_new = torch.relu(u)
        return l_new, u_new
    
    def tanh(self, l, u):
        # l_new = torch.zeros(l.shape)
        # u_new = u.clone()

        # indices = (l>0)
        # l_new[indices] = l[indices]

        # indices = (u<0)
        # u_new[indices] = 0

        l_new = torch.tanh(l)
        u_new = torch.tanh(u)
        return l_new, u_new
    
    def fc_deriv(self, L, U, W, b):
        S = (W >= 0) * 1
        W1 = S * W    
        W2 = (1 - S) * W

        L_new = W1 @ L + W2 @ U
        U_new = W2 @ L + W1 @ U

        return L_new, U_new
    
    def relu_deriv(self, l, u, L, U):
        # L_relu and U_relu are derivatives of just the ReLU layer.
        # The lower bounds are zero by default but 1 for the cases where the lower bound of input
        # is more than equal to 0.
        L_relu = (l >= 0) * 1

        # The upper bounds are 1 by default but 0 for the cases where the upper bound of input
        # is less than equal to 0.
        U_relu = 1 - ((u <= 0) * 1)

        ## TODO: Check if we need to the handle the l (or u) equal to 0 case differently as ReLU is not differentiable there.

        L_new_1 = L * L_relu.view(-1,1)
        L_new_2 = L * U_relu.view(-1,1)
        L_new = torch.min(L_new_1, L_new_2)

        U_new_1 = U * L_relu.view(-1,1)
        U_new_2 = U * U_relu.view(-1,1)
        U_new = torch.max(U_new_1, U_new_2)

        return L_new, U_new
    
    def tanh_deriv(self, l, u, L, U):
        deriv_l = 1 - torch.square(torch.tanh(l)) 
        deriv_u = 1 - torch.square(torch.tanh(u))

        L_tanh = torch.min(deriv_l, deriv_u) 
        U_tanh = torch.max(deriv_l, deriv_u) 

        indices = (l<0) and (u>0)
        U_tanh[indices] = 1

        L_new_1 = L * L_tanh.view(-1,1)
        L_new_2 = L * U_tanh.view(-1,1)
        L_new = torch.min(L_new_1, L_new_2)

        U_new_1 = U * L_tanh.view(-1,1)
        U_new_2 = U * U_tanh.view(-1,1)
        U_new = torch.max(U_new_1, U_new_2)

        return L_new, U_new
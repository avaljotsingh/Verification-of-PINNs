from model import MyModel  
from domain import Domain, Interval
import sys 
import torch
import matplotlib.pyplot as plt
import numpy as np
import time

    
def u0(x):
    u = -torch.sin(torch.pi * x)
    return u

class IntervalDomainHandler():
    def add(self, i1, i2):
        return (i1[0]+i2[0], i1[1]+i2[1])
    
    def subtract(self, i1, i2):
        return (i1[0]-i2[1], i1[1]-i2[0])
    
    def mul(self, i1, i2):
        op1 = i1[0] * i2[0]
        op2 = i1[0] * i2[1]
        op3 = i1[1] * i2[0]
        op4 = i1[1] * i2[1]
        
        return (torch.min(op1, torch.min(op2, torch.min(op3, op4))), torch.max(op1, torch.max(op2, torch.max(op3, op4))))


def residual_loss_per_partition(model, mu, interval_handler, inpx_range, inpt_range, device):
    lows = torch.tensor([inpx_range[0], inpt_range[0]], device=device).reshape(-1,1)
    highs = torch.tensor([inpx_range[1], inpt_range[1]], device=device).reshape(-1,1)
    val_bounds, deriv_bounds, deriv2_bounds = model.forward_interval(lows, highs, device)

    ul = val_bounds[0][0][0]
    ur = val_bounds[1][0][0]
    
    uxl = deriv_bounds[0][0][0]
    utl = deriv_bounds[0][0][1]
    
    uxr = deriv_bounds[1][0][0]
    utr = deriv_bounds[1][0][1]
    
    uxxl = deriv2_bounds[0][0][0]
    uxxr = deriv2_bounds[1][0][0]
    
    # ut + u*ux - mu*uxx
    residual_bounds = interval_handler.add((utl, utr), interval_handler.mul((ul, ur), (uxl, uxr)))
    residual_bounds = interval_handler.subtract(residual_bounds, (mu*uxxl, mu*uxxr))

    return max(abs(residual_bounds[0]), abs(residual_bounds[1]))

def certif_residual_loss(model, residual_points, mu, eps):
    interval_handler = IntervalDomainHandler()
    res_loss = 0
    for i in range((residual_points.shape[0])):
        point = residual_points[i]
        x = (point[0].item() - eps/2, point[0].item() + eps/2)
        t = (point[1].item() - eps/2, point[1].item() + eps/2)
        res_loss += residual_loss_per_partition(model, mu, interval_handler, x, t, device)
    return res_loss / residual_points.shape[0]


def loss_function(model, bcps, icps, residual_points, mu, eps, device): 
        
    # Boundary conditions loss
    bcp_xs = bcps[:,0].reshape(-1, 1)
    bcp_ts = bcps[:,1].reshape(-1, 1)
    boundary_values = model(torch.cat([bcp_xs, bcp_ts], dim=1))
    boundary_loss = torch.mean(torch.square(boundary_values))
    
    # Initial condition loss
    icp_xs = icps[:,0].reshape(-1, 1)
    icp_ts = icps[:,1].reshape(-1, 1)
    initial_values = model(torch.cat([icp_xs, icp_ts], dim=1))
    initial_ground_truth = u0(icp_xs)
    initial_loss = torch.nn.MSELoss()(initial_values, initial_ground_truth)
    
    # Residual loss using derivatives (the interesting part)
    # res_loss = certif_residual_loss(model, residual_points, mu, eps)
    res_loss = 0
    loss_value = boundary_loss + initial_loss + res_loss
    print(boundary_loss, initial_loss, res_loss)
    
    return loss_value


if __name__=='__main__':

    x_min = -1.0
    x_max = 1.0
    t_min = 0.0
    t_max = 1.0
    mu = .01 / np.pi
    eps = 0.00001

    device = 'cpu'

    boundary_condition_points = [torch.tensor([x_min, t]) for t in torch.linspace(t_min, t_max, 500)]
    boundary_condition_points.extend([torch.tensor([x_max, t]) for t in torch.linspace(t_min, t_max, 500)])
    boundary_condition_points = torch.stack(boundary_condition_points, dim = 0)

    initial_condition_points = [torch.tensor([x, t_min]) for x in torch.linspace(x_min, x_max, 500)]
    initial_condition_points = torch.stack(initial_condition_points, dim = 0)

    residual_points = [torch.tensor([x,t]) for x in torch.linspace(x_min + 0.001, x_max - 0.001, 100) for t in torch.linspace(t_min + 0.001, t_max - 0.001, 100)]
    residual_points = torch.stack(residual_points, dim = 0)
    residual_points.requires_grad = True 

    
    # 2. set the model
    torch.manual_seed(23939)
    model = MyModel(2, [15, 15, 15], ['tanh', 'tanh', 'tanh'], 1)

    # 3. set the optimizer
    lr = 0.01
    opt = torch.optim.Adam(model.parameters(), lr)
    n_epochs = 100

    loss_history = []
    start = time.time()
    for i in range(n_epochs):
        opt.zero_grad()
        loss = loss_function(model, boundary_condition_points, initial_condition_points, residual_points, mu, eps, device)
        loss_history.append(loss.item())
        
        loss.backward()
        opt.step()

        if i % 1 == 0:
            print(f'epoch {i}, loss = {loss}')

    final_loss = loss_function(model, boundary_condition_points, initial_condition_points, residual_points, mu, eps, device)
    print('final_loss = ', final_loss)

    torch.save(model, "../trained_models/pinn-burgers_certif_sampling_test.pt")
    print(time.time()-start)
    plt.plot(loss_history)
    plt.title("Loss progression")
    plt.xlabel('optimization step')
    plt.ylabel('loss')
    plt.show()


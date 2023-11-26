from model import MyModel  
from domain import Domain, Interval
import sys 
import torch
import matplotlib.pyplot as plt
import numpy as np
import time

    
def u0(x):
    """initial condition"""
    u = -torch.sin(torch.pi * x)
    return u


def residual_loss(model, residual_points, mu):
    residual_pts_xs = residual_points[:,0].reshape(-1, 1)
    residual_pts_ts = residual_points[:,1].reshape(-1, 1)
    
    output_on_residual_points = model(torch.cat((residual_pts_xs, residual_pts_ts), dim=1))
    

    
    grad_t = torch.autograd.grad(outputs=output_on_residual_points,
                                  inputs=residual_pts_ts, 
                                  grad_outputs=torch.ones_like(residual_pts_ts, requires_grad=False), 
                                  retain_graph=True, 
                                  create_graph=True, allow_unused=True)[0]
    
    grad_x = torch.autograd.grad(outputs=output_on_residual_points,
                              inputs=residual_pts_xs,
                              grad_outputs=torch.ones_like(residual_pts_xs, requires_grad=False),
                              retain_graph=True,
                              create_graph=True, allow_unused=True)[0]

    grad_xx = torch.autograd.grad(outputs = grad_x, 
                                  inputs = residual_pts_xs, 
                                  grad_outputs=torch.ones_like(residual_pts_xs, requires_grad=False),
                                  retain_graph=True, 
                                  create_graph=True, allow_unused=True)[0]
    
    lhs = grad_t + output_on_residual_points * grad_x
    rhs = mu * grad_xx

    return torch.nn.MSELoss()(lhs, rhs)

def pgd_residual_loss(model, residual_points, num_pgd_iterations, mu, eps, alpha):
    ori_data = residual_points.data
        
    for i in range(num_pgd_iterations) :    
        residual_points.requires_grad = True
        loss = residual_loss(model, residual_points, mu)
        loss.backward()
        model.zero_grad()
        adv_images = residual_points + alpha*residual_points.grad.sign()
        eta = torch.clamp(adv_images - ori_data, min=-eps, max=eps)
        residual_points = (ori_data + eta).detach_()
    return residual_points 



def loss_function(model, bcps, icps, residual_points, num_pgd_iterations, mu, eps, alpha): 
    
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
    residual_points = pgd_residual_loss(model, residual_points, num_pgd_iterations, mu, eps, alpha)
    residual_points.requires_grad = True
    res_loss = residual_loss(model, residual_points, mu)
    loss_value = boundary_loss + initial_loss + res_loss
    
    return loss_value


if __name__=='__main__':

    x_min = -1.0
    x_max = 1.0
    t_min = 0.0
    t_max = 1.0
    mu = .01 / np.pi
    alpha = 0.01
    eps = 0.01
    num_pgd_iterations = 100

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
    n_epochs = 1000

    loss_history = []
    start = time.time()
    for i in range(n_epochs):
        opt.zero_grad()
        loss = loss_function(model, boundary_condition_points, initial_condition_points, residual_points, num_pgd_iterations, mu, eps, alpha)
        loss_history.append(loss.item())
        
        loss.backward()
        opt.step()

        if i % 1 == 0:
            print(f'epoch {i}, loss = {loss}')

    final_loss = loss_function(model, boundary_condition_points, initial_condition_points, residual_points, num_pgd_iterations, mu, eps, alpha)
    print('final_loss = ', final_loss)

    torch.save(model, "../trained_models/pinn-burgers_pgd_train_1.pt")
    print(time.time()-start)
    plt.plot(loss_history)
    plt.title("Loss progression")
    plt.xlabel('optimization step')
    plt.ylabel('loss')
    plt.show()


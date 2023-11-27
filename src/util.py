import matplotlib.pyplot as plt
import numpy as np
import torch
from interpret import Abs_interpreter
from domain import Interval

## Visualize residual errors by sampling points and finding the true residual error at that point!
def visualize_residual_error(model_name, inpx_range=(-1.0, 1.0), inpt_range=(0.0, 1.0)):
    model = torch.load(model_name)
    model_name_base = model_name.split("/")[-1]
    mu = .01 / np.pi
    plt.figure(figsize = (10, 5))
    n = 2000
    xs, ts = torch.meshgrid(torch.linspace(inpx_range[0], inpx_range[1], n), 
                            torch.linspace(inpt_range[0], inpt_range[1], n))
    xs = xs.reshape(-1, 1).requires_grad_(True)
    ts = ts.reshape(-1, 1).requires_grad_(True)
    f = model(torch.cat([xs, ts], dim=1))
    grad_t = torch.autograd.grad(outputs=f,
                                  inputs=ts, 
                                  grad_outputs=torch.ones_like(ts, requires_grad=False), 
                                  retain_graph=True, 
                                  create_graph=True)[0]
    
    grad_x = torch.autograd.grad(outputs=f,
                                inputs=xs,
                                grad_outputs=torch.ones_like(xs, requires_grad=False),
                                retain_graph=True,
                                create_graph=True)[0]
    
    grad_xx = torch.autograd.grad(outputs = grad_x.sum(), 
                              inputs = xs, 
                              retain_graph=True, 
                              create_graph=True)[0]
    
    
    residual_err = (grad_t + f * grad_x - mu * grad_xx).reshape(n,n)
    
    print("Min:", residual_err.min())
    print("Max:", residual_err.max())
    print("Mean:", residual_err.mean())
    
    heatmap = plt.imshow(residual_err.detach().numpy(), extent=[inpt_range[0],inpt_range[1],inpx_range[0],inpx_range[1]], cmap='seismic', interpolation='nearest', aspect='auto')
    plt.colorbar(heatmap)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$x$')
    plt.title(r'$u_t + u * u_x - \mu*u_{xx}$ for ' + model_name_base)
    plt.show()
    

# Class to handle interval domain operations
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
        
        return (min(op1, op2, op3, op4), max(op1, op2, op3, op4))

mu = .01 / np.pi

def compute_pinn_burgers_residual_bounds(model_name, 
                                         inpx_range = (-1 + 0.001, 1 - 0.001), 
                                         inpt_range = (0 + 0.001, 1 - 0.001), 
                                         abstract_domain = 'Interval'):
    model = torch.load(model_name)
    
    if abstract_domain == 'Interval':
        our_domain = Interval()
    else:
        raise NotImplementedError("Unknown abstract domain: " + our_domain)
        
    abs_interpreter = Abs_interpreter(model, our_domain)
    lows = torch.tensor([inpx_range[0], inpt_range[0]]).reshape(-1,1)
    highs = torch.tensor([inpx_range[1], inpt_range[1]]).reshape(-1,1)
    val_bounds, deriv_bounds, deriv2_bounds = abs_interpreter.forward_pass(lows, highs)

    ul = val_bounds[0][0][0].item()
    ur = val_bounds[1][0][0].item()
    
    uxl = deriv_bounds[0][0][0].item()
    utl = deriv_bounds[0][0][1].item()
    
    uxr = deriv_bounds[1][0][0].item()
    utr = deriv_bounds[1][0][1].item()
    
    uxxl = deriv2_bounds[0][0][0].item()
    uxxr = deriv2_bounds[1][0][0].item()
        
    interval_handler = IntervalDomainHandler()
    
    # ut + u*ux - mu*uxx
    residual_bounds = interval_handler.add((utl, utr), interval_handler.mul((ul, ur), (uxl, uxr)))
    residual_bounds = interval_handler.subtract(residual_bounds, (mu*uxxl, mu*uxxr))
    
    return residual_bounds

def compute_pinn_bounds_using_input_splitting(model_name, 
                                              num_partitions, 
                                              inpx_range = (-1 + 0.001, 1 - 0.001), 
                                              inpt_range = (0 + 0.001, 1 - 0.001),
                                              abstract_domain1 = 'Interval'):
    xeps = (inpx_range[1] - inpx_range[0])/num_partitions
    teps = (inpt_range[1] - inpt_range[0])/num_partitions
    model_name_base = model_name.split("/")[-1]

    l_final = None
    u_final = None

    for i in range(num_partitions):
        for j in range(num_partitions):
            bounds = compute_pinn_burgers_residual_bounds(model_name, 
                                                          inpx_range=(inpx_range[0] + xeps*i, inpx_range[0] + xeps*(i+1)),
                                                          inpt_range=(inpt_range[0] + teps*j, inpt_range[0] + teps*(j+1)),
                                                          abstract_domain = abstract_domain1)

            if l_final is None: l_final = bounds[0]
            if u_final is None: u_final = bounds[1]

            l_final = min(l_final, bounds[0])
            u_final = max(u_final, bounds[1])
    
    print("Bounds for {} after {} partitions: ({}, {})".format(model_name_base, num_partitions, l_final, u_final))
    return l_final, u_final


def plot_actual_model(model_name):
    model = torch.load(model_name)
    plt.figure(figsize = (10, 5))
    n = 1000
    xs, ts = torch.meshgrid(torch.linspace(-1.0, 1, n), torch.linspace(0.0, 1.0, n))
    xs = xs.reshape(-1, 1).requires_grad_(False)
    ts = ts.reshape(-1, 1).requires_grad_(False)
    f = model(torch.cat([xs, ts], dim=1)).reshape((n ,n))
    heatmap = plt.imshow(f.detach().numpy(), extent=[0,1,-1,1], cmap='seismic', interpolation='nearest', aspect='auto')
    plt.colorbar(heatmap)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$x$')
    plt.title(r'$u(x,t)$')
    plt.show()
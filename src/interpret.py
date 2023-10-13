from model import MyModel  
from domain import Interval 
import sys 
import torch

class Abs_interpreter():
    def __init__(self, model, domain):
        self.model = model 
        self.domain = domain 
    
    def forward_pass(self, low, high):
        l = torch.ones((self.model.input_size, 1))*low   
        u = torch.ones((self.model.input_size, 1))*high
        L = torch.eye(self.model.input_size)
        U = torch.eye(self.model.input_size)
        
        for layer in self.model.layers:
            if type(layer) == torch.nn.modules.linear.Linear:
                params = list(layer.parameters())
                W = params[0]
                b = params[1]
                l, u = self.domain.fc(l, u, W, b)
                L, U = self.domain.fc_deriv(L, U, W, b)
            elif type(layer) == torch.nn.modules.activation.ReLU:
                L, U = self.domain.relu_deriv(l, u, L, U) 
                l, u = self.domain.relu(l, u)
            else:
                raise NotImplementedError("Forward logic not implemented for layer type: ", type(layer))

        return (l, u), (L, U)

if __name__ == '__main__':
    file_name = sys.argv[1]
    model = torch.load(file_name)
    domain = Interval()
    abs_interpreter = Abs_interpreter(model, domain)
    val_bounds, deriv_bounds = abs_interpreter.forward_pass(0, 1)
    print("Val bounds:", val_bounds)
    print("Deriv. bounds:", deriv_bounds)

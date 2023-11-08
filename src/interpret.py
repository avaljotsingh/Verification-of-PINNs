from model import MyModel  
from domain import Domain, Interval
import sys 
import torch

class Abs_interpreter():
    def __init__(self, model: MyModel, domain: Domain):
        self.model = model 
        self.domain = domain 
    
    def forward_pass(self, low, high):
        l = low   
        u = high

        L = torch.eye(self.model.input_size)
        U = torch.eye(self.model.input_size)

        L2 = torch.zeros((self.model.input_size, self.model.input_size))
        U2 = torch.zeros((self.model.input_size, self.model.input_size))
        
        # for layer in self.model.layers:
        #     if type(layer) == torch.nn.modules.linear.Linear:
        #         params = list(layer.parameters())
        #         W = params[0]
        #         b = params[1]
        #         l, u = self.domain.fc(l, u, W, b)
        #         L, U = self.domain.fc_deriv(L, U, W, b)
        #     elif type(layer) == torch.nn.modules.activation.ReLU:
        #         L, U = self.domain.relu_deriv(l, u, L, U) 
        #         l, u = self.domain.relu(l, u)
        #     elif type(layer) == torch.nn.modules.activation.Tanh:
        #         L, U = self.domain.tanh_deriv(l, u, L, U)
        #         l, u = self.domain.tanh(l, u)
        #     else:
        #         raise NotImplementedError("Forward logic not implemented for layer type: ", type(layer))

        # return (l, u), (L, U)

        for layer in self.model.layers:
            if type(layer) == torch.nn.modules.linear.Linear:
                params = list(layer.parameters())
                W = params[0]
                b = params[1]
                l, u, L, U, L2, U2 = self.domain.fc(l, u, L, U, L2, U2, W, b)
            elif type(layer) == torch.nn.modules.activation.ReLU:
                l, u, L, U, L2, U2 = self.domain.relu(l, u, L, U, L2, U2)
            elif type(layer) == torch.nn.modules.activation.Tanh:
                l, u, L, U, L2, U2 = self.domain.tanh(l, u, L, U, L2, U2)
            else:
                raise NotImplementedError("Forward logic not implemented for layer type: ", type(layer))
            
        return (l, u), (L, U), (L2, U2)

if __name__ == '__main__':
    file_name = sys.argv[1]
    model = torch.load(file_name)
    domain = Interval()
    abs_interpreter = Abs_interpreter(model, domain)
    val_bounds, deriv_bounds = abs_interpreter.forward_pass(0, 1)
    print("Val bounds:", val_bounds)
    print("Deriv. bounds:", deriv_bounds)

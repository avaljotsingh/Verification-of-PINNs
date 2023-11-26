import numpy as np
import torch
from domain import Interval 

class MyModel(torch.nn.Module):
    def __init__(self, input_size, widths, activations, output_size):
        super(MyModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.domain = Interval()

        # Create hidden layers dynamically using a loop
        layers = []
        for i in range(len(widths)):
            if i == 0:
                # First hidden layer connected to the input
                layers.append(torch.nn.Linear(input_size, widths[i]))
            else:
                # Subsequent hidden layers connected to the previous hidden layer
                layers.append(torch.nn.Linear(widths[i-1], widths[i]))

            activation = torch.nn.ReLU()
            if activations[i] == 'tanh':
                activation = torch.nn.Tanh()
            elif activations[i] == 'elu':
                activation = torch.nn.ELU()
            elif activations[i] == 'sigmoid':
                activation = torch.nn.Sigmoid()

            layers.append(activation)

        # Output layer
        layers.append(torch.nn.Linear(widths[-1], output_size))

        # Combine all layers into a sequential module
        self.layers = torch.nn.Sequential(*layers)

    def forward_interval(self, low, high, device):
        l = low   
        u = high

        L = torch.eye(self.input_size, device=device)
        U = torch.eye(self.input_size, device=device)

        L2 = torch.zeros((self.input_size, self.input_size), device=device)
        U2 = torch.zeros((self.input_size, self.input_size), device=device)

        for layer in self.layers:
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
        
    def forward(self, x):
        return self.layers(x)
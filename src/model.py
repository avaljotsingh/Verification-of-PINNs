import numpy as np
import torch

class MyModel(torch.nn.Module):
    def __init__(self, input_size, widths, activations, output_size):
        super(MyModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

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
        
    def forward(self, x):
        return self.layers(x)
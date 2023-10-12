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
            elif type(layer) == torch.nn.modules.activations.ReLU:
                L, U = self.domain.relu_deriv(L, U, W, b) 
                l, u = self.domain.relu(l, u, W, b)
        return l, u, L, U

if __name__ == '__main__':
    file_name = sys.argv[1]
    print(file_name)
    input_size = 2
    widths = [8, 16, 8]
    activations = ['relu', 'relu', 'relu']
    output_size = 1
    model = MyModel(input_size, widths, activations, output_size)
    # model.load_state_dict(torch.load('model_file.pkl'))
    # domain = Interval()
    # abs_interpreter = Abs_interpreter(model, domain)
    # print(abs_interpreter.forward_pass(0, 1))
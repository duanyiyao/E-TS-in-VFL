import torch
from torchvision import datasets, transforms
from torch import nn, optim
import numpy as np

class SplitNN(torch.nn.Module):
    
    def __init__(self, models, data_owner, label_owner, server,device,model_locations):
        super().__init__()
        self.label_owner = label_owner
        self.data_owners = data_owner
        self.models = models
        self.server = server
        self.device = device
        self.optimizers = {location: optim.Adam(self.models[location].parameters(), lr=0.001) for location in model_locations}
      
    
    def activate_model(self, device):
        self.to(device)
        for owner, model in self.models.items():
            self.models[owner] = model.to(device)
        self.device = device
        
    def forward(self, data_pointer):
        
        embedding = {}
        remote_outputs = []
        #iterate over each client and pass thier inputs to respective model segment and move outputs to server
        for owner in self.data_owners:
            embedding[owner] = self.models[owner](data_pointer[owner].reshape([-1, 28*4]))
            embedding[owner] = embedding[owner].view(-1,64,7,1)
            remote_outputs.append(embedding[owner].requires_grad_())
        remote_outputs = torch.cat(remote_outputs, -1)
        server_output = self.models["server"](remote_outputs.view(-1, 64, 7, 7))
        server_output = server_output.view(-1, 512)
        # move to label_owner
        server_output = server_output.requires_grad_()
        pred = self.models["label_owner"](server_output)
        return pred   

    def forward_client(self, data_pointer):
        embedding = {}
        for owner in self.data_owners:
            embedding[owner] = self.models[owner](data_pointer[owner].reshape([-1, 28*4]))
            embedding[owner] = embedding[owner].view(-1,64,7,1)
        return embedding
    
    def forward_server(self, input):
        server_input = []
        for client in self.data_owners:
            server_input.append(input[client])
        server_inputs = torch.cat(server_input, -1)
        server_output = self.models["server"](server_inputs.view(-1, 64, 7, 7))
        server_output = server_output.view(-1, 512)
        server_output = server_output.requires_grad_()
        pred = self.models["label_owner"](server_output)
        return pred
        
    def zero_grads(self):
        for opt in self.optimizers.values():
            opt.zero_grad()
    def steps(self):
        for opt in self.optimizers.values():
            opt.step()
    def train(self):
        for model in self.models.values():
            model.train()
    def eval(self):
        for model in self.models.values():
            model.eval()
    

def models_generate(data_owners, input_size, hidden_size):
    models = {}

    for i in range(len(data_owners)):
        models[data_owners[i]] = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                                    )
    
    models["server"] =      nn.Sequential(
                            # 3
                            nn.Conv2d(64, 128, kernel_size=2, padding=1),
                            nn.BatchNorm2d(128),
                            nn.ReLU(True),
                            # 4
                            nn.Conv2d(128, 128, kernel_size=3, padding=1),
                            nn.BatchNorm2d(128),
                            nn.ReLU(True),
                            nn.MaxPool2d(kernel_size=2, stride=2),
                            # 5
                            nn.Conv2d(128, 256, kernel_size=3, padding=1),
                            nn.BatchNorm2d(256),
                            nn.ReLU(True),
                            # 6
                            nn.Conv2d(256, 256, kernel_size=3, padding=1),
                            nn.BatchNorm2d(256),
                            nn.ReLU(True),
                            # 7
                            nn.Conv2d(256, 256, kernel_size=3, padding=1),
                            nn.BatchNorm2d(256),
                            nn.ReLU(True),
                            nn.MaxPool2d(kernel_size=2, stride=2),
                            # 8
                            nn.Conv2d(256, 512, kernel_size=3, padding=1),
                            nn.BatchNorm2d(512),
                            nn.ReLU(True),
                            # 9
                            nn.Conv2d(512, 512, kernel_size=3, padding=1),
                            nn.BatchNorm2d(512),
                            nn.ReLU(True),
                            # 10
                            nn.Conv2d(512, 512, kernel_size=3, padding=1),
                            nn.BatchNorm2d(512),
                            nn.ReLU(True),
                            nn.MaxPool2d(kernel_size=2, stride=2),
                            )

    models["label_owner"] = nn.Sequential(
                            # 11
                            nn.Linear(512, 512),
                            nn.ReLU(True),
                            nn.Dropout(0.5),
                            # 12
                            nn.Linear(512, 512),
                            nn.ReLU(True),
                            nn.Dropout(0.5),
                            # 13
                            nn.Linear(512, 10),
                            )
    return models
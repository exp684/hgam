import numpy
import torch
import random
from torch.utils.data import Dataset
from torch import linalg as LA

class OPDataset(Dataset):
    # code copied from https://github.com/wouterkool/attention-learn-to-route/blob/master/problems/vrp/problem_vrp.py
    def __init__(self, size, num_samples, scores='Distance'):

        # Making the code device-agnostic
        if torch.cuda.is_available(): 
            torch.set_default_device('cuda') 
            device='cuda'
        else: 
            torch.set_default_device('cpu') 
            device='cpu'

        Uniform, Constant = False, False
        if scores == 'Uniform':
            Uniform = True # True, scores come from a discrete uniform distribution function. False, scores are constant=1.
        elif scores == 'Constant':
            Constant = True # All scores are equal to 1.
        else:
            Distance = True # Scores are proportional to the distance from depot.  

        MAXIMUM_LENGTH = {
            10: 2.0,
            21: 2.0,
            51: 3.0,
            101: 4.0
        }

        self.locations = []

        self.scores = []

        # Available time budget per path.
        self.Tmax = []
         
        # Number of teams/vehicles.
        self.m = []

        for sample in range(num_samples):
            self.m.append( torch.tensor(random.randint(2,4)).to(device) )            
            #self.Tmax.append( torch.tensor(random.randint(1,1)).to(device) )            
            self.Tmax.append( MAXIMUM_LENGTH[size]/self.m[sample] )
            self.locations.append( torch.FloatTensor(size, 2).uniform_(0, 1).to(device) )
            self.locations[sample][1] = self.locations[sample][0]   #Start point = finish point. To compare with Kool.
                        
            if Uniform: # Uniform probability distribution function. 
                self.scores.append( torch.round(torch.FloatTensor(size).uniform_(0,1), decimals=5).to(device) )                            
            elif Constant:   # Constant distribution. 
                self.scores.append( torch.ones(size).to(device) )
            else:   # Distance function.
                d = torch.tensor(  [LA.vector_norm(self.locations[sample][node] - self.locations[sample][0]) for node in range (size)])
                self.scores.append( ( 1 + (d / d.max()*99).to(torch.int32) )/100 )                
                        
            # Depot always has score = 0. 
            self.scores[sample][0] = 0           
            self.scores[sample][1] = 0
            
        self.num_samples = len(self.locations)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.locations[idx], self.scores[idx], self.Tmax[idx], self.m[idx]


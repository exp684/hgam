import time
import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import linalg as LA
from Decoder import Decoder
from Encoder import TransformerEncoder
import numpy as np
import split_byBatch as sp
from numba import jit

class AttentionModel(nn.Module):
    def __init__(self, embedding_dim, n_layers, n_head, dim_feedforward, C, dropout=0.1):
        super(AttentionModel, self).__init__()
        self.embedding_dim = embedding_dim  
        self.n_layers = n_layers    
        self.n_head = n_head    
        self.dim_feedforward = dim_feedforward  
        self.decode_mode = "sample"
        self.dropout = dropout
        self.C = C
        self.input_dim = 2

        self.scores_embedding = nn.Linear(1, self.embedding_dim)

        self.city_embedding = nn.Linear(self.input_dim, self.embedding_dim)

        self.encoder = TransformerEncoder(self.n_layers, self.n_head, self.embedding_dim,
                                          self.dim_feedforward, self.dropout)
        self.decoder = Decoder(self.n_head, self.embedding_dim, self.decode_mode, self.C)
        
        #------------------------------------------------------
        self.accelerator = Accelerator()
        self.encoder, self.decoder = self.accelerator.prepare(self.encoder, self.decoder)
        #------------------------------------------------------

    def forward(self, inputs):
        """
        :param inputs : (locations, demands, Tmax, m)
               (locations : [batch_size, seq_len, input_dim],
                scores : [batch_size, seq_len, 1],
                Tmax : [batch_size])
                m : [batch_size])

        :return: raw_logits : [batch_size, seq_len, seq_len],
                 log_prob : [batch_size],
                 solutions : [batch_size, seq_len]
        """

        inputs, scores, Tmax, m = inputs
        scor = scores.unsqueeze(-1)

        data = self.encoder(self.city_embedding(inputs) + self.scores_embedding(scor))

        raw_logits, log_prob, solution = self.decoder((data, scores, Tmax, m))

        return raw_logits, log_prob, solution

    def set_decode_mode(self, mode):
        self.decode_mode = mode
        self.decoder.decode_mode = mode

    def samplingLoop(self, data: DataLoader, decode_mode="sample"):
        tour_scores = torch.tensor([])
        self.eval()
        self.set_decode_mode(decode_mode)
        cpu = time.time()
        runs = 100  # Sampling size.  

        for batch_id, batch in enumerate(tqdm(data)):
            locations, scores, Tmax, m = batch
            locations = locations.to(self.accelerator.device)
            scores = scores.to(self.accelerator.device)
            Tmax = Tmax.to(self.accelerator.device)
            m=m.to(self.accelerator.device)
            inputs = (locations, scores, Tmax.float(),m)    # Data sent to GPU device. 

            # Using numba to compute the split procedure ---
            locations = locations.detach().cpu().numpy()
            scores = scores.detach().cpu().numpy()
            Tmax = Tmax.detach().cpu().numpy()
            m = m.detach().cpu().numpy()
            # ----

            #To samplig the best solution out of several runs.
            nb = len(scores) # Number of batches.
            maxs = torch.ones(nb)
            sampling_ts = torch.zeros(nb,1)
            for u in range(runs):
                _, _, solution = self(inputs)
                # Using numba to compute the split procedure ---
                solution = solution.detach().cpu().numpy()  
                sc = self.splitByBatch(locations, scores, Tmax, m, solution)
                bts = torch.tensor(sc).view(-1)

                sampling_ts = torch.cat((sampling_ts, bts.unsqueeze(1)), dim=1)
            
            for u in range(nb):
                maxs[u] = torch.max(sampling_ts[u])     # max score over sampling.

            tour_scores = torch.cat((tour_scores, maxs), dim=0)

        cpu = time.time() - cpu
        return {
            "tour_scores": tour_scores,
            "avg_ts": tour_scores.mean().item(),
            "std_ts": tour_scores.std().item(),
            "cpu": cpu
        }
    @staticmethod
    @jit(nopython=True, parallel=True) # Set "nopython" mode for best performance.
    #@jit(nopython=True, parallel=True, cache=True, fastmath=True)  # Set "nopython" mode for best performance.
    def splitByBatch(locations, scores, Tmax, m, solution):
    #def splitByBatch(self, locations, scores, Tmax, m, solution):
        myRoute = np.zeros((len(solution), (len(solution[0]) - 2)), dtype=np.uint64) # A route without the depot (positions 0 and 1).
        distance_from_start = np.zeros((len(solution), (len(solution[0]) - 2)), dtype=float)  # Distance from start to a node.
        distance_to_next = np.zeros((len(solution), (len(solution[0]) - 2)), dtype=float)  # Distance between nodes.
        distance_to_finish = np.zeros((len(solution), (len(solution[0]) - 2)), dtype=float)  # Distance from a node to finishing node.
        # Computing route and distances.
        myRoute = sp.extractRoute(solution, myRoute)
        distance_from_start = sp.distanceFrom(locations, myRoute, distance_from_start, 0)
        distance_to_next = sp.distanceBetween(locations, myRoute, distance_to_next)
        distance_to_finish = sp.distanceFrom(locations, myRoute, distance_to_finish, 1)

        # 1 - build vector of size (vl), profit (vp), length/cost (vc) and successor (vs) for each saturated tour.                    
        vp = np.zeros((len(solution), (len(solution[0]) - 2)), dtype=float)
        vc = np.zeros((len(solution), (len(solution[0]) - 2)), dtype=float)            
        vs = np.zeros((len(solution), (len(solution[0]) - 2)), dtype=np.uint64)
                    
        vp, vc, vs = sp.buildVectors(scores, Tmax, myRoute, distance_from_start, distance_to_finish, distance_to_next, vp, vc, vs)
                    
        # 2 and 3 - dynamic programming to find the maximum-weighted independent set and taking optimal solution.
        maxNumbTours = np.max(m)
        mprofit = np.zeros((len(solution), (len(solution[0]) - 2), (maxNumbTours)), dtype=float)        

        #sc = np.zeros([len(solution), 1], dtype=float)  # Scores per route after solution.
        sc = np.zeros((len(solution), 1), dtype=np.float64)  # Scores per route after solution.        
        sc = sp.takeOptimal(m, vp, vs, mprofit, sc)

        return sc
    
    def splitNumba(self, instance_data, solution): 
        #Team orienteering problem
        locations, scores, Tmax, m = instance_data
        #locations = locations.to(self.accelerator.device)
        #scores = scores.to(self.accelerator.device)
        #Tmax = Tmax.to(self.accelerator.device)
        #m=m.to(self.accelerator.device)
        #inputs = (locations, scores, Tmax.float(),m)

        # Using numba to compute the split procedure ---
        locations = locations.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()
        Tmax = Tmax.detach().cpu().numpy()
        m = m.detach().cpu().numpy()
        # Using numba to compute the split procedure ---
        solution = solution.detach().cpu().numpy()  

        myRoute = np.zeros((len(solution), (len(solution[0]) - 2)), dtype=np.uint64) # A route without the depot (positions 0 and 1).
        distance_from_start = np.zeros((len(solution), (len(solution[0]) - 2)), dtype=float)  # Distance from start to a node.
        distance_to_next = np.zeros((len(solution), (len(solution[0]) - 2)), dtype=float)  # Distance between nodes.
        distance_to_finish = np.zeros((len(solution), (len(solution[0]) - 2)), dtype=float)  # Distance from a node to finishing node.
        # Computing route and distances.
        myRoute = sp.extractRoute(solution, myRoute)
        distance_from_start = sp.distanceFrom(locations, myRoute, distance_from_start, 0)
        distance_to_next = sp.distanceBetween(locations, myRoute, distance_to_next)
        distance_to_finish = sp.distanceFrom(locations, myRoute, distance_to_finish, 1)

        # 1 - build vector of size (vl), profit (vp), length/cost (vc) and successor (vs) for each saturated tour.                    
        vp = np.zeros((len(solution), (len(solution[0]) - 2)), dtype=float)
        vc = np.zeros((len(solution), (len(solution[0]) - 2)), dtype=float)            
        vs = np.zeros((len(solution), (len(solution[0]) - 2)), dtype=np.uint64)
                    
        vp, vc, vs = sp.buildVectors(scores, Tmax, myRoute, distance_from_start, distance_to_finish, distance_to_next, vp, vc, vs)
                    
        # 2 and 3 - dynamic programming to find the maximum-weighted independent set and taking optimal solution.
        maxNumbTours = np.max(m)
        mprofit = np.zeros((len(solution), (len(solution[0]) - 2), (maxNumbTours)), dtype=float)
                    
        sc = np.zeros([len(solution), 1], dtype=float)  # Scores per route after solution.
        sc = sp.takeOptimal(m, vp, vs, mprofit, sc)
        sctensor = torch.tensor(sc, device=self.accelerator.device)

        return sctensor.view(-1)
    
import csv
import time
import math
import openpyxl
import numpy as np
import torch
from scipy.stats import ttest_rel
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import plot
from Baseline import Baseline
from Model import AttentionModel
from OPDataset import OPDataset
from accelerate import Accelerator
import pandas as pd

class Trainer:
    def __init__(self, graph_size, n_epochs, batch_size, nb_train_samples,
                 nb_val_samples, n_layers, n_heads, embedding_dim,
                 dim_feedforward, C, dropout, learning_rate, RESUME, BASELINE, SCORE_TYPE
                 ):
        self.RESUME = RESUME
        self.BASELINE = BASELINE
        self.SCORE_TYPE = SCORE_TYPE
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.graph_size = graph_size
        self.nb_train_samples = nb_train_samples
        self.nb_val_samples = nb_val_samples

        #-------------------------------------
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.embedding_dim = embedding_dim
        self.dim_feedforward = dim_feedforward
        self.C = C
        self.dropout = dropout
        #-------------------------------------

        self.max_grad_norm = 1.0
        self.accelerator = Accelerator()

        self.model = AttentionModel(embedding_dim, n_layers, n_heads,
                                    dim_feedforward,
                                    C, dropout)  

        # This ia a rollout baseline
        baseline_model = AttentionModel(embedding_dim, n_layers, n_heads,
                                        dim_feedforward,
                                        C, dropout)
        self.baseline = Baseline(baseline_model, graph_size, nb_val_samples)

        self.baseline.load_state_dict(self.model.state_dict())

        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)

        self.baseline, self.optimizer = self.accelerator.prepare(self.baseline, self.optimizer)

        log_file_name = "{}-{}-logs.csv".format("vrp", graph_size)

        f = open(log_file_name, 'w', newline='')
        self.log_file = csv.writer(f, delimiter=",")

        header = ["epoch", "losses_per_batch", "avg_tl_batch_train", "avg_tl_epoch_train", "avg_tl_epoch_val"]
        self.log_file.writerow(header)

    def train(self):
        validation_dataset = OPDataset(size=self.graph_size, num_samples=self.nb_val_samples, scores=self.SCORE_TYPE)
        print("Validation dataset created with {} samples".format(len(validation_dataset)))
        validation_dataloader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
                                           
        benefits = []        
        avg_tour_score_batch = []
        avg_tour_score_epoch = []
        avg_sc_epoch_val = []
        begin_epoch = 0

        if self.RESUME == True:        
            data = torch.load('SC_TOP21_Epoch_18.pt')    #Current training epoch dictionary.
            if(self.BASELINE == 'EXP'):
                self.baseline=(data['baseline']).to(self.accelerator.device)
            else:
                self.baseline.load_state_dict(data['baseline'])
            self.model.load_state_dict(data['model'])
            self.model.to(self.accelerator.device)
            self.optimizer.load_state_dict(data['optimizer'])
            self.optimizer = self.accelerator.prepare(self.optimizer)
            begin_epoch = data['epoch'] + 1

        for epoch in range(begin_epoch, self.n_epochs):

            cpu = time.time()

            all_tour_scores = torch.tensor([], dtype=torch.float32, device=self.accelerator.device)

            # Put model in train mode!
            self.model.set_decode_mode("sample")
            self.model, self.optimizer, validation_dataloader = self.accelerator.prepare(self.model, self.optimizer,
                                                                                         validation_dataloader)
            
            if(self.BASELINE == 'CRITIC'):
                self.baseline.model.set_decode_mode("greedy")
                self.baseline.model = self.accelerator.prepare(self.baseline.model)
            
            self.model.train()

            # Generate new training data for each epoch
            train_dataset = OPDataset(size=self.graph_size, num_samples=self.nb_train_samples, scores=self.SCORE_TYPE)

            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)                                     
            
            nb_batches = len(validation_dataloader)

            beta = 0.2 #Smoothing value

            for batch_id, batch in enumerate(tqdm(train_dataloader)):
                locations, scores, Tmax, m = batch

                inputs = (locations.to(self.accelerator.device), scores.to(self.accelerator.device),
                          Tmax.float().to(self.accelerator.device), m.to(self.accelerator.device))

                _, log_prob, solution = self.model(inputs)

                with torch.no_grad():
                    #-----------------------------------------------------------------------------
                    tour_score = self.model.splitNumba(inputs, solution) 

                    if(self.BASELINE == 'CRITIC'):
                        baseline_tour_score = self.baseline.evaluate(inputs, False)
                        advantage = tour_score - baseline_tour_score[0:len(tour_score)] #showme
                    
                    if(self.BASELINE == 'EXP'):
                        if (batch_id == 0):
                            if(self.RESUME == True and begin_epoch == epoch):
                                baseline_tour_score = self.baseline
                            else:
                                baseline_tour_score = tour_score                                
                        else:
                            baseline_tour_score = baseline_tour_score*beta + (1-beta)*(tour_score)
                        advantage = tour_score - baseline_tour_score

                benefit = advantage * (-log_prob)
                benefit = benefit.mean()

                self.optimizer.zero_grad()
                self.accelerator.backward(benefit)

                for group in self.optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(
                        group['params'],
                        self.max_grad_norm if self.max_grad_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
                        norm_type=2
                    )
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=2)
                                
                self.optimizer.step()

                benefits.append(benefit.item())
                avg_tour_score_batch.append(tour_score.mean().item())
                all_tour_scores = torch.cat((all_tour_scores, tour_score), dim=0)

            avg_tour_score_epoch.append(all_tour_scores.mean().item())

            print(
                "\nEpoch: {}\t\nAverage tour score model : {}\nAverage tour score baseline : {}\n".format(
                    epoch + 1, tour_score.mean(), baseline_tour_score.mean()
                ))

            print("Validation and rollout update check\n")
            
            if(self.BASELINE == 'CRITIC'):  #Critic roll-out baseline.
                # t-test :
                self.model.set_decode_mode("sample")
                self.baseline.set_decode_mode("sample")
                self.model.eval()
                self.baseline.eval()
                with torch.no_grad():
                    rollout_tl = torch.tensor([], dtype=torch.float32, device=self.accelerator.device)
                    policy_tl = torch.tensor([], dtype=torch.float32, device=self.accelerator.device)
                    for batch_id, batch in enumerate(tqdm(validation_dataloader)):
                        locations, scores, Tmax, m = batch

                        inputs = (locations, scores, Tmax.float(), m)

                        _, _, solution = self.model(inputs)

                        tour_score = self.model.splitNumba(inputs, solution)
                        baseline_tour_score = self.baseline.evaluate(inputs, False)

                        rollout_tl = torch.cat((rollout_tl, baseline_tour_score.view(-1)), dim=0)
                        policy_tl = torch.cat((policy_tl, tour_score.view(-1)), dim=0)

                    rollout_tl = rollout_tl.cpu().numpy()
                    policy_tl = policy_tl.cpu().numpy()

                    avg_ptl = np.mean(policy_tl)
                    avg_rtl = np.mean(rollout_tl)

                    avg_sc_epoch_val.append(avg_ptl.item())

                    cpu = time.time() - cpu
                    print(
                        "CPU: {}\n"
                        "Benefit: {}\n"
                        "Average tour score by policy: {}\n"
                        "Average tour score by rollout: {}\n".format(cpu, benefit, avg_ptl, avg_rtl))

                    self.log_file.writerow([epoch, benefits[-nb_batches:],
                                            avg_tour_score_batch[-nb_batches:],
                                            avg_tour_score_epoch[-1],
                                            avg_ptl.item()
                                            ])
                    
                    if (avg_ptl - avg_rtl) > 0:
                        _, pvalue = ttest_rel(policy_tl, rollout_tl)
                        pvalue = pvalue / 2  # one-sided ttest [refer to the original implementation]
                        if pvalue < 0.05:
                            print("Rollout network update...\n")
                            self.baseline.load_state_dict(self.model.state_dict())
                            self.baseline.reset()
                            print("Generate new validation dataset\n")

                            validation_dataset = OPDataset(size=self.graph_size, num_samples=self.nb_val_samples, scores=self.SCORE_TYPE)

                            validation_dataloader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False,
                                                            num_workers=0)
                
            else:   #EXP baseline.
                self.model.set_decode_mode("sample")
                self.model.eval()
                with torch.no_grad():
                    policy_tl = torch.tensor([], dtype=torch.float32, device=self.accelerator.device)
                    for batch_id, batch in enumerate(tqdm(validation_dataloader)):
                        locations, scores, Tmax, m = batch
                        inputs = (locations, scores, Tmax.float(), m)

                        _, _, solution = self.model(inputs)

                        tour_score = self.model.splitNumba(inputs, solution)  
                        policy_tl = torch.cat((policy_tl, tour_score.view(-1)), dim=0)

                    policy_tl = policy_tl.cpu().numpy()
                    avg_ptl = np.mean(policy_tl)
                    avg_sc_epoch_val.append(avg_ptl.item())

                    cpu = time.time() - cpu
                    print(
                        "CPU: {}\n"
                        "Benefit: {}\n"
                        "Average tour score by policy: {}\n".format(cpu, benefit, avg_ptl))                        

                    self.log_file.writerow([epoch, benefits[-nb_batches:],
                                            avg_tour_score_batch[-nb_batches:],
                                            avg_tour_score_epoch[-1],
                                            avg_ptl.item()
                                            ])
       
            model_name = "output/"+"SC_{}{}_Epoch_{}.pt".format("TOP", self.graph_size, epoch + 1)
            if(self.BASELINE == 'CRITIC'):
                torch.save({
                'epoch': epoch,
                'baseline': self.baseline.state_dict(),
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
                }, model_name)
            else: #Exponential baseline.
                torch.save({
                'epoch': epoch,
                'baseline': baseline_tour_score,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
                }, model_name)

        plot.plot_stats(benefits, "{}-SC-Benefits per batch {}".format("op", self.graph_size), "Batch", "Benefit")
        plot.plot_stats(
                        avg_tour_score_epoch,
                        "{} Average tour score per epoch train {}".format("op", self.graph_size),
                        "Epoch", "Average tour score")
        plot.plot_stats(
                         avg_tour_score_batch,                        
                        "{} Average tour score per batch train {}".format("op", self.graph_size),
                        "Batch", "Average tour score")
        plot.plot_stats(
                         avg_sc_epoch_val,
                        "{} Average tour score per epoch validation {}".format("op", self.graph_size),
                        "Epoch", "Average tour score")
      
        
    def clip_grad_norms(param_groups, max_norm=math.inf):
        """
        Clips the norms for all param groups to max_norm and returns gradient norms before clipping
        :param optimizer:
        :param max_norm:
        :param gradient_norms_log:
        :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
        """
        showMax = max_norm
        grad_norms = [
            torch.nn.utils.clip_grad_norm_(
                group['params'],
                max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
                norm_type=2
            )
            for group in param_groups
        ]
        grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
        return grad_norms, grad_norms_clipped



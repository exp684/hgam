import torch
from Model import AttentionModel
from torch.optim import Adam

from Trainer import Trainer

graph_size = 21
n_epochs = 100
batch_size = 1000
nb_train_samples = 800000
nb_val_samples = 1000
n_layers = 3    
n_heads = 8 
embedding_dim = 128 
dim_feedforward = 512
C = 10
dropout = 0.1
learning_rate = 1e-5
seed = 1234

if __name__ == "__main__":

    RESUME = False   #Resume the training process from a given epoch.
    BASELINE = 'CRITIC'    # Type of baseline. EXP = Exponential, CRITIC = Critic.
    SCORE_TYPE = 'Constant'  # Type of scores: Uniform, Constant, Distance. 

    torch.set_num_threads(32)
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(seed)
    trainer = Trainer(graph_size, n_epochs, batch_size, nb_train_samples, nb_val_samples,
                      n_layers, n_heads, embedding_dim, dim_feedforward, C,
                      dropout, learning_rate, RESUME, BASELINE, SCORE_TYPE)
        
    trainer.train()

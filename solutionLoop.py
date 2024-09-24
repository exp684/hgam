import os
import torch
from torch.utils.data import DataLoader
import math
import plot
import train
from Model import AttentionModel
from OPDataset import OPDataset
import pandas as pd

if __name__ == "__main__":
    """
        This class is used to measure the performance of the best model obtained after training and evaluation.
    """
    # Making the code device-agnostic
    if torch.cuda.is_available(): torch.set_default_device('cuda') 
    else: torch.set_default_device('cpu') 

    graph_size = 21
    batch_size = 1000
    nb_test_samples = 100000
    test_samples = "random"  # "xml100", "random", "literature"
    SCORE_TYPE = 'Constant'  # Type of scores: Uniform, Constant, Distance.
    n_layers = train.n_layers
    n_heads = train.n_heads
    embedding_dim = train.embedding_dim
    dim_feedforward = train.dim_feedforward
    decode_mode = "sample"
    C = train.C
    dropout = train.dropout
    seed = 1234
    torch.manual_seed(seed)
    
    print("CUDA supported version by system--- ", torch.cuda.is_available())  
    print(f"CUDA version: {torch.version.cuda}")

    test_dataset = OPDataset(size=graph_size, num_samples=nb_test_samples, scores=SCORE_TYPE)
    print("Number of test samples : ", test_dataset.num_samples)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "heuristic"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"

    #Writing instances to a file.  Two type of formats.  
    """
    for u in range(len(test_dataset)):
        # Space format
        
        address = "Instances/opval-n22_spc_fmt/opval"+ str(u+1) +".txt"
        out = open(address, 'w')
        out.write('n\t'+str(graph_size)+'\n')
        out.write('m\t'+str(1)+'\n')
        out.write('tmax\t'+str(test_dataset.Tmax[u])+'\n')
        
        # Rym format
        address = "Instances/topval-n"+str(graph_size)+"_Rym_format/topval"+ str(u+1) +".txt"
        outr = open(address, 'w')
        outr.write('n;'+str(graph_size)+'\n')
        outr.write('m;'+str(test_dataset.m[u].cpu().numpy())+'\n')
        outr.write('tmax;'+str(test_dataset.Tmax[u].cpu().numpy())+'\n')

        v = 0
        while( v < graph_size):
            if(v==1): v+=1            
            numbx = float(test_dataset.locations[u][v][0])
            strNumbx = str(round(numbx, 5))
            numby = float(test_dataset.locations[u][v][1])
            strNumby = str(round(numby, 5))            
            numScore = int(test_dataset.scores[u][v]*100000) #IDCH uses integer values. Thus, we multiply by 10X10⁵            
            #numScore = float(test_dataset.scores[u][v]*100000)
            if(numScore==0): #IDCH cannot handle multiple values equal to 0.
            #if(numScore<0.00001):
                #numScore = numScore + 0.00001 
                numScore = numScore + 1 
            #strScore = str(round(numScore, 5))
            strScore = str(numScore)  
            #--------------------------------------------
            #print("Integer value--\t",v,"\t", numScore, "\t", test_dataset.scores[u][v] )
            #--------------------------------------------            
            if(v==0):
                lastx = strNumbx
                lasty = strNumby
                strScore = str(0)
            #print("x-\t"+ strNumbx + "\t-y\t"+ strNumby)
            #out.write(strNumbx + "\t"+ strNumby + "\t"+ strScore +"\n")
            outr.write(strNumbx + ";"+ strNumby + ";"+ strScore +"\n")
            v += 1
        #out.write(lastx + "\t"+ lasty + "\t"+ str(0)+"\n")
        outr.write(lastx + ";"+ lasty + ";"+ str(0)+"\n")
        #out.close()
        outr.close()
    """
    
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=0)
    model = AttentionModel(embedding_dim, n_layers, n_heads, dim_feedforward, C, dropout)
    model = model.to(device='cuda')

    #Scores constants.
    data = torch.load('model/SC_TOP21_Epoch_88.pt')    #graph_size=21
    #data = torch.load('SC_TOP51_Epoch_99.pt')   #graph_size=51
    #data = torch.load('SC_TOP101_Epoch_98.pt')   #graph_size=101

    try:
        model.load_state_dict(data["model"])    
        results = model.samplingLoop(test_dataloader)        
        print('Results : Average {}, Std. Dev. {} in cpu = {}'.format(results["avg_ts"], results["std_ts"], results["cpu"]))

    except Exception as e:
        print(e)
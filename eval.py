import os
import time
import torch
from torch.utils.data import DataLoader
import plot
import train
from Model import AttentionModel
from OPDataset import OPDataset

if __name__ == "__main__":
    # Making the code device-agnostic
    if torch.cuda.is_available(): torch.set_default_device('cuda') 
    else: torch.set_default_device('cpu') 

    graph_size = 21
    batch_size = 1000
    nb_test_samples = 10000
    test_samples = "random"  # "xml100", "random", "literature"
    SCORE_TYPE = 'Constant'  # Type of scores: Uniform, Constant, Distance. 
    n_layers = train.n_layers
    n_heads = train.n_heads
    embedding_dim = train.embedding_dim
    dim_feedforward = train.dim_feedforward
    decode_mode = "greedy"
    C = train.C
    dropout = train.dropout
    seed = 1234
    torch.manual_seed(seed)

    print("CUDA supported version by system--- ", torch.cuda.is_available())  
    print(f"CUDA version: {torch.version.cuda}")
        
    # Storing ID of current CUDA device
    cuda_id = torch.cuda.current_device()
    print("ID of current CUDA device:",
      {torch.cuda.current_device()})
       
    print("Name of current CUDA device:",
      {torch.cuda.get_device_name(cuda_id)})
    
    cpu = time.time()   #Starting running time.  
    test_dataset = OPDataset(size=graph_size, num_samples=nb_test_samples, scores=SCORE_TYPE)

    print("Number of test samples : ", len(test_dataset))
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=0)

    random_model = AttentionModel(embedding_dim, n_layers, n_heads, dim_feedforward, C, dropout)

    random_model = random_model.to(device='cuda')
    
    print("Is the model in cuda?:  ", next(random_model.parameters()).is_cuda)

    tour_scores = {
        "random": random_model.samplingLoop(test_dataloader)
    }

    folder = os.path.abspath("output/exp(Clean-TOP)numba-parallel")    
    datasets = os.listdir(folder) #   "../"
    for dataset in sorted(datasets, key=lambda x: int(x.split('_')[-1].split('.')[0]) if x.endswith('.pt') else 0):
        if not dataset.endswith('.pt'): continue
        print("Testing model : ", dataset)
        data = torch.load(os.path.join(folder, dataset))

        try:
            model = AttentionModel(embedding_dim, n_layers, n_heads, dim_feedforward, C, dropout)
            model = model.to(device='cuda')
            model.load_state_dict(data["model"])
            results = model.samplingLoop(test_dataloader)
            tour_scores[int(dataset.split('_')[-1].split('.')[0])] = results
            print('{} : {} in cpu = {}'.format(dataset, results["avg_ts"], results["cpu"]))
        except Exception as e:
            print(e)
            continue
    
    cpu = time.time() - cpu
    print("===========================================================================/n")
    print("Total running time ---- ", cpu)
    print("===========================================================================/n")

    plot.plot_stats([results["avg_ts"] for dataset, results in tour_scores.items()],
                    "{} Average tour scores per epoch evaluation {}".format("TOP", graph_size),
                    "Epoch", "Average tour score", folder)

    sorted_tour_lengths = sorted(tour_scores.items(), key=lambda x: x[1]["avg_ts"])

    print('Sorted tour scores per model')
    for model_name, results in sorted_tour_lengths:
        print('{} : {} in cpu = {}'.format(model_name, results["avg_ts"], results["cpu"]))

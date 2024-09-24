import math 
import pandas as pd
import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.distributions import Categorical
from Encoder import MultiHeadAttention

class Decoder(nn.Module):
    """
        This class contains the decoder that will be used to compute the probability distribution from which we will sample
        which city to visit next.
    """

    def __init__(self, n_head, embedding_dim, decode_mode="sample", C=10):
        super(Decoder, self).__init__()
        self.scale = math.sqrt(embedding_dim)
        self.decode_mode = decode_mode
        self.C = C  
        self.n_head = n_head

        self.vl = nn.Parameter(
            torch.FloatTensor(size=[1, 1, embedding_dim]).uniform_(-1. / embedding_dim, 1. / embedding_dim),
            requires_grad=True)
        self.vf = nn.Parameter(
            torch.FloatTensor(size=[1, 1, embedding_dim]).uniform_(-1. / embedding_dim, 1. / embedding_dim),
            requires_grad=True)
        
        #------------------------------------------------------------------------------------------------------
        self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))    
        self.W_placeholder.data.uniform_(-1, 1)  
        
        #Projecting the context (after computing the mean of the embedding inputs)
        self.project_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        #Projecting the context (after adding the last client included into the sequence)
        self.project_context_update = nn.Linear(2 * embedding_dim, embedding_dim, bias=False)
        #Projecting node embeddings to compute attention (glimpse_key, glimpse_val and logit_key)
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        #Projecting node embeddings to compute attention (glimpse_key, glimpse_val and logit_key)
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)
        #------------------------------------------------------------------------------------------------------
        self.glimpse = MultiHeadAttention(n_head, embedding_dim, 3 * embedding_dim, embedding_dim, embedding_dim)
        self.project_k = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        #------------------------------------------------------------------------------------------------------
        self.accelerate = Accelerator()
        self.glimpse, self.project_k, self.cross_entropy, self.project_context, self.project_context_update, self.project_node_embeddings, self.project_out = \
                self.accelerate.prepare(self.glimpse, self.project_k, self.cross_entropy, self.project_context, 
                self.project_context_update, self.project_node_embeddings, self.project_out)        
        #------------------------------------------------------------------------------------------------------

    def forward(self, inputs):
        """
        :param inputs: (encoded_inputs, demands, capacities) ([batch_size, seq_len, embedding_dim],[batch_size, seq_len],[batch_size])
        :return: log_prob, solutions
        """
        #encoded_inputs: decision variables.
        encoded_inputs, scores, Tmax, m = inputs

        batch_size, seq_len, embedding_dim = encoded_inputs.size()  

        h_hat = encoded_inputs.mean(-2, keepdim=True)   

        #------------------------------------------------------------------------------------------------------
        h_hat = self.project_context(h_hat) #[Batch_size, 1, embedding_dim]
        outputLog = []
        #------------------------------------------------------------------------------------------------------

        city_index = None

        mask = torch.zeros([batch_size, seq_len]).bool()

        solution = torch.tensor([batch_size, 1], dtype=torch.int64)  
        
        log_probabilities = torch.zeros(batch_size, dtype=torch.float32)

        #------------------------------------------------------------------------------------------------------
        first = torch.zeros(batch_size, 1, dtype=torch.int64)  
        last = torch.zeros(batch_size, 1, dtype=torch.int64)  
        #------------------------------------------------------------------------------------------------------
        
        raw_logits = torch.tensor([])
        t = 0  # time steps;  two positions in the sequence are already assigned.

        while torch.sum(mask) < batch_size * seq_len:
            t += 1

            #------------------------------------------------------------------------------------------------------
            if(t==1):                 
                embfl = self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1))                           
            else:           
                firstLast = torch.cat((first, last), dim=1)[:, :, None] .expand(batch_size, 2, encoded_inputs.size(-1))
                embfl = encoded_inputs.gather(1, firstLast).view(batch_size, 1, -1)        
            
            h_hat_update = self.project_context_update(embfl)   #[batch_size, 1, embedding_dim] v1, vf
            query = h_hat + h_hat_update

            glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(encoded_inputs[:, None, :, :]).chunk(3, dim=-1)
            glimpse_K = self._make_heads(glimpse_key_fixed)
            glimpse_V = self._make_heads(glimpse_val_fixed)
            logit_K = logit_key_fixed.contiguous()

            key_size = val_size = embedding_dim // self.n_head

            # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
            glimpse_Q = query.view(batch_size, 1, self.n_head, 1, key_size).permute(2, 0, 1, 3, 4)

            # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
            compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
            compatibility[mask[None, :, None, None, :].expand_as(compatibility)] = -math.inf

            # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
            heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

            # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)        
            glimpse = self.project_out(
                        heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, 1, 1, self.n_head * val_size))

            # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
            # logits = 'compatibility'
            logits = torch.matmul(glimpse, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(glimpse.size(-1))

            # From the logits compute the probabilities by clipping, masking and softmax
            logits = torch.tanh(logits) * self.C       
            logits[mask[: ,None, :].expand_as(logits)] = -math.inf
            #------------------------------------------------------------------------------------------------------
                        
            logits = torch.log_softmax(logits, dim=-1)

            assert not torch.isnan(logits).any(), "fuck up again...."
            
            probas = logits.exp()[:, 0, :]

            if self.decode_mode == "greedy":
                proba, city_index = self.greedy_decoding(probas)
            elif self.decode_mode == "sample":
                city_index = self.new_sample_decoding(probas, mask)

            outputLog.append(logits.squeeze(1)) 

            if (t==1):
                solution = city_index
            else:
                solution = torch.cat((solution, city_index), dim=1)
            
            # next node for the query            
            last = city_index

            # update mask
            mask = mask.scatter(1, city_index, True)

            if t == 1:
                first = last

        outputLog = torch.stack(outputLog, 1)
        log_probabilities = self._calc_log_likelihood(outputLog, solution)

        return raw_logits, log_probabilities, solution

    @staticmethod
    def greedy_decoding(probas):
        """
        :param probas: [batch_size, seq_len]
        :return: probas : [batch_size],  city_index: [batch_size,1]
        """
        probas, city_index = torch.max(probas, dim=1)

        return probas, city_index.view(-1, 1)

    @staticmethod
    def sample_decoding(probas):
        """
        :param probas: [ batch_size, seq_len]
        :return: probas : [batch_size],  city_index: [batch_size,1]
        """
        batch_size = probas.size(0)
        m = Categorical(probas)
        city_index = m.sample()
        probas = probas[[i for i in range(batch_size)], city_index]

        return probas, city_index.view(-1, 1)
    
    @staticmethod
    def new_sample_decoding(probs, mask):
        """
        :param probas: [ batch_size, seq_len]
        :param mask: [ batch_size, seq_len]
        :return: city_index: [batch_size,1]
        """
        assert (probs == probs).all(), "Probs should not contain any nans"
        city_index = probs.multinomial(1).squeeze(1)
        # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232 
        while mask.gather(1, city_index.unsqueeze(-1)).data.any():
            print('Sampled bad values, resampling!')
            city_index = probs.multinomial(1).squeeze(1)

        return city_index.view(-1, 1)

    def _make_heads(self, v):
        myTorchResult = v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_head, -1).permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)

        return (myTorchResult)
    
    def _calc_log_likelihood(self, _log_p, a):

        # Get log_p corresponding to selected actions
        squezeeda = a.unsqueeze(-1)
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)


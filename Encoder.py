from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as functional
from accelerate import Accelerator

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, embedding_dim, q_dim, k_dim, v_dim):
        super(MultiHeadAttention, self).__init__()
        #-----------------------------------------------------------------
        # Making the code device-agnostic
        if torch.cuda.is_available(): torch.set_default_device('cuda') 
        else: torch.set_default_device('cpu') 
        #-----------------------------------------------------------------
        self.n_head = n_heads
        self.hidden_dim = embedding_dim // self.n_head
        
        self.queries_transformation = nn.Linear(q_dim, embedding_dim, bias=False)
        self.keys_transformation = nn.Linear(k_dim, embedding_dim, bias=False)
        self.values_transformation = nn.Linear(v_dim, embedding_dim, bias=False)
        self.out = nn.Linear(embedding_dim, embedding_dim, bias=False)

        #------------------------------------------------------   
        self.accelerate = Accelerator()
        self.queries_transformation, self.keys_transformation, self.values_transformation, self.out = \
        self.accelerate.prepare(self.queries_transformation, self.keys_transformation, self.values_transformation, self.out)
        #------------------------------------------------------

    def forward(self, query, key, value, mask=None):
        """
        :param query: [batch_size]
        :param key: [seq_len]
        :param value: [embedding_dim]
        :param mask : [batch_size, seq_len]
        :return: out : [batch_size, seq_len, embedding_dim]
        """
        batch_size = query.size(0)

        q1 = self.queries_transformation(query) 
        q  = q1.reshape(batch_size, -1, self.n_head, self.hidden_dim).permute(0, 2, 1, 3)
        k = self.keys_transformation(key).reshape(batch_size, -1, self.n_head, self.hidden_dim).permute(0, 2, 1, 3)
        v = self.values_transformation(value).reshape(batch_size, -1, self.n_head, self.hidden_dim).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(3, 2)) / sqrt(self.hidden_dim)        

        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        attention = torch.nn.functional.softmax(scores, dim=-1)

        output = torch.matmul(attention, v)

        output_contiguous = output.transpose(1, 2).contiguous()
        concat_output = output_contiguous.view(batch_size, -1, self.n_head * self.hidden_dim)

        out = self.out(concat_output)

        return out

class FeedForwardLayer(nn.Module):
    def __init__(self, embedding_dim, dim_feedforward):
        super(FeedForwardLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.dim_feedforward = dim_feedforward

        self.lin1 = nn.Linear(self.embedding_dim, self.dim_feedforward)
        self.lin2 = nn.Linear(self.dim_feedforward, self.embedding_dim)

    def forward(self, inputs):
        """
        :param inputs: [batch_size, seq_len, embedding_dim]
        :return: out : [batch_size, seq_len, embedding_dim]
        """
        return self.lin2(functional.relu(self.lin1(inputs)))

class EncoderLayer(nn.Module):
    def __init__(self, n_head, embedding_dim, dim_feedforward, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(n_head,
                                                       embedding_dim,
                                                       embedding_dim,
                                                       embedding_dim,
                                                       embedding_dim)
        self.feed_forward_layer = FeedForwardLayer(embedding_dim, dim_feedforward)

        self.dropout1 = nn.Dropout(dropout)  # MHA dropout
        self.dropout2 = nn.Dropout(dropout)  # FFL dropout

        self.bn1 = nn.BatchNorm1d(embedding_dim, affine=True)
        self.bn2 = nn.BatchNorm1d(embedding_dim, affine=True)

    def forward(self, x):
        """
        :param x: [batch_size, seq_len, embedding_dim]
        :return: out : [batch_size, seq_len, embedding_dim]
        """
        x = x + self.dropout1(self.multi_head_attention(x, x, x))
        x = self.bn1(  x.view( -1, x.size(-1) )  ).view( *x.size() )

        x = x + self.dropout2(self.feed_forward_layer(x))
        x = self.bn2(x.view(-1, x.size(-1))).view(*x.size())

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, n_layers, n_head, embedding_dim, dim_feedforward, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = [EncoderLayer(n_head, embedding_dim, dim_feedforward, dropout) for _ in
                       range(n_layers)]
        
        self.transformer_encoder = nn.Sequential(*self.layers)

    def forward(self, inputs):
        """
        :param inputs: [batch_size, seq_len, embedding_dim]
        :return: [batch_size, seq_len, embedding_dim]
        """

        return self.transformer_encoder(inputs)

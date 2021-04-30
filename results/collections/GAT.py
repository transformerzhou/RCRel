
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


d_model = 768 
d_k = d_v = 64 
n_heads = 12
device = torch.device("cuda:0" if (torch.cuda.is_available() ) else "cpu")


class GATLayer(nn.Module):
    def __init__(self):
        super(GATLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)  
        enc_outputs = self.pos_ffn(enc_outputs)  
        return enc_outputs, attn


class GAT_module(nn.Module):
    def __init__(self, n_layers=1):
        super(GAT_module, self).__init__()
        self.layers = nn.ModuleList([GATLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):  
        batch_size, num_entity, num_rel = enc_inputs.shape[0], enc_inputs.shape[1], int(enc_inputs.shape[2] / 768)
        enc_outputs = enc_inputs.reshape([batch_size*num_entity, num_rel, 768])
        enc_self_attn_mask = get_attn_pad_mask(enc_outputs[:, :, 0], enc_outputs[:, :, 0])

        for layer in self.layers:
            outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
        return outputs

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  
    return pad_attn_mask.expand(batch_size, len_q, len_k) 


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            d_k) 
        scores.masked_fill_(attn_mask, -1e9) 
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=768, n_heads=12):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.n_heads = n_heads
        self.layer_norm = nn.LayerNorm(d_model)
        self.em = torch.nn.Embedding(24, d_model)
        self.em.load_state_dict(torch.load('NYT.em'))

    def forward(self, Q, K, V, attn_mask):
        N = Q.shape[0]
        rel_tensor = torch.tensor(list(range(24)), device=device)
#         rel_tensor = rel_tensor.cuda()
        rel_em = self.em(rel_tensor)
        rel_em = torch.unsqueeze(rel_em, dim=0)
        rel_em = torch.repeat_interleave(rel_em, N , dim=0)
#         print(Q.shape, rel_em.shape)
        rel_em = torch.reshape(rel_em, Q.shape)
        residual, batch_size = Q, Q.size(0)
        
        Q += rel_em
        K += rel_em
        V += rel_em

        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, d_k).transpose(1,
                                                                            2) 
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, d_k).transpose(1,
                                                                            2) 
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, d_v).transpose(1,
                                                                            2)  

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,self.n_heads * d_v)  
        output = self.linear(context)
        return self.layer_norm(output + residual), attn 


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_ff=2048):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs  
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)



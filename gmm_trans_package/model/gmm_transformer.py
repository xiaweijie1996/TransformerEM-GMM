import torch.nn as nn
import numpy as np
import torch

class CustomSigmoid(nn.Module):
    """
    Custom Sigmoid function with alpha and beta parameters   
    alpha: scaling factor
    beta: shifting factor
    a: scaling factor for the output
    b: shifting factor for the output 
    """
    def __init__(self, alpha=1.0, beta=0.0, a=1, b=0):
        super(CustomSigmoid, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.a = a
        self.b = b

    def forward(self, x):
        return self.a * (1 / (1 + torch.exp(-self.alpha * (x - self.beta)))) + self.b

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization Layer
    'https://arxiv.org/abs/1910.07467'
    """
    def __init__(self, eps=1e-15):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.rand(1))
        
    def forward(self, x):
        # calculate the root mean square normalization
        norm = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        # normalize and scale
        x_normalized = x / norm
        return x_normalized * self.weight


class Small_MLP(nn.Module):
    """
    Small MLP for the input and output mapping
    input: in dimension
    hidden: mid dimension
    output: out dimension

    """
    def __init__(self, in_dim, mid_dim, out_dim):
        super(Small_MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            RMSNorm(),
            nn.GELU(),
            
            nn.Linear(mid_dim, mid_dim),
            RMSNorm(),
            nn.GELU(),
            
            nn.Linear(mid_dim, mid_dim),
            RMSNorm(),
            nn.GELU(),
            nn.Linear(mid_dim, out_dim),  
                
        )
    def forward(self, x):
        return self.mlp(x)

class MSA(nn.Module):
    """
    Multi-head self-attention layer
    d: hidden dimension
    n_heads: number of heads
    """
    def __init__(self, d, n_heads):
        super(MSA, self).__init__()
        self.d = d
        self.n_heads = n_heads
        
        self.d_head = int(d // n_heads)
        self.q_map = nn.ModuleList([nn.Linear(self.d_head, self.d_head) for _ in range(n_heads)])
        self.k_map = nn.ModuleList([nn.Linear(self.d_head, self.d_head) for _ in range(n_heads)])
        self.v_map = nn.ModuleList([nn.Linear(self.d_head, self.d_head) for _ in range(n_heads)])
        self.softmax = nn.Softmax(dim=-1)   
    
    def forward(self, sequences):
        # split the sequences into n_heads
        q = [q_map(sequences[:, :, i*self.d_head:(i+1)*self.d_head]) for i, q_map in enumerate(self.q_map)]
        k = [k_map(sequences[:, :, i*self.d_head:(i+1)*self.d_head]) for i, k_map in enumerate(self.k_map)]
        v = [v_map(sequences[:, :, i*self.d_head:(i+1)*self.d_head]) for i, v_map in enumerate(self.v_map)]
        
        results = []
        for i in range(self.n_heads):
            # calculate the attention score
            attn_score = torch.bmm(q[i], k[i].transpose(1, 2)) / np.sqrt(self.d_head)
            attn_score = self.softmax(attn_score)
            # calculate the output
            output = torch.bmm(attn_score, v[i])
            results.append(output)
        return torch.cat(results, dim=2)   


class Vit_block(nn.Module):
    """
    A transformer with Multi-head self-attention and MLP
    hidden_d: hidden dimension
    n_heads: number of heads
    mlp_ratio: mlp ratio for the hidden dimension
    
    """
    def __init__(self, hidden_d, n_heads, mlp_ratio=4.0):
        super(Vit_block, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads
        self.msa = MSA(hidden_d, n_heads)
        self.norm1 = RMSNorm() #nn.LayerNorm([length, hidden_d]) # RMSNorm(hidden_d )
        self.norm2 = RMSNorm()  #nn.LayerNorm([length, hidden_d]) # RMSNorm(hidden_d) 
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, int(hidden_d * mlp_ratio)),
            RMSNorm(),
            nn.GELU(),
            nn.Linear(int(hidden_d * mlp_ratio), int(hidden_d * mlp_ratio)),
            RMSNorm(),
            nn.GELU(),
            nn.Linear(int(hidden_d * mlp_ratio), hidden_d),
            
        )

    def forward(self, x):
        x = x + self.msa(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT_encodernopara(nn.Module):
    """
    A transformer encoder
    chw: input data shape (1, num_days, num_time_steps+1), channel is always 1, the last dimension is the time steps+1 because of the date embedding has 1 more dimension
    hidden_d: hidden dimension
    out_d: output dimension (number of the time steps)
    n_heads: number of heads
    mlp_ratio: mlp ratio for the hidden dimension
    n_blocks: number of transformer blocks
    alpha: scaling factor for the sigmoid
    beta: shifting factor for the sigmoid
    """
    def __init__(self, 
                 chw = (1, 24, 24),
                 hidden_d = 96,
                 out_d = 2,
                 n_heads = 6,
                 mlp_ratio = 4.0,
                 n_blocks = 3,
                 alpha=1, 
                 beta=0.5
                 ):
        
        # Super constructor
        super(ViT_encodernopara, self).__init__()
        
        # input data shape (N, 365, 24)
        self.chw = chw # channel, height, width = 1, 365, 24
        self.hidden_d = hidden_d
        self.out_d = out_d
        self.linear_map_in = Small_MLP(self.chw[2], self.hidden_d, self.hidden_d) # nn.Linear(self.chw[2], self.hidden_d)
        self.linear_map_out2 = Small_MLP(self.hidden_d, self.hidden_d, self.out_d) # nn.Linear(self.hidden_d, self.out_d)
        
        # Vit block
        self.n_heads = n_heads
        self.mlp_ratio = mlp_ratio
        self.n_blocks = n_blocks
        self.vit_blocks = nn.ModuleList([Vit_block(self.hidden_d, self.n_heads, 
                                   self.mlp_ratio) for _ in range(self.n_blocks)])
        
        # output adding layer
        self.sig = CustomSigmoid(alpha, beta) # nn.Sigmoid() #CustomSigmoid(alpha, beta)
        self.bias = 0.000001
        
    def forward(self, images):
        _images = images
        tokens = self.linear_map_in(_images)
        
        for block in self.vit_blocks:
            tokens = block(tokens)
        
        tokens = self.linear_map_out2(tokens)
        return tokens
    
    def output_adding_layer(self, _new_para, _param):
        b, _, _ = _new_para.shape
        _new_para = _new_para.view(b, -1) + _param.view(b, -1)
        _new_para =  self.sig(_new_para) + self.bias
        return _new_para
    
 
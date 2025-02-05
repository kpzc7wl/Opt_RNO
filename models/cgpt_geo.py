#!/usr/bin/env python
#-*- coding:utf-8 _*-
import math
import numpy as np
import torch
import torch.nn as nn
import dgl
from einops import repeat, rearrange
from torch.nn import functional as F
from torch.nn import GELU, ReLU, Tanh, Sigmoid
from torch.nn.utils.rnn import pad_sequence
import scipy as sp

from utils import MultipleTensors
from models.mlp import MLP




class GPTConfig():
    """ base GPT config, params common to all GPT versions """
    def __init__(self,attn_type='linear', embd_pdrop=0.0, resid_pdrop=0.0,attn_pdrop=0.0, n_embd=128, 
                n_head=1, n_layer=3, block_size=128, n_inner=512,
                gamma_attn=0.3, gamma_phi=0.1, n_slice = 128, modes=8,
                act='gelu', branch_sizes=1,n_inputs=1):
        self.attn_type = attn_type
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.attn_pdrop = attn_pdrop
        self.n_embd = n_embd  # 64
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.n_inner = 4 * self.n_embd
        self.act = act
        self.branch_sizes = branch_sizes
        self.n_inputs = n_inputs
        self.gamma_attn = gamma_attn
        self.gamma_phi = gamma_phi
        self.n_slice = n_slice
        self.modes = modes





'''
    X: N*T*C --> N*(4*n + 3)*C 
'''
def horizontal_fourier_embedding(X, n=3):
    freqs = 2**torch.linspace(-n, n, 2*n+1).to(X.device)
    freqs = freqs[None,None,None,...]
    X_ = X.unsqueeze(-1).repeat([1,1,1,2*n+1])
    X_cos = torch.cos(freqs * X_)
    X_sin = torch.sin(freqs * X_)
    X = torch.cat([X.unsqueeze(-1), X_cos, X_sin],dim=-1).view(X.shape[0],X.shape[1],-1)
    return X

'''
    Random Feature Map
'''
def RFM(X, D=64, gamma=0.3):    
    # sigma = sigma / torch.tensor(2).sqrt() # gamma^2 = 2 * sigma^2 
    sigma = torch.tensor(2).sqrt() / gamma
    B, T, n = X.shape # n is spatial dimention, 2, 3...
    W = np.random.normal(0, sigma, size=(B, 1, D, n))
    W = torch.from_numpy(W).float().to(X.device)
    X_ = (X.unsqueeze(-2) * W).sum(dim=-1) # B, T, D
    # b = torch.rand((B, 1, D)) * 2 * np.pi
    # X_ = X_ + b
    X_cos = torch.cos(X_)
    X_sin = torch.sin(X_)
    X = 1 / torch.tensor(D).sqrt() * torch.cat([X_sin, X_cos], dim=-1).view(B, T, -1) # B, T, 2 * D
    # X = 1 / torch.tensor(D).sqrt() * X_cos.view(B, T, -1) # B, T, 2 * D
    return X


class LinearAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super(LinearAttention, self).__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head

        self.attn_type = 'l1'

    '''
        Linear Attention and Linear Cross Attention (if y is provided)
    '''
    def forward(self, x, y=None, layer_past=None):
        y = x if y is None else y
        B, T1, C = x.size()
        _, T2, _ = y.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x).view(B, T1, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = self.key(y).view(B, T2, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(y).view(B, T2, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)


        if self.attn_type == 'l1':
            q = q.softmax(dim=-1)
            k = k.softmax(dim=-1)   #
            k_cumsum = k.sum(dim=-2, keepdim=True)
            D_inv = 1. / (q * k_cumsum).sum(dim=-1, keepdim=True)       # normalized
        elif self.attn_type == "galerkin":
            q = q.softmax(dim=-1)
            k = k.softmax(dim=-1)  #
            D_inv = 1. / T2                                           # galerkin
        elif self.attn_type == "l2":                                   # still use l1 normalization
            q = q / q.norm(dim=-1,keepdim=True, p=1)
            k = k / k.norm(dim=-1,keepdim=True, p=1)
            k_cumsum = k.sum(dim=-2, keepdim=True)
            D_inv = 1. / (q * k_cumsum).abs().sum(dim=-1, keepdim=True)  # normalized
        else:
            raise NotImplementedError

        context = k.transpose(-2, -1) @ v
        y = self.attn_drop((q @ context) * D_inv + q)

        # output projection
        y = rearrange(y, 'b h n d -> b n (h d)')
        y = self.proj(y)
        return y



class LinearCrossAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super(LinearCrossAttention, self).__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.keys = nn.Linear(config.n_embd, config.n_embd)
        # self.keys = nn.ModuleList([nn.Linear(config.n_embd, config.n_embd) for _ in range(config.n_inputs)])
        self.values = nn.ModuleList([nn.Linear(config.n_embd, config.n_embd) for _ in range(config.n_inputs)])
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_inputs = config.n_inputs
        self.gamma = config.gamma_attn

        self.attn_type = 'l1'

    '''
        Linear Attention and Linear Cross Attention (if y is provided)
    '''
    def forward(self, x1, x2, y=None, layer_past=None, dist=None):
        B, T1, C = x1.size()
        _, T2, _ = x2.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x1).view(B, T1, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = self.keys(x2).view(B, T2, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-1)  #
        out = q

        v = torch.zeros_like(q).to(q.device)
        for i in range(self.n_inputs):
            _, T2, _ = y[i].size()
            v_ = self.values[i](y[i]).view(B, T2, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
            v = v + v_
        
        if dist != None:
            # # x_in = sum_{m, l, j} q_im * k_mj * h_il * h_lj * v_jn
            # #      = sum_{m, l} q_im * h_il * H_mln
            # #      = sum_{m} q_im * H_imn
            x_embed, _ = dist
            # print('k, x_embed, v: ', k.shape, x_embed.shape, v.shape)
            H_mln = torch.einsum('...jm, ...jl, ...jn->...mln', k, x_embed, v)
            H_imn = torch.einsum('...il, ...mln->...imn', x_embed, H_mln)
            q_k_v = torch.einsum('...im, ...imn->...in', q, H_imn)
        else:
            k_v = (k.transpose(-2, -1) @ v) # B, nh, hs, hs
            q_k_v = q @ k_v

        # print(i)
        k_cumsum = k.sum(dim=-2, keepdim=True)
        D_inv = 1. / (q * k_cumsum).sum(dim=-1, keepdim=True)  # normalized
        out = q_k_v * D_inv

        # output projection
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        return out


class CrossAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super(CrossAttention, self).__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads        
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.keys = nn.Linear(config.n_embd, config.n_embd)
        self.values = nn.ModuleList([nn.Linear(config.n_embd, config.n_embd) for _ in range(config.n_inputs)])
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_inputs = config.n_inputs
        self.gamma = config.gamma_attn
        self.eps = torch.exp(- torch.tensor(2.0).square())

        self.attn_type = 'l1'

    '''
        Cross Attention (if y is provided)
    '''
    def forward(self, x_q, x_r, y, mask=None, dist=None, layer_past=None):
        qs, ks, vs = x_q, x_r, y

        B, T1, C = qs.size() 
        _, T2, _ = ks.size()  
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(qs).view(B, T1, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = self.keys(ks).view(B, T2, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        attn = q @ k.transpose(-2, -1) / np.sqrt(C // self.n_head)    
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9) 
        attn = attn.softmax(dim=-1)       
        if dist is not None:
            attn = attn * ( torch.exp(- (dist / self.gamma).square()))  # modify attn by distance

        out = torch.zeros((1, 1, 1, 1)).to(qs.device)
        v_sum = torch.zeros_like(out)
        for i in range(self.n_inputs):
            v = self.values[i](vs[i]).view(B, T2, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)            
            # out = out + attn @ v
            v_sum = v_sum + v
        out = attn @ v_sum
        
        # output projection
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        return out


class PhysicsAttention(nn.Module):
    """
        Physics attention implementation (TranSolver)
    """

    def __init__(self, config):
        super(PhysicsAttention, self).__init__()
        assert config.n_embd % config.n_head == 0
        self.size_head = config.n_embd // config.n_head
        # key, query, value projections for all heads      
        self.query = nn.Linear(self.size_head, self.size_head)
        self.keys = nn.Linear(self.size_head, self.size_head)
        self.values = nn.Linear(self.size_head, self.size_head)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        # output projection
        self.out_proj = nn.Linear(config.n_embd, config.n_embd) 
        self.in_proj = nn.Linear(config.n_embd, config.n_embd)
        self.proj_slice = nn.Linear(self.size_head, config.n_slice)
        self.temperature = nn.Parameter(torch.ones([1, config.n_head, 1, 1]) / torch.sqrt(self.size_head * torch.ones(1)))
        self.n_head = config.n_head
        self.n_inputs = config.n_inputs
        self.gamma = config.gamma_attn
        self.eps = torch.exp(- torch.tensor(2.0).square())

        self.attn_type = 'l1'
        

    '''
        Physics Attention
    '''
    def forward(self, x_q, x_r, y, mask=None, dist=None, layer_past=None):
        q = x_q

        B, T, C = q.size() 

        # slice
        # qs = qs.view(B, T, self.n_head, C).transpose(1, 2)
        tmp_x = self.in_proj(q).view(B, T, self.n_head, self.size_head).transpose(1, 2)  # (B, nh, T, hs)    # xi
        slice_weights = (self.proj_slice(tmp_x) / self.temperature).softmax(dim=-1)   # B nh T S
        slice_norm = slice_weights.sum(2)  # B nh G
        slice = torch.einsum("bhnc,bhng->bhgc", tmp_x, slice_weights)
        token = slice / (slice_norm[:, :, :, None] + 1e-5)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(token) # (B, nh, S, hs)
        k = self.keys(token) # (B, nh, S, hs)
        v = self.values(token)  # (B, nh, S, hs)   
        attn = q @ k.transpose(-2, -1) / np.sqrt(self.size_head)    
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9) 
        attn = attn.softmax(dim=-1) 
        # q_norm = q.square().sum(dim=-1, keepdim=True) / np.sqrt(self.size_head)
        # k_norm = k.square().sum(dim=-1, keepdim=True) / np.sqrt(self.size_head)
        # attn = (2 * attn - q_norm - k_norm).exp() # exp(-|q-k|^2)
        if dist is not None:
            attn = attn * ( torch.exp(- (dist / self.gamma).square()))  # modify attn by distance
        out = attn @ v

        # output projection
        out = torch.einsum("b h g c, b h n g -> b h n c", out, slice_weights)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.out_proj(out)
        return out

# class PhysicsFourier(nn.Module):
#     """
#         Physics Fourier implementation.
#     """

#     def __init__(self, config):
#         super(PhysicsFourier, self).__init__()
#         assert config.n_embd % config.n_head == 0
#         self.size_head = config.n_embd // config.n_head
#         # output projection
#         self.out_proj = nn.Linear(config.n_embd, config.n_embd) 
#         self.in_proj = nn.Linear(config.n_embd, config.n_embd)
#         self.proj_slice = nn.Linear(self.size_head, config.n_slice)
#         # self.temperature = nn.Parameter(torch.ones([1, config.n_head, 1, 1]) * 0.5)
#         self.temperature = nn.Parameter(torch.ones([1, config.n_head, 1, 1]) / torch.sqrt(self.size_head * torch.ones(1)))

#         self.n_head = config.n_head
#         self.eps = torch.exp(- torch.tensor(2.0).square())

#         self.modes1 = config.modes
#         self.scale = (1 / (config.n_embd*config.n_embd))
#         # self.weights1 = nn.Parameter(self.scale * torch.rand(self.size_head, self.size_head, self.modes1, 2))
#         self.weights1 = nn.Parameter(self.scale * torch.rand(config.n_head, self.size_head, self.size_head, self.modes1, dtype=torch.cfloat))
        

#     '''
#         Physics Fourier
#     '''
#     def forward(self, x_q, x_r, y, mask=None, dist=None, layer_past=None):
#         q = x_q

#         B, T, C = q.size() 

#         # project all points onto slices
#         tmp_x = self.in_proj(q).view(B, T, self.n_head, self.size_head).transpose(1, 2)  # (B, nh, T, hs)    # xi
#         slice_weights = (self.proj_slice(tmp_x) / self.temperature).softmax(dim=-1)   # B nh T S
#         slice_norm = slice_weights.sum(2)  # B nh S
#         slice = torch.einsum("bhnc,bhns->bhsc", tmp_x, slice_weights)
#         z = slice / (slice_norm[:, :, :, None] + 1e-5)


#         z = z.transpose(2, 3).view(B, self.n_head, self.size_head, -1) # (B, nh, hs, S)
#         z_fft = torch.fft.rfft(z)

#         # Multiply relevant Fourier modes
#         out_ft = torch.zeros(B, self.n_head, self.size_head, z.size(-1)//2 + 1, device=z.device)
#         out = torch.einsum("bhix,hiox->bhox", z_fft[..., :self.modes1], self.weights1) 
#         # print('out.shape: ', out.shape)
#         out_ft[..., :self.modes1] = out

#         #Return to physical space
#         out = torch.fft.irfft(out_ft, n=z.size(-1)).view(B, self.n_head, self.size_head, -1)
#         out = out.transpose(2, 3) # (B, nh, S, hs)
#         # print('out.shape: ', out.shape)

#         # output projection
#         out = torch.einsum("b h g c, b h n g -> b h n c", out, slice_weights)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         out = self.out_proj(out)
#         return out


class VirtualFourier(nn.Module):
    """
        Virtual-Fourier implementation.
    """
    def __init__(self, config):
        super(VirtualFourier, self).__init__()
        assert config.n_embd % config.n_head == 0
        self.size_head = config.n_embd // config.n_head
        # output projection
        self.out_proj = nn.Linear(config.n_embd, config.n_embd) 
        self.in_proj = nn.Linear(config.n_embd, config.n_embd)
        self.proj_slice = nn.Linear(self.size_head, config.n_slice)
        self.temperature = nn.Parameter(torch.ones([1, config.n_head, 1, 1]) / torch.sqrt(self.size_head * torch.ones(1)))

        self.n_head = config.n_head
        self.eps = torch.exp(- torch.tensor(2.0).square())

        self.modes1 = config.modes
        self.scale = (1 / (config.n_embd*config.n_embd))
        self.weights1 = nn.Parameter(self.scale * torch.rand(config.n_head, self.size_head, self.size_head, self.modes1, dtype=torch.cfloat))  

    def forward(self, x_q, x_r, y, mask=None, dist=None, layer_past=None):
        q = x_q

        B, T, C = q.size() 

        # project all points onto slices
        tmp_x = self.in_proj(q).view(B, T, self.n_head, self.size_head).transpose(1, 2)  # (B, nh, T, hs)    # xi
        logits = self.proj_slice(tmp_x)   # B nh T S
        logits = (logits / self.temperature)

        z = torch.einsum("bhnc,bhns->bhsc", tmp_x, logits.softmax(dim=-2))
        z = z.transpose(2, 3).view(B, self.n_head, self.size_head, -1) # (B, nh, hs, S)
        z_fft = torch.fft.rfft(z)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(B, self.n_head, self.size_head, z.size(-1)//2 + 1, device=z.device)
        out = torch.einsum("bhix,hiox->bhox", z_fft[..., :self.modes1], self.weights1) 
        out_ft[..., :self.modes1] = out

        #Return to physical space
        out = torch.fft.irfft(out_ft, n=z.size(-1)).view(B, self.n_head, self.size_head, -1)
        out = out.transpose(2, 3) # (B, nh, S, hs)

        # output projection
        out = torch.einsum("b h g c, b h n g -> b h n c", out, logits.softmax(dim=-1))
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.out_proj(out)
        return out

'''
    Self and Cross Attention block for CGPT, contains  a cross attention block and a self attention block
'''
class CrossAttentionBlock(nn.Module):
    def __init__(self, config):
        super(CrossAttentionBlock, self).__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2_branch = nn.ModuleList([nn.LayerNorm(config.n_embd) for _ in range(config.n_inputs)])
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.n_inputs = config.n_inputs
        self.ln3 = nn.LayerNorm(config.n_embd)

        if config.attn_type == 'linear':
            print('Using Linear Attention')
            self.crossattn = LinearCrossAttention(config)
        elif config.attn_type == 'physics':
            print('Using Physics Attention')
            self.crossattn = PhysicsAttention(config)
        elif config.attn_type == 'physicsfourier':
            print('Using Physics Fourier')
            self.crossattn = PhysicsFourier(config)
        elif config.attn_type == 'virtualfourier':
            print('Using Virtual Fourier')
            self.crossattn = VirtualFourier(config)
        else:
            print('Using Nonlinear Attention')
            self.crossattn = CrossAttention(config)

        if config.act == 'gelu':
            self.act = GELU
        elif config.act == "tanh":
            self.act = Tanh
        elif config.act == 'relu':
            self.act = ReLU
        elif config.act == 'sigmoid':
            self.act = Sigmoid

        self.resid_drop1 = nn.Dropout(config.resid_pdrop)
        # self.mlp1 = nn.Sequential(
        #     nn.Linear(config.n_embd, config.n_inner),
        #     self.act(),
        #     nn.Linear(config.n_inner, config.n_embd),            
        # )
        self.linear = nn.Linear(config.n_embd, config.n_embd)
        self.act = self.act()


    def ln_branchs(self, y):
        if isinstance(y, MultipleTensors):
            out = MultipleTensors([self.ln2_branch[i](y[i]) for i in range(self.n_inputs)])
            # print('self.n_inputs, len(y): ', self.n_inputs, len(y))
        else:
            raise ValueError('y is not a MultipleTensors.')
            # out = MultipleTensors([self.ln4(y)])

        return out


    def forward(self, x_1, x_2, y, mask=None, dist=None):
        x = self.resid_drop1(self.crossattn(self.ln1(x_1), 
                                                self.ln2(x_2),
                                                self.ln_branchs(y),
                                                # mask=mask, 
                                                dist=dist
                                                )
                                    )
        # x = x + self.mlp1(self.ln3(x))
        # x = self.mlp1(self.ln3(x))
        x = self.act(self.linear(x) + self.ln3(x))

        return x



class RNO(nn.Module):
    def __init__(self,
                 trunk_size=2,
                 branch_sizes=None,
                 space_dim=2,
                 output_size=3,
                 n_layers=2,
                 n_hidden=64,
                 n_head=1,
                 n_experts = 2,
                 n_slice = 128,
                 modes = 8,
                 n_inner = 4,
                 mlp_layers=2,
                 attn_type='linear',
                 gamma=0.3,
                 act = 'gelu',
                 ffn_dropout=0.0,
                 attn_dropout=0.0,
                 horiz_fourier_dim = 0,
                 ):
        super(RNO, self).__init__()

        self.horiz_fourier_dim = horiz_fourier_dim
        self.trunk_size = trunk_size * (4*horiz_fourier_dim + 3) if horiz_fourier_dim>0 else trunk_size
        # self.branch_sizes = [bsize * (4*horiz_fourier_dim + 3) for bsize in branch_sizes] if horiz_fourier_dim > 0 else branch_sizes
        self.branch_sizes = branch_sizes
        # self.n_inputs = len(self.branch_sizes)
        # self.n_inputs = 3
        self.n_inputs = 1 # sum all input functions as one input
        self.output_size = output_size
        self.space_dim = space_dim

        

        self.trunk_mlp = MLP(self.trunk_size, n_hidden, n_hidden, n_layers=mlp_layers,act=act)
        self.branch_mlps = nn.ModuleList([MLP(bsize, n_hidden, n_hidden, n_layers=mlp_layers,act=act) for bsize in self.branch_sizes])
        # self.P_mlp = MLP(n_hidden * 2 + 6, n_hidden, n_hidden, n_layers=mlp_layers)

        self.config = GPTConfig(attn_type=attn_type,embd_pdrop=ffn_dropout, resid_pdrop=ffn_dropout, 
                                           attn_pdrop=attn_dropout,n_embd=n_hidden, n_head=n_head,
                                            block_size=128,act=act, branch_sizes=branch_sizes,
                                            gamma_attn=gamma, gamma_phi=0.1, n_slice = n_slice,
                                            n_inputs=self.n_inputs, n_inner=n_inner, modes=modes) 
        self.blocks = nn.Sequential(*[CrossAttentionBlock(self.config) for _ in range(n_layers)])        

        self.out_mlp = MLP(n_hidden, n_hidden, output_size, n_layers=mlp_layers)
        # self.apply(self._init_weights)

        self.__name__ = 'RNO'



    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.0002)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)



    def forward(self, gs, gs_r, u_p, inputs, no_ref=0.0):
        device = gs[0].device
        x_q = [_g.ndata['x'] for _g in gs]
        # Padding all inputs
        x = pad_sequence(x_q).permute(1, 0, 2).float().to(device)   # B, T_q, F
        x_r = [_g.ndata['x'] for _g in gs_r]  
        # Padding all inputs
        x_r = pad_sequence(x_r).permute(1, 0, 2).float().to(device)   # B, T_q, F
        assert x.shape == x_r.shape, f'x shape={x.shape} and x_r shape={x_r.shape} does not match.' 
        
        y_r = [_g.ndata['y'] for _g in gs_r] 
        y_r = pad_sequence(y_r).permute(1, 0, 2).float().to(device) # B, T_q, dim_y
        y_r = y_r[..., :self.output_size] # If y includes sensitivity, remove it     
        
        delta_a = x - x_r

        # # Fourier embedding
        if self.horiz_fourier_dim > 0:
            x = horizontal_fourier_embedding(x, self.horiz_fourier_dim)     
        x = self.trunk_mlp(x)      


        ########## CrossBlock #######
        z = [x_r, y_r, delta_a]
        z = [self.branch_mlps[i](z[i]) for i in range(len(z))] 
        
        if no_ref > 0:
            if torch.rand(1) < no_ref:
                # print('Drop reference.')
                z = [] # w/o reference
        
        x = torch.stack([x] + z, dim=0) # Pure MLP
        x = x.sum(dim=0)

        dist_x = None # No Distance-Aware (DA)
        for i, block in enumerate(self.blocks):
            y = MultipleTensors([x]) # sum all functions before attn layers 
            x = x + block(x, x, y, dist=dist_x) # query-key uses x
            # x = block(x, x, y, dist=dist_x)


        x = self.out_mlp(x)  # No delta_u

        x_out = torch.cat([x[i, :len(x_)] for i, x_ in enumerate(x_q)],dim=0)
        y_ref = torch.cat([y_r[i, :len(x_), :] for i, x_ in enumerate(x_q)],dim=0)


        return x_out, y_ref


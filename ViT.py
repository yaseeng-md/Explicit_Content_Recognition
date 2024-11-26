import einops
from einops import repeat
import torch 
from torchvision.datasets import OxfordIIITPet
from random import random
from torchvision.transforms import Resize, ToTensor
from torchvision.transforms.functional import to_pil_image
to_tensor = [Resize((244,244)),ToTensor()]
from torch import Tensor
from einops.layers.torch import Rearrange
from torch import nn


# Patch Embedding
class patch_Embedding(nn.Module):
    def __init__(self,in_channels,emb_size = 128,patch_size = 8):
        super().__init__()
        self.in_channels = in_channels
        self.emb_size = emb_size
        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',p1 = patch_size, p2 = patch_size),
            nn.Linear(in_features = (patch_size * patch_size * in_channels), out_features = self.emb_size)
        )
    def forward(self,x : Tensor):
        x = self.projection(x)
        return x


# Self Attension Mechnaism 
class Attension(nn.Module):
    def __init__(self,dim,n_heads,dropout):
        super().__init__()
        self.n_heads = n_heads
        self.attension = nn.MultiheadAttention(embed_dim=dim,num_heads=self.n_heads,dropout=dropout)
        self.q = nn.Linear(dim,dim)
        self.k = nn.Linear(dim,dim)
        self.v = nn.Linear(dim,dim)
    def forward(self,x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attention_output, attention_weight = self.attension(q,k,v)
        return attention_output

# -------- > Normalization < --------
class PreNorm(nn.Module):
    def __init__(self,dim,fn):
        super().__init__()
        self.fn = fn
        self.dim = dim
        self.norm = nn.LayerNorm(self.dim)
    def forward(self,x,**kwargs):
        return self.fn(self.norm(x),**kwargs)

# ----------> Feed Forward <-----------
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


# ----------> Residual Block <-----------
class ResidualAdd(nn.Module):
    def __init__(self,fn):
        super().__init__()
        self.fn = fn
    def forward(self,x,**kwargs):
        res = x
        x = self.fn(x,**kwargs)
        x += res
        return x

# ---------> Transformer Block <-----------
class ViT(nn.Module):
    def __init__(self,ch=3,img_size = 224, patch_size = 16, emb_size = 32, n_layers = 7, output_dim = 1, dropout = 0.1, heads = 2):
        super(ViT,self).__init__()

        # Attributes
        self.channels = ch
        self.img_size = img_size
        self.width = img_size
        self.width = img_size
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.dropout = dropout
        self.n_heads = heads

        # Patching
        self.patch_embedding = patch_Embedding(in_channels=self.channels, emb_size = self.emb_size, patch_size = self.patch_size)

        # Patching + Positional Embedding + CLS Token
        num_patches = (self.img_size // self.patch_size) ** 2
            
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, self.emb_size))
        self.cls_token = nn.Parameter(torch.rand(1,1,self.emb_size))

        # Transformer Encoders
        self.layers = nn.ModuleList([])
        for _ in range (self.n_layers):
            transformer_block = nn.Sequential(
                ResidualAdd(PreNorm(self.emb_size, Attension(emb_size, n_heads=self.n_heads, dropout = self.dropout))),
                ResidualAdd(PreNorm(self.emb_size, FeedForward(self.emb_size,self.emb_size, dropout = self.dropout)))
            )
            self.layers.append(transformer_block)

        # Classification Head
        self.head = nn.Sequential(nn.LayerNorm(self.emb_size), nn.Linear(emb_size, output_dim))
    
    
    
    def forward(self,img):
        x = self.patch_embedding(img)
        b, n, _ = x.shape
        cls_token = repeat(self.cls_token, "1 1 d -> b 1 d", b = b)
        x = torch.cat((cls_token,x),dim = 1)
        x += self.pos_embedding[:,:(n+1)]

        # Transformer Layers
        for i in range (self.n_layers):
            x = self.layers[i](x)
        return self.head(x[:,0,:])

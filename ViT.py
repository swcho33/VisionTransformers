import torch
from torch import nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

class MSA(nn.Module):
    def __init__(self):
        pass
    
    def forward(self, x):
        pass


class PatchEmbedding(nn.Module):
    def __init__(self, in_chs=3, p_size=4, emb_size = 128, img_size = 32):
        super().__init__()
        self.patch_size = p_size
        self.projection = nn.Sequential(
            nn.Conv2d(in_chs, emb_size, kernel_size=p_size, stride=p_size),
            Rearrange('b e (h) (w) -> b (h w) e')
        )
        self.cls_token = nn.Parameter(torch.rand(1,1,emb_size))
        self.positions = nn.Parameter(torch.rand((img_size//p_size) **2+1, emb_size))
        
    
    def forward(self, x):
        b, _,_,_ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b = b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions
        
        return x

class TransformerEncoder(nn.Module):
    def __init__(self):
        pass
    
    def forward(self, x):
        pass
  

class MLPHead(nn.Module):
    def __init__(self):
        pass
    
    def forward(self, x):
        pass
    
    
if __name__ == "__main__":
    x_tensor = torch.randn(8,3,32,32)
    patch_embedding = PatchEmbedding(3, 4, 128, 32)
    x_out = patch_embedding(x_tensor)
    print(x_out.shape)
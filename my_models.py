import math

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# ViT
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# ViT & CrossViT
# PreNorm class for layer normalization before passing the input to another function (fn)
class PreNorm(nn.Module):    
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# ViT & CrossViT
# FeedForward class for a feed forward neural network consisting of 2 linear layers, 
# where the first has a GELU activation followed by dropout, then the next linear layer
# followed again by dropout
class FeedForward(nn.Module):       #for MLP
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# ViT & CrossViT
# Attention class for multi-head self-attention mechanism with softmax and dropout
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        # set heads and scale (=sqrt(dim_head))
        
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.scaled = math.sqrt(dim_head)
        self.dropout = dropout
        self.wq = nn.Linear(dim, dim*heads)
        self.wk = nn.Linear(dim, dim*heads)
        self.wv = nn.Linear(dim, dim*heads)
        self.attn_drop = nn.Dropout(dropout)

        # as well as the q linear layer
       

        # and the k/v linear layer (can be realized as one single linear layer
        # or as two individual ones)
       

        # and the output linear layer followed by dropout
        

        self.out_linear_layer = nn.Sequential(nn.Linear(dim_head * heads, dim),
                                              nn.Dropout(dropout))      # 16,17,512 -> 16, 17, 64

        # we need softmax layer and dropout
        
        self.attn_drop = nn.Dropout(dropout)


    def forward(self, x, context = None, kv_include_self = False):
        # now compute the attention/cross-attention
        # in cross attention: x = class token, context = token embeddings
        # don't forget the dropout after the attention 
        # and before the multiplication w. 'v'
        # the output should be in the shape 'b n (h d)'
        b, n, _, h = *x.shape, self.heads
        if context is None:
            context = x

        if kv_include_self:
            # cross attention requires CLS token to be included itself as key / value and only that CLS token as query
            context = torch.cat((x, context), dim = 1) 
        
        # Attention

        q= self.wq(x).reshape(b, n, self.heads, self.dim).permute(0, 2, 1, 3)    # 16, 17, 512 -> 16, 17, 8, 64 -> 16, 8, 17, 64
        k= self.wk(context).reshape(b, n, self.heads, self.dim).permute(0, 2, 1, 3) # provision for cross ViT
        v= self.wv(context).reshape(b, n, self.heads, self.dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scaled
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(b, n, h * self.dim_head)  # 16,8,17,64 -> 16,17,512  concatenates multi-head o/p

        out = x + self.out_linear_layer(attn)   # 16, 17, 64
        #print(f' shape of x:' ,{x.shape})
        return out


# ViT & CrossViT
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# CrossViT
# projecting CLS tokens, in the case that small and large patch tokens have different dimensions
class ProjectInOut(nn.Module):
    """
    Adapter class that embeds a callable (layer) and handles mismatching dimensions
    """
    def __init__(self, dim_outer, dim_inner, fn):
        """
        Args:
            dim_outer (int): Input (and output) dimension. -> CLS dim of original branch
            dim_inner (int): Intermediate dimension (expected by fn). -> CLS dim adapted to the compared branch
            fn (callable): A callable object (like a layer). = Attention block
        """
        super().__init__()
        self.fn = fn
        need_projection = dim_outer != dim_inner
        self.project_in = nn.Linear(dim_outer, dim_inner) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_inner, dim_outer) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):
        """
        Args:
            *args, **kwargs: to be passed on to fn

        Notes:
            - after calling fn, the tensor has to be projected back into it's original shape   
            - fn(W_in) * W_out
        """
        

        x = self.project_out(self.fn(self.project_in(x))+x)

        return x

# CrossViT
# cross attention transformer x L   # depth = L
class CrossTransformer(nn.Module):
    # This is a special transformer block
    def __init__(self, sm_dim, lg_dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList([])

        # create # depth encoders using ProjectInOut
        for _ in range(depth):
            self.layers.append(nn.ModuleList([ProjectInOut(sm_dim, lg_dim, Attention(lg_dim, heads, dim_head, dropout)),
                                              ProjectInOut(lg_dim, sm_dim, Attention(sm_dim, heads, dim_head, dropout)),
                                              ]))
                                                 #for storing all the encrypted classes
        # Note: no positional FFN here

    def forward(self, sm_tokens, lg_tokens):
        (sm_cls, sm_patch_tokens), (lg_cls, lg_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]), (sm_tokens, lg_tokens))


        for sm_lg, lg_sm in self.layers:     # Forward pass through the layers,
                                                            # cross attend to
            sm_cls = sm_lg(sm_cls, lg_patch_tokens)     # 1. small cls token to large patches and
            lg_cls = lg_sm(lg_cls, sm_patch_tokens)     # 2. large cls token to small patches


        
        sm_tokens = torch.cat((sm_cls, sm_patch_tokens), dim=-1)  # finally concat sm/lg cls tokens with patch tokens
        lg_tokens = torch.cat((lg_cls,lg_patch_tokens), dim=-1)

        return sm_tokens, lg_tokens

# CrossViT
# multi-scale encoder   x K depth
class MultiScaleEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth,
        sm_dim,
        lg_dim,
        sm_enc_params,
        lg_enc_params,
        cross_attn_heads,
        cross_attn_depth,
        cross_attn_dim_head = 64,
        dropout = 0.
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        temp_sm = sm_enc_params.copy()
        temp_sm['dim'] =128

        temp_lg = lg_enc_params.copy()
        temp_lg['dim'] = 128

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(128, temp_sm['depth'], temp_sm['heads'], temp_sm['dim_head'], temp_sm['mlp_dim']),
                Transformer(128, temp_lg['depth'], temp_lg['heads'], temp_lg['dim_head'], temp_lg['mlp_dim']),# 2 transformer branches, one for small, one for large patches
                CrossTransformer(sm_dim, lg_dim, cross_attn_depth, cross_attn_heads, cross_attn_dim_head,dropout)# + 1 cross transformer block
            ]))

    def forward(self, sm_tokens, lg_tokens):
        # forward through the transformer encoders and cross attention block
        for sm_tf, lg_tf, cross_tf in self.layers:
            sm_tokens = sm_tf(sm_tokens)
            lg_tokens = lg_tf(lg_tokens)
            sm_tokens, lg_tokens = cross_tf(sm_tokens, lg_tokens)


        return sm_tokens, lg_tokens

# CrossViT (could actually also be used in ViT)
# helper function that makes the embedding from patches
# have a look at the image embedding in ViT
class ImageEmbedder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        image_size,
        patch_size,
        dropout = 0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2
        patch_height, patch_width = pair(patch_size)

        # create layer that re-arranges the image patches
        # and embeds them with layer norm + linear projection + layer norm 
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )
        # create/initialize #dim-dimensional positional embedding (will be learned)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))     
        # create #dim cls tokens (for each patch embedding)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))   
        # create dropput layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, img):
        x = self.to_patch_embedding(img)     # forward through patch embedding layer
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)  # b = batch = no. of images
        x = torch.cat((cls_tokens, x), dim=1)   # concat class tokens
        x += self.pos_embedding[:, :(n + 1)]        # and add positional embedding

        return self.dropout(x)      # x -> (16,17,64)


# normal ViT
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # initialize patch embedding
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),      # 16x16x64
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))   #'E' ->.Parameter instructs PyTorch to treat the tensor as learnable
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # create transformer blocks
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()      # keeps a copy of output as a residual to be added later as skip connection

        # create mlp head (layer norm + linear layer)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # patch embedding
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # concat class token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)    #b = batch = no. of images
        x = torch.cat((cls_tokens, x), dim=1)
        # add positional embedding
        x += self.pos_embedding[:, :(n + 1)]
        # apply dropout
        x = self.dropout(x)

        # forward through the transformer blocks
        x = self.transformer(x)

        # decide if x is the mean of the embedding 
        # or the class token
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        # transfer via an identity layer the cls tokens or the mean
        x = self.to_latent(x)   # to a latent space, which can then be used as input
        return self.mlp_head(x)     # to the mlp head


# CrossViT
class CrossViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        num_classes,
        sm_dim,
        lg_dim,
        sm_patch_size=12,
        sm_enc_depth=1,
        sm_enc_heads=8,
        sm_enc_mlp_dim=2048,
        sm_enc_dim_head=64,
        lg_patch_size=16,
        lg_enc_depth=4,
        lg_enc_heads=8,
        lg_enc_mlp_dim=2048,
        lg_enc_dim_head=64,
        cross_attn_depth=2,
        cross_attn_heads=8,
        cross_attn_dim_head=64,
        depth=3,
        dropout=0.1,
        emb_dropout=0.1
    ):
        super().__init__()
        # create ImageEmbedder for small and large patches
        self.sm_img_token = ImageEmbedder(dim=sm_dim, image_size=image_size, patch_size=sm_patch_size, dropout=emb_dropout)# TODO
        self.lg_img_token = ImageEmbedder(dim=lg_dim, image_size=image_size, patch_size=lg_patch_size, dropout=emb_dropout)
        #print(sm_enc_depth)
        # create MultiScaleEncoder
        self.multi_scale_encoder = MultiScaleEncoder(
            depth=depth,
            sm_dim=sm_dim,
            lg_dim=lg_dim,
            cross_attn_heads=cross_attn_heads,
            cross_attn_dim_head=cross_attn_dim_head,
            cross_attn_depth=cross_attn_depth,
            sm_enc_params=dict(
                depth=sm_enc_depth,
                heads=sm_enc_heads,
                mlp_dim=sm_enc_mlp_dim,
                dim_head=sm_enc_dim_head
            ),
            lg_enc_params=dict(
                depth=lg_enc_depth,
                heads=lg_enc_heads,
                mlp_dim=lg_enc_mlp_dim,
                dim_head=lg_enc_dim_head
            ),
            dropout=dropout
        )

        # create mlp heads for small and large patches
        self.sm_mlp_head = nn.Sequential(nn.LayerNorm(sm_dim), nn.Linear(sm_dim, num_classes))
        self.lg_mlp_head = nn.Sequential(nn.LayerNorm(lg_dim), nn.Linear(lg_dim, num_classes))

    def forward(self, img):
        # apply image embedders
        sm_tokens = self.sm_img_token(img)     # TODO
        lg_tokens = self.lg_img_token(img)

        # and the multi-scale encoder
        sm_token, lg_token = self.multi_scale_encoder(sm_tokens, lg_tokens) # TODO

        # call the mlp heads w. the class tokens 
        sm_logits = self.sm_mlp_head(sm_token)  # TODO
        lg_logits = self.lg_mlp_head(lg_token)
        
        return sm_logits + lg_logits


if __name__ == "__main__":
    x = torch.randn(16, 3, 32, 32)
    vit = ViT(image_size = 32, patch_size = 8, num_classes = 10, dim = 64, depth = 2, heads = 8, mlp_dim = 128, dropout = 0.1, emb_dropout = 0.1)
    cvit = CrossViT(image_size = 32, num_classes = 10, sm_dim = 64, lg_dim = 128, sm_patch_size = 8,
                    sm_enc_depth = 2, sm_enc_heads = 8, sm_enc_mlp_dim = 128, sm_enc_dim_head = 64,
                    lg_patch_size = 16, lg_enc_depth = 2, lg_enc_heads = 8, lg_enc_mlp_dim = 128,
                    lg_enc_dim_head = 64, cross_attn_depth = 2, cross_attn_heads = 8, cross_attn_dim_head = 64,
                    depth = 3, dropout = 0.1, emb_dropout = 0.1)
    #print(vit(x).shape)
    #print(cvit(x).shape)

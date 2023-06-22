import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model
from timm.models.vision_transformer import Mlp, DropPath, _cfg
from timm.models.layers import lecun_normal_, trunc_normal_, to_2tuple
from timm.models.helpers import named_apply
from functools import partial
from collections import OrderedDict
import torch.utils.checkpoint as checkpoint
import numpy as np


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.get_v = nn.Conv2d(head_dim, head_dim, kernel_size=3, stride=1, padding=1,groups=head_dim)
        nn.init.zeros_(self.get_v.weight)
        nn.init.zeros_(self.get_v.bias)
        
    def get_local_pos_embed(self, x):
        B, _, N, C = x.shape
        H = W = int(np.sqrt(N-1))
        x = x[:, :, 1:].transpose(-2, -1).contiguous().reshape(B * self.num_heads, -1, H, W)
        local_pe = self.get_v(x).reshape(B, -1, C, N-1).transpose(-2, -1).contiguous() # B, H, N-1, C
        local_pe = torch.cat((torch.zeros((B, self.num_heads, 1, C), device=x.device), local_pe), dim=2) # B, H, N, C
        return local_pe
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        local_pe = self.get_local_pos_embed(v) 
        x = ((attn @ v + local_pe)).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
    
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
    
    def flops(self):
        Ho, Wo = self.grid_size
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class GlobalPosEmbed(nn.Module):
    def __init__(self, embed_dim, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.embed_dim = embed_dim // 2
        self.normalize = normalize
        self.temperature = temperature
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.embed_layer = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding= 1, groups = embed_dim)
        
    def forward(self, x):
        b, n, c = x.shape
        patch_n = int((n-1) ** 0.5)
        not_mask = torch.ones((b, patch_n, patch_n), device = x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.embed_dim, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.embed_dim)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2) # B, C, H, W
        pos = self.embed_layer(pos).reshape(b, c, -1).transpose(1, 2)
        pos_cls = torch.zeros((b, 1, c), device = x.device)
        pos =  torch.cat((pos_cls, pos),dim=1)
        return pos + x
    

class ResFormer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size= (224, ), patch_size= 16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', use_checkpoint = False):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size)
            
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.use_checkpoint = use_checkpoint
        
        self.patch_size = patch_size
    
        self.patch_embed = embed_layer(img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches
        self.pos_embed = GlobalPosEmbed(embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
    
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads,mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
          
        self.depth = depth
        
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()
        
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_res = None
        self.init_weights(weight_init)

    def forward_features(self, x, distillation_target = 'logit'):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(self.pos_embed(x))

        if self.use_checkpoint:
            for i  in range(self.depth):
                x = checkpoint.checkpoint(self.blocks[i], x)
        else:
            x = self.blocks(x)
            
        x = self.norm(x)
        
        if distillation_target == 'gap':
            return  self.pre_logits(x[:, 0]), F.adaptive_avg_pool1d(x[:, self.num_tokens:].transpose(1,2), (1,)).flatten(1).squeeze(-1)
        else:
            return self.pre_logits(x[:, 0]), None
        
    
    def forward(self, x, distillation_target = 'logit'):
        x_distill = None
        x, x_distill = self.forward_features(x, distillation_target)
        
        if distillation_target == 'cls' :
            x_distill = x
    
        x = self.head(x)
        
        if distillation_target == 'logit' :
            x_distill = x
        
        if self.training:
            return x, x_distill

        else:
            return x
            
    
    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            if self.cls_token != None:
                trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    
    @torch.jit.ignore
    def no_weight_decay(self):
        no_weight_decay_params = set()
        # keywords = {'cls_token'}
        keywords = {'pos_embed', 'get_v', 'cls_token'}
        for name, param in self.named_parameters():
            if param.requires_grad:
                for key in keywords:
                    if key in name:
                        no_weight_decay_params.add(name)
                        break
        return no_weight_decay_params

def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)

def detach_module(module):
    for param in module.parameters():
        param.requires_grad = False
        

@register_model
def resformer_tiny_patch16(pretrained=False, **kwargs):
    model = ResFormer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def resformer_small_patch16(pretrained=False, **kwargs):
    model = ResFormer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def resformer_base_patch16(pretrained=False, **kwargs):
    model = ResFormer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model



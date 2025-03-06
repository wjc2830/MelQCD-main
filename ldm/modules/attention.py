from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any

from ldm.modules.diffusionmodules.util import checkpoint


try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

# CrossAttn precision handling
import os
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")


def sinusoidal_positional_embedding(token_sequence_size, token_embedding_dim, n=10000.0):

    if token_embedding_dim % 2 != 0:
        raise ValueError("Sinusoidal positional embedding cannot apply to odd token embedding dim (got dim={:d})".format(token_embedding_dim))

    T = token_sequence_size
    d = token_embedding_dim #d_model=head_num*d_k, not d_q, d_k, d_v

    positions = torch.arange(0, T).unsqueeze_(1)
    embeddings = torch.zeros(T, d)

    denominators = torch.pow(n, 2*torch.arange(0, d//2)/d) # 10000^(2i/d_model), i is the index of embedding
    embeddings[:, 0::2] = torch.sin(positions/denominators) # sin(pos/10000^(2i/d_model))
    embeddings[:, 1::2] = torch.cos(positions/denominators) # cos(pos/10000^(2i/d_model))

    return embeddings


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)



class CrossAttention_Video_time_aware(nn.Module):
    def __init__(self, query_dim, context_dim=None, 
        heads=8, dim_head=64, dropout=0., video_dim=None):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        video_dim = default(video_dim, 1024)
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.query_dim = query_dim
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(video_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(video_dim, inner_dim, bias=False)
        
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

        self.map = {4096: [128, 32], 1024: [64, 16], 256: [32, 8], 64: [16, 4]}
        # self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)))

    def forward(self, x, context=None, mask=None, hint_control=None):
        # context is torch.Size([4, 77, 768]), hint_control is torch.Size([4, 150, 1024])
        W, H = self.map[x.shape[1]]
        # H W -> W H
        x = rearrange(x, 'b (h w) c -> b (w h) c', w=W, h=H)

        h = self.heads
        q = self.to_q(x)

        mask_j = 1 - (hint_control.sum(dim=2) == 0) * 1.0 # 4, 150
        mask_j = mask_j.unsqueeze(1).float()# 4， 1， 150
        
        mask_i = F.interpolate(mask_j, size=(W,), mode='linear') # 4， 1，128
        mask_i = (mask_i > 0.5).float()
        # if W == 128:
        #     f_control_sample_vis = mask_i.unsqueeze(-1).repeat(1, 256, 1, 1).float()
        #     f_control_sample_vis[f_control_sample_vis==0]=1*(-1)
        #     f_control_sample_vis = einops.rearrange(f_control_sample_vis, 'b h w c -> b c h w').clone()
        #     torchvision.utils.save_image(f_control_sample_vis, 'bin_128_vis_att.png')
        mask_i = torch.repeat_interleave(mask_i, H, dim=-1) # 0,1,0->0,0,1,1,0,0    0,1,0,0,1,0  128 * 32=4096
        # if W == 128:
        #     f_control_sample_vis = mask_i.unsqueeze(-1).repeat(1, 256, 1, 1).float()
        #     f_control_sample_vis[f_control_sample_vis==0]=1*(-1)
        #     f_control_sample_vis = einops.rearrange(f_control_sample_vis, 'b h w c -> b c h w').clone()
        #     torchvision.utils.save_image(f_control_sample_vis, 'bin_128_vis_att_repeat.png')

        mask = einsum('b d i, b d j -> b i j', mask_i, mask_j)
        mask = torch.repeat_interleave(mask, h, dim=0)
        
        hint_control = default(hint_control, x)
        k = self.to_k(hint_control)
        v = self.to_v(hint_control)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # torch.Size([32, 256, 64]) torch.Size([32, 77, 64]) torch.Size([32, 77, 64])
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale  # i: 

        # mask
        mask_sim = torch.zeros_like(sim)
        mask_sim[mask==0] = -1e9 
        sim = sim + mask_sim
        
        del q, k
    
        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        out = self.to_out(out)
        out = rearrange(out, 'b (w h) c -> b (h w) c', w=W, h=H)
        return out
        # return torch.tanh(self.alpha_attn) * out


class CrossAttention_Video_Partially(nn.Module):
    def __init__(self, query_dim, context_dim=None, 
        heads=8, dim_head=64, dropout=0., video_dim=None):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        video_dim = default(video_dim, 1024)
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.query_dim = query_dim
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(video_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(video_dim, inner_dim, bias=False)
        
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

        self.w_size = 4
        self.pe = sinusoidal_positional_embedding(300, video_dim)

    def forward(self, x, context=None, mask=None, hint_control=None):
        # context is torch.Size([4, 77, 768]), hint_control is torch.Size([4, 150, 1024])
        h = self.heads
        b = x.shape[0]
        q = self.to_q(x)

        # hint_control_global = torch.mean(hint_control, dim=1).unsqueeze(1)
        # hint_control = torch.cat([hint_control, hint_control_global], dim=1)
        hint_control = hint_control + self.pe[:hint_control.shape[1], :].unsqueeze(0).to(hint_control.device)
        
        hint_control = default(hint_control, x)
        k = self.to_k(hint_control)
        v = self.to_v(hint_control)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        # print(f'===> q.shape is {q.shape}')
        # print(f'===> k.shape is {k.shape}')

        # torch.Size([32, 256, 64]) torch.Size([32, 77, 64]) torch.Size([32, 77, 64])
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale  # i: 
        
        seg_len = q.shape[1]
        mask_one = torch.ones(q.shape[1], k.shape[1]).to(hint_control.device)
        for i in range(self.w_size):
            mask_one[seg_len*i//self.w_size:seg_len*(i+1)//self.w_size, k.shape[1]*i//self.w_size:k.shape[1]*(i+1)//self.w_size] = 0

        mask = mask_one.unsqueeze(0).repeat(b * h, 1, 1)
        mask[mask==1] = -1e9 
        sim = sim + mask
        
        del q, k
    
        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        out = self.to_out(out)
        return out


class CrossAttention_Video_PE(nn.Module):
    def __init__(self, query_dim, context_dim=None, 
        heads=8, dim_head=64, dropout=0., video_dim=None):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        video_dim = default(video_dim, 1024)
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.query_dim = query_dim
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(video_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(video_dim, inner_dim, bias=False)
        
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

        self.pe = sinusoidal_positional_embedding(300, video_dim)

    def forward(self, x, context=None, mask=None, hint_control=None):
        # context is torch.Size([4, 77, 768]), hint_control is torch.Size([4, 150, 1024])
        h = self.heads
        q = self.to_q(x)

        # print(f'====> pe.shape is {self.pe[:hint_control.shape[0], :].unsqueeze(0).shape}')
        # print(f'====> hint_control.shape is {hint_control.shape}')
        hint_control = hint_control + self.pe[:hint_control.shape[1], :].unsqueeze(0).to(hint_control.device)
        
        hint_control = default(hint_control, x)
        k = self.to_k(hint_control)
        v = self.to_v(hint_control)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # torch.Size([32, 256, 64]) torch.Size([32, 77, 64]) torch.Size([32, 77, 64])
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale  # i: 
        
        del q, k
    
        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        out = self.to_out(out)
        return out


class CrossAttention_Video(nn.Module):
    def __init__(self, query_dim, context_dim=None, 
        heads=8, dim_head=64, dropout=0., video_dim=None):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        video_dim = default(video_dim, 1024)
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.query_dim = query_dim
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(video_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(video_dim, inner_dim, bias=False)
        
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None, hint_control=None):
        # context is torch.Size([4, 77, 768]), hint_control is torch.Size([4, 150, 1024])
        h = self.heads
        q = self.to_q(x)
        
        hint_control = default(hint_control, x)
        k = self.to_k(hint_control)
        v = self.to_v(hint_control)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # torch.Size([32, 256, 64]) torch.Size([32, 77, 64]) torch.Size([32, 77, 64])
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale  # i: 
        
        del q, k
    
        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        out = self.to_out(out)
        return out


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., video_dim=None):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None, hint_control=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        del q, k
    
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention,
        "softmax-VideoCA": CrossAttention_Video_PE # not used
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False, attn_mode="softmax", video_dim=1024):
        super().__init__()
        # attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        # assert attn_mode in self.ATTENTION_MODES
        # attn_cls = self.ATTENTION_MODES[attn_mode]

        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None,
                              video_dim=video_dim)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim, video_dim=video_dim
                              )
        if attn_mode == "softmax-VideoCA":
            self.attn3 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                                video_dim=video_dim
                                )
            self.norm4 = nn.LayerNorm(dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None, hint_control=None):
        return checkpoint(self._forward, (x, context, hint_control), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, hint_control=None):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None, hint_control=None) + x
        x = self.attn2(self.norm2(x), context=context, hint_control=None) + x
        if hint_control.size() != (2, 2):
            x = self.attn3(self.norm4(x), context=context, hint_control=hint_control) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True, attn_mode="softmax", video_dim=1024):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint, attn_mode=attn_mode, video_dim=video_dim)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None, hint_control=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
            hint_control = [hint_control]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i], hint_control=hint_control[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


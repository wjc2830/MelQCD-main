import torch
from torch import nn, einsum
from torch.nn import functional as F
import numpy as np
from einops import rearrange, repeat


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

    
class RelativePositionBias(nn.Module):
    """ from https://arxiv.org/abs/2111.09883 """

    def __init__(
        self,
        *,
        dim,
        heads,
        layers = 3
    ):
        super().__init__()
        self.net = nn.ModuleList([])
        self.net.append(nn.Sequential(nn.Linear(1, dim), nn.SiLU()))

        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), nn.SiLU()))

        self.net.append(nn.Linear(dim, heads))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, i, j, usi=1, usj=1):
        usr = int(max(usi, usj))
        seql = int(i / usi)
        assert i / usi == seql
        assert j / usj == seql
        assert usr % usi == 0
        assert usr % usj == 0
        device = self.device

        i_pos = torch.arange(i, device = device) * int(usr / usi)
        j_pos = torch.arange(j, device = device) * int(usr / usj)

        rel_pos = (rearrange(i_pos, 'i -> i 1') - rearrange(j_pos, 'j -> 1 j'))
        rel_pos = rel_pos + seql * usr - 1

        x = torch.arange(-seql * usr + 1, seql * usr, device = device).float() / usr
        x = rearrange(x, '... -> ... 1')

        for layer in self.net:
            x = layer(x)

        x = x[rel_pos]
        return rearrange(x, 'i j h -> h i j')

    
class Attend(nn.Module):
    def __init__(self):
        super().__init__()
            
    def forward(self, q, k, v, mask = None, attn_bias = None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """
        n, device = q.shape[-2], q.device
        scale = q.shape[-1] ** -0.5

        # similarity
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale

        # attention bias
        if attn_bias is not None:
            sim = sim + attn_bias

        # attention mask
        if mask is not None:
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # attention
        attn = sim.softmax(dim=-1)

        # aggregate values
        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        return out    

    
class Attention(nn.Module):
    def __init__(
        self,
        embed_dim=768, 
        num_heads=8
    ):
        super().__init__()
        self.heads = num_heads

        self.to_q = nn.Linear(embed_dim, embed_dim, bias = False)
        self.to_kv = nn.Linear(embed_dim, embed_dim * 2, bias = False)

        self.attend = Attend()

        self.to_out = nn.Linear(embed_dim, embed_dim, bias = True)

    def forward(
        self,
        x,
        context = None,
        attn_mask = None,
        attn_bias = None
    ):
        
        b, n, _, device = *x.shape, x.device
        kv_input = context

        # project for queries, keys, values
        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1)

        # split for multi-headed attention
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.heads)

        # attention
        out = self.attend(q, k, v, attn_bias = attn_bias, mask = attn_mask)

        # merge heads

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_size=768, heads=8, dropout=0.1, forward_expansion=4, window_length=25):
        super(TransformerEncoderBlock, self).__init__()
        self.num_heads = heads

        self.attention = Attention(embed_dim=embed_size, num_heads=heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.GELU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, effect_lengths, us_rate=1, attn_bias=None):
        B, L, C = x.shape 
        
        # Prepare attention mask
        attn_mask_max = torch.ones((L, L), device=x.device, dtype=torch.bool)
        attn_mask = torch.zeros((B, L, L), device=x.device, dtype=torch.bool)
        for i in range(B):
            attn_mask[i, :effect_lengths[i] * us_rate, :effect_lengths[i] * us_rate] = attn_mask_max[:effect_lengths[i] * us_rate, :effect_lengths[i] * us_rate]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        
        # Attention part
        xn = self.norm1(x) 
        attention = self.attention(xn, xn, attn_mask=attn_mask, attn_bias=attn_bias)  # (batch_size, seq_length, embed_dim)
        x = self.dropout1(attention) + x   # residual connection
                
        # Feed Forward part
        xn = self.norm2(x)
        forward = self.feed_forward(xn)
        x = self.dropout2(forward) + x    # residual connection
                
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_size=768, heads=8, dropout=0.1, forward_expansion=4, window_length=25):
        super(TransformerDecoderBlock, self).__init__()
        self.attn_type = attn_type
        self.num_heads = heads
        self.use_rpb = use_rpb
 
        self.attention1 = Attention(embed_dim=embed_size, num_heads=heads)
        self.attention2 = Attention(embed_dim=embed_size, num_heads=heads)
            
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm4 = nn.LayerNorm(embed_size)
        # Context normalization
        self.cnorm = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.GELU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

    def forward(self, x, c, effect_lengths, us_rate=1, attn_bias_self=None, attn_bias_cross=None):
        assert x.shape[0] == c.shape[0]
        assert x.shape[-1] == c.shape[-1]
        B, Lx, C = x.shape 
        B, S, C = c.shape
        L = S * us_rate
        
        # Prepare attention mask      
        attn_mask_ar = torch.ones((L, L), device=x.device, dtype=torch.bool)
        attn_mask_cross = torch.ones((S, S), device=x.device, dtype=torch.bool)
        
        attn_mask_cross = attn_mask_cross.unsqueeze(1).repeat(1, us_rate, 1).contiguous().view(L, S)
                
        attn_mask1 = torch.zeros((B, L, L), device=x.device, dtype=torch.bool)
        for i in range(B):
            attn_mask1[i, :effect_lengths[i] * us_rate, :effect_lengths[i] * us_rate] = attn_mask_ar[:effect_lengths[i] * us_rate, :effect_lengths[i] * us_rate]
        attn_mask1 = attn_mask1.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        
        attn_mask2 = torch.zeros((B, L, S), device=x.device, dtype=torch.bool)
        for i in range(B):
            attn_mask2[i, :effect_lengths[i] * us_rate, :effect_lengths[i]] = attn_mask_cross[:effect_lengths[i] * us_rate, :effect_lengths[i]]
        attn_mask2 = attn_mask2.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        
        if Lx < L:
            attn_mask1 = attn_mask1[:, :, :Lx, :Lx]
            attn_mask2 = attn_mask2[:, :, :Lx, :]
            if self.attn_type == 2:
                attn_mask3 = attn_mask3[:, :, :Lx, :Lx]
        
        # Attention part 
        xn = self.norm1(x)
        attention = self.attention1(xn, xn, attn_mask=attn_mask1, attn_bias=attn_bias_self)
        x = self.dropout1(attention) + x   # residual connection
        
        xn = self.norm2(x)
        c = self.cnorm(c)
        attention = self.attention2(xn, c, attn_mask=attn_mask2, attn_bias=attn_bias_cross)
        x = self.dropout2(attention) + x
                
        # Feed Forward part
        xn = self.norm4(x)
        forward = self.feed_forward(xn)
        x = self.dropout4(forward) + x    # residual connection
                
        return x

    
class CodePredictor(nn.Module):
    def __init__(self, us_rate=4, n_classifiers=2, n_labels=1024, n_layers=6, input_size=768, tseq_size=768, embed_size=768, heads=8, dropout=0.1, forward_expansion=4, window_length=25, seq_length=250):
        super(CodePredictor, self).__init__()
        self.seq_length = seq_length
        # sequence length for vfeat (default=250, fps=25)
        self.us_rate = us_rate
        # upsample rate for output
        self.n_classifiers = n_classifiers
        # number of classifiers
        
        self.proj_in = nn.Linear(input_size, embed_size)
    
        self.position_enc = nn.Identity() 
        self.rel_pos_bias = RelativePositionBias(dim = embed_size // 2, heads = heads)
            
        self.layers_en = nn.ModuleList([
            TransformerEncoderBlock(embed_size, heads, dropout, forward_expansion, window_length)
            for _ in range(n_layers)])
        self.layer_norm_en = nn.LayerNorm(embed_size)
        
        self.proj_out = nn.ModuleList([
            nn.Linear(embed_size, n_labels)
            for _ in range(n_classifiers)])
        
        self.proj_ms = nn.ModuleList([
            nn.Linear(embed_size, embed_size//2),
            nn.ReLU(),
            nn.Linear(embed_size//2, embed_size//4),
            nn.ReLU(),
            nn.Linear(embed_size//4, 2)])
    
    
    def encode(self, vfeat, effect_lengths):
        if self.us_rate != 1:
            vfeat = F.interpolate(vfeat.permute(0, 2, 1), scale_factor=self.us_rate, mode='linear')
            vfeat = vfeat.permute(0, 2, 1)        
        enc_output = self.proj_in(vfeat)

        B, L, C = enc_output.shape
  
        attn_bias_enc = self.rel_pos_bias(L, L, int(L/self.seq_length), int(L/self.seq_length))
        attn_bias_enc = attn_bias_enc.unsqueeze(0).repeat(B, 1, 1, 1)
                                      
        for enc_layer in self.layers_en:
            enc_output = enc_layer(enc_output, effect_lengths, us_rate=int(L/self.seq_length), attn_bias=attn_bias_enc)
        enc_output = self.layer_norm_en(enc_output)
        return enc_output
    
    def decode(self, enc_output, effect_lengths, tseq=None):
        B, L, C = enc_output.shape
        
        dec_output = enc_output
        
        seq_logits = []
        for proj_out in self.proj_out:
            seq_logits.append(proj_out(dec_output))
        
        ms_pred = dec_output
        for proj_ms_layer in self.proj_ms:
            ms_pred = proj_ms_layer(ms_pred)
        return seq_logits, ms_pred
    
    
    def forward(self, vfeat, effect_lengths, tseq=None):
        enc_output = self.encode(vfeat, effect_lengths)
        seq_logits, ms_pred = self.decode(enc_output, effect_lengths, tseq)  
        return seq_logits, ms_pred
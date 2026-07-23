import torch
import torch.nn as nn
import math
from einops import rearrange, repeat
import numbers
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from dinov2.dinov2_model.pre_dino import DINOv2
from denoising_diffusion_pytorch.simple_diffusion import ResnetBlock
from functools import partial

class Upsample(nn.Module):
    def __init__(
            self,
            dim,
            dim_out=None,
            factor=2
    ):
        super().__init__()
        self.factor = factor
        self.factor_squared = factor ** 2

        dim_out = dim if dim_out is None else dim_out
        conv = nn.Conv2d(dim, dim_out * self.factor_squared, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(factor)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // self.factor_squared, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o r) ...', r=self.factor_squared)

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)

def modulate_2d(x, shift, scale):
    return x * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=1000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb       

class MDTAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(MDTAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x):
        b,c,h,w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads).contiguous()
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads).contiguous()
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads).contiguous()
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w).contiguous()
        out = self.project_out(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
    
class DiTLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        return x

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c').contiguous()

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w).contiguous()

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)
    def forward(self, x):
        x = self.proj(x)
        return x

class FinalLayer(nn.Module):
    def __init__(self, dim_t,hidden_size, out_channels):
        super().__init__()
        self.norm_final = DiTLayerNorm(hidden_size)
        self.conv2d = nn.Conv2d(hidden_size, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_t, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate_2d(self.norm_final(x), shift, scale)
        x = self.conv2d(x)
        return x

class Self_attention_block(nn.Module):
    def __init__(self, dim, dim_t, num_heads, ffn_expansion_factor=2, self_bias=False, ffn_bias = False):
        super().__init__()
        self.norm1 = DiTLayerNorm(dim)
        self.self_attention = MDTAttention(dim=dim,num_heads=num_heads,bias=self_bias)
        self.norm2 = DiTLayerNorm(dim)
        self.ffn1 = FeedForward(dim=dim,ffn_expansion_factor=ffn_expansion_factor,bias=ffn_bias)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_t, 6 * dim, bias=True)
        )
    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(-1).unsqueeze(-1) * self.self_attention(modulate_2d(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(-1).unsqueeze(-1) * self.ffn1(modulate_2d(self.norm2(x), shift_mlp, scale_mlp))
        return x

class CrossAttention(nn.Module):
    def __init__(self, query_dim, feat_dim, num_heads=8, ffn_mult=2, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q_proj = nn.Conv2d(query_dim, query_dim, kernel_size=1, bias=bias)
        self.q_dw = nn.Conv2d(query_dim, query_dim, kernel_size=3, padding=1, groups=query_dim, bias=bias)

        self.kv_proj = nn.Conv2d(feat_dim, query_dim * 2, kernel_size=1, bias=bias)
        self.kv_dw = nn.Conv2d(query_dim * 2, query_dim * 2, kernel_size=3, padding=1, groups=query_dim * 2, bias=bias)

        self.project_out = nn.Conv2d(query_dim, query_dim, kernel_size=1, bias=bias)

    def forward(self, x, feat):
        B, C, H, W = x.shape
        q = self.q_dw(self.q_proj(x))
        kv = self.kv_dw(self.kv_proj(feat))
        k, v = kv.chunk(2, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads).contiguous()
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads).contiguous()
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads).contiguous()
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W).contiguous()
        out = self.project_out(out)
        return out
    
class Cross_attention_block(nn.Module):
    def __init__(self, dim, dim_cross_in, dim_t, num_heads, ffn_expansion_factor=2, stride=1, self_bias=False, ffn_bias = False):
        super().__init__()

        self.normq = DiTLayerNorm(dim)
        self.normkv = LayerNorm(dim_cross_in, LayerNorm_type='WithBias')
        self.cross_attention = CrossAttention(query_dim=dim,feat_dim=dim_cross_in,num_heads=num_heads,bias=False)
        self.norm2 = DiTLayerNorm(dim)
        self.ffn = FeedForward(dim=dim,ffn_expansion_factor=ffn_expansion_factor,bias=ffn_bias)
        self.adaLN_modulation = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(dim_t, 6 * dim, bias=True)
                )
    def forward(self, x, c, cond):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(-1).unsqueeze(-1) * self.cross_attention(modulate_2d(self.normq(x), shift_msa, scale_msa), self.normkv(cond))
        x = x + gate_mlp.unsqueeze(-1).unsqueeze(-1) * self.ffn(modulate_2d(self.norm2(x), shift_mlp, scale_mlp))
        return x
    
def cosine_similarity_loss(f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
    f1 = f1.permute(0, 2, 3, 1).reshape(-1, f1.size(1))  # [B*H*W, C]
    f2 = f2.permute(0, 2, 3, 1).reshape(-1, f2.size(1))  # [B*H*W, C]

    f1 = F.normalize(f1, dim=1)
    f2 = F.normalize(f2, dim=1)

    cos_sim = (f1 * f2).sum(dim=1)  # [B*H*W]
    return 1 - cos_sim.mean()

def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )

class net(nn.Module):
    def __init__(
        self,
        input_channels=1,
        output_channels=1,
        dim=[32,64,256,768],
        dim_t=256,
        num_heads=[4,8,16,32],
        num_block=[2,4,6,8],
        ffn_expansion_factor=2,
        dim_cross= 768,
        is_fre = True
    ):
        super().__init__()
        self.in_channels  = input_channels
        self.out_channels = output_channels
        self.gen_cond_img = DINOv2(is_fre=is_fre)
        resnet_block = partial(ResnetBlock, groups=8)

        self.x_embedder   = OverlapPatchEmbed(in_c=input_channels,embed_dim=dim[0])

        self.t_embedder   = TimestepEmbedder(dim_t)

        self.cls_proj = build_mlp(dim_cross+dim_t, dim_t*4, dim_t)
        self.encoder_level1 = nn.ModuleList([nn.ModuleList([
                                                Self_attention_block(dim=dim[0], dim_t=dim_t, num_heads=num_heads[2], ffn_expansion_factor=ffn_expansion_factor),
                                                Cross_attention_block(dim=dim[0], dim_cross_in=dim_cross//16, dim_t=dim_t, num_heads=num_heads[0], ffn_expansion_factor=ffn_expansion_factor)
                                            ]) for _ in range(num_block[0])])
        self.encoder_level2 = nn.ModuleList([nn.ModuleList([
                                                Self_attention_block(dim=dim[1], dim_t=dim_t, num_heads=num_heads[1], ffn_expansion_factor=ffn_expansion_factor),
                                                Cross_attention_block(dim=dim[1], dim_cross_in=dim_cross//4, dim_t=dim_t, num_heads=num_heads[1], ffn_expansion_factor=ffn_expansion_factor)
                                            ]) for _ in range(num_block[1])])
        self.encoder_level3 = nn.ModuleList([nn.ModuleList([
                                                Self_attention_block(dim=dim[2], dim_t=dim_t, num_heads=num_heads[2], ffn_expansion_factor=ffn_expansion_factor),
                                                Cross_attention_block(dim=dim[2], dim_cross_in=dim_cross//2, dim_t=dim_t, num_heads=num_heads[2], ffn_expansion_factor=ffn_expansion_factor)
                                            ]) for _ in range(num_block[2])])
        self.latent = nn.ModuleList([nn.ModuleList([
                                                Self_attention_block(dim=dim[3], dim_t=dim_t, num_heads=num_heads[3], ffn_expansion_factor=ffn_expansion_factor),
                                                Cross_attention_block(dim=dim[3], dim_cross_in=dim_cross, dim_t=dim_t, num_heads=num_heads[3], ffn_expansion_factor=ffn_expansion_factor)
                                            ]) for _ in range(num_block[3])])

        self.decoder_level3 = nn.ModuleList([Self_attention_block(dim=dim[2], dim_t=dim_t, num_heads=num_heads[2], ffn_expansion_factor=ffn_expansion_factor)
                                             for _ in range(num_block[2])])

        self.decoder_level2 = nn.ModuleList([Self_attention_block(dim=dim[1], dim_t=dim_t, num_heads=num_heads[1], ffn_expansion_factor=ffn_expansion_factor) for _ in range(num_block[1])])

        self.decoder_level1 = nn.ModuleList([Self_attention_block(dim=dim[0], dim_t=dim_t, num_heads=num_heads[0], ffn_expansion_factor=ffn_expansion_factor) for _ in range(num_block[0])])

        self.output = FinalLayer(hidden_size=dim[0],dim_t=dim_t,out_channels=output_channels)

        self.down = nn.ModuleList([
            nn.Sequential(
                ConvModule(dim[0], dim[1], kernel_size=7, padding=3, stride=4,
                           norm_cfg=dict(type='BN', requires_grad=True)),
                ),
            nn.Sequential(
                ConvModule(dim[1], dim[2], kernel_size=3, padding=1, stride=2,
                           norm_cfg=dict(type='BN', requires_grad=True)),
                ),
            nn.Sequential(
                ConvModule(dim[2], dim[3], kernel_size=3, padding=1, stride=2,
                           norm_cfg=dict(type='BN', requires_grad=True)),
                )
        ])
        self.up_img3 = Upsample(dim_cross,dim_cross//2)
        self.norm3 = DiTLayerNorm(dim=dim_cross)
        self.up_img2 = Upsample(dim_cross,dim_cross//4, factor=4)
        self.norm2 = DiTLayerNorm(dim=dim_cross)
        self.up_img1 = Upsample(dim_cross,dim_cross//16, factor=16)
        self.norm1 = DiTLayerNorm(dim=dim_cross)
        self.up = nn.ModuleList([
            nn.Sequential(
                Upsample(dim[1], dim[0], factor=4),
                resnet_block(dim[0], dim[0], time_emb_dim=dim_t),
                ConvModule(dim[0]*2, dim[0], kernel_size=1, norm_cfg=dict(type='BN', requires_grad=True)),
                resnet_block(dim[0], dim[0], time_emb_dim=dim_t)
            ),
            nn.Sequential(
                Upsample(dim[2], dim[1], factor=2),
                resnet_block(dim[1], dim[1], time_emb_dim=dim_t),
                ConvModule(dim[1]*2, dim[1], kernel_size=1, norm_cfg=dict(type='BN', requires_grad=True)),
                resnet_block(dim[1], dim[1], time_emb_dim=dim_t)
            ),
            nn.Sequential(
                Upsample(dim[3], dim[2], factor=2),
                resnet_block(dim[2], dim[2], time_emb_dim=dim_t),
                ConvModule(dim[2] * 2, dim[2], kernel_size=1, norm_cfg=dict(type='BN', requires_grad=True)),
                resnet_block(dim[2], dim[2], time_emb_dim=dim_t)
            ),
        ])

        self.initialize_weights()
    
    def initialize_weights(self):
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        nn.init.normal_(self.cls_proj[0].weight, std=0.02)
        nn.init.normal_(self.cls_proj[2].weight, std=0.02)
        nn.init.normal_(self.cls_proj[4].weight, std=0.02)
            
        for block in self.encoder_level1:
            nn.init.constant_(block[0].adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block[0].adaLN_modulation[-1].bias, 0)
            
            nn.init.constant_(block[1].adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block[1].adaLN_modulation[-1].bias, 0)
        
        for block in self.encoder_level2:
            nn.init.constant_(block[0].adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block[0].adaLN_modulation[-1].bias, 0)
            
            nn.init.constant_(block[1].adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block[1].adaLN_modulation[-1].bias, 0)

        for block in self.encoder_level3:
            nn.init.constant_(block[0].adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block[0].adaLN_modulation[-1].bias, 0)
            
            nn.init.constant_(block[1].adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block[1].adaLN_modulation[-1].bias, 0)

        for block in self.latent:
            nn.init.constant_(block[0].adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block[0].adaLN_modulation[-1].bias, 0)
            
            nn.init.constant_(block[1].adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block[1].adaLN_modulation[-1].bias, 0)
        
        for block in self.decoder_level3:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        for block in self.decoder_level2:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        for block in self.decoder_level1:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.output.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.output.adaLN_modulation[-1].bias, 0)
        
    def forward(self,x,t,img, x0, return_feat=False):
        loss_x = torch.tensor(0., device=x.device)
        with torch.no_grad():
            out = self.gen_cond_img(img)
            img = out[:4]
            cls = out[4]
            if x0 is not None:
                x_resized = x0.repeat(1, 3, 1, 1)
                x_out = self.gen_cond_img(x_resized)
                x_feat = x_out[3]
                x_feat = F.interpolate(x_feat, size=(28,28), mode='bicubic', align_corners=False)
                      
        img1,img2,img3,img4 = img[0],img[1],img[2],img[3]
        img1 = F.interpolate(self.up_img1(self.norm1(img1)), size=(448,448), mode='bicubic', align_corners=False)
        img2 = F.interpolate(self.up_img2(self.norm2(img2)), size=(112,112), mode='bicubic', align_corners=False)
        img3 = F.interpolate(self.up_img3(self.norm3(img3)), size=(56,56), mode='bicubic', align_corners=False)
        img4 = F.interpolate(img4, size=(28,28), mode='bicubic', align_corners=False)
                
        c = self.t_embedder(t)
        c = torch.concat([c, cls], dim=1)
        c = self.cls_proj(c)
        in_en_level1  = self.x_embedder(x)                
        for block in self.encoder_level1:
            out_en_level1 = block[0](in_en_level1,c)           
            out_en_level1 = block[1](out_en_level1,c, img1) 

        in_en_level2  = self.down[0](out_en_level1)        
        for block in self.encoder_level2:
            out_en_level2 = block[0](in_en_level2,c)
            out_en_level2 = block[1](out_en_level2,c, img2) 

        in_en_level3  = self.down[1](out_en_level2)         
        for block in self.encoder_level3:
            out_en_level3 = block[0](in_en_level3,c)        
            out_en_level3 = block[1](out_en_level3,c, img3)          
        
        in_en_level4  = self.down[2](out_en_level3)          

        if x0 is not None:
            loss_x = cosine_similarity_loss(in_en_level4, x_feat)
        if return_feat:
            mid_x = in_en_level4
            mid_gt = x_feat
        for block in self.latent:
            latent        = block[0](in_en_level4,c)   
            latent        = block[1](latent, c, img4) 
        
        in_de_level3  = self.up[2][0](latent)                
        in_de_level3  = self.up[2][1](in_de_level3, c)
        in_de_level3  = torch.cat([in_de_level3, out_en_level3],dim=1)  
        in_de_level3  = self.up[2][2](in_de_level3)          
        in_de_level3  = self.up[2][3](in_de_level3, c)         
        for block in self.decoder_level3:
            out_de_level3 = block(in_de_level3,c)             

        in_de_level2  = self.up[1][0](out_de_level3)       
        in_de_level2  = self.up[1][1](in_de_level2, c)
        in_de_level2  = torch.cat([in_de_level2, out_en_level2],dim=1)
        in_de_level2  = self.up[1][2](in_de_level2)          
        in_de_level2  = self.up[1][3](in_de_level2, c)
        for block in self.decoder_level2:
            out_de_level2 = block(in_de_level2,c)  

        in_de_level1  = self.up[0][0](out_de_level2)        
        in_de_level1  = self.up[0][1](in_de_level1, c)
        in_de_level1  = torch.cat([in_de_level1, out_en_level1],dim=1) 
        in_de_level1  = self.up[0][2](in_de_level1)           
        in_de_level1  = self.up[0][3](in_de_level1, c)
        for block in self.decoder_level1:
            out_de_level1= block(in_de_level1,c)        

        out_x         = self.output(out_de_level1,c) 
        out_x = torch.sigmoid(out_x)
        if return_feat:
            return out_x, loss_x, mid_x, mid_gt
        else:
            return out_x, loss_x
    
    @torch.inference_mode()
    def sample_net(self, x, t, img, x0 = None):
        return self.forward(x, t, img, x0)[0]
        
    def extract_features(self, cond_img):
        return cond_img

class EmptyObject(object):
    def __init__(self, *args, **kwargs):
        pass
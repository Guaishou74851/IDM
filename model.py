import torch, types
from torch import nn
import torch.nn.functional as F
from utils import *
from backprop import RevModule, VanillaBackProp, RevBackProp
from forward import MyUNet2DConditionModel_SD_v1_5_forward, \
                    MyCrossAttnDownBlock2D_SD_v1_5_forward, \
                    MyCrossAttnUpBlock2D_SD_v1_5_forward, \
                    MyResnetBlock2D_SD_v1_5_forward, \
                    MyTransformer2DModel_SD_v1_5_forward

class Injector(nn.Module):
    def __init__(self, nf, r, T):
        super().__init__()
        self.f2i = nn.ModuleList([nn.Sequential(
            nn.PixelShuffle(r),
            nn.Conv2d(nf//(r*r), 1, 1),
        ) for _ in range(T)])
        self.i2f = nn.ModuleList([nn.Sequential(
            nn.Conv2d(3, nf//(r*r), 1),
            nn.PixelUnshuffle(r),
        ) for _ in range(T)])

    def forward(self, x_in):
        x = self.f2i[t-1](x_in)
        x = torch.cat([x, AT(A(x)), ATy], dim=1)
        return x_in + self.i2f[t-1](x)

class Step(RevModule):
    def __init__(self, t):
        super().__init__()
        self.t = t

    def body(self, x):
        with torch.cuda.amp.autocast(enabled=use_amp, cache_enabled=False):
            global t
            t = self.t
            cur_alpha_bar = alpha_bar[t]
            prev_alpha_bar = alpha_bar[t-1]
            e = F.pixel_shuffle(unet(F.pixel_unshuffle(x, 2)), 2) # 0. Noise Estimation (epsilon)
            x = (x - (1 - cur_alpha_bar).pow(0.5) * e) / cur_alpha_bar.pow(0.5) # 1. Denoising
            x = x - AT(A(x) - y) # 2. RND
            return prev_alpha_bar.pow(0.5) * x + (1 - prev_alpha_bar).pow(0.5) * e # 3. DDIM Sampling

class Net(nn.Module):
    def __init__(self, T, unet):
        super().__init__()
        del unet.time_embedding, unet.mid_block
        unet.down_blocks = unet.down_blocks[:-2]
        unet.down_blocks[-1].downsamplers = None
        unet.up_blocks = unet.up_blocks[2:]
        self.body = nn.ModuleList([Step(T-i) for i in range(T)])
        self.input_help_scale_factor = nn.Parameter(torch.tensor([1.0]))
        self.merge_scale_factor = nn.Parameter(torch.tensor([0.0]))
        self.alpha = nn.Parameter(torch.full((T,), 0.5))
        self.unet = unet
        self.unet_add_down_rev_modules_and_injectors(T)
        self.unet_add_up_rev_modules_and_injectors(T)
        self.unet_remove_resnet_time_emb_proj()
        self.unet_remove_cross_attn()
        self.unet_set_inplace_to_true()
        self.unet_replace_forward_methods()

    def unet_add_down_rev_modules_and_injectors(self, T):
        self.unet.down_blocks[0].register_module("injectors", nn.ModuleList([Injector(320, 2, T) for _ in range(4)]))
        self.unet.down_blocks[1].register_module("injectors", nn.ModuleList([Injector(640, 4, T) for _ in range(4)]))
        for i in range(2):
            self.unet.down_blocks[i].register_module("rev_module_lists", nn.ModuleList([]))
            self.unet.down_blocks[i].register_parameter("input_help_scale_factor", nn.Parameter(torch.ones(1,)))
            self.unet.down_blocks[i].register_parameter("merge_scale_factors", nn.Parameter(torch.zeros(2,)))
            for j in range(2):
                rev_module_list = nn.ModuleList([])
                if self.unet.down_blocks[i].resnets[j].in_channels == self.unet.down_blocks[i].resnets[j].out_channels:
                    rev_module_list.append(RevModule(self.unet.down_blocks[i].resnets[j]))
                rev_module_list.append(RevModule(self.unet.down_blocks[i].injectors[2*j]))
                rev_module_list.append(RevModule(self.unet.down_blocks[i].attentions[j]))
                rev_module_list.append(RevModule(self.unet.down_blocks[i].injectors[2*j+1]))
                self.unet.down_blocks[i].rev_module_lists.append(rev_module_list)

    def unet_add_up_rev_modules_and_injectors(self, T):
        self.unet.up_blocks[0].register_module("injectors", nn.ModuleList([Injector(640, 4, T) for _ in range(6)]))
        self.unet.up_blocks[1].register_module("injectors", nn.ModuleList([Injector(320, 2, T) for _ in range(6)]))
        for i in range(2):
            self.unet.up_blocks[i].register_parameter("input_help_scale_factor", nn.Parameter(torch.ones(1,)))
            self.unet.up_blocks[i].register_parameter("merge_scale_factor", nn.Parameter(torch.zeros(1,)))
            rev_module_list = nn.ModuleList([])
            for j in range(3):
                if j > 0:
                    rev_module_list.append(RevModule(self.unet.up_blocks[i].resnets[j]))
                rev_module_list.append(RevModule(self.unet.up_blocks[i].injectors[2*j]))
                rev_module_list.append(RevModule(self.unet.up_blocks[i].attentions[j]))
                rev_module_list.append(RevModule(self.unet.up_blocks[i].injectors[2*j+1]))
            self.unet.up_blocks[i].register_module("rev_module_list", rev_module_list)

    def unet_replace_forward_methods(self):
        from diffusers.models.unets.unet_2d_blocks import CrossAttnDownBlock2D
        from diffusers.models.unets.unet_2d_blocks import CrossAttnUpBlock2D
        from diffusers.models.resnet import ResnetBlock2D
        from diffusers.models.transformers.transformer_2d import Transformer2DModel
        def replace_forward_methods(module):
            if isinstance(module, CrossAttnDownBlock2D):
                module.forward = types.MethodType(MyCrossAttnDownBlock2D_SD_v1_5_forward, module)
            elif isinstance(module, CrossAttnUpBlock2D):
                module.forward = types.MethodType(MyCrossAttnUpBlock2D_SD_v1_5_forward, module)
            elif isinstance(module, ResnetBlock2D):
                module.forward = types.MethodType(MyResnetBlock2D_SD_v1_5_forward, module)
            elif isinstance(module, Transformer2DModel):
                module.forward = types.MethodType(MyTransformer2DModel_SD_v1_5_forward, module)
        self.unet.apply(replace_forward_methods)
        self.unet.forward = types.MethodType(MyUNet2DConditionModel_SD_v1_5_forward, self.unet)

    def unet_remove_resnet_time_emb_proj(self):
        from diffusers.models.resnet import ResnetBlock2D
        def ResnetBlock2D_remove_time_emb_proj(module):
            if isinstance(module, ResnetBlock2D):
                module.time_emb_proj = None
        self.unet.apply(ResnetBlock2D_remove_time_emb_proj)

    def unet_remove_cross_attn(self):
        from diffusers.models.attention import BasicTransformerBlock
        def BasicTransformerBlock_remove_cross_attn(module):
            if isinstance(module, BasicTransformerBlock):
                module.attn2 = module.norm2 = None
        self.unet.apply(BasicTransformerBlock_remove_cross_attn)
    
    def unet_set_inplace_to_true(self):
        def set_inplace_to_true(module):
            if isinstance(module, nn.Dropout) or isinstance(module, nn.SiLU):
                module.inplace = True
        self.unet.apply(set_inplace_to_true)

    def forward(self, y_, A_, AT_, use_amp_=True):
        global y, A, AT, unet, ATy, alpha_bar, use_amp
        y, A, AT, unet, use_amp = y_, A_, AT_, self.unet, use_amp_
        alpha_bar = torch.cat([torch.ones(1, device=y.device), self.alpha.cumprod(dim=0)])
        x = AT(y)
        ATy = x
        x = alpha_bar[-1].pow(0.5) * torch.cat([x, self.input_help_scale_factor * x], dim=1)
        x = RevBackProp.apply(x, self.body)
        return x[:, :1] + self.merge_scale_factor * x[:, 1:]
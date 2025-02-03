import torch
from backprop import RevModule, VanillaBackProp, RevBackProp

def MyCheckpoint(module, x):
    def f(m, z):
        return m(z)
    return torch.utils.checkpoint.checkpoint(f, module, x, use_reentrant=False)

def MyUNet2DConditionModel_SD_v1_5_forward(self, x):
    x = self.conv_in(x)
    skip = [x]
    for i, down in enumerate(self.down_blocks):
        x, cur_skip = down(x)
        skip += cur_skip
    x = x.repeat(1, 2, 1, 1)
    for i, up in enumerate(self.up_blocks):
        x = up(x, skip[-len(up.resnets):])
        skip = skip[:-len(up.resnets)]
    x = self.conv_norm_out(x)
    x = self.conv_act(x)
    x = self.conv_out(x)
    return x

def MyCrossAttnDownBlock2D_SD_v1_5_forward(self, x):
    skip = []
    if self.resnets[0].in_channels != self.resnets[0].out_channels:
        x = MyCheckpoint(self.resnets[0], x)
    x = torch.cat([x, self.input_help_scale_factor * x], dim=1)
    for i in range(2):
        x = RevBackProp.apply(x, self.rev_module_lists[i])
        x_split = x.chunk(2, dim=1)
        x_merge = x_split[0] + self.merge_scale_factors[i] * x_split[1]
        skip.append(x_merge)
    x = x_merge
    if self.downsamplers is not None:
        x = MyCheckpoint(self.downsamplers[0], x)
        skip.append(x)
    return x, skip

def MyCrossAttnUpBlock2D_SD_v1_5_forward(self, x, skip):
    x = MyCheckpoint(self.resnets[0], torch.cat([x, skip[-1]], dim=1))
    self.resnets[1].register_buffer("skip", skip[-2], persistent=False)
    self.resnets[2].register_buffer("skip", skip[-3], persistent=False)
    x = torch.cat([x, self.input_help_scale_factor * x], dim=1)
    x = RevBackProp.apply(x, self.rev_module_list)
    x_split = x.chunk(2, dim=1)
    x = x_split[0] + self.merge_scale_factor * x_split[1]
    if self.upsamplers is not None:
        x = MyCheckpoint(self.upsamplers[0], x)
    return x

def MyResnetBlock2D_SD_v1_5_forward(self, x_in):
    if hasattr(self, "skip"):
        x_in = torch.cat([x_in, self.skip], dim=1)
    x = self.norm1(x_in)
    x = self.nonlinearity(x)
    x = self.conv1(x)
    x = self.norm2(x)
    x = self.nonlinearity(x)
    x = self.conv2(x)
    if self.in_channels == self.out_channels:
        return x + x_in
    return x + self.conv_shortcut(x_in)

def MyTransformer2DModel_SD_v1_5_forward(self, x_in):
    b, c, h, w = x_in.shape
    x = self.norm(x_in)
    x = self.proj_in(x)
    x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
    for block in self.transformer_blocks:
        x = x + block.attn1(block.norm1(x))
        x = x + block.ff(block.norm3(x))
    x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
    x = self.proj_out(x)
    return x + x_in
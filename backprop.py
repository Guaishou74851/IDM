import torch
from torch import nn
from torch.cuda.amp import autocast, custom_fwd, custom_bwd

class RevModule(nn.Module):
    def __init__(self, body=None, v=0.5):
        super().__init__()
        if body is not None:
            self.body = body
        self.v = nn.Parameter(torch.tensor([v]))

    def forward(self, x1, x2):
        return (1 - self.v) * self.body(x1) + self.v * x2, x1
    
    def backward_pass(self, y1, y2, dy1, dy2):
        """
        F = (1 - v) * body(x1)
        y1 = F + v * x2
        y2 = x1
        """
        with torch.no_grad():
            x1 = y2.detach()
            del y2
        with torch.enable_grad():
            x1.requires_grad = True
            F = (1 - self.v) * self.body(x1)
            F.backward(dy1, retain_graph=True)
        with torch.no_grad():
            dx1 = x1.grad + dy2
            del x1.grad, dy2
            dx2 = self.v * dy1
            x2 = (y1 - F) / self.v
            del y1, F
            self.v.grad += (x2 * dy1).sum()
            del dy1
        return x1, x2, dx1, dx2

class VanillaBackProp(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def apply(x, layers):
        x1, x2 = x.chunk(2, dim=1)
        for layer in layers:
            x1, x2 = layer(x1, x2)
        return torch.cat([x1, x2], dim=1)

class RevBackProp(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, x, layers):
        with torch.no_grad():
            x1, x2 = x.chunk(2, dim=1)
            for layer in layers:
                x1, x2 = layer(x1, x2)
        ctx.save_for_backward(x1.detach(), x2.detach())
        ctx.layers = layers
        return torch.cat([x1, x2], dim=1)

    @staticmethod
    @custom_bwd
    def backward(ctx, dx):
        dx1, dx2 = dx.chunk(2, dim=1)
        x1, x2 = ctx.saved_tensors
        for layer in ctx.layers[::-1]:
            x1, x2, dx1, dx2 = layer.backward_pass(x1, x2, dx1, dx2)
        return torch.cat([dx1, dx2], dim=1), None
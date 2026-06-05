import torch
from torch import nn
import math
from einops import einsum, rearrange


class CSAM(nn.Module):

    ### Convolution Self-Attention Module (CSAM)

    # 1. input BxHxWxC is reshaped to BxLxC
    # 2. a 1d convolution is applied with 1x1 kernels (why are the kernels 1x1 if it is 1d?)
    # 3. GELU activation function is applied

    ### Self-Attention

    # 4. Split q, k and v vectors into multiple heads
    # 5. compute attention scores and weight values accordingly
    # 6. fuse heads back together
    # 7. perform last 1d convolution on the output

    def __init__(self, c, num_heads):
        super().__init__()

        assert c%num_heads == 0, 'channel dimension needs to be divisable by the number of heads'

        self.num_heads = num_heads
        self.head_dim = c//num_heads
        self.to_qkv = nn.Conv1d(in_channels=c, out_channels=3*c, kernel_size=1)
        self.out_conv = nn.Conv1d(in_channels=c, out_channels=c, kernel_size=1)

    def forward(self, x):

        # reshape
        B, C, _, _ = x.shape
        x = torch.reshape(x, (B, C, -1))

        # appply the 1d conv
        x = self.to_qkv(x)

        # apply GELU
        x = nn.functional.gelu(x)

        # split into q, k and v then into multiple heads
        q, k, v = map(
            lambda t : rearrange(t, 'b (h d) l -> b h l d', h=self.num_heads, d=self.head_dim),
            torch.chunk(x, chunks=3, dim=1)
        )

        # compute attention scores
        attn = einsum(q, k, 'b h l1 d, b h l2 d -> b h l1 l2') / math.sqrt(self.head_dim)
        attn = nn.functional.softmax(attn, dim=-1)

        # weight v according to attention scores
        out = einsum(attn, v, 'b h l l1, b h l1 d -> b h l d')

        # rearrange the output
        out = rearrange(out, 'b h l d -> b (h d) l')

        # apply last convolution
        out = self.out_conv(out)

        return out


class AFIOM(nn.Module):

    ### Adaptive Feature Integration and Output Module

    # 1. perform Global Average Pooling (GAP) on the input feature map
    # 2. concatenate the features from all models in the assembly
    # 3. use a FC layer to map inputs down to a snigle dimension
    # 4. apply sigmoid to obtain a probability

    def __init__(self, out_channels):

        super().__init__()
        self.fc = nn.Linear(sum(out_channels), 1)


    def forward(self, xs):

        gs = []

        # perform GAP for each input from the assembly
        for x in xs:

            # x has shape BxCxL, for each channel compute the average

            gs.append(torch.mean(x, dim=2))

        g = torch.cat(gs, dim=1)
        out = self.fc(g)

        return out


class MMSEN(nn.Module):

    def __init__(self, assembly, num_heads, out_channels):

        super().__init__()

        self.assembly = nn.ModuleList(assembly)
        self.csam = nn.ModuleList([CSAM(out_channels[i], num_heads) for i in range(len(assembly))])
        self.afiom = AFIOM(out_channels)

    def forward(self, x):

        xs = []

        for model in self.assembly:
            xs.append(model(x))

        ys = []

        for x, csam in zip(xs, self.csam):
            ys.append(csam(x))

        return self.afiom(ys)
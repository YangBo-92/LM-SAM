import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MDP_Adapter(nn.Module):

    def __init__(
            self,
            embedding_dim=96,
            mlp_dim=96,
            config=None,
            d_model=None,
            bottleneck=None,
            dropout=0.0,
            init_option="lora",
            adapter_scalar="1.0",
            adapter_layernorm_option="in"
    ):
        super().__init__()
        self.n_embd = embedding_dim
        self.mlp_dim = mlp_dim

        self.adapter_layernorm_option = adapter_layernorm_option
        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.mlp_dim)
        self.non_linear_func = nn.ReLU(inplace=True)
        self.up_proj = nn.Linear(self.mlp_dim, self.n_embd)

        # project deep prior feature to adapter hidden dimension
        self.adapt_projection = nn.Linear(self.n_embd, self.mlp_dim)

        self.att2 = TripletAttention()
        self.bn = nn.BatchNorm2d(mlp_dim)

        self.dropout = dropout

        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

                nn.init.kaiming_uniform_(self.adapt_projection.weight, a=math.sqrt(5))
                nn.init.zeros_(self.adapt_projection.bias)

    def forward(self, x, deep_adapt=None, layer_idx=None, add_residual=True, residual=None):
        residual = x if residual is None else residual

        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        # select multi-scale deep prior feature according to layer depth
        if deep_adapt is not None and layer_idx is not None:
            B, H, W, C = deep_adapt.shape
            deep_adapt_conv = deep_adapt.permute(0, 3, 1, 2)

            if layer_idx < 4:
                selected_adapt = deep_adapt_conv
            elif layer_idx < 8:
                selected_adapt = F.adaptive_avg_pool2d(deep_adapt_conv, (H // 2, W // 2))
            else:
                selected_adapt = F.adaptive_avg_pool2d(deep_adapt_conv, (H // 4, W // 4))

            selected_adapt = selected_adapt.permute(0, 2, 3, 1)
        else:
            selected_adapt = deep_adapt

        down = self.down_proj(x)
        down = self.non_linear_func(down)

        # align spatial size and fuse deep prior feature
        if selected_adapt is not None:
            if selected_adapt.shape[1:3] != down.shape[1:3]:
                selected_adapt = selected_adapt.permute(0, 3, 1, 2)
                selected_adapt = F.interpolate(
                    selected_adapt,
                    size=down.shape[1:3],
                    mode='bilinear',
                    align_corners=False
                )
                selected_adapt = selected_adapt.permute(0, 2, 3, 1)

            selected_adapt = self.adapt_projection(selected_adapt)
            down = down + selected_adapt

        down = down.permute(0, 3, 1, 2)
        down = self.bn(down)
        down = self.att2(down)
        down = down.permute(0, 2, 3, 1)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)

        up = self.up_proj(down)
        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output


# triplet attention module
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, 7, stride=1, padding=3, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()

        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()

        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x_out
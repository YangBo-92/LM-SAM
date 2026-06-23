import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LGM_Adapter(nn.Module):

    def __init__(
            self,
            embedding_dim=768,
            mlp_dim=192,
            dropout=0.1,
            init_option="lora",
            adapter_scalar="1.0"
    ):
        super().__init__()

        self.down_proj = nn.Linear(embedding_dim, mlp_dim)
        self.act = nn.GELU()

        # fixed Laplacian kernel (dataset-agnostic structural prior)
        self.struct_probe = nn.Conv2d(
            in_channels=mlp_dim,
            out_channels=mlp_dim,
            kernel_size=3,
            padding=1,
            groups=mlp_dim,
            bias=False
        )

        laplacian_kernel = torch.tensor([
            [0., 1., 0.],
            [1., -4., 1.],
            [0., 1., 0.]
        ])

        with torch.no_grad():
            self.struct_probe.weight.copy_(
                laplacian_kernel.view(1, 1, 3, 3).repeat(mlp_dim, 1, 1, 1)
            )

        # learnable non-linear gate (modality-adaptive)
        self.raw_gate = nn.Sequential(
            nn.Conv2d(mlp_dim, mlp_dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(mlp_dim // 4, 1, kernel_size=1)
        )

        self.gate_scale = nn.Parameter(torch.ones(1))
        self.gate_shift = nn.Parameter(torch.zeros(1))

        self.norm = nn.BatchNorm2d(mlp_dim)
        self.up_proj = nn.Linear(mlp_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self._init_weights(init_option)

    def _init_weights(self, init_option):
        if init_option == "lora":
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.up_proj.bias)

            for m in self.raw_gate.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

            nn.init.constant_(self.gate_scale, 1.0)
            nn.init.constant_(self.gate_shift, 0.0)

        elif init_option == "bert":
            nn.init.normal_(self.down_proj.weight, std=0.02)
            nn.init.normal_(self.up_proj.weight, std=0.02)
            for m in self.raw_gate.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.02)

    def forward(self, x, residual=None, add_residual=True):
        if residual is None:
            residual = x

        down = self.down_proj(x)
        down = self.act(down)

        B, H, W, C = down.shape
        feat = down.permute(0, 3, 1, 2)

        # fixed Laplacian-based structure extraction
        struct_response = self.struct_probe(feat)

        # learnable gating mechanism
        raw_gate_value = self.raw_gate(struct_response)
        gate = torch.sigmoid(
            raw_gate_value * self.gate_scale + self.gate_shift
        )

        # 8-neighborhood aggregation
        neighbors = []
        for dx, dy in [(-1, -1), (-1, 0), (-1, 1),
                       (0, -1), (0, 1),
                       (1, -1), (1, 0), (1, 1)]:
            neighbors.append(
                torch.roll(feat, shifts=(dx, dy), dims=(2, 3))
            )

        neighbors = torch.stack(neighbors, dim=1)
        aggregated = torch.mean(neighbors, dim=1)

        # gate-controlled fusion
        gate_expanded = gate.unsqueeze(1)
        aligned_feat = feat * gate_expanded + aggregated * (1 - gate_expanded)
        aligned_feat = self.norm(aligned_feat)

        aligned_feat = aligned_feat.permute(0, 2, 3, 1)

        up = self.up_proj(aligned_feat)
        up = self.dropout(up)
        up = up * self.scale

        if add_residual:
            output = up + residual
        else:
            output = up

        return output
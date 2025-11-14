import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter


class CustomLayerNorm(nn.Module):
    def __init__(self, input_dims, stat_dims=(1,), num_dims=4, eps=1e-5):
        super().__init__()
        assert isinstance(input_dims, tuple) and isinstance(stat_dims, tuple)
        assert len(input_dims) == len(stat_dims)
        param_size = [1] * num_dims
        for input_dim, stat_dim in zip(input_dims, stat_dims):
            param_size[stat_dim] = input_dim
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps
        self.stat_dims = stat_dims
        self.num_dims = num_dims

    def forward(self, x):
        assert x.ndim == self.num_dims, print(
            "Expect x to have {} dimensions, but got {}".format(self.num_dims, x.ndim))

        mu_ = x.mean(dim=self.stat_dims, keepdim=True)  # [B,1,T,F]
        std_ = torch.sqrt(
            x.var(dim=self.stat_dims, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat


class LinearAttention(nn.Module):
    """Lightweight linear attention for efficient sequence modeling"""

    def __init__(self, emb_dim, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = (emb_dim // heads) ** -0.5
        self.to_qkv = nn.Linear(emb_dim, emb_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (batch, seq_len, dim)
        B, N, D = x.shape
        H = self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(B, N, H, D // H).transpose(1, 2), qkv)

        # Linear attention: Q(K^T V) instead of softmax(QK^T)V
        k = k.softmax(dim=-2)  # Normalize along sequence dimension
        context = torch.einsum('bhnd,bhne->bhde', k, v)
        out = torch.einsum('bhnd,bhde->bhne', q, context)

        out = out.transpose(1, 2).reshape(B, N, D)
        return self.to_out(out)


class LightTransformerBlock(nn.Module):
    """Efficient transformer block with linear attention"""

    def __init__(self, emb_dim, heads=4, expansion=2, dropout=0.1):
        super().__init__()
        # Self-attention with linear complexity
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = LinearAttention(emb_dim, heads, dropout)

        # Feed-forward network
        self.norm2 = nn.LayerNorm(emb_dim)
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * expansion),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * expansion, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Residual connection + attention
        x = x + self.attn(self.norm1(x))
        # Residual connection + feed-forward
        x = x + self.ff(self.norm2(x))
        return x


class LightTransformer(nn.Module):
    """Lightweight transformer to replace GRU"""

    def __init__(self, emb_dim, hidden_dim, num_layers=2, dropout_p=0.1):
        super().__init__()
        # Adaptive head calculation - ensure at least 2 heads
        heads = max(2, emb_dim // 16)

        self.transformer_blocks = nn.Sequential(*[
            LightTransformerBlock(
                emb_dim=emb_dim,
                heads=heads,
                expansion=2,  # Fixed expansion factor
                dropout=dropout_p
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        # x: (batch, seq_len, dim)
        return self.transformer_blocks(x)


class DualPathTransformer(nn.Module):
    """Dual-Path processing with lightweight transformers"""

    def __init__(self, emb_dim, hidden_dim, n_freqs=32, dropout_p=0.1, num_layers=2):
        super().__init__()

        # Intra-frame processing (frequency-wise)
        self.intra_norm = nn.LayerNorm((n_freqs, emb_dim))
        self.intra_transformer = LightTransformer(
            emb_dim, hidden_dim // 2, num_layers, dropout_p
        )

        # Inter-frame processing (time-wise)
        self.inter_norm = nn.LayerNorm((n_freqs, emb_dim))
        self.inter_transformer = LightTransformer(
            emb_dim, hidden_dim, num_layers, dropout_p
        )

    def forward(self, x):
        # x: (b, d, t, f)
        B, D, T, F = x.size()
        x = x.permute(0, 2, 3, 1)  # (b, t, f, d)

        # Intra-frame (frequency-wise) processing
        x_res = x
        x = self.intra_norm(x)
        x = x.reshape(B * T, F, D)  # (b*t, f, d)
        x = self.intra_transformer(x)
        x = x.reshape(B, T, F, D)
        x = x + x_res

        # Inter-frame (time-wise) processing
        x_res = x
        x = self.inter_norm(x)
        x = x.permute(0, 2, 1, 3)  # (b, f, t, d)
        x = x.reshape(B * F, T, D)
        x = self.inter_transformer(x)
        x = x.reshape(B, F, T, D).permute(0, 2, 1, 3)  # (b, t, f, d)
        x = x + x_res

        x = x.permute(0, 3, 1, 2)  # (b, d, t, f)
        return x


class ConvolutionalGLU(nn.Module):
    def __init__(self, emb_dim, n_freqs=32, expansion_factor=2, dropout_p=0.1):
        super().__init__()
        hidden_dim = int(emb_dim * expansion_factor)
        self.norm = CustomLayerNorm((emb_dim, n_freqs), stat_dims=(1, 3))
        self.fc1 = nn.Conv2d(emb_dim, hidden_dim * 2, 1)
        self.dwconv = nn.Sequential(
            nn.ConstantPad2d((1, 1, 2, 0), value=0.0),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, groups=hidden_dim),
        )
        self.act = nn.Mish()
        self.fc2 = nn.Conv2d(hidden_dim, emb_dim, 1)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        # x:(b,d,t,f)
        res = x
        x = self.norm(x)
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.act(self.dwconv(x)) * v
        x = self.dropout(x)
        x = self.fc2(x)
        x = x + res
        return x


class DPR(nn.Module):
    """Dual-Path Transformer with Convolutional GLU"""

    def __init__(self, emb_dim=16, hidden_dim=24, n_freqs=32, dropout_p=0.1, num_layers=2):
        super().__init__()
        self.dp_transformer = DualPathTransformer(
            emb_dim, hidden_dim, n_freqs, dropout_p, num_layers
        )
        self.conv_glu = ConvolutionalGLU(
            emb_dim, n_freqs=n_freqs, expansion_factor=2, dropout_p=dropout_p
        )

    def forward(self, x):
        x = self.dp_transformer(x)
        x = self.conv_glu(x)
        return x
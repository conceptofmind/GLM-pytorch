import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum, nn

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# normalization
# they use layernorm with bias, different from PaLM

class PostNormResidual(nn.Module):
    def __init__(self, dim, fn, scale_residual = 1.):
        super().__init__()
        self.fn = fn
        self.scale_residual = scale_residual
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        residual = x * self.scale_residual
        out = self.fn(x, *args, **kwargs) + residual
        return self.norm(out)


# deepnet init

def deepnorm_init(transformer, beta, module_name_match_list = ['.ff_out.', '.v_out', '.attn_out']):
    for name, module in transformer.named_modules():
        if type(module) != nn.Linear:
            continue

        needs_beta_gain = any(map(lambda substr: substr in name, module_name_match_list))
        gain = beta if needs_beta_gain else 1
        nn.init.xavier_normal_(module.weight.data, gain = gain)

        if exists(module.bias):
            nn.init.constant_(module.bias.data, 0)


# rotary positional embedding
# https://arxiv.org/abs/2104.09864

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)

def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


# classic Noam Shazeer paper, use GEGLU for gating the feedforward
# https://arxiv.org/abs/2002.05202

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


# parallel attention and feedforward with residual
# discovered by Wang et al + EleutherAI from GPT-J fame

class ParallelAttention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4):
        super().__init__()

        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult

        self.heads = heads
        self.scale = dim_head**-0.5
        self.rotary_emb = RotaryEmbedding(dim_head)

        self.to_q = nn.Linear(dim, attn_inner_dim, bias = False)
        self.to_k = nn.Linear(dim, dim_head, bias=False)
        self.to_v = nn.Linear(dim, dim_head, bias=False)
        self.to_ff = nn.Linear(dim, ff_inner_dim * 2, bias=False)

        self.attn_out = nn.Linear(attn_inner_dim, dim)

        self.ff_out = nn.Sequential(
            GEGLU(),
            nn.Linear(ff_inner_dim, dim)
        )

        # for caching causal mask and rotary embeddings

        self.register_buffer("mask", None, persistent=False)
        self.register_buffer("pos_emb", None, persistent=False)

    def get_mask(self, n, device):
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def get_rotary_embedding(self, n, device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n]

        pos_emb = self.rotary_emb(n, device=device)
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb

    def forward(self, x):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, device, h = x.shape[1], x.device, self.heads

        # attention queries, keys, values, and feedforward inner

        q, k, v, ff = self.to_q(x), self.to_k(x), self.to_v(x), self.to_ff(x)

        # split heads
        # they use multi-query single-key-value attention, yet another Noam Shazeer paper
        # they found no performance loss past a certain scale, and more efficient decoding obviously
        # https://arxiv.org/abs/1911.02150

        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        # rotary embeddings

        positions = self.get_rotary_embedding(n, device)
        q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))

        # scale

        q = q * self.scale

        # similarity

        sim = einsum("b h i d, b j d -> b h i j", q, k)

        # causal mask

        causal_mask = self.get_mask(n, device)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        # merge heads

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.attn_out(out) + self.ff_out(ff)


# transformer

class ParallelTransformer(nn.Module):
    def __init__(
        self, 
        dim, 
        depth, 
        heads, 
        dim_head, 
        ff_mult=4, 
        scale_residual=1.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                PostNormResidual(dim, ParallelAttention(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult), scale_residual=scale_residual)
            )

    def forward(self, x):
        for block in self.layers:
            x = block(x)
        return x


# Model

class GLM(nn.Module):
    def __init__(
        self, 
        dim, 
        num_tokens, 
        depth, 
        dim_head=64, 
        heads=8, 
        ff_mult=4
    ):
        super().__init__()

        #dec_scale_residual = default(dec_scale_residual, (3 * depth) ** 0.25)

        self.net = nn.Sequential(
            nn.Embedding(num_tokens, dim),
            ParallelTransformer(dim=dim, depth=depth, heads=heads, dim_head=dim_head, ff_mult=ff_mult),
            nn.Linear(dim, num_tokens)
        )

    def forward(self, x):
        # they used embedding weight tied projection out to logits, not common, but works
        self.net[-1].weight = self.net[0].weight

        nn.init.normal_(self.net[0].weight, std=0.02)
        return self.net(x)


if __name__ == "__main__":

    glm = GLM(
        num_tokens = 20000,
        dim = 512,
        depth = 1,
        heads = 8,
        dim_head = 64,
    )

    tokens = torch.randint(0, 20000, (1, 2048))
    logits = glm(tokens) # (1, 2048, 20000)

    n_params_torch = sum(
        p.numel() for p in glm.parameters() if p.requires_grad
    )

    print(f"Number of parameters in torch model: {n_params_torch}")

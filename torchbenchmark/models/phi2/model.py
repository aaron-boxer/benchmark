from dataclasses import dataclass
from typing import Optional
import math
import torch_directml
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 51200
    n_layer: int = 32
    n_head: int = 32
    dim: int = 2560
    intermediate_size: int = 10240
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        else:
            self.n_kv_groups = self.n_head // self.n_local_heads
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head
    
    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        # fuzzy search
        config = [config for config in transformer_configs if config in str(name).upper() or config in str(name)]
        assert len(config) == 1, name
        return cls(**transformer_configs[config[0]])

transformer_configs = {
    "phi-2": dict(block_size=2048, n_layer=32, n_head=32, dim=2560, intermediate_size=10240, rope_base=10000),
}


class KVCache(nn.Module):
    def __init__(self):
        super().__init__()
        self.cache = {}

    def update(self, k_val, v_val):
        if 'key' not in self.cache:
            self.cache['key'] = k_val
            self.cache['val'] = v_val
        else:
            self.cache['key'] = torch.cat([self.cache['key'], k_val], dim=2)
            self.cache['val'] = torch.cat([self.cache['val'], v_val], dim=2)
        return self.cache['key'], self.cache['val']


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = nn.LayerNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size)

        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1
        self.partial_rotatory_factor = 0.4
    
    def setup_caches(self, max_batch_size, max_seq_length, dtype=torch.float32):
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache()
            b.attention._init_rope(dtype=dtype)

        self.causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
        self.attn_mask = torch.ones([1,1,1,1]).to(torch.bool)

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        if input_pos.size(0) > 1:
            mask = self.causal_mask[None, None, input_pos, :input_pos.size(0)]
        else:
            mask = self.attn_mask
        x = self.tok_embeddings(idx)

        for _, layer in enumerate(self.layers):
            x = layer(x, input_pos, mask)
        x = self.norm(x)
        logits = self.output(x)
        return logits
    
    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = nn.LayerNorm(config.dim, config.norm_eps)

    def forward(self, x: Tensor, input_pos: Tensor, mask: Tensor) -> Tensor:
        hidden_states = self.attention_norm(x)
        attn_outputs = self.attention(hidden_states, mask, input_pos)
        ffn_hidden_states = self.feed_forward(hidden_states)
        return attn_outputs + ffn_hidden_states + x


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, 3 * config.dim)
        self.wo = nn.Linear(config.dim, config.dim)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *argspy):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def _init_rope(self, dtype=torch.float32):
        self.rotary_emb = PhiRotaryEmbedding(self.n_head, dtype=dtype)  # original

    def forward(self, x: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        bsz, seqlen, _ = x.shape

        q, k, v = self.wqkv(x).split([self.dim, self.dim, self.dim], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1,2)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim).transpose(1,2)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim).transpose(1,2)
        
        cos, sin = self.rotary_emb(v, input_pos.unsqueeze(0))

        query_rot, query_pass = (
            q[..., : self.rotary_emb.dim],
            q[..., self.rotary_emb.dim :],
        )
        key_rot, key_pass = (
            k[..., : self.rotary_emb.dim],
            k[..., self.rotary_emb.dim :],
        )

        # [batch_size, seq_length, num_heads, head_dim // config.partial_rotary_factor]
        query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin)

        # [batch_size, seq_length, num_heads, head_dim]
        q = torch.cat((query_rot, query_pass), dim=-1)
        k = torch.cat((key_rot, key_pass), dim=-1)

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(k, v)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        y = self.wo(y)
        return y


class NewGELUActivation(nn.Module):
    """        xxor Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size)
        self.w2 = nn.Linear(config.intermediate_size, config.dim)
        self.act = NewGELUActivation()

    def forward(self, x: Tensor) -> Tensor:
        x = self.w1(x)
        x = self.act(x)
        x = self.w2(x)
        return x


class PhiRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, dtype=torch.float32, device=None):
        super().__init__()
        self.device = device if device is not None else torch_directml.device(torch_directml.default_device())
        self.dtype = dtype
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=dtype).to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, pos=None):
        return (
            self.cos_cached[pos].to(dtype=x.dtype),
            self.sin_cached[pos].to(dtype=x.dtype),
        )


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


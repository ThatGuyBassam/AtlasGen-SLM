# model/transformer.py
# AtlasGen-SLM Phase 1 — Encoder-only Transformer for genomic MLM pretraining.
# Raw PyTorch. No HuggingFace wrappers.
#
# Final Phase 1 architecture (~43M params):
#   Vocab:       4107 tokens
#   d_model:     512
#   n_heads:     8
#   n_layers:    12
#   d_ff:        1536  (SwiGLU feedforward)
#   max_length:  256
#   dropout:     0.1
#   norm:        RMSNorm
#   positions:   RoPE applied to Q/K inside attention
#   attention:   torch.nn.functional.scaled_dot_product_attention
#
# Notes:
# - This keeps the same public model interface as the previous AtlasGenSLM:
#       model(input_ids, attention_mask) -> MLM logits
#       model(input_ids, attention_mask, return_hidden=True) -> hidden states
# - Weight tying is used between token embeddings and MLM decoder.
# - PAD embedding row is explicitly zeroed and protected from gradient updates.

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Rotary Position Embeddings ────────────────────────────────────────────────

class RotaryPositionEmbedding(nn.Module):
    """
    Rotary positional embedding cache/generator for attention Q/K tensors.

    Input attention tensors use shape:
        [B, n_heads, T, d_head]

    RoPE is applied pairwise over the final dimension:
        even/odd channels rotate together.

    The cosine/sine tensors are cached per sequence length, device, and dtype so
    each layer does not repeatedly rebuild the same position table.
    """

    def __init__(self, d_head, base=10_000.0):
        super().__init__()

        if d_head % 2 != 0:
            raise ValueError("RoPE requires an even d_head.")

        inv_freq = 1.0 / (
            base ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head)
        )

        self.d_head = d_head
        self.base = base

        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, seq_len, device, dtype):
        """
        Returns:
            cos, sin with shape [1, 1, T, d_head/2]
        """
        cache_is_valid = (
            self._cos_cached is not None
            and self._sin_cached is not None
            and self._seq_len_cached >= seq_len
            and self._cos_cached.device == device
            and self._sin_cached.device == device
            and self._cos_cached.dtype == dtype
            and self._sin_cached.dtype == dtype
        )

        if cache_is_valid:
            return (
                self._cos_cached[:, :, :seq_len, :],
                self._sin_cached[:, :, :seq_len, :],
            )

        positions = torch.arange(
            seq_len,
            device=device,
            dtype=self.inv_freq.dtype,
        )

        freqs = torch.outer(positions, self.inv_freq.to(device))

        cos = freqs.cos().to(dtype=dtype)[None, None, :, :]
        sin = freqs.sin().to(dtype=dtype)[None, None, :, :]

        self._seq_len_cached = seq_len
        self._cos_cached = cos
        self._sin_cached = sin

        return cos, sin


def apply_rope(x, cos, sin):
    """
    Apply RoPE to tensor x with explicit even/odd interleaving.

    x:   [B, n_heads, T, d_head]
    cos: [1, 1, T, d_head/2]
    sin: [1, 1, T, d_head/2]
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd = x_even * sin + x_odd * cos

    x_out = torch.empty_like(x)
    x_out[..., 0::2] = x_rot_even
    x_out[..., 1::2] = x_rot_odd

    return x_out


# ── RMSNorm ───────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Unlike LayerNorm, RMSNorm does not re-center the input (no mean
    subtraction) and has no bias term — it only rescales by the RMS of the
    input, then applies a learned per-channel gain.

        y = x / sqrt(mean(x^2) + eps) * weight

    Cheaper than LayerNorm (no mean, no bias) and standard in modern
    transformer stacks (LLaMA, etc.). Used everywhere nn.LayerNorm previously
    appeared in this model: embeddings, block norms, final norm, MLM head.
    """

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # Compute in float32 for numerical stability, then cast back.
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (x.to(input_dtype)) * self.weight


# ── Embedding Layer ───────────────────────────────────────────────────────────

class GenomicEmbedding(nn.Module):
    """
    Token embedding layer.

    Positional information is not added here. RoPE is applied to Q/K inside
    self-attention.

    PAD handling:
    - token_embeddings uses padding_idx=0.
    - PAD positions are zeroed before LayerNorm using attention_mask.
    - This prevents PAD vectors from contaminating embedding normalization.
    """

    def __init__(self, vocab_size, d_model, dropout):
        super().__init__()

        self.token_embeddings = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=0,
        )

        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids:      [B, T]
        attention_mask: [B, T] where 1 = real token, 0 = PAD

        returns:        [B, T, d_model]
        """
        x = self.token_embeddings(input_ids)

        # Important: mask before norm.
        # RMSNorm has no bias term, so a zero input vector stays exactly zero
        # (0 / sqrt(eps) * weight = 0) — no special-casing needed.
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)

        x = self.norm(x)
        x = self.dropout(x)

        return x


# ── Multi-Head Self-Attention ─────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention using PyTorch SDPA.

    Advantages over explicit scores/softmax/matmul:
    - Lets PyTorch dispatch to optimized CUDA attention kernels when available.
    - Avoids manually materializing attention probabilities in Python.
    - Keeps the same Q/K/V/O parameterization as the original implementation.

    RoPE is applied to Q and K before attention.
    """

    def __init__(self, d_model, n_heads, dropout, rope_base=10_000.0):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionEmbedding(
            d_head=self.d_head,
            base=rope_base,
        )

    def forward(self, x, attention_mask=None):
        """
        x:              [B, T, d_model]
        attention_mask: [B, T] where 1 = real token, 0 = PAD

        returns:        [B, T, d_model]
        """
        B, T, _ = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        cos, sin = self.rope(
            seq_len=T,
            device=x.device,
            dtype=Q.dtype,
        )

        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)

        attn_mask = None

        if attention_mask is not None:
            # Shape [B, 1, 1, T], broadcast over heads and query positions.
            # For boolean SDPA masks: True = keep, False = masked out.
            attn_mask = attention_mask[:, None, None, :].bool()

        context = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )

        context = (
            context
            .transpose(1, 2)
            .contiguous()
            .view(B, T, self.d_model)
        )

        out = self.W_o(context)
        out = self.dropout(out)

        return out


# ── Feedforward Block ─────────────────────────────────────────────────────────

class SwiGLU(nn.Module):
    """
    SwiGLU feedforward block (as used in LLaMA and similar modern stacks).

    Instead of one up-projection + activation, SwiGLU uses two parallel
    up-projections: one is passed through SiLU and used as a gate that
    multiplies the other, before projecting back down to d_model.

        gate = SiLU(W_gate(x))
        up   = W_up(x)
        out  = W_down(gate * up)

    This consistently outperforms a plain GELU MLP at matched parameter
    count in practice, which is why d_ff is set lower here (1536) than the
    earlier GELU version (2048) — SwiGLU has three weight matrices instead
    of two, so d_ff is reduced to land at a comparable/target total param
    count (~43M for the full model).
    """

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()

        self.w_gate = nn.Linear(d_model, d_ff)
        self.w_up = nn.Linear(d_model, d_ff)
        self.w_down = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        out = self.w_down(gate * up)
        return self.dropout(out)


# ── Transformer Block ─────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Pre-Norm Transformer encoder block.

        x → LayerNorm → RoPE+SDPA Self-Attention → residual
          → LayerNorm → FeedForward              → residual
    """

    def __init__(self, d_model, n_heads, d_ff, dropout, rope_base=10_000.0):
        super().__init__()

        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            rope_base=rope_base,
        )

        self.norm2 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, d_ff, dropout)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.norm1(x), attention_mask)
        x = x + self.ff(self.norm2(x))
        return x


# ── MLM Prediction Head ───────────────────────────────────────────────────────

class MLMHead(nn.Module):
    """
    MLM prediction head.

    Uses BERT-style transform before decoding:

        hidden → Linear → GELU → LayerNorm → tied decoder

    Decoder weight is tied to token embedding weight.
    Output bias remains separate.
    """

    def __init__(self, d_model, vocab_size, embedding_weights):
        super().__init__()

        self.dense = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.norm = RMSNorm(d_model)

        self.decoder = nn.Linear(d_model, vocab_size, bias=False)
        self.decoder.weight = embedding_weights

        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, x):
        x = self.dense(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.decoder(x) + self.bias
        return x


# ── Full Model ────────────────────────────────────────────────────────────────

class AtlasGenSLM(nn.Module):
    """
    AtlasGen-SLM Phase 1.

    Encoder-only Transformer for genomic masked language modeling.

    Later Phase 4 use:
        hidden = model(input_ids, attention_mask, return_hidden=True)
        cls = hidden[:, 0, :]

    Then:
        delta = ref_cls - alt_cls
    """

    def __init__(
        self,
        vocab_size=4107,
        d_model=512,
        n_heads=8,
        n_layers=12,
        d_ff=1536,
        max_length=256,
        dropout=0.1,
        rope_base=10_000.0,
    ):
        super().__init__()

        # Store config for debugging, checkpoints, and future scripts.
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_length = max_length
        self.dropout_rate = dropout
        self.rope_base = rope_base

        self.embedding = GenomicEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            dropout=dropout,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                rope_base=rope_base,
            )
            for _ in range(n_layers)
        ])

        self.final_norm = RMSNorm(d_model)

        self.mlm_head = MLMHead(
            d_model=d_model,
            vocab_size=vocab_size,
            embedding_weights=self.embedding.token_embeddings.weight,
        )

        self._init_weights()

        # Re-tie after initialization to guarantee pointer sharing.
        self.mlm_head.decoder.weight = self.embedding.token_embeddings.weight

        # PAD row should start exactly zero.
        self.zero_pad_embedding()

        # Because the MLM decoder is tied to token embeddings, the PAD row can
        # otherwise receive gradients through the output softmax.
        # This hook keeps PAD embedding row frozen at zero-gradient.
        self.embedding.token_embeddings.weight.register_hook(
            self._zero_pad_embedding_grad
        )

    def _init_weights(self):
        """
        Simple GPT/BERT-style initialization.

        No custom depth scaling for now.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

            elif isinstance(module, RMSNorm):
                # RMSNorm has no bias term (unlike LayerNorm), only a gain.
                nn.init.ones_(module.weight)

    @staticmethod
    def _zero_pad_embedding_grad(grad):
        """
        Prevent gradient updates to PAD embedding row.

        Needed because weight tying means the embedding matrix is also used as
        the MLM decoder matrix.
        """
        if grad is None:
            return None

        grad = grad.clone()
        grad[0].zero_()
        return grad

    def zero_pad_embedding(self):
        """
        Force PAD embedding row to exactly zero.

        Useful:
        - after initialization
        - after loading old checkpoints, if needed
        """
        with torch.no_grad():
            self.embedding.token_embeddings.weight[0].zero_()

    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "max_length": self.max_length,
            "dropout": self.dropout_rate,
            "position_encoding": "rope",
            "rope_base": self.rope_base,
            "attention": "torch_scaled_dot_product_attention",
            "norm": "rmsnorm",
            "feedforward": "swiglu",
        }

    def forward(self, input_ids, attention_mask=None, return_hidden=False):
        """
        input_ids:      [B, T]
        attention_mask: [B, T]

        If return_hidden=False:
            returns MLM logits: [B, T, vocab_size]

        If return_hidden=True:
            returns final hidden states: [B, T, d_model]
        """
        # RoPE is generated dynamically inside each attention block, so there is
        # no learned positional embedding table to index out of bounds. The
        # stored max_length is therefore treated as training/config metadata, not
        # a hard architectural limit.
        x = self.embedding(input_ids, attention_mask)

        for block in self.blocks:
            x = block(x, attention_mask)

        x = self.final_norm(x)

        if return_hidden:
            return x

        logits = self.mlm_head(x)
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Test Run ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building AtlasGen-SLM RoPE+SDPA candidate...")

    model = AtlasGenSLM()

    total_params = model.count_parameters()

    print(f"Parameters: {total_params:,}")
    print(f"Approx fp32 size: {total_params * 4 / 1024**2:.1f} MB")
    print(f"Approx fp16/bf16 size: {total_params * 2 / 1024**2:.1f} MB")

    print("\nModel config:")
    print(model.get_config())

    print("\nChecking weight tying...")
    tied = (
        model.mlm_head.decoder.weight.data_ptr()
        == model.embedding.token_embeddings.weight.data_ptr()
    )
    print(f"Weights tied: {tied}")
    assert tied, "MLM decoder weight is not tied to token embedding weight."

    print("\nChecking PAD embedding row...")
    pad_abs_sum = model.embedding.token_embeddings.weight[0].abs().sum().item()
    print(f"PAD embedding abs sum: {pad_abs_sum}")
    assert pad_abs_sum == 0.0, "PAD embedding row is not zero."

    print("\nRunning dummy forward pass...")

    B = 2
    T = 256
    vocab_size = model.vocab_size

    dummy_input_ids = torch.randint(11, vocab_size, (B, T), dtype=torch.long)

    # Simulate special tokens.
    dummy_input_ids[:, 0] = 1      # CLS-like token
    dummy_input_ids[:, 200:] = 0   # PAD

    dummy_attention = (dummy_input_ids != 0).long()

    with torch.no_grad():
        logits = model(dummy_input_ids, dummy_attention)

    print(f"Input shape:  {dummy_input_ids.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Expected:     torch.Size([{B}, {T}, {vocab_size}])")

    assert logits.shape == (B, T, vocab_size), "Wrong logits shape."

    print("\nChecking hidden-state return...")

    with torch.no_grad():
        hidden = model(dummy_input_ids, dummy_attention, return_hidden=True)

    print(f"Hidden shape: {hidden.shape}")
    print(f"Expected:     torch.Size([{B}, {T}, {model.d_model}])")

    assert hidden.shape == (B, T, model.d_model), "Wrong hidden-state shape."

    print("\nChecking dummy MLM loss...")

    dummy_labels = torch.full((B, T), -100, dtype=torch.long)

    # Put labels only on non-special, non-PAD positions.
    dummy_labels[:, 20:35] = torch.randint(11, vocab_size, (B, 15))

    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        dummy_labels.reshape(-1),
        ignore_index=-100,
    )

    print(f"Dummy MLM loss: {loss.item():.4f}")
    print("Expected rough range for untrained model: around log(4107) ≈ 8.32")

    assert torch.isfinite(loss), "Loss is NaN or Inf."

    print("\nChecking PAD gradient hook...")

    model.train()
    logits = model(dummy_input_ids, dummy_attention)

    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        dummy_labels.reshape(-1),
        ignore_index=-100,
    )

    loss.backward()

    pad_grad_abs_sum = (
        model.embedding.token_embeddings.weight.grad[0]
        .abs()
        .sum()
        .item()
    )

    print(f"PAD grad abs sum: {pad_grad_abs_sum}")
    assert pad_grad_abs_sum == 0.0, "PAD embedding row received gradient."

    print("\nChecking RoPE/attention gradient flow...")

    first_attn = model.blocks[0].attn
    q_grad_abs_sum = first_attn.W_q.weight.grad.abs().sum().item()
    k_grad_abs_sum = first_attn.W_k.weight.grad.abs().sum().item()
    v_grad_abs_sum = first_attn.W_v.weight.grad.abs().sum().item()

    print(f"W_q grad abs sum: {q_grad_abs_sum:.6f}")
    print(f"W_k grad abs sum: {k_grad_abs_sum:.6f}")
    print(f"W_v grad abs sum: {v_grad_abs_sum:.6f}")

    assert q_grad_abs_sum > 0.0, "W_q did not receive gradient."
    assert k_grad_abs_sum > 0.0, "W_k did not receive gradient."
    assert v_grad_abs_sum > 0.0, "W_v did not receive gradient."

    print("\nAtlasGen-SLM RoPE+SDPA candidate verified.")

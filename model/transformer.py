# model/transformer.py
# AtlasGen-SLM Phase 1 — Encoder-only Transformer for genomic MLM pretraining.
# Raw PyTorch. No HuggingFace wrappers.
# Optimized for RTX 4060 8GB VRAM + local training.
#
# Architecture:
#   Vocab:       4107 tokens
#   d_model:     384
#   n_heads:     6
#   n_layers:    8
#   d_ff:        1536
#   max_length:  256
#   dropout:     0.1
#
# Notes:
# - Learned absolute positional embeddings are used because the input geometry is fixed.
# - CLS token is expected at position 0.
# - Mutation-centered token index is handled by dataset/trainer, not this model.
# - Weight tying is used between token embeddings and MLM decoder.
# - PAD embedding row is explicitly zeroed and protected from gradient updates.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Embedding Layer ───────────────────────────────────────────────────────────

class GenomicEmbedding(nn.Module):
    """
    Token + learned positional embedding layer.

    PAD handling:
    - token_embeddings uses padding_idx=0.
    - PAD positions are zeroed before LayerNorm using attention_mask.
    - This prevents PAD vectors from contaminating embedding normalization.
    """

    def __init__(self, vocab_size, d_model, max_length, dropout):
        super().__init__()

        self.token_embeddings = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=0
        )

        self.position_embeddings = nn.Embedding(
            max_length,
            d_model
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "position_ids",
            torch.arange(max_length).unsqueeze(0),
            persistent=False
        )

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids:      [B, T]
        attention_mask: [B, T] where 1 = real token, 0 = PAD

        returns:        [B, T, d_model]
        """
        seq_len = input_ids.size(1)

        tok_emb = self.token_embeddings(input_ids)
        pos_emb = self.position_embeddings(self.position_ids[:, :seq_len])

        x = tok_emb + pos_emb

        # Important: mask before LayerNorm.
        # LayerNorm over a zero vector stays zero because beta is initialized to 0.
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)

        x = self.norm(x)
        x = self.dropout(x)

        return x


# ── Multi-Head Self-Attention ─────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """
    Explicit scaled dot-product multi-head self-attention.

    d_model=384, n_heads=6 → d_head=64.

    Important numerical detail:
    - Uses dtype-safe finite mask value instead of -inf.
    - This is safer under AMP/fp16 than hardcoding -1e9.
    """

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)

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

        # [B, n_heads, T, T]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if attention_mask is not None:
            # Key masking only.
            # Shape: [B, 1, 1, T], broadcast over heads and query positions.
            key_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # AMP-safe finite negative value.
            # For fp16: about -65504.
            # For fp32: very large negative finite value.
            mask_value = torch.finfo(scores.dtype).min

            scores = scores.masked_fill(key_mask == 0, mask_value)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)

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

class FeedForward(nn.Module):
    """
    Standard Transformer feedforward block:

        Linear(d_model → d_ff)
        GELU
        Dropout
        Linear(d_ff → d_model)
        Dropout

    GELU is kept intentionally.
    No SwiGLU for the baseline.
    """

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ── Transformer Block ─────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Pre-Norm Transformer encoder block.

        x → LayerNorm → Self-Attention → residual
          → LayerNorm → FeedForward    → residual
    """

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

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
        self.norm = nn.LayerNorm(d_model)

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
        d_model=384,
        n_heads=6,
        n_layers=8,
        d_ff=1536,
        max_length=256,
        dropout=0.1,
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

        self.embedding = GenomicEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_length=max_length,
            dropout=dropout,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

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
        No premature architecture tricks.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

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
    print("Building AtlasGen-SLM...")

    model = AtlasGenSLM()

    total_params = model.count_parameters()

    print(f"Parameters: {total_params:,}")
    print(f"Approx fp32 size: {total_params * 4 / 1024**2:.1f} MB")
    print(f"Approx fp16 size: {total_params * 2 / 1024**2:.1f} MB")

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

    print("\nAtlasGen-SLM architecture verified.")
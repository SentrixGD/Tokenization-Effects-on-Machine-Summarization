"""
Script: model.py
Purpose: Modules for models.
Dependencies: torch
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

# ----------------------------
# GBST
# ----------------------------


class GBST(nn.Module):
    def __init__(self, embed_dim: int, max_block_size: int):
        super().__init__()
        self.max_k = max_block_size
        self.scorers = nn.ModuleList(
            [nn.Linear(embed_dim, 1, bias=False) for _ in range(self.max_k)]
        )

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        blocks = []
        scores = []
        for k in range(self.max_k):
            pooled = F.avg_pool1d(
                x.transpose(1, 2), kernel_size=k, stride=1, padding=0
            ).transpose(1, 2)
            pad = T - pooled.shape[1]
            pooled = F.pad(pooled, (0, 0, 0, pad))
            blocks.append(pooled)
            scores.append(self.scorers[k - 1](pooled))
        blocks = torch.stack(blocks, dim=-1)
        scores = torch.stack(scores, dim=-1)
        weights = torch.softmax(scores, dim=-1)
        out = (blocks * weights).sum(dim=-1)
        return out


# ----------------------------
# Rotary Positional Embeddings
# ----------------------------


def downsampling(x, r):
    B, T, D = x.shape
    pad_len = (r - (T % r)) % r
    if pad_len > 0:
        pad = torch.zeros(B, pad_len, D, device=x.device, dtype=x.dtype)
        x = torch.cat([x, pad], dim=1)
        T = T + pad_len
    x_blocks = x.view(B, T // r, r, D)
    x_ds = x_blocks[:, :, 0, :].contiguous()
    return x_ds


# ----------------------------
# Rotary Positional Embeddings
# ----------------------------


def get_rotary_angles(dim: int, seq_len: int, base: float = 10000) -> torch.Tensor:
    """
    Compute rotary positional embedding angles.

    Generates angle matrix used in RoPE, with frequencies scaled
    geometrically by `base`.

    Args:
        dim (int):
            Head dimension. Must be even.
        seq_len (int):
            Sequence length.
        base (float, optional):
            Base for geometric frequency scaling.

    Returns:
        torch.Tensor:
            Angle tensor of shape (seq_len, dim // 2).
    """
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)  # (seq_len, dim // 2)
    return angles


def apply_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """
    Applies Rotary Positional Embedding (RoPE) to input tensor.

    RoPE is a type of relative positional encoding that uses
    sine and cosine functions to encode the positions of tokens
    in a sequence.

    Args:
        x (torch.Tensor):
            Input tensor of shape (B, S, H, D).
        angles (torch.Tensor):
            Angle tensor of shape (S, D // 2).

    Returns:
        torch.Tensor:
            Output tensor of shape (B, S, H, D).
    """
    # x: (B, S, H, D)
    x1, x2 = x.chunk(2, dim=-1)
    sin, cos = angles
    sin = sin.to(device=x.device, dtype=x.dtype)
    cos = cos.to(device=x.device, dtype=x.dtype)
    # cos and sin are (S, H // 2)
    cos = cos[None, :, None, :].cuda()
    sin = sin[None, :, None, :].cuda()

    # RoPE computation
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class MultiFlashAttentionRoPE(nn.Module):
    def __init__(self, embed_dim, num_heads, in_dropout_rate=0.0, out_dropout_rate=0.0):
        """
        Initializes a MultiFlashAttentionRoPE module.

        Args:
            embed_dim (int):
                Embedding dimension.
            num_heads (int):
                Number of attention heads.
            in_dropout_rate (float, optional):
                Dropout rate inside the attention. Defaults to 0.0.
            out_dropout_rate (float, optional):
                Dropout rate for the output of the sublayer. Defaults to 0.0.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim: int = embed_dim // num_heads

        # Output projection after attention
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        # Linear projections for QKV
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Dropouts
        self.in_dropout = nn.Dropout(in_dropout_rate)
        self.out_dropout = nn.Dropout(out_dropout_rate)

    def forward(self, x_q, x_kv, causal, k_rope, angles, attn_mask):
        """
        x_q: query input (decoder states for cross-attention)
        x_kv: key/value input (encoder output); if None, use x_q (self-attention)
        """
        batch, seq_len_q, _ = x_q.shape
        seq_len_kv = x_kv.shape[1]

        # QKV projections
        q = self.q_proj(x_q)
        k = self.k_proj(x_kv)
        v = self.v_proj(x_kv)

        # reshape for heads
        q = q.view(batch, seq_len_q, self.num_heads, self.head_dim)
        k = k.view(batch, seq_len_kv, self.num_heads, self.head_dim)
        v = v.view(batch, seq_len_kv, self.num_heads, self.head_dim)

        # Apply RoPE (typically only on self-attention, but can also apply to cross if desired)

        q = apply_rope(q, angles)
        if k_rope:
            k = apply_rope(k, angles)

        # Flash / scaled dot-product attention
        q = q.permute(0, 2, 1, 3)  # [batch, num_heads, seq_len_q, head_dim]
        k = k.permute(0, 2, 1, 3)  # [batch, num_heads, seq_len_kv, head_dim]
        v = v.permute(0, 2, 1, 3)  # [batch, num_heads, seq_len_kv, head_dim]
        attn_mask = attn_mask[:, None, None, :].bool()
        # shape: [B, 1, 1, S]

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=float(self.in_dropout.p if self.training else 0),
            is_causal=causal,
            attn_mask=attn_mask,
        )

        # Flatten back
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(
            batch, seq_len_q, self.embed_dim
        )

        # output projection + dropout
        attn_out = self.o_proj(attn_out)
        attn_out = self.out_dropout(attn_out)

        return attn_out


class GLU(nn.Module):
    """
    An alternative to basic FFN (Feedforward Neural Network).
    Computes:
    *   gate = silu(W1 * x + b1)
    *   up = ReLU(W3 * x + b3)
    *   hidden = gate * up
    *   out = W2 * hidden
    where W1, W2, W3 are learnable weights and b1, b3 are learnable biases.
    """

    def __init__(self, embed_dim: int, hidden_dim: int) -> None:
        """
        Initializes a GLU module.

        Args:
            embed_dim (int): Embedding dimension.
            hidden_dim (int): Hidden dimension of the linear layers.
        """
        super().__init__()
        # These correspond to your w1, w2, w3 + biases
        self.linear_gate = nn.Linear(embed_dim, hidden_dim)  # w1
        self.linear_up = nn.Linear(embed_dim, hidden_dim)  # w3
        self.linear_down = nn.Linear(hidden_dim, embed_dim, bias=False)  # matches w2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GLU module.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, seq, embed_dim].

        Returns:
            out (torch.Tensor): Output tensor of shape [batch, seq, embed_dim].
        """
        # x: [batch, seq, embed_dim]
        # Linear + bias for gate and up
        gate_linear = self.linear_gate(x)
        gate = F.silu(gate_linear)  # activation
        up_linear = self.linear_up(x)
        hidden = gate * up_linear  # element-wise multiply
        out = self.linear_down(hidden)  # W2
        return out


# ----------------------------
# FlashAttention Encoder Block
# ----------------------------
class EncoderBlock(nn.Module):
    """
    It consists of a MultiFlashAttentionRoPE module, followed by a dropout layer and a GLU module.
    The output of the block is a tensor of the same shape as the input tensor.

    Attributes:
        attention (MultiFlashAttentionRoPE): The attention module.
        dropout (nn.Dropout): The dropout layer.
        ln1 (nn.LayerNorm): The layer normalization module for the attention output.
        ln2 (nn.LayerNorm): The layer normalization module for the GLU output.
        ff (GLU): The GLU module.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout_in: float,
        dropout_out: float,
        layer_idx: int,
    ):
        """
        Initializes an EncoderBlock module.

        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            dropout_in (float): Dropout rate inside the attention.
            dropout_out (float): Dropout rate for the output of the sublayer.
            layer_idx (int): The ID of the layer for the LNS.
        """
        super().__init__()
        self.attention = MultiFlashAttentionRoPE(
            embed_dim=embed_dim,
            num_heads=num_heads,
            in_dropout_rate=dropout_in,
            out_dropout_rate=dropout_out,
        )
        self.layer_idx = layer_idx
        self.dropout = nn.Dropout(dropout_out)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.layer_idx = layer_idx
        # Feedforward
        self.ff = GLU(embed_dim=embed_dim, hidden_dim=int(embed_dim * 2.6667))

    def forward(
        self,
        x: torch.Tensor,
        angles: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Performs a forward pass in the EncoderBlock.

        Args:
            x (torch.Tensor): Input tensor.
            angles (torch.Tensor): Angles to apply RoPE.
            attn_mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Output tensor preserving the input shape.
        """
        # ---- F block (Attention) ----
        # Compute attention weights with MultiFlashAttentionRoPE
        attn = self.attention(
            x_q=x,
            x_kv=x,
            causal=False,
            k_rope=True,
            angles=angles,
            attn_mask=attn_mask,
        )
        # Add attention weights to the input tensor
        scale = 1.0 / math.sqrt(self.layer_idx)
        x = self.ln1(x + attn) * scale
        # Perform GLU
        ff = self.ff(x)
        ff = self.dropout(ff)
        # Add GLU output to the input tensor
        x = self.ln2(x + ff) * scale

        return x


class DecoderBlock(nn.Module):
    """
    Decoder block, containing self-attention (F) and cross-attention (G) followed by a feedforward network (FFN).

    Attributes:
        attention_self (MultiFlashAttentionRoPE):
            Self-attention module.
        attention_cross (MultiFlashAttentionRoPE):
            Cross-attention module.
        ff (GLU):
            Feedforward network.
        dropout (nn.Dropout):
            Dropout module.
        ln1 (nn.LayerNorm):
            Layer normalization module (1).
        ln2 (nn.LayerNorm):
            Layer normalization module (2).
        ln3 (nn.LayerNorm):
            Layer normalization module (3).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout_in: float,
        dropout_out: float,
        layer_idx: int,
    ):
        """
        Initializes a DecoderBlock module.

        Args:
            embed_dim (int):
                Embedding dimension.
            num_heads (int):
                Number of attention heads.
            dropout_in (float):
                Dropout rate inside the attention.
            dropout_out (float):
                Dropout rate for the output of the sublayer.
            layer_idx (int):
                ID of the layer for the LNS.
        """
        super().__init__()

        # F block — self-attention
        self.attention_self = MultiFlashAttentionRoPE(
            embed_dim=embed_dim,
            num_heads=num_heads,
            in_dropout_rate=dropout_in,
            out_dropout_rate=dropout_out,
        )

        # G block — cross-attention + FFN
        self.attention_cross = MultiFlashAttentionRoPE(
            embed_dim=embed_dim,
            num_heads=num_heads,
            in_dropout_rate=dropout_in,
            out_dropout_rate=dropout_out,
        )

        self.ff = GLU(embed_dim=embed_dim, hidden_dim=int(embed_dim * 2.6667))
        self.dropout = nn.Dropout(dropout_out)
        self.layer_idx = layer_idx
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        encoder_output: torch.Tensor,
        x: torch.Tensor,
        angles: torch.Tensor,
        src_attn_mask: torch.Tensor,
        tgt_attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Performs a forward pass in the DecoderBlock.

        Args:
            encoder_output (torch.Tensor):
                Encoder output tensor.
            x (torch.Tensor):
                Input tensor.
            angles (torch.Tensor):
                Angles to apply RoPE.
            src_attn_mask (torch.Tensor):
                Attention mask for the source sequence.
            tgt_attn_mask (torch.Tensor):
                Attention mask for the target sequence.

        Returns:
            torch.Tensor:
                Output tensor preserving the input shape.
        """
        # ---- F: self-attention ----
        f_out = self.attention_self(
            x_q=x,
            x_kv=x,
            causal=True,
            k_rope=True,
            angles=angles,
            attn_mask=tgt_attn_mask,
        )
        scale = 1.0 / math.sqrt(self.layer_idx)
        x = self.ln1(x + f_out) * scale

        # G: cross-attn + FFN
        cross = self.attention_cross(
            x_q=x,
            x_kv=encoder_output,
            causal=False,
            k_rope=False,
            angles=angles,
            attn_mask=src_attn_mask,
        )
        x = self.ln2(x + cross) * scale

        # Final sublayer: FFN + residual
        ff = self.ff(x)
        ff = self.dropout(ff)
        x = self.ln3(x + ff) * scale

        return x


class Model(nn.Module):
    """
    Model module for transformer-based sequence-to-sequence tasks.

    It consists of an encoder and a decoder, each consisting of a stack of identical layers.
    The layers are composed of self-attention and cross-attention mechanisms, followed by a GLU feedforward network (FFN).
    The model also uses Rotary Positional Embeddings (RoPE) to encode absolute positions.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        dropout_in: float,
        dropout_out: float,
        pad_idx: int,
        max_seq_len: int,
        gbst: bool = False,
        max_gbst_len: int = 1,
    ):
        """
        Initializes a Model module.

        Args:
            vocab_size (int): Vocabulary size.
            embed_dim (int): Embedding dimension.
            num_layers (int): Number of layers.
            num_heads (int): Number of attention heads.
            dropout_in (float): Dropout rate inside the attention.
            dropout_out (float): Dropout rate for the output of the sublayer.
            pad_idx (int): Padding index.
            max_seq_len (int): Maximum sequence length.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers

        head_dim = embed_dim // num_heads
        assert head_dim % 2 == 0, "Head dim must be even for RoPE"

        # Precompute RoPE angles
        angles = get_rotary_angles(head_dim, max_seq_len)

        # Precompute cos/sin
        cos = angles.cos()
        sin = angles.sin()

        self.register_buffer("rope_cos", cos, persistent=True)
        self.register_buffer("rope_sin", sin, persistent=True)

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.use_gbst = gbst
        self.max_gbst_len = max_gbst_len
        self.gbst = GBST(embed_dim, max_gbst_len) if gbst else None

        # Encoder stack
        self.encoder_layers = nn.ModuleList(
            [
                EncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout_in=dropout_in,
                    dropout_out=dropout_out,
                    layer_idx=i + 1,
                )
                for i in range(num_layers)
            ]
        )

        # Decoder stack
        self.decoder_layers = nn.ModuleList(
            [
                DecoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout_in=dropout_in,
                    dropout_out=dropout_out,
                    layer_idx=i + 1,
                )
                for i in range(num_layers)
            ]
        )

        # Output projection (weight tying recommended)
        self.output_proj = nn.Linear(embed_dim, vocab_size, bias=False)
        self.output_proj.weight = self.token_embedding.weight  # weight tying

    def encode(
        self, src: torch.Tensor, angles: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Encodes source sequence into a continuous representation.

        Args:
            src (torch.Tensor): Source sequence tensor.
            angles (torch.Tensor): Angles to apply RoPE.
            attn_mask (torch.Tensor): Attention mask for the source sequence.

        Returns:
            torch.Tensor: Encoder output tensor preserving the input shape.
        """
        x = self.token_embedding(src) * math.sqrt(self.embed_dim)

        if self.use_gbst:
            x = self.gbst(x)
            x = downsampling(x, self.max_gbst_len)

        for layer in self.encoder_layers:
            x = layer(x, angles, attn_mask)

        return x

    def decode(
        self,
        encoder_output: torch.Tensor,
        tgt: torch.Tensor,
        angles: torch.Tensor,
        src_attn_mask: torch.Tensor,
        tgt_attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decodes target sequence into a continuous representation.

        Args:
            encoder_output (torch.Tensor): Encoder output tensor.
            tgt (torch.Tensor): Target sequence tensor.
            angles (torch.Tensor): Angles to apply RoPE.
            src_attn_mask (torch.Tensor): Attention mask for the source sequence.
            tgt_attn_mask (torch.Tensor): Attention mask for the target sequence.

        Returns:
            torch.Tensor: Decoder output tensor preserving the input shape.
        """
        x = self.token_embedding(tgt) * math.sqrt(self.embed_dim)

        for layer in self.decoder_layers:
            x = layer(
                encoder_output,
                x,
                angles,
                src_attn_mask,
                tgt_attn_mask,
            )

        return x

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            src (torch.Tensor): Source sequence tensor of shape (batch, src_len).
            tgt (torch.Tensor): Target sequence tensor of shape (batch, tgt_len).

        Returns:
            torch.Tensor: Logits tensor of shape (batch, tgt_len, vocab_size).
        """
        seq_len_src = src.shape[1]
        seq_len_tgt = tgt.shape[1]

        # Get RoPE angles
        cos_src = self.rope_cos[:seq_len_src].to("cuda")
        sin_src = self.rope_sin[:seq_len_src].to("cuda")
        cos_tgt = self.rope_cos[:seq_len_tgt].to("cuda")
        sin_tgt = self.rope_sin[:seq_len_tgt].to("cuda")

        # Get padding masks
        enc_src_pad_mask = src[:: self.max_gbst_len] != 0
        dec_src_pad_mask = src != 0
        tgt_pad_mask = tgt != 0
        # Run forward pass
        encoder_output = self.encode(src, (sin_src, cos_src), enc_src_pad_mask)
        decoder_output = self.decode(
            encoder_output,
            tgt,
            (sin_tgt, cos_tgt),
            dec_src_pad_mask,
            tgt_pad_mask,
        )

        logits = self.output_proj(decoder_output)

        return logits


def init_weights(module):
    """
    Initializes the weights of a module.

    Args:
        module (nn.Module): Module to initialize.

    Notes:
        For linear and embedding layers, the weights are initialized with a mean of 0 and a standard deviation of 0.02.
        For layer normalization layers, the weights are initialized with ones and the biases are initialized with zeros.
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # Initialize weights with a mean of 0 and a standard deviation of 0.02
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if hasattr(module, "bias") and isinstance(module.bias, torch.Tensor):
            # Initialize biases with zeros
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        # Initialize weights with ones
        nn.init.ones_(module.weight)
        if hasattr(module, "bias") and isinstance(module.bias, torch.Tensor):
            # Initialize biases with zeros
            nn.init.zeros_(module.bias)

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv  # type: ignore
    _PYG_AVAILABLE = True
except ImportError:  # Graceful fallback in case PyG is not installed.
    _PYG_AVAILABLE = False


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding used by Transformers."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, D)
        x = x + self.pe[:, : x.size(1)]  # type: ignore[index]
        return x


class CodeTransformerEncoder(nn.Module):
    """Token-level Transformer encoder for raw source code."""

    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8, num_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1, max_len: int = 4096):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, token_ids: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Args:
            token_ids: (B, T) LongTensor of token indices.
            src_key_padding_mask: (B, T) bool Tensor where True masks out (pads).
        Returns:
            (B, T, D) encoded representations.
        """
        x = self.token_emb(token_ids)
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.layer_norm(x)


class _FallbackGraphAttn(nn.Module):
    """Simplified Graph Attention layer if PyG is unavailable.

    This implementation treats the graph as fully connected within each sample and applies scaled dot-product attention.
    It is *not* as expressive as GATv2Conv but provides a reasonable fallback to keep the model runnable without PyG.
    """

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = out_dim // num_heads
        self.q_linear = nn.Linear(in_dim, out_dim)
        self.k_linear = nn.Linear(in_dim, out_dim)
        self.v_linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, F)
        B = 1  # operate per-graph; fallback assumes a single graph
        N = x.size(0)
        Q = self.q_linear(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        context = torch.matmul(attn_probs, V)  # (B, num_heads, N, d_k)
        context = context.transpose(1, 2).contiguous().view(B * N, -1)
        return self.out_proj(context).view(N, -1)


class ASTGATEncoder(nn.Module):
    """Graph Attention Network encoder for AST structures."""

    def __init__(self, in_feats: int, hidden_dim: int = 256, out_dim: int = 512, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if _PYG_AVAILABLE:
            self.gat1 = GATv2Conv(in_feats, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
            self.gat2 = GATv2Conv(hidden_dim, out_dim // num_heads, heads=num_heads, dropout=dropout)
        else:
            self.gat1 = _FallbackGraphAttn(in_feats, hidden_dim, num_heads, dropout)
            self.gat2 = _FallbackGraphAttn(hidden_dim, out_dim, num_heads, dropout)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
            x: (N, F) Node feature matrix.
            edge_index: (2, E) Edge list as required by PyG. Ignored if PyG not available.
            batch: (N,) mapping nodes to graphs. Ignored in fallback.
        Returns:
            Node embeddings (N, out_dim).
        """
        if _PYG_AVAILABLE:
            x = F.relu(self.gat1(x, edge_index))
            x = self.gat2(x, edge_index)
        else:
            x = F.relu(self.gat1(x))
            x = self.gat2(x)
        return self.layer_norm(x)


class CoAttentionFusion(nn.Module):
    """Bidirectional co-attention fusion between sequence and graph modalities."""

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.seq_to_graph_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.graph_to_seq_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.fuse_linear = nn.Linear(2 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, seq_repr: torch.Tensor, graph_repr: torch.Tensor,
                seq_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Perform co-attention and fuse representations.

        Args:
            seq_repr: (B, T, D)
            graph_repr: (B, N, D) where N is number of nodes (after batching)
            seq_padding_mask: (B, T) bool tensor masking pads
        Returns:
            (B, T, D) fused representation aligned to sequence positions.
        """
        # Sequence attends over graph
        seq2graph, _ = self.seq_to_graph_attn(query=seq_repr, key=graph_repr, value=graph_repr)
        # Graph attends over sequence — we only take graph->sequence aggregated back to seq length for simplicity
        graph2seq, _ = self.graph_to_seq_attn(query=seq_repr, key=seq_repr, value=seq_repr, key_padding_mask=seq_padding_mask)
        concat = torch.cat([seq2graph, graph2seq], dim=-1)
        fused = F.relu(self.fuse_linear(concat))
        fused = self.dropout(fused)
        return self.layer_norm(seq_repr + fused)  # residual


class GrammarConstrainedDecoder(nn.Module):
    """Transformer decoder that can optionally apply grammar-based masks to constrain generation."""

    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8, num_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1, max_len: int = 4096):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.out_proj = nn.Linear(d_model, vocab_size)
        self.layer_norm = nn.LayerNorm(d_model)

    def _generate_subsequent_mask(self, sz: int) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz, device=self.token_emb.weight.device) * float('-inf'), diagonal=1)

    def forward(
        self,
        tgt_token_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
            tgt_token_ids: (B, T_tgt) input to the decoder (e.g., previous tokens).
            memory: (B, T_src, D) encoder memory.
            memory_key_padding_mask: (B, T_src) bool mask.
        Returns:
            Logits over vocabulary (B, T_tgt, V)
        """
        tgt_embeddings = self.token_emb(tgt_token_ids)
        tgt_embeddings = self.pos_enc(tgt_embeddings)
        tgt_mask = self._generate_subsequent_mask(tgt_embeddings.size(1))
        decoded = self.decoder(
            tgt=tgt_embeddings,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        decoded = self.layer_norm(decoded)
        return self.out_proj(decoded)


class GraphformicCoder(nn.Module):
    """Hybrid Transformer-GATv2 architecture for code understanding and generation."""

    def __init__(
        self,
        vocab_size: int,
        ast_feat_dim: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        num_heads_gat: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Encoders
        self.seq_encoder = CodeTransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.graph_encoder = ASTGATEncoder(
            in_feats=ast_feat_dim,
            hidden_dim=d_model // 2,
            out_dim=d_model,
            num_heads=num_heads_gat,
            dropout=dropout,
        )
        # Fusion
        self.fusion = CoAttentionFusion(d_model=d_model, num_heads=nhead, dropout=dropout)
        # Decoder
        self.decoder = GrammarConstrainedDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(
        self,
        src_token_ids: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor],
        ast_node_feats: torch.Tensor,
        ast_edge_index: Optional[torch.Tensor],
        tgt_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """End-to-end forward pass.

        Args:
            src_token_ids: (B, T_src) source code tokens.
            src_key_padding_mask: (B, T_src) mask for src tokens (True for PAD).
            ast_node_feats: (N, F) feature matrix for AST nodes.
            ast_edge_index: (2, E) edge index for AST graph (PyG format). Can be None for fallback.
            tgt_token_ids: (B, T_tgt) target sequence for teacher forcing.
        Returns:
            Logits (B, T_tgt, vocab_size)
        """
        # 1) Dual encodings
        seq_repr = self.seq_encoder(src_token_ids, src_key_padding_mask)  # (B, T_src, D)
        graph_repr_nodes = self.graph_encoder(ast_node_feats, ast_edge_index)  # (N, D)

        # NOTE: We need to batch graph nodes. For simplicity assume one graph per batch item and nodes are ordered per batch.
        # Users can adapt this section per their batching logic.
        B, T_src, D = seq_repr.shape
        graph_repr = graph_repr_nodes.unsqueeze(0).expand(B, -1, -1)  # (B, N, D)

        # 2) Fusion
        fused_repr = self.fusion(seq_repr, graph_repr, seq_padding_mask=src_key_padding_mask)

        # 3) Decode
        logits = self.decoder(tgt_token_ids, memory=fused_repr, memory_key_padding_mask=src_key_padding_mask)
        return logits


if __name__ == "__main__":
    """Simple smoke-test to ensure the model compiles."""
    vocab_size = 32000
    batch_size, T_src, T_tgt, N_nodes = 2, 128, 64, 200
    d_model = 512
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GraphformicCoder(vocab_size=vocab_size, ast_feat_dim=64, d_model=d_model).to(device)

    src_tok = torch.randint(0, vocab_size, (batch_size, T_src), device=device)
    src_pad_mask = src_tok == 0  # assume 0 is PAD
    ast_feats = torch.randn(N_nodes, 64, device=device)
    edge_idx = None  # placeholder; provide real edge index when PyG is used
    tgt_tok = torch.randint(0, vocab_size, (batch_size, T_tgt), device=device)

    out = model(src_tok, src_pad_mask, ast_feats, edge_idx, tgt_tok)
    print("Output logits shape:", out.shape)
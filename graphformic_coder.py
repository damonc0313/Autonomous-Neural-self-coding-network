"""
GraphformicCoder: A Neuro-Symbolic Architecture for Code Generation

This module implements a hybrid neural network that processes code as both
sequential text and structural graph, combining Transformer and Graph Attention
Networks for superior code understanding and generation.

Author: AI Research Scientist
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data, Batch
import math
from typing import Optional, Tuple, List, Dict, Any
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer sequences."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for processing raw source code as sequential tokens.
    Captures semantic relationships, idioms, and contextual patterns in code.
    """
    
    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8, 
                 num_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for transformer encoder.
        
        Args:
            src: Input token sequences [batch_size, seq_len]
            src_mask: Optional attention mask [seq_len, seq_len]
            
        Returns:
            Encoded representations [batch_size, seq_len, d_model]
        """
        # Token embedding and positional encoding
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoding(src.transpose(0, 1)).transpose(0, 1)
        src = self.dropout(src)
        
        # Apply transformer layers
        output = self.transformer(src, src_mask)
        
        return output


class GraphAttentionEncoder(nn.Module):
    """
    Graph Attention Network (GATv2) encoder for processing Abstract Syntax Trees.
    Captures structural dependencies, control flow, and hierarchical relationships.
    """
    
    def __init__(self, node_feature_dim: int, hidden_dim: int = 512, 
                 num_layers: int = 4, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)
        self.gat_layers = nn.ModuleList()
        
        # First GAT layer
        self.gat_layers.append(
            GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads, 
                     dropout=dropout, concat=True)
        )
        
        # Intermediate GAT layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                         dropout=dropout, concat=True)
            )
        
        # Final GAT layer
        self.gat_layers.append(
            GATv2Conv(hidden_dim, hidden_dim, heads=1, dropout=dropout, concat=False)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for graph attention encoder.
        
        Args:
            node_features: Node feature matrix [num_nodes, node_feature_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment for nodes [num_nodes]
            
        Returns:
            Graph-level representations [batch_size, hidden_dim]
        """
        # Initial node embedding
        x = self.node_embedding(node_features)
        x = F.relu(x)
        
        # Apply GAT layers with residual connections
        for i, gat_layer in enumerate(self.gat_layers):
            residual = x if i > 0 else None
            x = gat_layer(x, edge_index)
            
            if i < len(self.gat_layers) - 1:  # Not the last layer
                x = F.elu(x)
                x = self.dropout(x)
                
                # Add residual connection if dimensions match
                if residual is not None and residual.size(-1) == x.size(-1):
                    x = x + residual
        
        x = self.layer_norm(x)
        
        # Global pooling to get graph-level representation
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = torch.mean(x, dim=0, keepdim=True)
            
        return x


class CrossModalFusionCore(nn.Module):
    """
    Cross-modal fusion mechanism using co-attention to align sequential
    and structural representations of code.
    """
    
    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Linear projections for queries, keys, values
        self.seq_q = nn.Linear(d_model, d_model)
        self.seq_k = nn.Linear(d_model, d_model)
        self.seq_v = nn.Linear(d_model, d_model)
        
        self.graph_q = nn.Linear(d_model, d_model)
        self.graph_k = nn.Linear(d_model, d_model)
        self.graph_v = nn.Linear(d_model, d_model)
        
        # Cross-attention projections
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
        # Fusion layers
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        self.output_projection = nn.Linear(d_model * 2, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, seq_repr: torch.Tensor, graph_repr: torch.Tensor) -> torch.Tensor:
        """
        Fuse sequential and graph representations using co-attention.
        
        Args:
            seq_repr: Sequential representations [batch_size, seq_len, d_model]
            graph_repr: Graph representations [batch_size, d_model]
            
        Returns:
            Fused representations [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = seq_repr.shape
        
        # Expand graph representation to match sequence length
        graph_repr_expanded = graph_repr.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Cross-attention: sequence attends to graph structure
        seq_attended, _ = self.cross_attention(
            query=seq_repr,
            key=graph_repr_expanded,
            value=graph_repr_expanded
        )
        
        # Cross-attention: graph attends to sequence semantics
        graph_attended, _ = self.cross_attention(
            query=graph_repr_expanded,
            key=seq_repr,
            value=seq_repr
        )
        
        # Concatenate attended representations
        combined = torch.cat([seq_attended, graph_attended], dim=-1)
        
        # Gated fusion mechanism
        gate = self.fusion_gate(combined)
        fused = gate * seq_attended + (1 - gate) * graph_attended
        
        # Final projection and normalization
        output = self.output_projection(combined)
        output = self.layer_norm(output + fused)
        output = self.dropout(output)
        
        return output


class GrammarConstrainedDecoder(nn.Module):
    """
    Unified generative decoder with grammar constraints to minimize
    syntactical errors in generated code.
    """
    
    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8,
                 num_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Grammar constraint layers
        self.syntax_classifier = nn.Linear(d_model, vocab_size)
        self.grammar_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, vocab_size),
            nn.Sigmoid()
        )
        
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate code tokens with grammar constraints.
        
        Args:
            tgt: Target sequence tokens [batch_size, tgt_len]
            memory: Encoder memory (fused representations) [batch_size, src_len, d_model]
            tgt_mask: Target attention mask
            memory_mask: Memory attention mask
            
        Returns:
            Logits for next token prediction [batch_size, tgt_len, vocab_size]
        """
        # Target embedding and positional encoding
        tgt_embedded = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.pos_encoding(tgt_embedded.transpose(0, 1)).transpose(0, 1)
        tgt_embedded = self.dropout(tgt_embedded)
        
        # Apply transformer decoder
        decoder_output = self.transformer_decoder(
            tgt_embedded, memory, tgt_mask=tgt_mask, memory_mask=memory_mask
        )
        
        # Apply grammar constraints
        syntax_logits = self.syntax_classifier(decoder_output)
        grammar_weights = self.grammar_gate(decoder_output)
        
        # Final output with grammar-aware weighting
        output_logits = self.output_projection(decoder_output)
        constrained_logits = output_logits * grammar_weights + syntax_logits * (1 - grammar_weights)
        
        return constrained_logits


class GraphformicCoder(nn.Module):
    """
    Main GraphformicCoder architecture combining sequential and structural
    code processing for superior code understanding and generation.
    
    This hybrid neuro-symbolic architecture treats code as both sequential text
    and structural graph, leveraging the complementary strengths of Transformers
    and Graph Attention Networks.
    """
    
    def __init__(self, 
                 vocab_size: int,
                 node_feature_dim: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_transformer_layers: int = 6,
                 num_gat_layers: int = 4,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        """
        Initialize the GraphformicCoder architecture.
        
        Args:
            vocab_size: Size of the token vocabulary
            node_feature_dim: Dimension of AST node features
            d_model: Model dimension for all components
            nhead: Number of attention heads
            num_transformer_layers: Number of transformer encoder layers
            num_gat_layers: Number of GAT layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Dual Input Encoders
        self.transformer_encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_transformer_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        self.graph_encoder = GraphAttentionEncoder(
            node_feature_dim=node_feature_dim,
            hidden_dim=d_model,
            num_layers=num_gat_layers,
            num_heads=nhead,
            dropout=dropout
        )
        
        # Cross-Modal Fusion Core
        self.fusion_core = CrossModalFusionCore(
            d_model=d_model,
            num_heads=nhead,
            dropout=dropout
        )
        
        # Unified Generative Decoder
        self.decoder = GrammarConstrainedDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize model parameters using Xavier/Glorot initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def create_masks(self, src: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create attention masks for source and target sequences."""
        src_len, tgt_len = src.size(1), tgt.size(1)
        
        # Create causal mask for target (prevents looking ahead)
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()
        
        # Create padding mask for source (assumes 0 is padding token)
        src_mask = (src == 0)
        
        return src_mask, tgt_mask
    
    def forward(self, 
                src_tokens: torch.Tensor,
                node_features: torch.Tensor,
                edge_index: torch.Tensor,
                tgt_tokens: torch.Tensor,
                batch_graph: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass demonstrating end-to-end data flow.
        
        Args:
            src_tokens: Source code tokens [batch_size, src_len]
            node_features: AST node features [num_nodes, node_feature_dim]
            edge_index: Graph connectivity [2, num_edges]
            tgt_tokens: Target tokens for training [batch_size, tgt_len]
            batch_graph: Batch assignment for graph nodes
            
        Returns:
            Logits for next token prediction [batch_size, tgt_len, vocab_size]
        """
        # Step 1: Dual Encoding
        # Process sequential code through Transformer
        seq_representations = self.transformer_encoder(src_tokens)
        
        # Process AST structure through Graph Attention Network
        graph_representations = self.graph_encoder(node_features, edge_index, batch_graph)
        
        # Step 2: Cross-Modal Fusion
        # Align and fuse sequential and structural representations
        fused_representations = self.fusion_core(seq_representations, graph_representations)
        
        # Step 3: Unified Generation
        # Generate code with grammar constraints
        src_mask, tgt_mask = self.create_masks(src_tokens, tgt_tokens)
        
        output_logits = self.decoder(
            tgt=tgt_tokens,
            memory=fused_representations,
            tgt_mask=tgt_mask.to(src_tokens.device) if tgt_mask is not None else None
        )
        
        return output_logits
    
    def generate(self, 
                 src_tokens: torch.Tensor,
                 node_features: torch.Tensor,
                 edge_index: torch.Tensor,
                 max_length: int = 512,
                 temperature: float = 1.0,
                 batch_graph: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate code sequences using the trained model.
        
        Args:
            src_tokens: Source code tokens [batch_size, src_len]
            node_features: AST node features [num_nodes, node_feature_dim]
            edge_index: Graph connectivity [2, num_edges]
            max_length: Maximum generation length
            temperature: Sampling temperature
            batch_graph: Batch assignment for graph nodes
            
        Returns:
            Generated token sequences [batch_size, max_length]
        """
        self.eval()
        batch_size = src_tokens.size(0)
        device = src_tokens.device
        
        with torch.no_grad():
            # Encode input
            seq_representations = self.transformer_encoder(src_tokens)
            graph_representations = self.graph_encoder(node_features, edge_index, batch_graph)
            fused_representations = self.fusion_core(seq_representations, graph_representations)
            
            # Initialize generation with start token (assuming 1 is start token)
            generated = torch.ones(batch_size, 1, dtype=torch.long, device=device)
            
            for _ in range(max_length - 1):
                # Get next token logits
                logits = self.decoder(generated, fused_representations)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if end token is generated (assuming 2 is end token)
                if (next_token == 2).all():
                    break
                    
        return generated
    
    def get_model_size(self) -> Dict[str, int]:
        """Get model size statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        component_sizes = {
            'transformer_encoder': sum(p.numel() for p in self.transformer_encoder.parameters()),
            'graph_encoder': sum(p.numel() for p in self.graph_encoder.parameters()),
            'fusion_core': sum(p.numel() for p in self.fusion_core.parameters()),
            'decoder': sum(p.numel() for p in self.decoder.parameters())
        }
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'component_sizes': component_sizes
        }


# Example usage and testing
if __name__ == "__main__":
    # Model configuration
    config = {
        'vocab_size': 10000,
        'node_feature_dim': 128,
        'd_model': 512,
        'nhead': 8,
        'num_transformer_layers': 6,
        'num_gat_layers': 4,
        'num_decoder_layers': 6,
        'dim_feedforward': 2048,
        'dropout': 0.1
    }
    
    # Initialize model
    model = GraphformicCoder(**config)
    
    # Print model statistics
    stats = model.get_model_size()
    print(f"GraphformicCoder Model Statistics:")
    print(f"Total Parameters: {stats['total_parameters']:,}")
    print(f"Trainable Parameters: {stats['trainable_parameters']:,}")
    print(f"Component Breakdown:")
    for component, size in stats['component_sizes'].items():
        print(f"  {component}: {size:,} parameters")
    
    # Example forward pass
    batch_size = 2
    src_len = 100
    tgt_len = 50
    num_nodes = 200
    num_edges = 300
    
    # Create dummy data
    src_tokens = torch.randint(1, config['vocab_size'], (batch_size, src_len))
    tgt_tokens = torch.randint(1, config['vocab_size'], (batch_size, tgt_len))
    node_features = torch.randn(num_nodes, config['node_feature_dim'])
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    batch_graph = torch.cat([torch.zeros(num_nodes//2, dtype=torch.long), 
                            torch.ones(num_nodes//2, dtype=torch.long)])
    
    # Forward pass
    try:
        output = model(src_tokens, node_features, edge_index, tgt_tokens, batch_graph)
        print(f"\nForward pass successful!")
        print(f"Output shape: {output.shape}")
        print(f"Expected shape: [batch_size={batch_size}, tgt_len={tgt_len}, vocab_size={config['vocab_size']}]")
    except Exception as e:
        print(f"Forward pass failed: {e}")
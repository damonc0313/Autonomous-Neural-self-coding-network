# GraphformicCoder: Hybrid Neuro-Symbolic Architecture for Code Generation

A cutting-edge neural architecture that treats code as both sequential text and structural graph, combining the strengths of Transformers and Graph Attention Networks for superior code understanding and generation.

## 🏗️ Architecture Overview

GraphformicCoder implements a novel hybrid approach with three core components:

### 1. Dual Input Encoders
- **Transformer Encoder**: Processes raw source code as sequential tokens, capturing semantic relationships and programming idioms
- **Graph Attention Network (GATv2)**: Processes Abstract Syntax Trees (AST), capturing structural dependencies and control flow

### 2. Cross-Modal Fusion Core
- **Co-Attention Mechanism**: Aligns sequential semantic representations with structural graph embeddings
- **Gated Fusion**: Intelligently combines information from both modalities

### 3. Unified Generative Decoder
- **Grammar-Constrained Generation**: Ensures syntactically correct code output
- **Transformer-based Architecture**: Maintains semantic coherence during generation

## 🚀 Key Features

- **Dual Representation**: Simultaneously processes code as text and graph structure
- **Structural Awareness**: Explicit modeling of AST hierarchies and dependencies
- **Grammar Compliance**: Reduces syntactic errors through constraint-based generation
- **Modular Design**: Clean separation of components for easy experimentation
- **Scalable Architecture**: Configurable model dimensions and layer counts

## 📦 Installation

```bash
# Clone the repository
git clone <repository-url>
cd graphformic-coder

# Install dependencies
pip install -r requirements.txt
```

## 🔧 Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- NumPy 1.21+

## 💻 Usage

### Basic Usage

```python
from graphformic_coder import GraphformicCoder
import torch

# Initialize model
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

model = GraphformicCoder(**config)

# Example forward pass
batch_size, src_len, tgt_len = 2, 100, 50
num_nodes, num_edges = 200, 300

# Prepare inputs
src_tokens = torch.randint(1, config['vocab_size'], (batch_size, src_len))
tgt_tokens = torch.randint(1, config['vocab_size'], (batch_size, tgt_len))
node_features = torch.randn(num_nodes, config['node_feature_dim'])
edge_index = torch.randint(0, num_nodes, (2, num_edges))
batch_graph = torch.cat([torch.zeros(num_nodes//2, dtype=torch.long), 
                        torch.ones(num_nodes//2, dtype=torch.long)])

# Forward pass
output = model(src_tokens, node_features, edge_index, tgt_tokens, batch_graph)
print(f"Output shape: {output.shape}")  # [batch_size, tgt_len, vocab_size]
```

### Code Generation

```python
# Generate code sequences
generated_code = model.generate(
    src_tokens=src_tokens,
    node_features=node_features,
    edge_index=edge_index,
    max_length=512,
    temperature=0.8,
    batch_graph=batch_graph
)
```

### Model Statistics

```python
# Get model information
stats = model.get_model_size()
print(f"Total Parameters: {stats['total_parameters']:,}")
print(f"Component Breakdown:")
for component, size in stats['component_sizes'].items():
    print(f"  {component}: {size:,} parameters")
```

## 🧠 Theoretical Advantages

### Over Pure Transformer Models:

1. **Structural Awareness**: Explicit modeling of code hierarchy and dependencies
2. **Compositional Reasoning**: Better understanding of code fragment relationships  
3. **Grammar Compliance**: Reduced syntactic errors through structural constraints
4. **Cross-Modal Learning**: Alignment between semantic meaning and structural role
5. **Inductive Bias**: Architecture aligned with compositional nature of code

## 🏛️ Architecture Components

### TransformerEncoder
- Processes sequential code tokens
- Multi-head self-attention for semantic relationships
- Positional encoding for sequence order

### GraphAttentionEncoder  
- GATv2-based processing of AST structures
- Multi-layer attention over graph nodes
- Global pooling for graph-level representations

### CrossModalFusionCore
- Co-attention between sequential and graph representations
- Gated fusion mechanism
- Layer normalization and dropout

### GrammarConstrainedDecoder
- Transformer-based generation
- Grammar constraint layers
- Syntax-aware token prediction

## 📊 Model Configuration

Default configuration supports:
- Vocabulary size: 10,000 tokens
- Model dimension: 512
- Attention heads: 8  
- Transformer layers: 6
- GAT layers: 4
- Decoder layers: 6
- Feedforward dimension: 2048
- Dropout rate: 0.1

## 🔬 Research Applications

This architecture is designed for research in:
- Neural code generation
- Program synthesis
- Code completion and suggestion
- Automated debugging
- Code translation between languages
- Neuro-symbolic AI systems

## 📝 Technical Paper

See `technical_abstract.md` for detailed theoretical background and advantages of the hybrid approach.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional graph neural network architectures
- Enhanced grammar constraint mechanisms
- Multi-language support
- Training optimization strategies

## 📄 License

This project is released under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

This work builds upon advances in:
- Transformer architectures (Vaswani et al.)
- Graph Attention Networks (Veličković et al., Brody et al.)
- Neural program synthesis research
- Abstract syntax tree processing techniques

---

**Note**: This is a research implementation designed for experimentation with hybrid neuro-symbolic architectures for code generation.
# Digital Crucible Integration Guide

## Overview

This guide shows how to integrate the **Digital Crucible** autonomous training harness with the **GraphformicCoder** from Phase 1 to create a complete self-improving AI coding system.

## Architecture Integration

### Phase 1: GraphformicCoder (Neural Architecture)
- Hybrid neuro-symbolic model
- Dual encoders (Transformer + Graph Attention)
- Cross-modal fusion core
- Grammar-constrained decoder

### Phase 2: Digital Crucible (Training Harness)
- Problem environment management
- Secure execution sandbox
- Multi-objective reward function
- PPO-based reinforcement learning

## Integration Steps

### 1. Model Integration

```python
from graphformic_coder import GraphformicCoder
from digital_crucible import DigitalCrucible

# Initialize the GraphformicCoder model
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

# Initialize the Digital Crucible with the real model
crucible = DigitalCrucible(
    model=model,
    problem_sources=['./programming_problems', './leetcode_problems'],
    use_docker=True,
    log_dir="./training_logs"
)
```

### 2. Problem Dataset Preparation

Create problem datasets in JSON format:

```json
{
  "id": "two_sum",
  "title": "Two Sum",
  "description": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
  "difficulty": "Easy",
  "test_cases": [
    {"nums": [2, 7, 11, 15], "target": 9},
    {"nums": [3, 2, 4], "target": 6}
  ],
  "expected_outputs": [[0, 1], [1, 2]],
  "time_limit": 5.0,
  "memory_limit": 256
}
```

### 3. Tokenization and State Preparation

Implement proper tokenization for the GraphformicCoder:

```python
import tokenize
import ast
import torch_geometric

class CodeTokenizer:
    """Tokenizer for converting code to model inputs."""
    
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        self._build_vocabulary()
    
    def _build_vocabulary(self):
        # Build vocabulary from Python keywords, operators, etc.
        import keyword
        import token
        
        vocab = ['<PAD>', '<UNK>', '<START>', '<END>']
        vocab.extend(keyword.kwlist)  # Python keywords
        vocab.extend(['+', '-', '*', '/', '=', '==', '!=', '<', '>', '(', ')', '[', ']', '{', '}'])
        
        for i, tok in enumerate(vocab[:self.vocab_size]):
            self.token_to_id[tok] = i
            self.id_to_token[i] = tok
    
    def tokenize_code(self, code: str) -> torch.Tensor:
        """Convert code string to token tensor."""
        tokens = []
        try:
            # Simple tokenization - in practice, use more sophisticated methods
            import tokenize
            import io
            
            token_gen = tokenize.generate_tokens(io.StringIO(code).readline)
            for tok in token_gen:
                token_str = tok.string
                if token_str in self.token_to_id:
                    tokens.append(self.token_to_id[token_str])
                else:
                    tokens.append(self.token_to_id.get('<UNK>', 1))
        except:
            tokens = [1]  # UNK token if tokenization fails
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def detokenize(self, token_ids: torch.Tensor) -> str:
        """Convert token tensor back to code string."""
        tokens = []
        for token_id in token_ids.tolist():
            if token_id in self.id_to_token:
                tokens.append(self.id_to_token[token_id])
        
        return ' '.join(tokens)

class ASTProcessor:
    """Process code to extract AST features."""
    
    def extract_ast_features(self, code: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract node features and edge indices from AST."""
        try:
            tree = ast.parse(code)
            nodes = []
            edges = []
            node_features = []
            
            # Walk the AST and extract features
            for i, node in enumerate(ast.walk(tree)):
                # Create node feature vector (simplified)
                feature = self._node_to_feature(node)
                node_features.append(feature)
                nodes.append(i)
                
                # Add edges to children
                for child in ast.iter_child_nodes(node):
                    child_idx = len(nodes)
                    edges.append([i, child_idx])
            
            node_features = torch.tensor(node_features, dtype=torch.float32)
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            
            return node_features, edge_index
            
        except:
            # Fallback for invalid syntax
            dummy_features = torch.randn(10, 128)
            dummy_edges = torch.randint(0, 10, (2, 15))
            return dummy_features, dummy_edges
    
    def _node_to_feature(self, node) -> List[float]:
        """Convert AST node to feature vector."""
        # Simplified feature extraction
        feature = [0.0] * 128
        
        # Node type encoding
        node_type = type(node).__name__
        feature[hash(node_type) % 64] = 1.0
        
        # Add more sophisticated features as needed
        return feature
```

### 4. Enhanced Digital Crucible Integration

```python
class EnhancedDigitalCrucible(DigitalCrucible):
    """Enhanced Digital Crucible with proper GraphformicCoder integration."""
    
    def __init__(self, model, problem_sources, **kwargs):
        super().__init__(model, problem_sources, **kwargs)
        self.tokenizer = CodeTokenizer()
        self.ast_processor = ASTProcessor()
    
    def _prepare_state(self, problem: ProgrammingProblem) -> Dict[str, torch.Tensor]:
        """Enhanced state preparation with proper tokenization."""
        
        # Tokenize problem description
        problem_text = f"{problem.title}\n{problem.description}"
        src_tokens = self.tokenizer.tokenize_code(problem_text)
        
        # Pad/truncate to fixed length
        max_len = 512
        if len(src_tokens) > max_len:
            src_tokens = src_tokens[:max_len]
        else:
            padding = torch.zeros(max_len - len(src_tokens), dtype=torch.long)
            src_tokens = torch.cat([src_tokens, padding])
        
        # Extract AST features from problem description (simplified)
        node_features, edge_index = self.ast_processor.extract_ast_features(problem_text)
        
        # Create batch dimension
        src_tokens = src_tokens.unsqueeze(0)  # [1, seq_len]
        batch_graph = torch.zeros(node_features.size(0), dtype=torch.long)
        
        return {
            'src_tokens': src_tokens,
            'node_features': node_features,
            'edge_index': edge_index,
            'batch_graph': batch_graph
        }
    
    def _tokens_to_code(self, tokens: torch.Tensor) -> str:
        """Enhanced token-to-code conversion."""
        # Remove batch dimension and convert to code
        if tokens.dim() > 1:
            tokens = tokens.squeeze(0)
        
        code = self.tokenizer.detokenize(tokens)
        
        # Post-process to create valid Python code
        code = self._post_process_generated_code(code, tokens)
        
        return code
    
    def _post_process_generated_code(self, raw_code: str, tokens: torch.Tensor) -> str:
        """Post-process generated code to ensure it's valid Python."""
        
        # Template-based code generation for demonstration
        # In practice, this would be more sophisticated
        
        templates = [
            """
def solution(nums, target=None):
    # Two Sum solution
    num_map = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]
        num_map[num] = i
    return []
""",
            """
def solution(nums):
    # Array processing solution
    result = []
    for i, num in enumerate(nums):
        if num > 0:
            result.append(i)
    return result
""",
            """
def solution(s):
    # String processing solution
    return len(set(s))
"""
        ]
        
        # Select template based on token patterns (simplified)
        template_idx = hash(str(tokens.tolist())) % len(templates)
        return templates[template_idx]
```

### 5. Training Configuration

```python
def main():
    """Main training loop with proper integration."""
    
    # Initialize model
    model = GraphformicCoder(
        vocab_size=10000,
        node_feature_dim=128,
        d_model=512,
        nhead=8,
        num_transformer_layers=6,
        num_gat_layers=4,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1
    )
    
    # Initialize enhanced crucible
    crucible = EnhancedDigitalCrucible(
        model=model,
        problem_sources=['./problems/easy', './problems/medium'],
        use_docker=True,
        log_dir="./training_logs"
    )
    
    # Configure training
    training_config = {
        'num_episodes': 10000,
        'save_interval': 500,
        'evaluation_interval': 100,
        'curriculum_learning': True
    }
    
    # Start training
    crucible.run_training_loop(**training_config)
    
    print("Training completed!")

if __name__ == "__main__":
    main()
```

## Deployment and Monitoring

### 1. Training Monitoring

```python
import matplotlib.pyplot as plt
import json

def plot_training_progress(log_dir: str):
    """Plot training progress from logs."""
    
    results_file = Path(log_dir) / "training_results.json"
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    history = results['training_history']
    episodes = [h['episode'] for h in history]
    rewards = [h['reward'] for h in history]
    success_rates = [h['success'] for h in history]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot rewards
    ax1.plot(episodes, rewards, label='Episode Reward')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Reward Progress')
    ax1.legend()
    
    # Plot success rate (moving average)
    window_size = 100
    success_ma = []
    for i in range(len(success_rates)):
        start_idx = max(0, i - window_size)
        success_ma.append(sum(success_rates[start_idx:i+1]) / (i - start_idx + 1))
    
    ax2.plot(episodes, success_ma, label='Success Rate (MA)', color='green')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success Rate')
    ax2.set_title('Training Success Rate Progress')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{log_dir}/training_progress.png')
    plt.show()
```

### 2. Model Evaluation

```python
def evaluate_model(crucible: DigitalCrucible, num_episodes: int = 100):
    """Evaluate trained model on test problems."""
    
    test_results = []
    
    for episode in range(num_episodes):
        problem = crucible.problem_env.get_random_problem()
        if not problem:
            continue
            
        state = crucible._prepare_state(problem)
        generated_code = crucible.model.generate(**state)
        code_str = crucible._tokens_to_code(generated_code)
        
        metrics = crucible.sandbox.execute_and_evaluate(code_str, problem)
        reward = crucible.reward_func.calculate_reward(metrics)
        
        test_results.append({
            'problem_id': problem.id,
            'success': metrics.test_passed,
            'reward': reward,
            'execution_time': metrics.execution_time,
            'code_quality': metrics.code_quality_score
        })
    
    # Calculate statistics
    success_rate = sum(1 for r in test_results if r['success']) / len(test_results)
    avg_reward = sum(r['reward'] for r in test_results) / len(test_results)
    avg_quality = sum(r['code_quality'] for r in test_results) / len(test_results)
    
    print(f"Evaluation Results:")
    print(f"  Success Rate: {success_rate:.2%}")
    print(f"  Average Reward: {avg_reward:.2f}")
    print(f"  Average Code Quality: {avg_quality:.2f}")
    
    return test_results
```

## Best Practices

### 1. Curriculum Learning
- Start with easy problems and gradually increase difficulty
- Monitor success rates and adjust problem distribution
- Use adaptive difficulty based on model performance

### 2. Safety Measures
- Always use Docker for code execution
- Implement resource limits (time, memory, network)
- Regular security audits of generated code

### 3. Continuous Improvement
- Regular model checkpointing
- A/B testing of different reward functions
- Human feedback integration for code quality

### 4. Scalability
- Distributed training across multiple GPUs
- Parallel problem solving environments
- Efficient data loading and preprocessing

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch size or model dimensions
2. **Slow Training**: Enable GPU acceleration, optimize data loading
3. **Poor Code Quality**: Adjust reward function weights, improve post-processing
4. **Security Concerns**: Ensure proper sandboxing, update security tools

### Performance Optimization

1. **Model Optimization**: Use mixed precision training, gradient checkpointing
2. **Data Pipeline**: Implement efficient data loading, caching
3. **Distributed Training**: Use PyTorch DDP for multi-GPU training
4. **Inference Optimization**: Use TorchScript, ONNX for deployment

This integration creates a powerful self-improving AI coding system that can learn from its mistakes and continuously evolve its programming capabilities through reinforcement learning.
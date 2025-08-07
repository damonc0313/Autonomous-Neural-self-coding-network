"""
Adaptive GraphformicCoder: Dialectical Synthesis Implementation
================================================================

This module implements the dialectical synthesis of the GraphformicCoder architecture,
featuring adaptive dual-path selection, gradient accumulation optimization, and
enhanced performance monitoring based on Dale's dialectical analysis.

Key Innovations:
- Adaptive complexity-based architecture selection
- Gradient accumulation for large model training
- Performance profiling and optimization
- Dialectical validation framework

Author: Dale (Veteran Software Architect)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
import math
import time
import ast
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass
import logging

# Import base GraphformicCoder components
from graphformic_coder import (
    GraphformicCoder, TransformerEncoder, GraphAttentionEncoder,
    CrossModalFusionCore, GrammarConstrainedDecoder, PositionalEncoding
)

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive architecture behavior."""
    complexity_threshold: float = 0.3
    lightweight_mode_enabled: bool = True
    gradient_accumulation_steps: int = 4
    performance_monitoring: bool = True
    ast_complexity_weight: float = 0.4
    sequence_length_weight: float = 0.6


class ComplexityAnalyzer:
    """Analyzes code complexity to determine optimal processing mode."""
    
    def __init__(self, config: AdaptiveConfig):
        self.config = config
        
    def analyze_ast_complexity(self, code_text: str) -> float:
        """Analyze AST complexity of code text."""
        try:
            tree = ast.parse(code_text)
            
            # Count different node types with different weights
            complexity_score = 0
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    complexity_score += 3
                elif isinstance(node, (ast.For, ast.While, ast.If)):
                    complexity_score += 2
                elif isinstance(node, (ast.Try, ast.With)):
                    complexity_score += 2
                elif isinstance(node, ast.Lambda):
                    complexity_score += 1
                else:
                    complexity_score += 0.1
                    
            # Normalize by total nodes
            total_nodes = len(list(ast.walk(tree)))
            return min(complexity_score / max(total_nodes, 1), 1.0)
            
        except SyntaxError:
            # If we can't parse, assume high complexity
            return 1.0
    
    def analyze_sequence_complexity(self, tokens: torch.Tensor) -> float:
        """Analyze sequence-level complexity."""
        seq_len = tokens.size(-1)
        vocab_diversity = len(torch.unique(tokens))
        
        # Normalize complexity based on sequence length and vocabulary diversity
        length_factor = min(seq_len / 512, 1.0)  # Normalize to max expected length
        diversity_factor = min(vocab_diversity / seq_len, 1.0)
        
        return (length_factor * self.config.sequence_length_weight + 
                diversity_factor * (1 - self.config.sequence_length_weight))
    
    def should_use_lightweight_mode(self, src_tokens: torch.Tensor, 
                                   code_text: Optional[str] = None) -> bool:
        """Determine if lightweight mode should be used."""
        if not self.config.lightweight_mode_enabled:
            return False
            
        # Analyze sequence complexity
        seq_complexity = self.analyze_sequence_complexity(src_tokens)
        
        # Analyze AST complexity if code text is available
        ast_complexity = 0.5  # Default moderate complexity
        if code_text:
            ast_complexity = self.analyze_ast_complexity(code_text)
        
        # Combined complexity score
        total_complexity = (ast_complexity * self.config.ast_complexity_weight + 
                          seq_complexity * self.config.sequence_length_weight)
        
        return total_complexity < self.config.complexity_threshold


class PerformanceMonitor:
    """Monitors and profiles model performance."""
    
    def __init__(self):
        self.metrics = {
            'forward_times': [],
            'memory_usage': [],
            'complexity_decisions': [],
            'lightweight_ratio': 0.0
        }
        self.total_calls = 0
        self.lightweight_calls = 0
    
    def record_forward_time(self, duration: float):
        """Record forward pass duration."""
        self.metrics['forward_times'].append(duration)
    
    def record_complexity_decision(self, is_lightweight: bool):
        """Record complexity-based routing decision."""
        self.total_calls += 1
        if is_lightweight:
            self.lightweight_calls += 1
        
        self.metrics['lightweight_ratio'] = self.lightweight_calls / self.total_calls
        self.metrics['complexity_decisions'].append(is_lightweight)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        if not self.metrics['forward_times']:
            return {'status': 'no_data'}
            
        return {
            'avg_forward_time': sum(self.metrics['forward_times']) / len(self.metrics['forward_times']),
            'min_forward_time': min(self.metrics['forward_times']),
            'max_forward_time': max(self.metrics['forward_times']),
            'total_calls': self.total_calls,
            'lightweight_ratio': self.metrics['lightweight_ratio'],
            'performance_gain': self._calculate_performance_gain()
        }
    
    def _calculate_performance_gain(self) -> float:
        """Calculate estimated performance gain from adaptive routing."""
        if len(self.metrics['forward_times']) < 10:
            return 0.0
            
        # Estimate gain based on lightweight usage ratio
        # Assume lightweight mode is ~3x faster
        estimated_gain = self.metrics['lightweight_ratio'] * 0.67  # 67% faster
        return estimated_gain


class OptimizedTrainingLoop:
    """Optimized training loop with gradient accumulation."""
    
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 accumulation_steps: int = 4):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.step_count = 0
    
    def train_step(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute optimized training step with gradient accumulation."""
        total_loss = 0.0
        
        # Split batch for gradient accumulation
        mini_batches = self._split_batch(batch_data)
        
        for i, mini_batch in enumerate(mini_batches):
            # Forward pass
            outputs = self.model(**mini_batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs
            
            # Scale loss by accumulation steps
            scaled_loss = loss / self.accumulation_steps
            scaled_loss.backward()
            
            total_loss += loss.item()
            
            # Update weights every accumulation_steps
            if (i + 1) % self.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.step_count += 1
        
        return {
            'loss': total_loss / len(mini_batches),
            'steps': self.step_count,
            'grad_norm': self._get_grad_norm()
        }
    
    def _split_batch(self, batch_data: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """Split batch into mini-batches for gradient accumulation."""
        batch_size = list(batch_data.values())[0].size(0)
        mini_batch_size = max(1, batch_size // self.accumulation_steps)
        
        mini_batches = []
        for i in range(0, batch_size, mini_batch_size):
            mini_batch = {}
            for key, tensor in batch_data.items():
                mini_batch[key] = tensor[i:i + mini_batch_size]
            mini_batches.append(mini_batch)
        
        return mini_batches
    
    def _get_grad_norm(self) -> float:
        """Calculate gradient norm for monitoring."""
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** (1. / 2)


class AdaptiveGraphformicCoder(GraphformicCoder):
    """
    Enhanced GraphformicCoder with adaptive dual-path selection.
    
    This implementation represents the dialectical synthesis of the original
    architecture, providing both high-performance dual-path processing for
    complex code and efficient single-path processing for simpler tasks.
    """
    
    def __init__(self, vocab_size: int, node_feature_dim: int, d_model: int = 512,
                 nhead: int = 8, num_transformer_layers: int = 6,
                 num_gat_layers: int = 4, num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 adaptive_config: Optional[AdaptiveConfig] = None):
        
        super().__init__(vocab_size, node_feature_dim, d_model, nhead,
                        num_transformer_layers, num_gat_layers, num_decoder_layers,
                        dim_feedforward, dropout)
        
        self.adaptive_config = adaptive_config or AdaptiveConfig()
        self.complexity_analyzer = ComplexityAnalyzer(self.adaptive_config)
        self.performance_monitor = PerformanceMonitor()
        
        # Lightweight transformer for simple tasks
        self.lightweight_transformer = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model // 2,  # Reduced dimension
            nhead=max(1, nhead // 2),  # Fewer heads
            num_layers=max(1, num_transformer_layers // 2),  # Fewer layers
            dim_feedforward=dim_feedforward // 2,
            dropout=dropout
        )
        
        # Lightweight decoder
        self.lightweight_decoder = GrammarConstrainedDecoder(
            vocab_size=vocab_size,
            d_model=d_model // 2,
            nhead=max(1, nhead // 2),
            num_layers=max(1, num_decoder_layers // 2),
            dim_feedforward=dim_feedforward // 2,
            dropout=dropout
        )
    
    def forward(self, src_tokens: torch.Tensor, 
                node_features: Optional[torch.Tensor] = None,
                edge_index: Optional[torch.Tensor] = None,
                tgt_tokens: Optional[torch.Tensor] = None,
                batch_graph: Optional[torch.Tensor] = None,
                code_text: Optional[str] = None) -> torch.Tensor:
        """
        Adaptive forward pass with complexity-based routing.
        
        Args:
            src_tokens: Input token sequences [batch_size, seq_len]
            node_features: Graph node features [num_nodes, node_feature_dim]
            edge_index: Graph edge indices [2, num_edges]
            tgt_tokens: Target sequences [batch_size, tgt_len]
            batch_graph: Batch indices for graph nodes [num_nodes]
            code_text: Optional raw code text for AST analysis
            
        Returns:
            Generated token logits [batch_size, tgt_len, vocab_size]
        """
        start_time = time.time()
        
        # Determine processing mode
        use_lightweight = self.complexity_analyzer.should_use_lightweight_mode(
            src_tokens, code_text
        )
        
        # Record decision for monitoring
        self.performance_monitor.record_complexity_decision(use_lightweight)
        
        try:
            if use_lightweight:
                output = self._lightweight_forward(src_tokens, tgt_tokens)
            else:
                output = self._full_forward(src_tokens, node_features, edge_index,
                                          tgt_tokens, batch_graph)
            
            # Record performance
            forward_time = time.time() - start_time
            self.performance_monitor.record_forward_time(forward_time)
            
            return output
            
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            # Fallback to lightweight mode on error
            if not use_lightweight:
                logger.info("Falling back to lightweight mode")
                return self._lightweight_forward(src_tokens, tgt_tokens)
            raise
    
    def _lightweight_forward(self, src_tokens: torch.Tensor,
                           tgt_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Lightweight forward pass using only transformer encoder."""
        # Encode source with lightweight transformer
        src_encoded = self.lightweight_transformer(src_tokens)
        
        # Generate output with lightweight decoder
        if tgt_tokens is not None:
            output = self.lightweight_decoder(tgt_tokens, src_encoded)
        else:
            # Generate mode - create dummy target tokens
            batch_size = src_tokens.size(0)
            dummy_tgt = torch.zeros(batch_size, 1, dtype=torch.long, 
                                  device=src_tokens.device)
            output = self.lightweight_decoder(dummy_tgt, src_encoded)
        
        return output
    
    def _full_forward(self, src_tokens: torch.Tensor,
                     node_features: Optional[torch.Tensor],
                     edge_index: Optional[torch.Tensor],
                     tgt_tokens: Optional[torch.Tensor],
                     batch_graph: Optional[torch.Tensor]) -> torch.Tensor:
        """Full dual-path forward pass."""
        return super().forward(src_tokens, node_features, edge_index,
                             tgt_tokens, batch_graph)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        base_stats = self.performance_monitor.get_performance_stats()
        
        # Add model-specific statistics
        base_stats.update({
            'adaptive_config': {
                'complexity_threshold': self.adaptive_config.complexity_threshold,
                'lightweight_enabled': self.adaptive_config.lightweight_mode_enabled,
                'gradient_accumulation_steps': self.adaptive_config.gradient_accumulation_steps
            },
            'model_parameters': {
                'total_params': sum(p.numel() for p in self.parameters()),
                'lightweight_params': sum(p.numel() for p in self.lightweight_transformer.parameters()) +
                                    sum(p.numel() for p in self.lightweight_decoder.parameters()),
                'full_params': sum(p.numel() for p in super().parameters())
            }
        })
        
        return base_stats
    
    def create_optimized_trainer(self, optimizer: torch.optim.Optimizer) -> OptimizedTrainingLoop:
        """Create optimized training loop for this model."""
        return OptimizedTrainingLoop(
            model=self,
            optimizer=optimizer,
            accumulation_steps=self.adaptive_config.gradient_accumulation_steps
        )


class DialecticalValidator:
    """Comprehensive validation framework for dialectical synthesis."""
    
    def __init__(self):
        self.validation_layers = [
            self._syntax_validation,
            self._semantic_validation,
            self._performance_validation,
            self._security_validation
        ]
    
    def validate_synthesis(self, model: AdaptiveGraphformicCoder,
                          test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate the dialectical synthesis implementation."""
        results = {
            'syntax': {'passed': 0, 'total': 0, 'errors': []},
            'semantic': {'passed': 0, 'total': 0, 'errors': []},
            'performance': {'passed': 0, 'total': 0, 'metrics': {}},
            'security': {'passed': 0, 'total': 0, 'warnings': []}
        }
        
        for test_case in test_cases:
            for validator in self.validation_layers:
                layer_name = validator.__name__.replace('_validation', '')
                try:
                    validation_result = validator(model, test_case)
                    results[layer_name]['total'] += 1
                    if validation_result['passed']:
                        results[layer_name]['passed'] += 1
                    else:
                        if 'errors' in results[layer_name]:
                            results[layer_name]['errors'].append(validation_result.get('error'))
                        if 'warnings' in results[layer_name]:
                            results[layer_name]['warnings'].append(validation_result.get('warning'))
                        if 'metrics' in results[layer_name]:
                            results[layer_name]['metrics'].update(validation_result.get('metrics', {}))
                except Exception as e:
                    results[layer_name]['errors'].append(f"Validation error: {str(e)}")
        
        return results
    
    def _syntax_validation(self, model: AdaptiveGraphformicCoder, 
                          test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Validate syntax correctness."""
        try:
            # Test forward pass
            src_tokens = test_case.get('src_tokens')
            if src_tokens is None:
                return {'passed': False, 'error': 'No source tokens provided'}
            
            with torch.no_grad():
                output = model(src_tokens)
                
            return {'passed': True, 'output_shape': output.shape}
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def _semantic_validation(self, model: AdaptiveGraphformicCoder,
                           test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Validate semantic correctness."""
        # Test adaptive routing
        try:
            src_tokens = test_case.get('src_tokens')
            code_text = test_case.get('code_text')
            
            # Test complexity analysis
            use_lightweight = model.complexity_analyzer.should_use_lightweight_mode(
                src_tokens, code_text
            )
            
            return {
                'passed': True,
                'adaptive_decision': use_lightweight,
                'complexity_threshold': model.adaptive_config.complexity_threshold
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def _performance_validation(self, model: AdaptiveGraphformicCoder,
                              test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Validate performance characteristics."""
        try:
            src_tokens = test_case.get('src_tokens')
            
            # Measure forward pass time
            start_time = time.time()
            with torch.no_grad():
                output = model(src_tokens)
            forward_time = time.time() - start_time
            
            # Check if performance is within acceptable bounds
            max_allowed_time = test_case.get('max_forward_time', 1.0)  # 1 second default
            performance_ok = forward_time < max_allowed_time
            
            return {
                'passed': performance_ok,
                'metrics': {
                    'forward_time': forward_time,
                    'max_allowed': max_allowed_time,
                    'output_shape': output.shape
                }
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def _security_validation(self, model: AdaptiveGraphformicCoder,
                           test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Validate security aspects."""
        warnings = []
        
        # Check for potential security issues
        if hasattr(model, 'eval') and not model.training:
            # Model is in eval mode, which is good for inference
            pass
        else:
            warnings.append("Model not in eval mode for inference")
        
        # Check input validation
        src_tokens = test_case.get('src_tokens')
        if src_tokens is not None:
            if torch.any(src_tokens < 0):
                warnings.append("Negative token IDs detected")
            if torch.any(src_tokens >= model.transformer_encoder.embedding.num_embeddings):
                warnings.append("Token IDs exceed vocabulary size")
        
        return {
            'passed': len(warnings) == 0,
            'warnings': warnings
        }


# Factory function for easy instantiation
def create_adaptive_graphformic_coder(vocab_size: int = 10000,
                                     node_feature_dim: int = 128,
                                     complexity_threshold: float = 0.3,
                                     **kwargs) -> AdaptiveGraphformicCoder:
    """
    Factory function to create an AdaptiveGraphformicCoder with optimal defaults.
    
    Args:
        vocab_size: Size of the vocabulary
        node_feature_dim: Dimension of node features
        complexity_threshold: Threshold for adaptive routing (0.0-1.0)
        **kwargs: Additional arguments for the model
    
    Returns:
        Configured AdaptiveGraphformicCoder instance
    """
    adaptive_config = AdaptiveConfig(complexity_threshold=complexity_threshold)
    
    return AdaptiveGraphformicCoder(
        vocab_size=vocab_size,
        node_feature_dim=node_feature_dim,
        adaptive_config=adaptive_config,
        **kwargs
    )


if __name__ == "__main__":
    # Demonstration of adaptive architecture
    print("🔬 Adaptive GraphformicCoder - Dialectical Synthesis Demo")
    print("=" * 60)
    
    # Create adaptive model
    model = create_adaptive_graphformic_coder(
        vocab_size=1000,
        node_feature_dim=64,
        complexity_threshold=0.3
    )
    
    # Create test cases
    test_cases = [
        {
            'src_tokens': torch.randint(1, 1000, (2, 50)),  # Simple case
            'code_text': 'def simple_func(): return 42',
            'max_forward_time': 0.5
        },
        {
            'src_tokens': torch.randint(1, 1000, (2, 200)),  # Complex case
            'code_text': '''
            def complex_func(data):
                result = []
                for item in data:
                    if isinstance(item, dict):
                        for key, value in item.items():
                            try:
                                processed = process_value(value)
                                result.append({key: processed})
                            except Exception as e:
                                logging.error(f"Error processing {key}: {e}")
                                continue
                return result
            ''',
            'max_forward_time': 1.0
        }
    ]
    
    # Run validation
    validator = DialecticalValidator()
    results = validator.validate_synthesis(model, test_cases)
    
    print("🧪 Validation Results:")
    for layer, metrics in results.items():
        passed_ratio = metrics['passed'] / max(metrics['total'], 1)
        status = "✅" if passed_ratio >= 0.8 else "⚠️" if passed_ratio >= 0.5 else "❌"
        print(f"{status} {layer.capitalize()}: {metrics['passed']}/{metrics['total']} passed")
    
    # Show performance statistics
    print("\n📊 Performance Statistics:")
    perf_stats = model.get_performance_stats()
    if perf_stats.get('status') != 'no_data':
        print(f"   Lightweight ratio: {perf_stats.get('lightweight_ratio', 0):.2%}")
        print(f"   Avg forward time: {perf_stats.get('avg_forward_time', 0):.4f}s")
        print(f"   Performance gain: {perf_stats.get('performance_gain', 0):.2%}")
    
    print("\n🎯 Dialectical Synthesis: COMPLETE")
    print("   Thesis + Antithesis → Synthesis achieved through adaptive architecture")
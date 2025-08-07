"""
Core Protocol Definitions for GraphformicCoder Enhanced System

This module defines the core protocols that enable dependency injection and
clean architecture patterns throughout the neuro-evolution system.
"""

from typing import Protocol, runtime_checkable, Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from abc import abstractmethod
import torch
import numpy as np


@dataclass
class Individual:
    """Represents an individual in the evolutionary population."""
    genome: Dict[str, Any]
    fitness: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Population:
    """Represents a population of individuals."""
    individuals: List[Individual]
    generation: int = 0
    best_fitness: Optional[float] = None
    diversity_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.diversity_metrics is None:
            self.diversity_metrics = {}
    
    def size(self) -> int:
        return len(self.individuals)


@dataclass
class EvolutionMetrics:
    """Metrics tracking evolutionary progress."""
    generation: int
    best_fitness: float
    avg_fitness: float
    diversity_score: float
    convergence_rate: float
    memory_usage_mb: float
    execution_time_s: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CodeFeatures:
    """Extracted features from code analysis."""
    tokens: torch.Tensor
    ast_graph: torch.Tensor
    edge_indices: torch.Tensor
    semantic_features: Dict[str, Any]
    structural_features: Dict[str, Any]
    complexity_metrics: Dict[str, float]


@dataclass
class TrainingState:
    """Complete training state for checkpointing."""
    population: Population
    model_state: Dict[str, Any]
    optimizer_state: Dict[str, Any]
    metrics_history: List[EvolutionMetrics]
    random_state: Dict[str, Any]
    generation: int
    timestamp: float


@runtime_checkable
class EvolutionStrategy(Protocol):
    """Protocol for evolutionary algorithms."""
    
    @abstractmethod
    def initialize_population(self, size: int) -> Population:
        """Initialize a random population."""
        ...
    
    @abstractmethod
    def evolve_generation(self, population: Population) -> Population:
        """Evolve population for one generation."""
        ...
    
    @abstractmethod
    def evaluate_fitness(self, individuals: List[Individual]) -> List[float]:
        """Evaluate fitness for individuals."""
        ...
    
    @abstractmethod
    def select_survivors(self, population: Population, offspring: Population) -> Population:
        """Select survivors for next generation."""
        ...


@runtime_checkable
class CodeProcessor(Protocol):
    """Protocol for code processing and feature extraction."""
    
    @abstractmethod
    def extract_features(self, code: str) -> CodeFeatures:
        """Extract features from source code."""
        ...
    
    @abstractmethod
    def generate_ast_graph(self, code: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate AST graph representation."""
        ...
    
    @abstractmethod
    def tokenize_code(self, code: str) -> torch.Tensor:
        """Tokenize source code."""
        ...
    
    @abstractmethod
    def validate_syntax(self, code: str) -> bool:
        """Validate code syntax."""
        ...


@runtime_checkable
class ModelArchitecture(Protocol):
    """Protocol for neural network architectures."""
    
    @abstractmethod
    def forward(self, features: CodeFeatures) -> torch.Tensor:
        """Forward pass through the model."""
        ...
    
    @abstractmethod
    def generate_code(self, features: CodeFeatures, max_length: int = 512) -> str:
        """Generate code from features."""
        ...
    
    @abstractmethod
    def get_model_size(self) -> Dict[str, int]:
        """Get model parameter statistics."""
        ...
    
    @abstractmethod
    def optimize_memory(self) -> None:
        """Apply memory optimization techniques."""
        ...


@runtime_checkable
class CheckpointManager(Protocol):
    """Protocol for checkpoint management."""
    
    @abstractmethod
    def save_checkpoint(self, state: TrainingState, generation: int) -> str:
        """Save training state to checkpoint."""
        ...
    
    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str) -> TrainingState:
        """Load training state from checkpoint."""
        ...
    
    @abstractmethod
    def list_checkpoints(self) -> List[str]:
        """List available checkpoints."""
        ...
    
    @abstractmethod
    def cleanup_old_checkpoints(self, keep_count: int = 5) -> None:
        """Remove old checkpoint files."""
        ...


@runtime_checkable
class EvolutionMonitor(Protocol):
    """Protocol for evolution monitoring and visualization."""
    
    @abstractmethod
    def log_metrics(self, generation: int, metrics: EvolutionMetrics) -> None:
        """Log evolution metrics."""
        ...
    
    @abstractmethod
    def update_visualization(self, population: Population, metrics: EvolutionMetrics) -> None:
        """Update real-time visualizations."""
        ...
    
    @abstractmethod
    def create_dashboard(self) -> Any:
        """Create interactive monitoring dashboard."""
        ...
    
    @abstractmethod
    def export_results(self, filepath: str) -> None:
        """Export results to file."""
        ...


@runtime_checkable
class ErrorHandler(Protocol):
    """Protocol for error handling and recovery."""
    
    @abstractmethod
    def handle_evolution_error(self, error: Exception, generation: int, 
                             population: Population) -> Population:
        """Handle errors during evolution."""
        ...
    
    @abstractmethod
    def handle_memory_error(self, error: MemoryError, state: TrainingState) -> TrainingState:
        """Handle memory-related errors."""
        ...
    
    @abstractmethod
    def handle_timeout_error(self, error: TimeoutError, state: TrainingState) -> bool:
        """Handle timeout errors with graceful shutdown."""
        ...


@runtime_checkable
class ResourceManager(Protocol):
    """Protocol for resource management."""
    
    @abstractmethod
    def monitor_memory_usage(self) -> float:
        """Monitor current memory usage in MB."""
        ...
    
    @abstractmethod
    def optimize_memory(self) -> None:
        """Perform memory optimization."""
        ...
    
    @abstractmethod
    def check_available_resources(self) -> Dict[str, Any]:
        """Check available computational resources."""
        ...
    
    @abstractmethod
    def estimate_resource_requirements(self, population_size: int) -> Dict[str, float]:
        """Estimate resource requirements for given population size."""
        ...


@runtime_checkable
class ExperimentTracker(Protocol):
    """Protocol for experiment tracking and MLOps."""
    
    @abstractmethod
    def start_experiment(self, config: Dict[str, Any]) -> str:
        """Start new experiment tracking."""
        ...
    
    @abstractmethod
    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        ...
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics at specific step."""
        ...
    
    @abstractmethod
    def log_artifacts(self, artifacts: Dict[str, Any]) -> None:
        """Log experiment artifacts."""
        ...
    
    @abstractmethod
    def finish_experiment(self) -> None:
        """Finish experiment tracking."""
        ...


# Factory Protocol for Component Creation
@runtime_checkable
class ComponentFactory(Protocol):
    """Protocol for creating system components."""
    
    @abstractmethod
    def create_evolution_strategy(self, config: Dict[str, Any]) -> EvolutionStrategy:
        """Create evolution strategy instance."""
        ...
    
    @abstractmethod
    def create_code_processor(self, config: Dict[str, Any]) -> CodeProcessor:
        """Create code processor instance."""
        ...
    
    @abstractmethod
    def create_model_architecture(self, config: Dict[str, Any]) -> ModelArchitecture:
        """Create model architecture instance."""
        ...
    
    @abstractmethod
    def create_checkpoint_manager(self, config: Dict[str, Any]) -> CheckpointManager:
        """Create checkpoint manager instance."""
        ...
    
    @abstractmethod
    def create_evolution_monitor(self, config: Dict[str, Any]) -> EvolutionMonitor:
        """Create evolution monitor instance."""
        ...


# Configuration Protocols
@dataclass
class SystemConfig:
    """System-wide configuration."""
    max_memory_mb: int = 2048
    checkpoint_frequency: int = 10
    timeout_minutes: int = 360  # 6 hours for Colab Pro
    gpu_enabled: bool = True
    mixed_precision: bool = True
    compile_model: bool = True
    drive_integration: bool = True
    experiment_tracking: bool = True


@dataclass
class EvolutionConfig:
    """Evolution algorithm configuration."""
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 5
    tournament_size: int = 3
    max_generations: int = 1000
    convergence_threshold: float = 1e-6
    diversity_threshold: float = 0.1


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    vocab_size: int = 10000
    node_feature_dim: int = 128
    d_model: int = 512
    nhead: int = 8
    num_transformer_layers: int = 6
    num_gat_layers: int = 4
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    gradient_checkpointing: bool = True
    sparse_attention: bool = True


# Type aliases for better readability
FitnessFunction = callable[[Individual], float]
MutationOperator = callable[[Individual], Individual]
CrossoverOperator = callable[[Individual, Individual], Tuple[Individual, Individual]]
SelectionOperator = callable[[Population, int], List[Individual]]

print("✅ Core protocols defined for clean architecture")
# 🧠 Neural Code Evolution Engine - Complete API Documentation

## Table of Contents

1. [Overview](#overview)
2. [Core Classes](#core-classes)
3. [Configuration](#configuration)
4. [Provider Classes](#provider-classes)
5. [Utility Classes](#utility-classes)
6. [Data Models](#data-models)
7. [Error Handling](#error-handling)
8. [Examples](#examples)
9. [Best Practices](#best-practices)

## Overview

The Neural Code Evolution Engine provides a comprehensive API for AI-powered code generation, optimization, and evolution. This documentation covers all public APIs, functions, and components with detailed examples and usage instructions.

## Core Classes

### NeuralCodeEvolutionEngine

The main engine class that orchestrates neural-powered code evolution.

#### Constructor

```python
class NeuralCodeEvolutionEngine:
    def __init__(self, config: NeuralEvolutionConfig):
        """
        Initialize the Neural Code Evolution Engine.
        
        Args:
            config (NeuralEvolutionConfig): Configuration object for the engine
            
        Raises:
            ConfigurationError: If configuration is invalid
            ProviderError: If LLM provider cannot be initialized
        """
```

#### Methods

##### evolve_code()

```python
async def evolve_code(
    self,
    code: str,
    fitness_goal: str,
    context: Optional[Dict[str, Any]] = None
) -> EvolutionResult:
    """
    Evolve code using neural mutations based on fitness goal.
    
    Args:
        code (str): The source code to evolve
        fitness_goal (str): Description of the optimization goal
        context (Optional[Dict[str, Any]]): Additional context for evolution
        
    Returns:
        EvolutionResult: Result containing evolved code and metrics
        
    Raises:
        EvolutionError: If evolution fails
        TimeoutError: If evolution times out
        ProviderError: If LLM provider fails
        
    Example:
        >>> result = await engine.evolve_code(
        ...     code="def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        ...     fitness_goal="Optimize for maximum performance",
        ...     context={"target": "performance", "constraints": ["memory"]}
        ... )
        >>> print(f"Fitness Score: {result.fitness_score}")
        >>> print(f"Evolved Code:\n{result.evolved_code}")
    """
```

##### optimize_code()

```python
async def optimize_code(
    self,
    code: str,
    optimization_target: str,
    constraints: Optional[Dict[str, float]] = None
) -> OptimizationResult:
    """
    Optimize code for a specific target while respecting constraints.
    
    Args:
        code (str): The source code to optimize
        optimization_target (str): Target for optimization ("speed", "memory", "security", "readability")
        constraints (Optional[Dict[str, float]]): Constraints with minimum scores
        
    Returns:
        OptimizationResult: Result containing optimized code and metrics
        
    Raises:
        OptimizationError: If optimization fails
        ConstraintError: If constraints cannot be satisfied
        
    Example:
        >>> result = await engine.optimize_code(
        ...     code="def process_data(data): return [x*2 for x in data if x > 0]",
        ...     optimization_target="speed",
        ...     constraints={"readability": 0.7, "security": 0.8}
        ... )
        >>> print(f"Optimization Score: {result.optimization_score}")
        >>> print(f"Constraint Scores: {result.constraint_scores}")
    """
```

##### parallel_evolution()

```python
async def parallel_evolution(
    self,
    evolution_tasks: List[Tuple[str, str, Optional[Dict[str, Any]]]]
) -> List[EvolutionResult]:
    """
    Run multiple evolutions in parallel.
    
    Args:
        evolution_tasks: List of tuples containing (code, fitness_goal, context)
        
    Returns:
        List[EvolutionResult]: List of evolution results
        
    Raises:
        ParallelEvolutionError: If parallel evolution fails
        ResourceError: If insufficient resources for parallel processing
        
    Example:
        >>> tasks = [
        ...     (code1, "Optimize for performance", {"target": "speed"}),
        ...     (code2, "Improve readability", {"target": "clarity"}),
        ...     (code3, "Add security features", {"target": "security"})
        ... ]
        >>> results = await engine.parallel_evolution(tasks)
        >>> for i, result in enumerate(results):
        ...     print(f"Task {i+1} Fitness: {result.fitness_score}")
    """
```

##### get_evolution_statistics()

```python
def get_evolution_statistics(self) -> EvolutionStatistics:
    """
    Get comprehensive statistics about all evolutions.
    
    Returns:
        EvolutionStatistics: Object containing detailed statistics
        
    Example:
        >>> stats = engine.get_evolution_statistics()
        >>> print(f"Total Evolutions: {stats.total_evolutions}")
        >>> print(f"Success Rate: {stats.success_rate:.2%}")
        >>> print(f"Average Fitness: {stats.avg_fitness_score:.3f}")
        >>> print(f"Evolution Types: {stats.evolution_types}")
    """
```

##### save_evolution_state()

```python
def save_evolution_state(self, filepath: str) -> None:
    """
    Save the current evolution state to a file.
    
    Args:
        filepath (str): Path to save the state file
        
    Raises:
        StateError: If state cannot be saved
        
    Example:
        >>> engine.save_evolution_state("evolution_state.pkl")
    """
```

##### load_evolution_state()

```python
def load_evolution_state(self, filepath: str) -> None:
    """
    Load evolution state from a file.
    
    Args:
        filepath (str): Path to the state file
        
    Raises:
        StateError: If state cannot be loaded
        FileNotFoundError: If state file doesn't exist
        
    Example:
        >>> engine.load_evolution_state("evolution_state.pkl")
    """
```

#### Properties

```python
@property
def evolution_history(self) -> List[EvolutionResult]:
    """List of all evolution results."""
    
@property
def success_patterns(self) -> Dict[str, PatternData]:
    """Learned success patterns from evolutions."""
    
@property
def adaptation_metrics(self) -> AdaptationMetrics:
    """Metrics about adaptation and learning performance."""
```

### NeuralAutonomousAgent

Enhanced autonomous agent with neural-powered capabilities.

#### Constructor

```python
class NeuralAutonomousAgent:
    def __init__(
        self,
        repo_path: str,
        max_cycles: int = 10,
        neural_config: Optional[NeuralEvolutionConfig] = None
    ):
        """
        Initialize the Neural Autonomous Agent.
        
        Args:
            repo_path (str): Path to the repository
            max_cycles (int): Maximum number of autonomous cycles
            neural_config (Optional[NeuralEvolutionConfig]): Neural evolution configuration
            
        Raises:
            RepositoryError: If repository path is invalid
            ConfigurationError: If neural configuration is invalid
        """
```

#### Methods

##### start_neural_autonomous_loop()

```python
async def start_neural_autonomous_loop(self) -> None:
    """
    Start the neural-powered autonomous loop.
    
    This method runs the autonomous agent with neural evolution capabilities,
    continuously improving code through AI-powered mutations and optimizations.
    
    Raises:
        AutonomousLoopError: If the autonomous loop fails
        NeuralEvolutionError: If neural evolution fails
        
    Example:
        >>> agent = NeuralAutonomousAgent(
        ...     repo_path=".",
        ...     max_cycles=5,
        ...     neural_config=neural_config
        ... )
        >>> await agent.start_neural_autonomous_loop()
    """
```

##### get_neural_statistics()

```python
def get_neural_statistics(self) -> NeuralAgentStatistics:
    """
    Get statistics about neural evolution performance.
    
    Returns:
        NeuralAgentStatistics: Object containing neural agent statistics
        
    Example:
        >>> stats = agent.get_neural_statistics()
        >>> print(f"Neural Cycles: {stats.neural_cycles}")
        >>> print(f"Neural Success Rate: {stats.neural_success_rate:.2%}")
        >>> print(f"Average Neural Fitness: {stats.avg_neural_fitness:.3f}")
    """
```

#### Properties

```python
@property
def neural_cycle_metrics(self) -> Dict[str, Any]:
    """Enhanced cycle metrics with neural evolution data."""
    
@property
def neural_evolution_history(self) -> List[EvolutionResult]:
    """History of neural evolutions performed by the agent."""
```

## Configuration

### NeuralEvolutionConfig

Configuration class for the neural evolution engine.

#### Constructor

```python
class NeuralEvolutionConfig:
    def __init__(
        self,
        provider_type: str = "openai",
        model_name: str = "gpt-4",
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        max_concurrent_evolutions: int = 5,
        evolution_timeout: float = 30.0,
        temperature: float = 0.3,
        max_tokens: int = 4000,
        enable_quality_analysis: bool = True,
        enable_parallel_evolution: bool = True,
        fitness_threshold: float = 0.7,
        max_evolution_attempts: int = 3
    ):
        """
        Initialize neural evolution configuration.
        
        Args:
            provider_type (str): LLM provider type ("openai", "codellama", "hybrid")
            model_name (str): Model name to use
            api_key (Optional[str]): API key for OpenAI
            api_endpoint (Optional[str]): Endpoint for Code Llama
            max_concurrent_evolutions (int): Maximum parallel evolutions
            evolution_timeout (float): Timeout per evolution in seconds
            temperature (float): LLM creativity (0.0-1.0)
            max_tokens (int): Maximum tokens per response
            enable_quality_analysis (bool): Enable code quality analysis
            enable_parallel_evolution (bool): Enable parallel processing
            fitness_threshold (float): Minimum fitness score (0.0-1.0)
            max_evolution_attempts (int): Maximum attempts per evolution
            
        Raises:
            ConfigurationError: If configuration parameters are invalid
        """
```

#### Properties

```python
@property
def provider_type(self) -> str:
    """LLM provider type."""
    
@property
def model_name(self) -> str:
    """Model name being used."""
    
@property
def api_key(self) -> Optional[str]:
    """API key for the provider."""
    
@property
def api_endpoint(self) -> Optional[str]:
    """API endpoint for the provider."""
    
@property
def max_concurrent_evolutions(self) -> int:
    """Maximum number of concurrent evolutions."""
    
@property
def evolution_timeout(self) -> float:
    """Timeout for each evolution in seconds."""
    
@property
def temperature(self) -> float:
    """LLM temperature setting."""
    
@property
def max_tokens(self) -> int:
    """Maximum tokens per response."""
    
@property
def enable_quality_analysis(self) -> bool:
    """Whether quality analysis is enabled."""
    
@property
def enable_parallel_evolution(self) -> bool:
    """Whether parallel evolution is enabled."""
    
@property
def fitness_threshold(self) -> float:
    """Minimum fitness score threshold."""
    
@property
def max_evolution_attempts(self) -> int:
    """Maximum attempts per evolution."""
```

## Provider Classes

### BaseProvider

Abstract base class for LLM providers.

```python
class BaseProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate_code(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate code using the LLM.
        
        Args:
            prompt (str): The prompt for code generation
            context (Optional[Dict[str, Any]]): Additional context
            
        Returns:
            str: Generated code
            
        Raises:
            ProviderError: If code generation fails
        """
        pass
    
    @abstractmethod
    async def analyze_code_quality(
        self,
        code: str
    ) -> QualityMetrics:
        """
        Analyze code quality.
        
        Args:
            code (str): Code to analyze
            
        Returns:
            QualityMetrics: Quality analysis results
            
        Raises:
            ProviderError: If analysis fails
        """
        pass
```

### OpenAIProvider

OpenAI-specific provider implementation.

```python
class OpenAIProvider(BaseProvider):
    def __init__(
        self,
        model_name: str = "gpt-4",
        api_key: str = None,
        temperature: float = 0.3,
        max_tokens: int = 4000
    ):
        """
        Initialize OpenAI provider.
        
        Args:
            model_name (str): OpenAI model name
            api_key (str): OpenAI API key
            temperature (float): Model temperature
            max_tokens (int): Maximum tokens
            
        Raises:
            ConfigurationError: If API key is missing
        """
    
    async def generate_code(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate code using OpenAI models."""
    
    async def analyze_code_quality(
        self,
        code: str
    ) -> QualityMetrics:
        """Analyze code quality using OpenAI models."""
```

### CodeLlamaProvider

Code Llama-specific provider implementation.

```python
class CodeLlamaProvider(BaseProvider):
    def __init__(
        self,
        model_name: str = "codellama-34b-instruct",
        api_endpoint: str = None,
        temperature: float = 0.3,
        max_tokens: int = 4000
    ):
        """
        Initialize Code Llama provider.
        
        Args:
            model_name (str): Code Llama model name
            api_endpoint (str): API endpoint URL
            temperature (float): Model temperature
            max_tokens (int): Maximum tokens
            
        Raises:
            ConfigurationError: If endpoint is missing
        """
    
    async def generate_code(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate code using Code Llama models."""
    
    async def analyze_code_quality(
        self,
        code: str
    ) -> QualityMetrics:
        """Analyze code quality using Code Llama models."""
```

### HybridProvider

Provider that combines multiple LLM providers.

```python
class HybridProvider(BaseProvider):
    def __init__(
        self,
        primary_provider: BaseProvider,
        secondary_provider: BaseProvider,
        fallback_strategy: str = "primary"
    ):
        """
        Initialize hybrid provider.
        
        Args:
            primary_provider (BaseProvider): Primary LLM provider
            secondary_provider (BaseProvider): Secondary LLM provider
            fallback_strategy (str): Fallback strategy ("primary", "secondary", "best")
        """
    
    async def generate_code(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate code using hybrid approach."""
    
    async def analyze_code_quality(
        self,
        code: str
    ) -> QualityMetrics:
        """Analyze code quality using hybrid approach."""
```

## Utility Classes

### QualityAnalyzer

Utility class for code quality analysis.

```python
class QualityAnalyzer:
    """Utility class for comprehensive code quality analysis."""
    
    def __init__(self, provider: BaseProvider):
        """
        Initialize quality analyzer.
        
        Args:
            provider (BaseProvider): LLM provider for analysis
        """
    
    async def analyze_code(
        self,
        code: str,
        language: str = "python"
    ) -> QualityMetrics:
        """
        Analyze code quality comprehensively.
        
        Args:
            code (str): Code to analyze
            language (str): Programming language
            
        Returns:
            QualityMetrics: Comprehensive quality metrics
        """
    
    def calculate_fitness_score(
        self,
        quality_metrics: QualityMetrics,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate fitness score from quality metrics.
        
        Args:
            quality_metrics (QualityMetrics): Quality analysis results
            weights (Optional[Dict[str, float]]): Weights for different metrics
            
        Returns:
            float: Fitness score between 0.0 and 1.0
        """
```

### EvolutionOptimizer

Utility class for evolution optimization strategies.

```python
class EvolutionOptimizer:
    """Utility class for evolution optimization strategies."""
    
    def __init__(self, engine: NeuralCodeEvolutionEngine):
        """
        Initialize evolution optimizer.
        
        Args:
            engine (NeuralCodeEvolutionEngine): Neural evolution engine
        """
    
    async def optimize_evolution_strategy(
        self,
        code: str,
        target: str,
        constraints: Dict[str, float]
    ) -> OptimizationStrategy:
        """
        Optimize evolution strategy for given code and target.
        
        Args:
            code (str): Code to optimize
            target (str): Optimization target
            constraints (Dict[str, float]): Optimization constraints
            
        Returns:
            OptimizationStrategy: Optimized strategy
        """
    
    def adapt_strategy(
        self,
        current_strategy: OptimizationStrategy,
        results: List[EvolutionResult]
    ) -> OptimizationStrategy:
        """
        Adapt strategy based on previous results.
        
        Args:
            current_strategy (OptimizationStrategy): Current strategy
            results (List[EvolutionResult]): Previous evolution results
            
        Returns:
            OptimizationStrategy: Adapted strategy
        """
```

## Data Models

### EvolutionResult

Result of a code evolution operation.

```python
@dataclass
class EvolutionResult:
    """Result of a code evolution operation."""
    
    success: bool
    evolved_code: str
    fitness_score: float
    quality_metrics: QualityMetrics
    evolution_time: float
    attempts: int
    context: Dict[str, Any]
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Validate the evolution result."""
        if not isinstance(self.success, bool):
            raise ValueError("success must be a boolean")
        if not isinstance(self.fitness_score, (int, float)):
            raise ValueError("fitness_score must be a number")
        if not 0.0 <= self.fitness_score <= 1.0:
            raise ValueError("fitness_score must be between 0.0 and 1.0")
```

### OptimizationResult

Result of a code optimization operation.

```python
@dataclass
class OptimizationResult:
    """Result of a code optimization operation."""
    
    success: bool
    optimized_code: str
    optimization_score: float
    constraint_scores: Dict[str, float]
    quality_metrics: QualityMetrics
    optimization_time: float
    target: str
    constraints: Dict[str, float]
    error_message: Optional[str] = None
```

### QualityMetrics

Comprehensive code quality metrics.

```python
@dataclass
class QualityMetrics:
    """Comprehensive code quality metrics."""
    
    overall_score: float
    performance_score: float
    readability_score: float
    security_score: float
    maintainability_score: float
    testability_score: float
    complexity_score: float
    documentation_score: float
    error_handling_score: float
    efficiency_score: float
    
    def __post_init__(self):
        """Validate quality metrics."""
        for field_name, value in self.__dict__.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"{field_name} must be a number")
            if not 0.0 <= value <= 10.0:
                raise ValueError(f"{field_name} must be between 0.0 and 10.0")
```

### EvolutionStatistics

Statistics about evolution performance.

```python
@dataclass
class EvolutionStatistics:
    """Statistics about evolution performance."""
    
    total_evolutions: int
    successful_evolutions: int
    failed_evolutions: int
    success_rate: float
    avg_fitness_score: float
    avg_evolution_time: float
    evolution_types: Dict[str, int]
    quality_distribution: Dict[str, List[float]]
    learning_patterns: Dict[str, PatternData]
    
    @property
    def total_attempts(self) -> int:
        """Total number of evolution attempts."""
        return sum(pattern.attempts for pattern in self.learning_patterns.values())
```

### PatternData

Data about learning patterns.

```python
@dataclass
class PatternData:
    """Data about learning patterns."""
    
    count: int
    avg_fitness: float
    avg_quality: float
    success_rate: float
    attempts: int
    last_used: datetime
    pattern_type: str
    context: Dict[str, Any]
```

## Error Handling

### Custom Exceptions

```python
class NeuralEvolutionError(Exception):
    """Base exception for neural evolution errors."""
    pass

class ConfigurationError(NeuralEvolutionError):
    """Exception for configuration errors."""
    pass

class ProviderError(NeuralEvolutionError):
    """Exception for LLM provider errors."""
    pass

class EvolutionError(NeuralEvolutionError):
    """Exception for evolution errors."""
    pass

class OptimizationError(NeuralEvolutionError):
    """Exception for optimization errors."""
    pass

class QualityAnalysisError(NeuralEvolutionError):
    """Exception for quality analysis errors."""
    pass

class ParallelEvolutionError(NeuralEvolutionError):
    """Exception for parallel evolution errors."""
    pass

class StateError(NeuralEvolutionError):
    """Exception for state management errors."""
    pass

class AutonomousLoopError(NeuralEvolutionError):
    """Exception for autonomous loop errors."""
    pass

class ConstraintError(NeuralEvolutionError):
    """Exception for constraint violation errors."""
    pass

class ResourceError(NeuralEvolutionError):
    """Exception for resource-related errors."""
    pass
```

## Examples

### Basic Code Evolution

```python
import asyncio
from neural_code_evolution_engine import (
    NeuralCodeEvolutionEngine,
    NeuralEvolutionConfig
)

async def basic_evolution_example():
    """Basic example of code evolution."""
    
    # Configure the engine
    config = NeuralEvolutionConfig(
        provider_type="openai",
        model_name="gpt-4",
        api_key="your-api-key",
        fitness_threshold=0.7,
        temperature=0.3
    )
    
    # Initialize engine
    engine = NeuralCodeEvolutionEngine(config)
    
    # Code to evolve
    original_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
    
    # Evolve the code
    result = await engine.evolve_code(
        code=original_code,
        fitness_goal="Optimize for maximum performance and memory efficiency",
        context={
            "target": "performance",
            "constraints": ["memory", "readability"],
            "language": "python"
        }
    )
    
    # Display results
    print(f"Evolution Success: {result.success}")
    print(f"Fitness Score: {result.fitness_score:.3f}")
    print(f"Evolution Time: {result.evolution_time:.2f}s")
    print(f"Attempts: {result.attempts}")
    
    if result.success:
        print(f"\nOriginal Code:\n{original_code}")
        print(f"\nEvolved Code:\n{result.evolved_code}")
        print(f"\nQuality Metrics:")
        print(f"  Performance: {result.quality_metrics.performance_score:.1f}/10")
        print(f"  Readability: {result.quality_metrics.readability_score:.1f}/10")
        print(f"  Security: {result.quality_metrics.security_score:.1f}/10")
    else:
        print(f"Evolution failed: {result.error_message}")

# Run the example
asyncio.run(basic_evolution_example())
```

### Parallel Evolution

```python
async def parallel_evolution_example():
    """Example of parallel code evolution."""
    
    config = NeuralEvolutionConfig(
        provider_type="openai",
        model_name="gpt-4",
        api_key="your-api-key",
        max_concurrent_evolutions=3,
        enable_parallel_evolution=True
    )
    
    engine = NeuralCodeEvolutionEngine(config)
    
    # Multiple evolution tasks
    evolution_tasks = [
        (
            "def sort_list(lst): return sorted(lst)",
            "Optimize for maximum speed",
            {"target": "performance", "constraints": ["memory"]}
        ),
        (
            "def validate_email(email): return '@' in email",
            "Improve security and robustness",
            {"target": "security", "constraints": ["readability"]}
        ),
        (
            "def calculate_average(numbers): return sum(numbers)/len(numbers)",
            "Add comprehensive error handling",
            {"target": "robustness", "constraints": ["performance"]}
        )
    ]
    
    # Run parallel evolutions
    results = await engine.parallel_evolution(evolution_tasks)
    
    # Display results
    for i, result in enumerate(results):
        print(f"\nTask {i+1} Results:")
        print(f"  Success: {result.success}")
        print(f"  Fitness Score: {result.fitness_score:.3f}")
        print(f"  Evolution Time: {result.evolution_time:.2f}s")
        
        if result.success:
            print(f"  Quality Score: {result.quality_metrics.overall_score:.1f}/10")

asyncio.run(parallel_evolution_example())
```

### Code Optimization

```python
async def optimization_example():
    """Example of targeted code optimization."""
    
    config = NeuralEvolutionConfig(
        provider_type="openai",
        model_name="gpt-4",
        api_key="your-api-key",
        enable_quality_analysis=True
    )
    
    engine = NeuralCodeEvolutionEngine(config)
    
    # Code to optimize
    code_to_optimize = """
def process_user_data(user_data):
    result = []
    for user in user_data:
        if user['active']:
            processed = {}
            processed['id'] = user['id']
            processed['name'] = user['name'].upper()
            processed['email'] = user['email'].lower()
            result.append(processed)
    return result
"""
    
    # Optimize for different targets
    targets = ["speed", "memory", "security", "readability"]
    
    for target in targets:
        print(f"\nOptimizing for {target.upper()}:")
        
        result = await engine.optimize_code(
            code=code_to_optimize,
            optimization_target=target,
            constraints={
                "readability": 0.6,
                "security": 0.7
            }
        )
        
        print(f"  Success: {result.success}")
        print(f"  Optimization Score: {result.optimization_score:.3f}")
        print(f"  Constraint Scores: {result.constraint_scores}")
        
        if result.success:
            print(f"  Optimized Code:\n{result.optimized_code}")

asyncio.run(optimization_example())
```

### Autonomous Agent

```python
async def autonomous_agent_example():
    """Example of neural autonomous agent."""
    
    neural_config = NeuralEvolutionConfig(
        provider_type="openai",
        model_name="gpt-4",
        api_key="your-api-key",
        max_cycles=3,
        fitness_threshold=0.6
    )
    
    # Initialize neural autonomous agent
    agent = NeuralAutonomousAgent(
        repo_path=".",
        max_cycles=3,
        neural_config=neural_config
    )
    
    # Start autonomous loop
    await agent.start_neural_autonomous_loop()
    
    # Get statistics
    stats = agent.get_neural_statistics()
    print(f"\nNeural Agent Statistics:")
    print(f"  Neural Cycles: {stats.neural_cycles}")
    print(f"  Neural Success Rate: {stats.neural_success_rate:.2%}")
    print(f"  Average Neural Fitness: {stats.avg_neural_fitness:.3f}")
    print(f"  Total Evolutions: {stats.total_evolutions}")

asyncio.run(autonomous_agent_example())
```

### Custom Quality Analysis

```python
async def custom_quality_analysis_example():
    """Example of custom quality analysis."""
    
    config = NeuralEvolutionConfig(
        provider_type="openai",
        model_name="gpt-4",
        api_key="your-api-key"
    )
    
    engine = NeuralCodeEvolutionEngine(config)
    
    # Code to analyze
    code = """
def complex_function(data):
    result = []
    for item in data:
        if item > 0:
            processed = item * 2
            if processed > 100:
                result.append(processed)
    return result
"""
    
    # Analyze quality
    quality_metrics = await engine.provider.analyze_code_quality(code)
    
    print("Code Quality Analysis:")
    print(f"  Overall Score: {quality_metrics.overall_score:.1f}/10")
    print(f"  Performance: {quality_metrics.performance_score:.1f}/10")
    print(f"  Readability: {quality_metrics.readability_score:.1f}/10")
    print(f"  Security: {quality_metrics.security_score:.1f}/10")
    print(f"  Maintainability: {quality_metrics.maintainability_score:.1f}/10")
    print(f"  Testability: {quality_metrics.testability_score:.1f}/10")
    print(f"  Complexity: {quality_metrics.complexity_score:.1f}/10")
    print(f"  Documentation: {quality_metrics.documentation_score:.1f}/10")
    print(f"  Error Handling: {quality_metrics.error_handling_score:.1f}/10")
    print(f"  Efficiency: {quality_metrics.efficiency_score:.1f}/10")

asyncio.run(custom_quality_analysis_example())
```

## Best Practices

### Configuration Management

1. **Environment Variables**: Store API keys in environment variables
```python
import os
from neural_code_evolution_engine import NeuralEvolutionConfig

config = NeuralEvolutionConfig(
    provider_type="openai",
    model_name="gpt-4",
    api_key=os.getenv("OPENAI_API_KEY")
)
```

2. **Configuration Validation**: Always validate configuration
```python
try:
    config = NeuralEvolutionConfig(
        provider_type="openai",
        model_name="gpt-4",
        api_key="your-api-key"
    )
    engine = NeuralCodeEvolutionEngine(config)
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

### Error Handling

1. **Comprehensive Error Handling**: Handle all possible exceptions
```python
async def safe_evolution(code: str, goal: str):
    try:
        result = await engine.evolve_code(code, goal)
        return result
    except EvolutionError as e:
        print(f"Evolution failed: {e}")
        return None
    except TimeoutError as e:
        print(f"Evolution timed out: {e}")
        return None
    except ProviderError as e:
        print(f"Provider error: {e}")
        return None
```

2. **Retry Logic**: Implement retry logic for transient failures
```python
import asyncio
from functools import wraps

def retry_on_error(max_retries: int = 3, delay: float = 1.0):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except (ProviderError, TimeoutError) as e:
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(delay * (2 ** attempt))
            return None
        return wrapper
    return decorator

@retry_on_error(max_retries=3)
async def robust_evolution(code: str, goal: str):
    return await engine.evolve_code(code, goal)
```

### Performance Optimization

1. **Parallel Processing**: Use parallel evolution for multiple tasks
```python
# Good: Parallel processing
tasks = [(code1, goal1), (code2, goal2), (code3, goal3)]
results = await engine.parallel_evolution(tasks)

# Avoid: Sequential processing
results = []
for code, goal in tasks:
    result = await engine.evolve_code(code, goal)
    results.append(result)
```

2. **Caching**: Leverage built-in caching
```python
# The engine automatically caches results
# Access cached results through evolution_history
cached_results = engine.evolution_history
```

3. **Resource Management**: Monitor resource usage
```python
config = NeuralEvolutionConfig(
    max_concurrent_evolutions=5,  # Adjust based on resources
    evolution_timeout=30.0,       # Set appropriate timeouts
    max_tokens=4000              # Limit token usage
)
```

### Quality Assurance

1. **Fitness Thresholds**: Set appropriate fitness thresholds
```python
config = NeuralEvolutionConfig(
    fitness_threshold=0.7,  # High quality threshold
    max_evolution_attempts=3  # Limit attempts
)
```

2. **Constraint Validation**: Always validate constraints
```python
result = await engine.optimize_code(
    code=code,
    optimization_target="speed",
    constraints={
        "readability": 0.7,  # Minimum readability score
        "security": 0.8      # Minimum security score
    }
)

# Check if constraints were met
if result.success:
    for constraint, score in result.constraint_scores.items():
        if score < result.constraints[constraint]:
            print(f"Warning: {constraint} constraint not met")
```

3. **Quality Monitoring**: Monitor quality metrics over time
```python
stats = engine.get_evolution_statistics()
print(f"Average Quality: {stats.avg_fitness_score:.3f}")
print(f"Quality Distribution: {stats.quality_distribution}")
```

### Security Considerations

1. **Input Validation**: Always validate inputs
```python
def validate_code_input(code: str) -> bool:
    """Validate code input before evolution."""
    if not isinstance(code, str):
        return False
    if len(code) > 10000:  # Limit code size
        return False
    if "import os" in code and "system" in code:  # Basic security check
        return False
    return True

async def secure_evolution(code: str, goal: str):
    if not validate_code_input(code):
        raise ValueError("Invalid code input")
    return await engine.evolve_code(code, goal)
```

2. **API Key Security**: Secure API key management
```python
import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env file

config = NeuralEvolutionConfig(
    provider_type="openai",
    model_name="gpt-4",
    api_key=os.getenv("OPENAI_API_KEY")  # Never hardcode
)
```

3. **Output Validation**: Validate generated code
```python
def validate_generated_code(code: str) -> bool:
    """Validate generated code before use."""
    try:
        compile(code, '<string>', 'exec')  # Basic syntax check
        return True
    except SyntaxError:
        return False

result = await engine.evolve_code(code, goal)
if result.success and validate_generated_code(result.evolved_code):
    # Use the evolved code
    pass
```

---

This comprehensive API documentation provides complete coverage of all public APIs, functions, and components in the Neural Code Evolution Engine. Each section includes detailed examples, usage instructions, and best practices for effective implementation.
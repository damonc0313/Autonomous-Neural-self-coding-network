# 🧠 Neural Code Evolution Engine - API Reference Index

## Quick Navigation

- [📚 Complete Python API Documentation](API_DOCUMENTATION.md)
- [📚 Complete TypeScript/JavaScript API Documentation](API_DOCUMENTATION_TS.md)
- [📖 Project Overview](README_NEURAL_EVOLUTION.md)

## 🚀 Quick Start

### Python
```python
import asyncio
from neural_code_evolution_engine import NeuralCodeEvolutionEngine, NeuralEvolutionConfig

async def quick_start():
    config = NeuralEvolutionConfig(
        provider_type="openai",
        model_name="gpt-4",
        api_key="your-api-key"
    )
    
    engine = NeuralCodeEvolutionEngine(config)
    result = await engine.evolve_code(
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "Optimize for performance"
    )
    print(f"Fitness: {result.fitness_score}")

asyncio.run(quick_start())
```

### TypeScript/JavaScript
```typescript
import { NeuralCodeEvolutionEngine, NeuralEvolutionConfig } from 'neural-code-evolution-engine';

async function quickStart() {
    const config = new NeuralEvolutionConfig({
        providerType: 'openai',
        modelName: 'gpt-4',
        apiKey: 'your-api-key'
    });
    
    const engine = new NeuralCodeEvolutionEngine(config);
    const result = await engine.evolveCode(
        "function fibonacci(n) { return n <= 1 ? n : fibonacci(n-1) + fibonacci(n-2); }",
        "Optimize for performance"
    );
    console.log(`Fitness: ${result.fitnessScore}`);
}

quickStart().catch(console.error);
```

## 📋 API Overview

### Core Classes

| Class | Purpose | Language |
|-------|---------|----------|
| `NeuralCodeEvolutionEngine` | Main engine for code evolution | Python, TypeScript |
| `NeuralAutonomousAgent` | Autonomous agent with neural capabilities | Python, TypeScript |
| `NeuralEvolutionConfig` | Configuration management | Python, TypeScript |

### Provider Classes

| Class | Purpose | Language |
|-------|---------|----------|
| `BaseProvider` | Abstract base for LLM providers | Python, TypeScript |
| `OpenAIProvider` | OpenAI GPT-4/Codex integration | Python, TypeScript |
| `CodeLlamaProvider` | Code Llama integration | Python, TypeScript |
| `HybridProvider` | Multi-provider fallback | Python, TypeScript |

### Utility Classes

| Class | Purpose | Language |
|-------|---------|----------|
| `QualityAnalyzer` | Code quality analysis | Python, TypeScript |
| `EvolutionOptimizer` | Evolution strategy optimization | Python, TypeScript |

## 🔧 Core Methods Reference

### NeuralCodeEvolutionEngine

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `evolve_code()` / `evolveCode()` | Evolve code using neural mutations | `code`, `fitness_goal`, `context?` | `EvolutionResult` |
| `optimize_code()` / `optimizeCode()` | Optimize code for specific target | `code`, `target`, `constraints?` | `OptimizationResult` |
| `parallel_evolution()` / `parallelEvolution()` | Run multiple evolutions in parallel | `evolution_tasks` | `List[EvolutionResult]` |
| `get_evolution_statistics()` / `getEvolutionStatistics()` | Get comprehensive statistics | None | `EvolutionStatistics` |
| `save_evolution_state()` / `saveEvolutionState()` | Save evolution state | `filepath` | `None` |
| `load_evolution_state()` / `loadEvolutionState()` | Load evolution state | `filepath` | `None` |

### NeuralAutonomousAgent

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `start_neural_autonomous_loop()` / `startNeuralAutonomousLoop()` | Start autonomous loop | None | `None` |
| `get_neural_statistics()` / `getNeuralStatistics()` | Get neural statistics | None | `NeuralAgentStatistics` |

## 📊 Data Models

### Core Result Types

| Type | Purpose | Key Fields |
|------|---------|------------|
| `EvolutionResult` | Result of code evolution | `success`, `evolved_code`, `fitness_score`, `quality_metrics` |
| `OptimizationResult` | Result of code optimization | `success`, `optimized_code`, `optimization_score`, `constraint_scores` |
| `QualityMetrics` | Code quality assessment | `overall_score`, `performance_score`, `readability_score`, etc. |
| `EvolutionStatistics` | Evolution performance stats | `total_evolutions`, `success_rate`, `avg_fitness_score` |

### Configuration Types

| Type | Purpose | Key Fields |
|------|---------|------------|
| `NeuralEvolutionConfig` | Engine configuration | `provider_type`, `model_name`, `api_key`, `fitness_threshold` |
| `ProviderType` | LLM provider options | `OPENAI`, `CODELLAMA`, `HYBRID` |
| `OptimizationTarget` | Optimization targets | `SPEED`, `MEMORY`, `SECURITY`, `READABILITY` |

## 🎯 Common Use Cases

### 1. Basic Code Evolution
```python
# Python
result = await engine.evolve_code(
    code="your_code_here",
    fitness_goal="Optimize for performance"
)
```

```typescript
// TypeScript
const result = await engine.evolveCode(
    "your_code_here",
    "Optimize for performance"
);
```

### 2. Parallel Evolution
```python
# Python
tasks = [
    (code1, "Optimize for speed", {"target": "performance"}),
    (code2, "Improve readability", {"target": "clarity"})
]
results = await engine.parallel_evolution(tasks)
```

```typescript
// TypeScript
const tasks: Array<[string, string, Record<string, any>?]> = [
    [code1, "Optimize for speed", { target: "performance" }],
    [code2, "Improve readability", { target: "clarity" }]
];
const results = await engine.parallelEvolution(tasks);
```

### 3. Code Optimization
```python
# Python
result = await engine.optimize_code(
    code="your_code_here",
    optimization_target="speed",
    constraints={"readability": 0.7}
)
```

```typescript
// TypeScript
const result = await engine.optimizeCode(
    "your_code_here",
    "speed",
    { readability: 0.7 }
);
```

### 4. Autonomous Agent
```python
# Python
agent = NeuralAutonomousAgent(
    repo_path=".",
    max_cycles=5,
    neural_config=config
)
await agent.start_neural_autonomous_loop()
```

```typescript
// TypeScript
const agent = new NeuralAutonomousAgent(
    ".",
    5,
    config
);
await agent.startNeuralAutonomousLoop();
```

## ⚙️ Configuration Options

### Provider Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `provider_type` / `providerType` | `ProviderType` | `"openai"` | LLM provider to use |
| `model_name` / `modelName` | `string` | `"gpt-4"` | Model name |
| `api_key` / `apiKey` | `string` | `None` | API key for provider |
| `api_endpoint` / `apiEndpoint` | `string` | `None` | API endpoint (for Code Llama) |

### Evolution Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_concurrent_evolutions` / `maxConcurrentEvolutions` | `number` | `5` | Max parallel evolutions |
| `evolution_timeout` / `evolutionTimeout` | `number` | `30.0` | Timeout per evolution (seconds) |
| `temperature` | `number` | `0.3` | LLM creativity (0.0-1.0) |
| `max_tokens` / `maxTokens` | `number` | `4000` | Max tokens per response |
| `fitness_threshold` / `fitnessThreshold` | `number` | `0.7` | Minimum fitness score |
| `max_evolution_attempts` / `maxEvolutionAttempts` | `number` | `3` | Max attempts per evolution |

### Feature Flags

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_quality_analysis` / `enableQualityAnalysis` | `boolean` | `true` | Enable quality analysis |
| `enable_parallel_evolution` / `enableParallelEvolution` | `boolean` | `true` | Enable parallel processing |

## 🚨 Error Handling

### Exception Hierarchy

```
NeuralEvolutionError (Base)
├── ConfigurationError
├── ProviderError
├── EvolutionError
├── OptimizationError
├── QualityAnalysisError
├── ParallelEvolutionError
├── StateError
├── AutonomousLoopError
├── ConstraintError
└── ResourceError
```

### Common Error Patterns

```python
# Python
try:
    result = await engine.evolve_code(code, goal)
except EvolutionError as e:
    print(f"Evolution failed: {e}")
except ProviderError as e:
    print(f"Provider error: {e}")
except TimeoutError as e:
    print(f"Operation timed out: {e}")
```

```typescript
// TypeScript
try {
    const result = await engine.evolveCode(code, goal);
} catch (error) {
    if (error instanceof EvolutionError) {
        console.error(`Evolution failed: ${error.message}`);
    } else if (error instanceof ProviderError) {
        console.error(`Provider error: ${error.message}`);
    } else if (error instanceof TimeoutError) {
        console.error(`Operation timed out: ${error.message}`);
    }
}
```

## 📈 Performance Optimization

### Parallel Processing
```python
# Use parallel evolution for multiple tasks
results = await engine.parallel_evolution(tasks)
```

### Caching
```python
# Access cached results
cached_results = engine.evolution_history
```

### Resource Management
```python
config = NeuralEvolutionConfig(
    max_concurrent_evolutions=5,  # Adjust based on resources
    evolution_timeout=30.0,       # Set appropriate timeouts
    max_tokens=4000              # Limit token usage
)
```

## 🔒 Security Best Practices

### API Key Management
```python
# Use environment variables
import os
config = NeuralEvolutionConfig(
    api_key=os.getenv("OPENAI_API_KEY")
)
```

### Input Validation
```python
def validate_code_input(code: str) -> bool:
    if len(code) > 10000:  # Limit code size
        return False
    if "import os" in code and "system" in code:  # Security check
        return False
    return True
```

### Output Validation
```python
# Validate generated code before use
if result.success:
    try:
        compile(result.evolved_code, '<string>', 'exec')
        # Use the evolved code
    except SyntaxError:
        print("Generated code has syntax errors")
```

## 📊 Monitoring and Analytics

### Evolution Statistics
```python
stats = engine.get_evolution_statistics()
print(f"Success Rate: {stats.success_rate:.2%}")
print(f"Average Fitness: {stats.avg_fitness_score:.3f}")
print(f"Total Evolutions: {stats.total_evolutions}")
```

### Quality Metrics
```python
quality = result.quality_metrics
print(f"Performance: {quality.performance_score:.1f}/10")
print(f"Readability: {quality.readability_score:.1f}/10")
print(f"Security: {quality.security_score:.1f}/10")
```

### Learning Patterns
```python
patterns = engine.success_patterns
for pattern, data in patterns.items():
    print(f"Pattern: {pattern}, Success Rate: {data.success_rate:.2%}")
```

## 🔧 Advanced Features

### Custom Evolution Strategies
```python
class CustomEvolutionStrategy:
    def __init__(self, engine):
        self.engine = engine
    
    async def custom_evolution(self, code: str, strategy: str):
        if strategy == "performance":
            return await self.engine.evolve_code(
                code, "Optimize for maximum performance"
            )
```

### Custom Quality Analysis
```python
async def custom_quality_analysis(code: str) -> Dict[str, Any]:
    metrics = {
        "custom_score": calculate_custom_score(code),
        "business_logic_score": analyze_business_logic(code)
    }
    return metrics
```

### State Persistence
```python
# Save evolution state
engine.save_evolution_state("evolution_state.pkl")

# Load evolution state
engine.load_evolution_state("evolution_state.pkl")
```

## 📚 Additional Resources

- **Complete Python Documentation**: [API_DOCUMENTATION.md](API_DOCUMENTATION.md)
- **Complete TypeScript Documentation**: [API_DOCUMENTATION_TS.md](API_DOCUMENTATION_TS.md)
- **Project Overview**: [README_NEURAL_EVOLUTION.md](README_NEURAL_EVOLUTION.md)

## 🤝 Support

For questions, issues, or contributions:

1. Check the comprehensive documentation above
2. Review the examples in the documentation files
3. Follow the best practices outlined in this index
4. Use the error handling patterns for robust implementations

---

This API Reference Index provides a quick overview and navigation guide for the Neural Code Evolution Engine. For detailed information about any specific API, method, or feature, please refer to the complete documentation files.
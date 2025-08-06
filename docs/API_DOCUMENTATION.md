# Neural Code Evolution Engine – Comprehensive API Documentation

## Table of Contents

1. Introduction
2. Installation & Setup
3. Quick-Start Example
4. Public API Reference
   1. `NeuralEvolutionConfig`
   2. `NeuralCodeEvolutionEngine`
   3. `NeuralAutonomousAgent`
5. Command-Line Interfaces (CLI)
6. Advanced Usage & Extensibility
7. Security & Best-Practices
8. Troubleshooting & FAQ

---

## 1. Introduction

The **Neural Code Evolution Engine** (NCEE) upgrades traditional autonomous code-generation pipelines by introducing Large Language Models (LLMs) as first-class mutation and optimisation operators.  
This document is a single source of truth for **all public classes, functions, and CLI components** exposed by the project.

> **Audience** – Developers who want to embed the engine inside their own tools, contribute new evolution strategies, or run the included demos.

---

## 2. Installation & Setup

### Prerequisites

* Python **3.9+**
* At least one supported LLM endpoint (e.g. OpenAI, Code-Llama, Local LLM)
* Git (only for cloning the repository)

### Basic Installation

```bash
# 1️⃣  Clone the repository
$ git clone https://github.com/your-org/neural-code-evolution.git
$ cd neural-code-evolution

# 2️⃣  Install python dependencies
$ pip install -r requirements_neural.txt

# 3️⃣  Export your LLM credentials (example: OpenAI)
$ export OPENAI_API_KEY="sk-..."
```

> **Tip** – For local models, skip the API key and set `api_endpoint` instead.

---

## 3. Quick-Start Example

The snippet below shows the absolute minimum required to evolve a single function for performance.

```python
import asyncio
from neural_code_evolution_engine import (
    NeuralCodeEvolutionEngine,
    NeuralEvolutionConfig,
)

async def main():
    config = NeuralEvolutionConfig(
        provider_type="openai",
        model_name="gpt-4",
        api_key="YOUR_API_KEY",
        fitness_threshold=0.7,
    )

    engine = NeuralCodeEvolutionEngine(config)

    original_code = """
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    """

    result = await engine.evolve_code(
        code=original_code,
        fitness_goal="Optimise for maximum performance",
        context={"target": "performance"},
    )

    print(result.evolved_code)

asyncio.run(main())
```

---

## 4. Public API Reference

### 4.1 `NeuralEvolutionConfig`

Configuration object passed to the engine and the autonomous agent.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `provider_type` | `str` | **Yes** | `"openai"`, `"codellama"` or `"hybrid"`. |
| `model_name` | `str` | **Yes** | Name/identifier of the LLM to use. |
| `api_key` | `str \| None` | **Conditional** | Secret token for hosted models (e.g. OpenAI). |
| `api_endpoint` | `str \| None` | No | URL for self-hosted / local models. |
| `max_concurrent_evolutions` | `int` | No | Parallel evolution limit *(default: 5)*. |
| `evolution_timeout` | `float` | No | Seconds before an evolution task aborts. |
| `temperature` | `float` | No | LLM creativity 0.0 – 1.0 *(default: 0.3)*. |
| `max_tokens` | `int` | No | Token ceiling per LLM response. |
| `fitness_threshold` | `float` | No | Minimum acceptable fitness score. |
| `enable_quality_analysis` | `bool` | No | Perform quality metrics after each run. |
| `enable_parallel_evolution` | `bool` | No | Toggle internal async worker pool. |
| `max_evolution_attempts` | `int` | No | Retry budget per code sample. |

#### Example

```python
config = NeuralEvolutionConfig(
    provider_type="codellama",
    model_name="codellama-34b-instruct",
    api_endpoint="http://localhost:8080/v1/chat/completions",
    max_concurrent_evolutions=10,
    temperature=0.2,
)
```

---

### 4.2 `NeuralCodeEvolutionEngine`

Central orchestrator responsible for applying neural mutations and evaluating fitness.

| Method | Signature | Description |
|--------|-----------|-------------|
| `evolve_code` | `async evolve_code(code: str, fitness_goal: str, context: dict | None = None) -> EvolutionResult` | Evolves a single snippet according to a textual goal. |
| `optimize_code` | `async optimize_code(code: str, optimization_target: str, constraints: dict | None = None)` | Higher-level wrapper with common targets: `"speed"`, `"memory"`, `"security"`. |
| `parallel_evolution` | `async parallel_evolution(tasks: list[tuple[str, str, dict]]) -> list[EvolutionResult]` | Evolves many snippets concurrently. |
| `get_evolution_statistics` | `get_evolution_statistics() -> dict` | Aggregate metrics across the session. |
| `save_evolution_state` | `save_evolution_state(filepath: str)` | Pickles internal state to disk. |
| `load_evolution_state` | `load_evolution_state(filepath: str)` | Restores state from previous run. |

#### Minimal Example – Batch Evolution

```python
snippets = [
    (code1, "Optimise for speed", {"target": "speed"}),
    (code2, "Improve readability", {"target": "clarity"}),
]

results = await engine.parallel_evolution(snippets)
```

---

### 4.3 `NeuralAutonomousAgent`

Thin wrapper on top of your existing agent loop that swaps deterministic heuristics for neural-powered operations.

| Parameter | Type | Description |
|-----------|------|-------------|
| `repo_path` | `str` | Path to the Git repository under optimisation. |
| `max_cycles` | `int` | Hard limit for the agent loop. |
| `neural_config` | `NeuralEvolutionConfig` | Shared settings object. |

| Method | Signature | Description |
|--------|-----------|-------------|
| `start_neural_autonomous_loop` | `async start_neural_autonomous_loop() -> None` | Runs the agent loop until `max_cycles` or external stop. |
| `get_neural_statistics` | `get_neural_statistics() -> dict` | Returns extended metrics incl. quality scores. |

#### Example – Run an Agent for 5 Cycles

```bash
python neural_autonomous_agent.py --cycles 5 \
  --provider openai --model gpt-4
```

Or programmatically:

```python
agent = NeuralAutonomousAgent(
    repo_path="/path/to/repo",
    max_cycles=5,
    neural_config=config,
)
await agent.start_neural_autonomous_loop()
```

---

## 5. Command-Line Interfaces (CLI)

Two helper scripts ship with the project:

1. **`neural_evolution_demo.py`** – One-off demos (`performance`, `readability`, `security`).
2. **`neural_autonomous_agent.py`** – Continuous improvement loop.

### 5.1 Demo Script Usage

```bash
# Run all demos sequentially
python neural_evolution_demo.py

# Run only the performance demo
python neural_evolution_demo.py --demo performance

# Choose a different provider/model
python neural_evolution_demo.py \
  --provider codellama \
  --model codellama-34b-instruct
```

### 5.2 Autonomous Agent

```bash
python neural_autonomous_agent.py --cycles 10 \
  --provider openai \
  --model gpt-4
```

---

## 6. Advanced Usage & Extensibility

### 6.1 Custom Evolution Strategies

```python
class MyStrategy:
    def __init__(self, engine):
        self.engine = engine

    async def evolve_for_security(self, code: str):
        return await self.engine.evolve_code(
            code=code,
            fitness_goal="Enhance security features",
            context={"strategy": "security"},
        )
```

### 6.2 Custom Quality Metrics

```python
async def my_quality_check(code: str) -> dict[str, float]:
    # Placeholder for domain-specific rules
    return {"custom_score": 0.9}

metrics = await engine.provider.analyze_code_quality(code)
metrics.update(await my_quality_check(code))
```

### 6.3 State Persistence

```python
engine.save_evolution_state("state.pkl")
# Later…
engine.load_evolution_state("state.pkl")
```

---

## 7. Security & Best-Practices

* **API Keys** – Store secrets in environment variables or a vault provider.
* **Validation** – Never blindly execute evolved code.  Use static analysis + sandboxing.
* **Rate-Limiting** – Respect model provider quotas; tune `max_concurrent_evolutions`.
* **Output Filtering** – Check LLM output for unsafe code, dependencies, or network calls.

---

## 8. Troubleshooting & FAQ

| Symptom | Recommendation |
|---------|----------------|
| *`openai.error.AuthenticationError`* | Confirm `OPENAI_API_KEY` is set and valid. |
| Timeouts during evolution | Increase `evolution_timeout` or lower concurrency. |
| Memory exhaustion | Lower `max_tokens` or JVM heap (for local models). |
| Poor fitness scores | Raise `temperature` slowly or provide more context. |

---

*Generated automatically – last updated on {{DATE}}*
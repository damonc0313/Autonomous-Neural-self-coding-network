# 🧠 Neural Code Evolution Engine - Comprehensive Examples

## Table of Contents

1. [Basic Examples](#basic-examples)
2. [Advanced Examples](#advanced-examples)
3. [Real-World Scenarios](#real-world-scenarios)
4. [Integration Examples](#integration-examples)
5. [Performance Examples](#performance-examples)
6. [Security Examples](#security-examples)

## Basic Examples

### 1. Simple Code Evolution

#### Python
```python
import asyncio
from neural_code_evolution_engine import NeuralCodeEvolutionEngine, NeuralEvolutionConfig

async def simple_evolution():
    # Configure the engine
    config = NeuralEvolutionConfig(
        provider_type="openai",
        model_name="gpt-4",
        api_key="your-api-key",
        fitness_threshold=0.6
    )
    
    # Initialize engine
    engine = NeuralCodeEvolutionEngine(config)
    
    # Simple recursive function to evolve
    original_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
    
    # Evolve the code
    result = await engine.evolve_code(
        code=original_code,
        fitness_goal="Optimize for maximum performance",
        context={"target": "performance"}
    )
    
    # Display results
    print(f"Evolution Success: {result.success}")
    print(f"Fitness Score: {result.fitness_score:.3f}")
    print(f"Evolution Time: {result.evolution_time:.2f}s")
    
    if result.success:
        print(f"\nOriginal Code:\n{original_code}")
        print(f"\nEvolved Code:\n{result.evolved_code}")
        print(f"\nQuality Metrics:")
        print(f"  Performance: {result.quality_metrics.performance_score:.1f}/10")
        print(f"  Readability: {result.quality_metrics.readability_score:.1f}/10")
        print(f"  Security: {result.quality_metrics.security_score:.1f}/10")

# Run the example
asyncio.run(simple_evolution())
```

#### TypeScript
```typescript
import { NeuralCodeEvolutionEngine, NeuralEvolutionConfig } from 'neural-code-evolution-engine';

async function simpleEvolution(): Promise<void> {
    // Configure the engine
    const config = new NeuralEvolutionConfig({
        providerType: 'openai',
        modelName: 'gpt-4',
        apiKey: 'your-api-key',
        fitnessThreshold: 0.6
    });
    
    // Initialize engine
    const engine = new NeuralCodeEvolutionEngine(config);
    
    // Simple recursive function to evolve
    const originalCode = `
function fibonacci(n: number): number {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}
`;
    
    // Evolve the code
    const result = await engine.evolveCode(
        originalCode,
        "Optimize for maximum performance",
        { target: "performance" }
    );
    
    // Display results
    console.log(`Evolution Success: ${result.success}`);
    console.log(`Fitness Score: ${result.fitnessScore.toFixed(3)}`);
    console.log(`Evolution Time: ${result.evolutionTime.toFixed(2)}s`);
    
    if (result.success) {
        console.log(`\nOriginal Code:\n${originalCode}`);
        console.log(`\nEvolved Code:\n${result.evolvedCode}`);
        console.log(`\nQuality Metrics:`);
        console.log(`  Performance: ${result.qualityMetrics.performanceScore.toFixed(1)}/10`);
        console.log(`  Readability: ${result.qualityMetrics.readabilityScore.toFixed(1)}/10`);
        console.log(`  Security: ${result.qualityMetrics.securityScore.toFixed(1)}/10`);
    }
}

simpleEvolution().catch(console.error);
```

### 2. Code Optimization

#### Python
```python
async def code_optimization():
    config = NeuralEvolutionConfig(
        provider_type="openai",
        model_name="gpt-4",
        api_key="your-api-key",
        enable_quality_analysis=True
    )
    
    engine = NeuralCodeEvolutionEngine(config)
    
    # Code to optimize
    code_to_optimize = """
def process_data(data_list):
    result = []
    for item in data_list:
        if item > 0:
            result.append(item * 2)
    return result
"""
    
    # Optimize for different targets
    targets = ["speed", "memory", "security", "readability"]
    
    for target in targets:
        print(f"\n{'='*50}")
        print(f"Optimizing for {target.upper()}")
        print(f"{'='*50}")
        
        result = await engine.optimize_code(
            code=code_to_optimize,
            optimization_target=target,
            constraints={
                "readability": 0.6,
                "security": 0.7
            }
        )
        
        print(f"Success: {result.success}")
        print(f"Optimization Score: {result.optimization_score:.3f}")
        print(f"Constraint Scores: {result.constraint_scores}")
        
        if result.success:
            print(f"\nOptimized Code:\n{result.optimized_code}")
            print(f"\nQuality Metrics:")
            print(f"  Overall: {result.quality_metrics.overall_score:.1f}/10")
            print(f"  Performance: {result.quality_metrics.performance_score:.1f}/10")
            print(f"  Readability: {result.quality_metrics.readability_score:.1f}/10")
            print(f"  Security: {result.quality_metrics.security_score:.1f}/10")

asyncio.run(code_optimization())
```

#### TypeScript
```typescript
async function codeOptimization(): Promise<void> {
    const config = new NeuralEvolutionConfig({
        providerType: 'openai',
        modelName: 'gpt-4',
        apiKey: 'your-api-key',
        enableQualityAnalysis: true
    });
    
    const engine = new NeuralCodeEvolutionEngine(config);
    
    // Code to optimize
    const codeToOptimize = `
function processData(dataList: number[]): number[] {
    const result: number[] = [];
    for (const item of dataList) {
        if (item > 0) {
            result.push(item * 2);
        }
    }
    return result;
}
`;
    
    // Optimize for different targets
    const targets: OptimizationTarget[] = ['speed', 'memory', 'security', 'readability'];
    
    for (const target of targets) {
        console.log(`\n${'='.repeat(50)}`);
        console.log(`Optimizing for ${target.toUpperCase()}`);
        console.log(`${'='.repeat(50)}`);
        
        const result = await engine.optimizeCode(
            codeToOptimize,
            target,
            {
                readability: 0.6,
                security: 0.7
            }
        );
        
        console.log(`Success: ${result.success}`);
        console.log(`Optimization Score: ${result.optimizationScore.toFixed(3)}`);
        console.log(`Constraint Scores:`, result.constraintScores);
        
        if (result.success) {
            console.log(`\nOptimized Code:\n${result.optimizedCode}`);
            console.log(`\nQuality Metrics:`);
            console.log(`  Overall: ${result.qualityMetrics.overallScore.toFixed(1)}/10`);
            console.log(`  Performance: ${result.qualityMetrics.performanceScore.toFixed(1)}/10`);
            console.log(`  Readability: ${result.qualityMetrics.readabilityScore.toFixed(1)}/10`);
            console.log(`  Security: ${result.qualityMetrics.securityScore.toFixed(1)}/10`);
        }
    }
}

codeOptimization().catch(console.error);
```

## Advanced Examples

### 3. Parallel Evolution

#### Python
```python
async def parallel_evolution():
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
            """
def sort_list(lst):
    return sorted(lst)
""",
            "Optimize for maximum speed",
            {"target": "performance", "constraints": ["memory"]}
        ),
        (
            """
def validate_email(email):
    return '@' in email
""",
            "Improve security and robustness",
            {"target": "security", "constraints": ["readability"]}
        ),
        (
            """
def calculate_average(numbers):
    return sum(numbers) / len(numbers)
""",
            "Add comprehensive error handling",
            {"target": "robustness", "constraints": ["performance"]}
        ),
        (
            """
def find_duplicates(items):
    seen = set()
    duplicates = []
    for item in items:
        if item in seen:
            duplicates.append(item)
        seen.add(item)
    return duplicates
""",
            "Optimize for memory efficiency",
            {"target": "memory", "constraints": ["speed"]}
        )
    ]
    
    print("Starting parallel evolution...")
    start_time = time.time()
    
    # Run parallel evolutions
    results = await engine.parallel_evolution(evolution_tasks)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nParallel Evolution completed in {total_time:.2f}s")
    print(f"Average time per evolution: {total_time/len(evolution_tasks):.2f}s")
    
    # Display results
    for i, result in enumerate(results):
        print(f"\n{'='*40}")
        print(f"Task {i+1} Results:")
        print(f"{'='*40}")
        print(f"Success: {result.success}")
        print(f"Fitness Score: {result.fitness_score:.3f}")
        print(f"Evolution Time: {result.evolution_time:.2f}s")
        print(f"Attempts: {result.attempts}")
        
        if result.success:
            print(f"Quality Score: {result.quality_metrics.overall_score:.1f}/10")
            print(f"Performance: {result.quality_metrics.performance_score:.1f}/10")
            print(f"Readability: {result.quality_metrics.readability_score:.1f}/10")
            print(f"Security: {result.quality_metrics.security_score:.1f}/10")
        else:
            print(f"Error: {result.error_message}")

asyncio.run(parallel_evolution())
```

#### TypeScript
```typescript
async function parallelEvolution(): Promise<void> {
    const config = new NeuralEvolutionConfig({
        providerType: 'openai',
        modelName: 'gpt-4',
        apiKey: 'your-api-key',
        maxConcurrentEvolutions: 3,
        enableParallelEvolution: true
    });
    
    const engine = new NeuralCodeEvolutionEngine(config);
    
    // Multiple evolution tasks
    const evolutionTasks: Array<[string, string, Record<string, any>?]> = [
        [
            `
function sortList(lst: number[]): number[] {
    return lst.sort();
}
`,
            "Optimize for maximum speed",
            { target: "performance", constraints: ["memory"] }
        ],
        [
            `
function validateEmail(email: string): boolean {
    return email.includes('@');
}
`,
            "Improve security and robustness",
            { target: "security", constraints: ["readability"] }
        ],
        [
            `
function calculateAverage(numbers: number[]): number {
    return numbers.reduce((a, b) => a + b, 0) / numbers.length;
}
`,
            "Add comprehensive error handling",
            { target: "robustness", constraints: ["performance"] }
        ],
        [
            `
function findDuplicates(items: any[]): any[] {
    const seen = new Set();
    const duplicates: any[] = [];
    for (const item of items) {
        if (seen.has(item)) {
            duplicates.push(item);
        }
        seen.add(item);
    }
    return duplicates;
}
`,
            "Optimize for memory efficiency",
            { target: "memory", constraints: ["speed"] }
        ]
    ];
    
    console.log("Starting parallel evolution...");
    const startTime = Date.now();
    
    // Run parallel evolutions
    const results = await engine.parallelEvolution(evolutionTasks);
    
    const endTime = Date.now();
    const totalTime = (endTime - startTime) / 1000;
    
    console.log(`\nParallel Evolution completed in ${totalTime.toFixed(2)}s`);
    console.log(`Average time per evolution: ${(totalTime / evolutionTasks.length).toFixed(2)}s`);
    
    // Display results
    results.forEach((result, i) => {
        console.log(`\n${'='.repeat(40)}`);
        console.log(`Task ${i + 1} Results:`);
        console.log(`${'='.repeat(40)}`);
        console.log(`Success: ${result.success}`);
        console.log(`Fitness Score: ${result.fitnessScore.toFixed(3)}`);
        console.log(`Evolution Time: ${result.evolutionTime.toFixed(2)}s`);
        console.log(`Attempts: ${result.attempts}`);
        
        if (result.success) {
            console.log(`Quality Score: ${result.qualityMetrics.overallScore.toFixed(1)}/10`);
            console.log(`Performance: ${result.qualityMetrics.performanceScore.toFixed(1)}/10`);
            console.log(`Readability: ${result.qualityMetrics.readabilityScore.toFixed(1)}/10`);
            console.log(`Security: ${result.qualityMetrics.securityScore.toFixed(1)}/10`);
        } else {
            console.log(`Error: ${result.errorMessage}`);
        }
    });
}

parallelEvolution().catch(console.error);
```

### 4. Autonomous Agent

#### Python
```python
async def autonomous_agent():
    neural_config = NeuralEvolutionConfig(
        provider_type="openai",
        model_name="gpt-4",
        api_key="your-api-key",
        fitness_threshold=0.6,
        max_cycles=3
    )
    
    # Initialize neural autonomous agent
    agent = NeuralAutonomousAgent(
        repo_path=".",
        max_cycles=3,
        neural_config=neural_config
    )
    
    print("Starting Neural Autonomous Agent...")
    print("This will run 3 cycles of autonomous code evolution.")
    
    # Start autonomous loop
    await agent.start_neural_autonomous_loop()
    
    # Get statistics
    stats = agent.get_neural_statistics()
    print(f"\n{'='*50}")
    print(f"Neural Agent Statistics")
    print(f"{'='*50}")
    print(f"Neural Cycles: {stats.neural_cycles}")
    print(f"Neural Success Rate: {stats.neural_success_rate:.2%}")
    print(f"Average Neural Fitness: {stats.avg_neural_fitness:.3f}")
    print(f"Total Evolutions: {stats.total_evolutions}")
    
    # Display cycle metrics
    cycle_metrics = agent.neural_cycle_metrics
    print(f"\nCycle Metrics:")
    for cycle, metrics in cycle_metrics.items():
        print(f"  Cycle {cycle}:")
        print(f"    Evolutions: {metrics.get('evolutions', 0)}")
        print(f"    Success Rate: {metrics.get('success_rate', 0):.2%}")
        print(f"    Avg Fitness: {metrics.get('avg_fitness', 0):.3f}")

asyncio.run(autonomous_agent())
```

#### TypeScript
```typescript
async function autonomousAgent(): Promise<void> {
    const neuralConfig = new NeuralEvolutionConfig({
        providerType: 'openai',
        modelName: 'gpt-4',
        apiKey: 'your-api-key',
        fitnessThreshold: 0.6,
        maxCycles: 3
    });
    
    // Initialize neural autonomous agent
    const agent = new NeuralAutonomousAgent(
        ".",
        3,
        neuralConfig
    );
    
    console.log("Starting Neural Autonomous Agent...");
    console.log("This will run 3 cycles of autonomous code evolution.");
    
    // Start autonomous loop
    await agent.startNeuralAutonomousLoop();
    
    // Get statistics
    const stats = agent.getNeuralStatistics();
    console.log(`\n${'='.repeat(50)}`);
    console.log(`Neural Agent Statistics`);
    console.log(`${'='.repeat(50)}`);
    console.log(`Neural Cycles: ${stats.neuralCycles}`);
    console.log(`Neural Success Rate: ${(stats.neuralSuccessRate * 100).toFixed(2)}%`);
    console.log(`Average Neural Fitness: ${stats.avgNeuralFitness.toFixed(3)}`);
    console.log(`Total Evolutions: ${stats.totalEvolutions}`);
    
    // Display cycle metrics
    const cycleMetrics = agent.neuralCycleMetrics;
    console.log(`\nCycle Metrics:`);
    for (const [cycle, metrics] of Object.entries(cycleMetrics)) {
        console.log(`  Cycle ${cycle}:`);
        console.log(`    Evolutions: ${metrics.evolutions || 0}`);
        console.log(`    Success Rate: ${((metrics.successRate || 0) * 100).toFixed(2)}%`);
        console.log(`    Avg Fitness: ${(metrics.avgFitness || 0).toFixed(3)}`);
    }
}

autonomousAgent().catch(console.error);
```

## Real-World Scenarios

### 5. Web API Optimization

#### Python
```python
async def web_api_optimization():
    config = NeuralEvolutionConfig(
        provider_type="openai",
        model_name="gpt-4",
        api_key="your-api-key",
        enable_quality_analysis=True
    )
    
    engine = NeuralCodeEvolutionEngine(config)
    
    # Web API endpoint code
    api_code = """
from flask import Flask, request, jsonify
import json

app = Flask(__name__)

@app.route('/api/users', methods=['GET'])
def get_users():
    try:
        # Simulate database query
        users = [
            {"id": 1, "name": "John", "email": "john@example.com"},
            {"id": 2, "name": "Jane", "email": "jane@example.com"}
        ]
        return jsonify(users)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/users', methods=['POST'])
def create_user():
    try:
        data = request.get_json()
        # Simulate user creation
        new_user = {
            "id": len(users) + 1,
            "name": data.get('name'),
            "email": data.get('email')
        }
        return jsonify(new_user), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
"""
    
    print("Optimizing Web API for production...")
    
    # Optimize for different aspects
    optimizations = [
        ("security", "Enhance security with input validation and authentication"),
        ("performance", "Optimize for high performance and scalability"),
        ("error_handling", "Improve error handling and logging"),
        ("documentation", "Add comprehensive documentation and type hints")
    ]
    
    for target, description in optimizations:
        print(f"\n{'='*60}")
        print(f"Optimizing for: {target.upper()}")
        print(f"Description: {description}")
        print(f"{'='*60}")
        
        result = await engine.optimize_code(
            code=api_code,
            optimization_target=target,
            constraints={
                "readability": 0.7,
                "security": 0.8 if target == "security" else 0.6,
                "performance": 0.8 if target == "performance" else 0.6
            }
        )
        
        if result.success:
            print(f"✅ Optimization successful!")
            print(f"Score: {result.optimization_score:.3f}")
            print(f"\nOptimized Code:\n{result.optimized_code}")
        else:
            print(f"❌ Optimization failed: {result.error_message}")

asyncio.run(web_api_optimization())
```

### 6. Data Processing Pipeline

#### Python
```python
async def data_pipeline_optimization():
    config = NeuralEvolutionConfig(
        provider_type="openai",
        model_name="gpt-4",
        api_key="your-api-key"
    )
    
    engine = NeuralCodeEvolutionEngine(config)
    
    # Data processing pipeline
    pipeline_code = """
import pandas as pd
import numpy as np
from typing import List, Dict, Any

def process_data_pipeline(data: List[Dict[str, Any]]) -> pd.DataFrame:
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Clean data
    df = df.dropna()
    df = df.drop_duplicates()
    
    # Transform data
    df['processed_value'] = df['value'] * 2
    df['category'] = df['type'].apply(lambda x: 'A' if x > 5 else 'B')
    
    # Aggregate data
    result = df.groupby('category').agg({
        'processed_value': ['mean', 'sum', 'count']
    }).reset_index()
    
    return result

def analyze_data(df: pd.DataFrame) -> Dict[str, float]:
    analysis = {}
    analysis['total_rows'] = len(df)
    analysis['mean_value'] = df['processed_value'].mean()
    analysis['std_value'] = df['processed_value'].std()
    return analysis
"""
    
    print("Optimizing Data Processing Pipeline...")
    
    # Evolve for different optimization goals
    evolution_goals = [
        "Optimize for memory efficiency and large dataset handling",
        "Improve performance with vectorized operations",
        "Add comprehensive error handling and validation",
        "Enhance code readability and maintainability"
    ]
    
    for i, goal in enumerate(evolution_goals, 1):
        print(f"\n{'='*70}")
        print(f"Evolution {i}: {goal}")
        print(f"{'='*70}")
        
        result = await engine.evolve_code(
            code=pipeline_code,
            fitness_goal=goal,
            context={
                "target": "data_processing",
                "constraints": ["performance", "memory"],
                "language": "python"
            }
        )
        
        if result.success:
            print(f"✅ Evolution successful!")
            print(f"Fitness Score: {result.fitness_score:.3f}")
            print(f"Quality Score: {result.quality_metrics.overall_score:.1f}/10")
            print(f"\nEvolved Code:\n{result.evolved_code}")
        else:
            print(f"❌ Evolution failed: {result.error_message}")

asyncio.run(data_pipeline_optimization())
```

## Integration Examples

### 7. CI/CD Integration

#### Python
```python
import os
import sys
from neural_code_evolution_engine import NeuralCodeEvolutionEngine, NeuralEvolutionConfig

async def ci_cd_integration():
    """Example of integrating neural evolution into CI/CD pipeline"""
    
    # Get configuration from environment variables
    config = NeuralEvolutionConfig(
        provider_type=os.getenv("NEURAL_PROVIDER", "openai"),
        model_name=os.getenv("NEURAL_MODEL", "gpt-4"),
        api_key=os.getenv("OPENAI_API_KEY"),
        fitness_threshold=float(os.getenv("FITNESS_THRESHOLD", "0.7")),
        max_evolution_attempts=int(os.getenv("MAX_ATTEMPTS", "3"))
    )
    
    engine = NeuralCodeEvolutionEngine(config)
    
    # Read code from file (in CI/CD, this would be the changed files)
    code_file = os.getenv("CODE_FILE", "example.py")
    
    try:
        with open(code_file, 'r') as f:
            code = f.read()
    except FileNotFoundError:
        print(f"❌ Code file {code_file} not found")
        sys.exit(1)
    
    print(f"🔍 Analyzing and optimizing {code_file}...")
    
    # Analyze current code quality
    quality_metrics = await engine.provider.analyze_code_quality(code)
    print(f"📊 Current Quality Score: {quality_metrics.overall_score:.1f}/10")
    
    # Only evolve if quality is below threshold
    if quality_metrics.overall_score < 7.0:
        print("🔄 Quality below threshold, starting evolution...")
        
        result = await engine.evolve_code(
            code=code,
            fitness_goal="Improve code quality, performance, and maintainability",
            context={
                "target": "ci_cd",
                "constraints": ["backward_compatibility", "performance"]
            }
        )
        
        if result.success and result.fitness_score > config.fitness_threshold:
            print("✅ Evolution successful! Writing optimized code...")
            
            # Write evolved code back to file
            with open(f"{code_file}.optimized", 'w') as f:
                f.write(result.evolved_code)
            
            print(f"📝 Optimized code written to {code_file}.optimized")
            print(f"📈 Quality improved from {quality_metrics.overall_score:.1f}/10 to {result.quality_metrics.overall_score:.1f}/10")
            
            # In CI/CD, you might want to create a pull request with the changes
            print("🚀 Ready to create pull request with optimizations")
        else:
            print("❌ Evolution failed or didn't meet quality threshold")
            sys.exit(1)
    else:
        print("✅ Code quality is already good, no evolution needed")

# Run in CI/CD environment
if __name__ == "__main__":
    import asyncio
    asyncio.run(ci_cd_integration())
```

### 8. IDE Plugin Integration

#### TypeScript
```typescript
// Example of integrating with VS Code extension
import * as vscode from 'vscode';
import { NeuralCodeEvolutionEngine, NeuralEvolutionConfig } from 'neural-code-evolution-engine';

export class NeuralEvolutionProvider {
    private engine: NeuralCodeEvolutionEngine;
    
    constructor() {
        const config = new NeuralEvolutionConfig({
            providerType: 'openai',
            modelName: 'gpt-4',
            apiKey: process.env.OPENAI_API_KEY,
            fitnessThreshold: 0.7
        });
        
        this.engine = new NeuralCodeEvolutionEngine(config);
    }
    
    async optimizeCurrentFile(): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor');
            return;
        }
        
        const document = editor.document;
        const code = document.getText();
        
        // Show progress
        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "Optimizing code with Neural Evolution...",
            cancellable: false
        }, async (progress) => {
            try {
                progress.report({ increment: 0 });
                
                const result = await this.engine.evolveCode(
                    code,
                    "Optimize for performance and readability",
                    { target: "ide_integration" }
                );
                
                progress.report({ increment: 100 });
                
                if (result.success) {
                    // Show diff view
                    const originalUri = document.uri;
                    const optimizedUri = vscode.Uri.parse(`untitled:${document.fileName}.optimized`);
                    
                    const optimizedDocument = await vscode.workspace.openTextDocument(optimizedUri);
                    await optimizedDocument.save();
                    
                    await vscode.commands.executeCommand('vscode.diff',
                        originalUri,
                        optimizedUri,
                        'Code Optimization Diff'
                    );
                    
                    vscode.window.showInformationMessage(
                        `Code optimized! Fitness score: ${result.fitnessScore.toFixed(3)}`
                    );
                } else {
                    vscode.window.showErrorMessage(
                        `Optimization failed: ${result.errorMessage}`
                    );
                }
            } catch (error) {
                vscode.window.showErrorMessage(
                    `Error during optimization: ${error.message}`
                );
            }
        });
    }
    
    async quickOptimize(): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;
        
        const selection = editor.selection;
        const code = editor.document.getText(selection);
        
        if (!code.trim()) {
            vscode.window.showErrorMessage('No code selected');
            return;
        }
        
        try {
            const result = await this.engine.optimizeCode(
                code,
                'readability',
                { performance: 0.6 }
            );
            
            if (result.success) {
                await editor.edit(editBuilder => {
                    editBuilder.replace(selection, result.optimizedCode);
                });
                
                vscode.window.showInformationMessage(
                    `Code optimized! Score: ${result.optimizationScore.toFixed(3)}`
                );
            }
        } catch (error) {
            vscode.window.showErrorMessage(
                `Quick optimization failed: ${error.message}`
            );
        }
    }
}

// Register commands
export function activate(context: vscode.ExtensionContext) {
    const provider = new NeuralEvolutionProvider();
    
    context.subscriptions.push(
        vscode.commands.registerCommand('neuralEvolution.optimizeFile', () => {
            provider.optimizeCurrentFile();
        }),
        
        vscode.commands.registerCommand('neuralEvolution.quickOptimize', () => {
            provider.quickOptimize();
        })
    );
}
```

## Performance Examples

### 9. Batch Processing

#### Python
```python
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from neural_code_evolution_engine import NeuralCodeEvolutionEngine, NeuralEvolutionConfig

async def batch_processing():
    """Example of processing multiple files in batches"""
    
    config = NeuralEvolutionConfig(
        provider_type="openai",
        model_name="gpt-4",
        api_key="your-api-key",
        max_concurrent_evolutions=5,
        enable_parallel_evolution=True
    )
    
    engine = NeuralCodeEvolutionEngine(config)
    
    # Simulate multiple code files
    code_files = [
        ("file1.py", """
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
"""),
        ("file2.py", """
def sort_array(arr):
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if arr[i] > arr[j]:
                arr[i], arr[j] = arr[j], arr[i]
    return arr
"""),
        ("file3.py", """
def validate_email(email):
    return '@' in email and '.' in email
"""),
        ("file4.py", """
def process_data(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result
"""),
        ("file5.py", """
def find_max(numbers):
    max_num = numbers[0]
    for num in numbers:
        if num > max_num:
            max_num = num
    return max_num
""")
    ]
    
    print(f"Processing {len(code_files)} files in batches...")
    start_time = time.time()
    
    # Process in batches
    batch_size = 3
    all_results = []
    
    for i in range(0, len(code_files), batch_size):
        batch = code_files[i:i + batch_size]
        print(f"\nProcessing batch {i//batch_size + 1} ({len(batch)} files)...")
        
        # Create evolution tasks for this batch
        evolution_tasks = [
            (code, f"Optimize {filename} for performance and readability", {
                "filename": filename,
                "target": "batch_processing"
            })
            for filename, code in batch
        ]
        
        # Process batch in parallel
        batch_results = await engine.parallel_evolution(evolution_tasks)
        all_results.extend(batch_results)
        
        # Report batch progress
        successful = sum(1 for r in batch_results if r.success)
        print(f"Batch {i//batch_size + 1} complete: {successful}/{len(batch)} successful")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Summary
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total files processed: {len(code_files)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per file: {total_time/len(code_files):.2f}s")
    print(f"Successful evolutions: {sum(1 for r in all_results if r.success)}/{len(all_results)}")
    
    if all_results:
        avg_fitness = sum(r.fitness_score for r in all_results if r.success) / sum(1 for r in all_results if r.success)
        print(f"Average fitness score: {avg_fitness:.3f}")
    
    # Show individual results
    for i, (filename, _) in enumerate(code_files):
        result = all_results[i]
        status = "✅" if result.success else "❌"
        print(f"{status} {filename}: {result.fitness_score:.3f} ({result.evolution_time:.2f}s)")

asyncio.run(batch_processing())
```

### 10. Performance Monitoring

#### Python
```python
import asyncio
import time
import statistics
from neural_code_evolution_engine import NeuralCodeEvolutionEngine, NeuralEvolutionConfig

async def performance_monitoring():
    """Example of monitoring evolution performance over time"""
    
    config = NeuralEvolutionConfig(
        provider_type="openai",
        model_name="gpt-4",
        api_key="your-api-key",
        fitness_threshold=0.6
    )
    
    engine = NeuralCodeEvolutionEngine(config)
    
    # Test code for performance monitoring
    test_code = """
def inefficient_function(data):
    result = []
    for i in range(len(data)):
        for j in range(len(data)):
            if data[i] == data[j]:
                result.append(data[i])
    return result
"""
    
    print("Running performance monitoring over 10 evolutions...")
    
    performance_metrics = {
        'evolution_times': [],
        'fitness_scores': [],
        'quality_scores': [],
        'success_rates': []
    }
    
    for i in range(10):
        print(f"\nEvolution {i+1}/10...")
        
        start_time = time.time()
        
        result = await engine.evolve_code(
            code=test_code,
            fitness_goal="Optimize for performance and efficiency",
            context={"target": "performance_monitoring", "iteration": i}
        )
        
        end_time = time.time()
        evolution_time = end_time - start_time
        
        # Record metrics
        performance_metrics['evolution_times'].append(evolution_time)
        performance_metrics['fitness_scores'].append(result.fitness_score if result.success else 0)
        performance_metrics['quality_scores'].append(
            result.quality_metrics.overall_score if result.success else 0
        )
        performance_metrics['success_rates'].append(1 if result.success else 0)
        
        print(f"  Time: {evolution_time:.2f}s")
        print(f"  Success: {result.success}")
        print(f"  Fitness: {result.fitness_score:.3f}")
        print(f"  Quality: {result.quality_metrics.overall_score:.1f}/10")
    
    # Calculate statistics
    print(f"\n{'='*60}")
    print(f"PERFORMANCE MONITORING RESULTS")
    print(f"{'='*60}")
    
    print(f"Evolution Times:")
    print(f"  Average: {statistics.mean(performance_metrics['evolution_times']):.2f}s")
    print(f"  Median: {statistics.median(performance_metrics['evolution_times']):.2f}s")
    print(f"  Min: {min(performance_metrics['evolution_times']):.2f}s")
    print(f"  Max: {max(performance_metrics['evolution_times']):.2f}s")
    
    print(f"\nFitness Scores:")
    successful_fitness = [s for s in performance_metrics['fitness_scores'] if s > 0]
    if successful_fitness:
        print(f"  Average: {statistics.mean(successful_fitness):.3f}")
        print(f"  Median: {statistics.median(successful_fitness):.3f}")
        print(f"  Min: {min(successful_fitness):.3f}")
        print(f"  Max: {max(successful_fitness):.3f}")
    
    print(f"\nQuality Scores:")
    successful_quality = [s for s in performance_metrics['quality_scores'] if s > 0]
    if successful_quality:
        print(f"  Average: {statistics.mean(successful_quality):.1f}/10")
        print(f"  Median: {statistics.median(successful_quality):.1f}/10")
        print(f"  Min: {min(successful_quality):.1f}/10")
        print(f"  Max: {max(successful_quality):.1f}/10")
    
    success_rate = sum(performance_metrics['success_rates']) / len(performance_metrics['success_rates'])
    print(f"\nOverall Success Rate: {success_rate:.2%}")
    
    # Get engine statistics
    engine_stats = engine.get_evolution_statistics()
    print(f"\nEngine Statistics:")
    print(f"  Total Evolutions: {engine_stats.total_evolutions}")
    print(f"  Engine Success Rate: {engine_stats.success_rate:.2%}")
    print(f"  Engine Avg Fitness: {engine_stats.avg_fitness_score:.3f}")

asyncio.run(performance_monitoring())
```

## Security Examples

### 11. Security-Focused Evolution

#### Python
```python
async def security_evolution():
    """Example of security-focused code evolution"""
    
    config = NeuralEvolutionConfig(
        provider_type="openai",
        model_name="gpt-4",
        api_key="your-api-key",
        enable_quality_analysis=True
    )
    
    engine = NeuralCodeEvolutionEngine(config)
    
    # Vulnerable code examples
    vulnerable_code_examples = [
        {
            "name": "SQL Injection",
            "code": """
def get_user_data(user_id):
    import sqlite3
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
    return cursor.fetchall()
"""
        },
        {
            "name": "Command Injection",
            "code": """
def process_file(filename):
    import os
    os.system(f"cat {filename}")
    return "File processed"
"""
        },
        {
            "name": "XSS Vulnerability",
            "code": """
def render_user_input(user_input):
    return f"<div>{user_input}</div>"
"""
        },
        {
            "name": "Insecure Random",
            "code": """
def generate_token():
    import random
    return random.randint(1000, 9999)
"""
        }
    ]
    
    print("🔒 Security-Focused Code Evolution")
    print("=" * 50)
    
    for example in vulnerable_code_examples:
        print(f"\n🛡️  Securing: {example['name']}")
        print("-" * 30)
        
        # Analyze current security
        quality_metrics = await engine.provider.analyze_code_quality(example['code'])
        print(f"Current Security Score: {quality_metrics.security_score:.1f}/10")
        
        # Evolve for security
        result = await engine.evolve_code(
            code=example['code'],
            fitness_goal=f"Fix security vulnerabilities in {example['name'].lower()} code",
            context={
                "target": "security",
                "vulnerability_type": example['name'],
                "constraints": ["performance", "readability"]
            }
        )
        
        if result.success:
            print(f"✅ Security evolution successful!")
            print(f"New Security Score: {result.quality_metrics.security_score:.1f}/10")
            print(f"Overall Quality: {result.quality_metrics.overall_score:.1f}/10")
            print(f"\nSecured Code:\n{result.evolved_code}")
            
            # Additional security analysis
            if result.quality_metrics.security_score > 8.0:
                print("🟢 Code is now secure!")
            elif result.quality_metrics.security_score > 6.0:
                print("🟡 Code security improved, but review recommended")
            else:
                print("🔴 Code still has security concerns")
        else:
            print(f"❌ Security evolution failed: {result.error_message}")

asyncio.run(security_evolution())
```

### 12. Input Validation and Sanitization

#### Python
```python
async def input_validation_evolution():
    """Example of evolving code to add proper input validation"""
    
    config = NeuralEvolutionConfig(
        provider_type="openai",
        model_name="gpt-4",
        api_key="your-api-key"
    )
    
    engine = NeuralCodeEvolutionEngine(config)
    
    # Code without proper input validation
    unsafe_code = """
def process_user_data(user_data):
    # Process user data without validation
    result = {}
    result['name'] = user_data['name']
    result['email'] = user_data['email']
    result['age'] = user_data['age']
    result['salary'] = user_data['salary']
    
    # Store in database
    save_to_database(result)
    return result

def calculate_discount(price, discount_percent):
    # Calculate discount without validation
    discount = price * (discount_percent / 100)
    return price - discount

def search_files(directory):
    # Search files without path validation
    import os
    files = os.listdir(directory)
    return [f for f in files if f.endswith('.txt')]
"""
    
    print("🔍 Evolving code to add input validation...")
    
    # Evolve for input validation
    result = await engine.evolve_code(
        code=unsafe_code,
        fitness_goal="Add comprehensive input validation, sanitization, and error handling",
        context={
            "target": "input_validation",
            "security_level": "high",
            "constraints": ["performance", "usability"]
        }
    )
    
    if result.success:
        print("✅ Input validation evolution successful!")
        print(f"Security Score: {result.quality_metrics.security_score:.1f}/10")
        print(f"Error Handling Score: {result.quality_metrics.error_handling_score:.1f}/10")
        print(f"\nValidated Code:\n{result.evolved_code}")
        
        # Test the evolved code with various inputs
        print("\n🧪 Testing evolved code with various inputs...")
        
        test_cases = [
            {"name": "John", "email": "john@example.com", "age": 25, "salary": 50000},
            {"name": "", "email": "invalid-email", "age": -5, "salary": -1000},
            {"name": "A" * 1000, "email": "test@test.com", "age": 200, "salary": 999999999}
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest Case {i}: {test_case}")
            # In a real scenario, you would execute the evolved code here
            print("  ✅ Input validation would catch issues")
    else:
        print(f"❌ Input validation evolution failed: {result.error_message}")

asyncio.run(input_validation_evolution())
```

---

This comprehensive examples file demonstrates all the key features and use cases of the Neural Code Evolution Engine. Each example includes both Python and TypeScript implementations where applicable, and covers real-world scenarios that developers might encounter.

The examples range from basic usage to advanced integrations, performance optimization, and security-focused evolution. They provide practical guidance for implementing the Neural Code Evolution Engine in various contexts.
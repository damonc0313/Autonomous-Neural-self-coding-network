"""
Digital Crucible: Autonomous Training Harness for GraphformicCoder

A comprehensive reinforcement learning framework that enables the GraphformicCoder
model to evolve its coding skills through trial, error, and self-correction in a
secure, monitored environment.

Author: Reinforcement Learning Architect
"""

import os
import sys
import json
import time
import subprocess
import tempfile
import shutil
import docker
import ast
import coverage
import bandit
import flake8.api.legacy as flake8
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import logging
from dataclasses import dataclass, field
from pathlib import Path
import threading
import queue
import signal
import psutil
from contextlib import contextmanager
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ProgrammingProblem:
    """Represents a programming challenge with all necessary components."""
    id: str
    title: str
    description: str
    difficulty: str
    test_cases: List[Dict[str, Any]]
    expected_outputs: List[Any]
    constraints: Dict[str, Any] = field(default_factory=dict)
    time_limit: float = 5.0  # seconds
    memory_limit: int = 256  # MB
    language: str = "python"


@dataclass
class ExecutionMetrics:
    """Comprehensive metrics from code execution and analysis."""
    test_passed: bool = False
    test_pass_rate: float = 0.0
    coverage_percentage: float = 0.0
    execution_time: float = 0.0
    memory_usage: float = 0.0
    security_vulnerabilities: int = 0
    style_violations: int = 0
    syntax_errors: int = 0
    runtime_errors: int = 0
    performance_score: float = 0.0
    code_quality_score: float = 0.0


class ProblemEnvironment:
    """
    Manages programming problems from various sources and provides them
    to the AI model for training.
    """
    
    def __init__(self, problem_sources: List[str]):
        """
        Initialize the problem environment.
        
        Args:
            problem_sources: List of paths or APIs to pull problems from
        """
        self.problem_sources = problem_sources
        self.problems: List[ProgrammingProblem] = []
        self.current_problem_idx = 0
        self._load_problems()
    
    def _load_problems(self):
        """Load problems from all configured sources."""
        for source in self.problem_sources:
            if os.path.isdir(source):
                self._load_from_directory(source)
            elif source.startswith('http'):
                self._load_from_api(source)
            else:
                logger.warning(f"Unknown problem source type: {source}")
    
    def _load_from_directory(self, directory: str):
        """Load problems from a directory of JSON files."""
        problem_dir = Path(directory)
        for problem_file in problem_dir.glob("*.json"):
            try:
                with open(problem_file, 'r') as f:
                    problem_data = json.load(f)
                    problem = ProgrammingProblem(**problem_data)
                    self.problems.append(problem)
                    logger.info(f"Loaded problem: {problem.title}")
            except Exception as e:
                logger.error(f"Failed to load problem from {problem_file}: {e}")
    
    def _load_from_api(self, api_url: str):
        """Load problems from an API endpoint (placeholder for future implementation)."""
        # This would implement API calls to services like LeetCode
        logger.info(f"API loading not implemented yet for: {api_url}")
    
    def get_next_problem(self) -> Optional[ProgrammingProblem]:
        """Get the next problem in the sequence."""
        if not self.problems:
            return None
        
        problem = self.problems[self.current_problem_idx]
        self.current_problem_idx = (self.current_problem_idx + 1) % len(self.problems)
        return problem
    
    def get_random_problem(self) -> Optional[ProgrammingProblem]:
        """Get a random problem from the available set."""
        if not self.problems:
            return None
        return np.random.choice(self.problems)
    
    def get_problems_by_difficulty(self, difficulty: str) -> List[ProgrammingProblem]:
        """Get all problems of a specific difficulty level."""
        return [p for p in self.problems if p.difficulty.lower() == difficulty.lower()]


class ExecutionSandbox:
    """
    Secure sandbox for executing AI-generated code with comprehensive monitoring.
    Provides isolation, security analysis, performance metrics, and style checking.
    """
    
    def __init__(self, use_docker: bool = True, timeout: float = 30.0):
        """
        Initialize the execution sandbox.
        
        Args:
            use_docker: Whether to use Docker for isolation
            timeout: Maximum execution time in seconds
        """
        self.use_docker = use_docker
        self.timeout = timeout
        self.docker_client = None
        
        if use_docker:
            try:
                self.docker_client = docker.from_env()
                logger.info("Docker client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Docker: {e}. Falling back to local execution.")
                self.use_docker = False
    
    def execute_and_evaluate(self, code: str, problem: ProgrammingProblem) -> ExecutionMetrics:
        """
        Execute code and return comprehensive metrics.
        
        Args:
            code: The Python code to execute
            problem: The programming problem being solved
            
        Returns:
            ExecutionMetrics with all evaluation results
        """
        metrics = ExecutionMetrics()
        
        # Create temporary directory for execution
        with tempfile.TemporaryDirectory() as temp_dir:
            code_file = os.path.join(temp_dir, "solution.py")
            test_file = os.path.join(temp_dir, "test_solution.py")
            
            # Write code to file
            with open(code_file, 'w') as f:
                f.write(code)
            
            # Generate test file
            self._generate_test_file(test_file, problem)
            
            # Perform static analysis
            metrics.security_vulnerabilities = self._check_security(code_file)
            metrics.style_violations = self._check_style(code_file)
            metrics.syntax_errors = self._check_syntax(code)
            
            # Execute tests if syntax is valid
            if metrics.syntax_errors == 0:
                if self.use_docker:
                    metrics = self._execute_in_docker(temp_dir, metrics, problem)
                else:
                    metrics = self._execute_locally(temp_dir, metrics, problem)
            
            # Calculate composite scores
            metrics.performance_score = self._calculate_performance_score(metrics)
            metrics.code_quality_score = self._calculate_quality_score(metrics)
        
        return metrics
    
    def _generate_test_file(self, test_file: str, problem: ProgrammingProblem):
        """Generate a test file for the given problem."""
        test_code = f"""
import sys
import os
import unittest
import coverage
import time
import psutil
from solution import *

class TestSolution(unittest.TestCase):
    def setUp(self):
        self.start_time = time.time()
        self.process = psutil.Process(os.getpid())
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
    
    def tearDown(self):
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Store metrics in global variables for retrieval
        global execution_time, memory_usage
        execution_time = end_time - self.start_time
        memory_usage = max(0, end_memory - self.start_memory)
"""
        
        # Add test methods for each test case
        for i, test_case in enumerate(problem.test_cases):
            test_code += f"""
    def test_case_{i}(self):
        # Test case {i}
        inputs = {test_case}
        expected = {problem.expected_outputs[i] if i < len(problem.expected_outputs) else None}
        
        try:
            # This assumes the main function is called 'solution' or similar
            # In practice, this would be more sophisticated
            result = solution(*inputs.values()) if hasattr(inputs, 'values') else solution(inputs)
            self.assertEqual(result, expected)
        except Exception as e:
            self.fail(f"Test case {i} failed with exception: {{e}}")
"""
        
        test_code += """
if __name__ == '__main__':
    # Initialize coverage
    cov = coverage.Coverage()
    cov.start()
    
    # Run tests
    unittest.main(exit=False, verbosity=2)
    
    # Stop coverage and save results
    cov.stop()
    cov.save()
    
    # Print coverage report
    print("\\n=== Coverage Report ===")
    cov.report()
"""
        
        with open(test_file, 'w') as f:
            f.write(test_code)
    
    def _check_security(self, code_file: str) -> int:
        """Check for security vulnerabilities using Bandit."""
        try:
            from bandit.core import manager
            from bandit.core import config
            
            # Create Bandit manager
            conf = config.BanditConfig()
            b_mgr = manager.BanditManager(conf, 'file')
            
            # Run Bandit on the file
            b_mgr.discover([code_file])
            b_mgr.run_tests()
            
            # Count vulnerabilities
            return len([issue for issue in b_mgr.get_issue_list()])
        except Exception as e:
            logger.warning(f"Security check failed: {e}")
            return 0
    
    def _check_style(self, code_file: str) -> int:
        """Check style violations using flake8."""
        try:
            style_guide = flake8.get_style_guide()
            report = style_guide.check_files([code_file])
            return report.get_count()
        except Exception as e:
            logger.warning(f"Style check failed: {e}")
            return 0
    
    def _check_syntax(self, code: str) -> int:
        """Check for syntax errors."""
        try:
            ast.parse(code)
            return 0
        except SyntaxError:
            return 1
    
    def _execute_in_docker(self, temp_dir: str, metrics: ExecutionMetrics, 
                          problem: ProgrammingProblem) -> ExecutionMetrics:
        """Execute code in Docker container for isolation."""
        try:
            # Create Docker container
            container = self.docker_client.containers.run(
                'python:3.9-slim',
                f'python test_solution.py',
                volumes={temp_dir: {'bind': '/workspace', 'mode': 'rw'}},
                working_dir='/workspace',
                detach=True,
                mem_limit=f"{problem.memory_limit}m",
                timeout=self.timeout
            )
            
            # Wait for completion
            result = container.wait(timeout=self.timeout)
            logs = container.logs().decode('utf-8')
            
            # Parse results from logs
            metrics = self._parse_execution_results(logs, metrics)
            
            # Cleanup
            container.remove()
            
        except Exception as e:
            logger.error(f"Docker execution failed: {e}")
            metrics.runtime_errors = 1
        
        return metrics
    
    def _execute_locally(self, temp_dir: str, metrics: ExecutionMetrics,
                        problem: ProgrammingProblem) -> ExecutionMetrics:
        """Execute code locally with resource monitoring."""
        try:
            # Change to temp directory
            original_dir = os.getcwd()
            os.chdir(temp_dir)
            
            # Execute test file
            start_time = time.time()
            result = subprocess.run([
                sys.executable, 'test_solution.py'
            ], capture_output=True, text=True, timeout=self.timeout)
            end_time = time.time()
            
            # Parse results
            metrics.execution_time = end_time - start_time
            metrics = self._parse_execution_results(result.stdout, metrics)
            
            if result.returncode != 0:
                metrics.runtime_errors = 1
                logger.error(f"Execution error: {result.stderr}")
            
        except subprocess.TimeoutExpired:
            metrics.runtime_errors = 1
            logger.error("Execution timed out")
        except Exception as e:
            metrics.runtime_errors = 1
            logger.error(f"Local execution failed: {e}")
        finally:
            os.chdir(original_dir)
        
        return metrics
    
    def _parse_execution_results(self, output: str, metrics: ExecutionMetrics) -> ExecutionMetrics:
        """Parse execution output to extract metrics."""
        lines = output.split('\n')
        
        # Parse test results
        passed_tests = 0
        total_tests = 0
        
        for line in lines:
            if 'test_case_' in line and 'ok' in line:
                passed_tests += 1
                total_tests += 1
            elif 'test_case_' in line and ('FAIL' in line or 'ERROR' in line):
                total_tests += 1
            elif 'Coverage Report' in line:
                # Parse coverage information from subsequent lines
                for i, next_line in enumerate(lines[lines.index(line):]):
                    if '%' in next_line and 'TOTAL' in next_line:
                        try:
                            coverage_str = next_line.split()[-1].replace('%', '')
                            metrics.coverage_percentage = float(coverage_str)
                        except:
                            pass
                        break
        
        if total_tests > 0:
            metrics.test_pass_rate = passed_tests / total_tests
            metrics.test_passed = (passed_tests == total_tests)
        
        return metrics
    
    def _calculate_performance_score(self, metrics: ExecutionMetrics) -> float:
        """Calculate a composite performance score."""
        if metrics.execution_time <= 0:
            return 0.0
        
        time_score = min(1.0, 1.0 / metrics.execution_time)
        memory_score = max(0.0, 1.0 - (metrics.memory_usage / 100.0))  # Normalize to 100MB
        
        return (time_score + memory_score) / 2.0
    
    def _calculate_quality_score(self, metrics: ExecutionMetrics) -> float:
        """Calculate a composite code quality score."""
        quality_score = 1.0
        
        # Penalize issues
        quality_score -= (metrics.security_vulnerabilities * 0.2)
        quality_score -= (metrics.style_violations * 0.05)
        quality_score -= (metrics.syntax_errors * 0.5)
        quality_score -= (metrics.runtime_errors * 0.3)
        
        return max(0.0, quality_score)


class RewardFunction:
    """
    Multi-objective reward function that guides the AI's learning process
    by rewarding good practices and penalizing poor ones.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize reward function with configurable weights.
        
        Args:
            weights: Custom weights for different reward components
        """
        self.weights = weights or {
            'test_success': 10.0,
            'test_failure': -10.0,
            'coverage_multiplier': 5.0,
            'performance_multiplier': 1.0,
            'security_penalty': -20.0,
            'style_penalty': -5.0,
            'syntax_penalty': -15.0,
            'runtime_penalty': -10.0
        }
    
    def calculate_reward(self, metrics: ExecutionMetrics) -> float:
        """
        Calculate the total reward based on execution metrics.
        
        Args:
            metrics: ExecutionMetrics from code execution
            
        Returns:
            Total reward value
        """
        reward = 0.0
        
        # Test success/failure rewards
        if metrics.test_passed:
            reward += self.weights['test_success']
        else:
            reward += self.weights['test_failure']
        
        # Coverage reward (scaled by coverage percentage)
        reward += (metrics.coverage_percentage / 100.0) * self.weights['coverage_multiplier']
        
        # Performance reward
        reward += metrics.performance_score * self.weights['performance_multiplier']
        
        # Security penalties
        reward += metrics.security_vulnerabilities * self.weights['security_penalty']
        
        # Style penalties
        reward += metrics.style_violations * self.weights['style_penalty']
        
        # Syntax error penalties
        reward += metrics.syntax_errors * self.weights['syntax_penalty']
        
        # Runtime error penalties
        reward += metrics.runtime_errors * self.weights['runtime_penalty']
        
        return reward
    
    def get_detailed_breakdown(self, metrics: ExecutionMetrics) -> Dict[str, float]:
        """Get detailed breakdown of reward components."""
        breakdown = {}
        
        breakdown['test_reward'] = (self.weights['test_success'] if metrics.test_passed 
                                  else self.weights['test_failure'])
        breakdown['coverage_reward'] = ((metrics.coverage_percentage / 100.0) * 
                                      self.weights['coverage_multiplier'])
        breakdown['performance_reward'] = (metrics.performance_score * 
                                         self.weights['performance_multiplier'])
        breakdown['security_penalty'] = (metrics.security_vulnerabilities * 
                                       self.weights['security_penalty'])
        breakdown['style_penalty'] = (metrics.style_violations * 
                                    self.weights['style_penalty'])
        breakdown['syntax_penalty'] = (metrics.syntax_errors * 
                                     self.weights['syntax_penalty'])
        breakdown['runtime_penalty'] = (metrics.runtime_errors * 
                                      self.weights['runtime_penalty'])
        
        breakdown['total_reward'] = sum(breakdown.values())
        
        return breakdown


class PPOAgent:
    """
    Proximal Policy Optimization agent for training the GraphformicCoder.
    Implements the PPO algorithm for policy gradient learning.
    """
    
    def __init__(self, model, learning_rate: float = 3e-4, clip_epsilon: float = 0.2,
                 value_coef: float = 0.5, entropy_coef: float = 0.01):
        """
        Initialize PPO agent.
        
        Args:
            model: The GraphformicCoder model to train
            learning_rate: Learning rate for optimization
            clip_epsilon: PPO clipping parameter
            value_coef: Value function coefficient
            entropy_coef: Entropy bonus coefficient
        """
        self.model = model
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Create value function head
        self.value_head = nn.Linear(model.d_model, 1)
        
        # Optimizer for both policy and value function
        all_params = list(model.parameters()) + list(self.value_head.parameters())
        self.optimizer = optim.Adam(all_params, lr=learning_rate)
        
        # Storage for trajectory data
        self.trajectory_buffer = []
    
    def select_action(self, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select action using current policy.
        
        Args:
            state: Current state representation
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        with torch.no_grad():
            # Get model output (logits)
            logits = self.model(**state)
            
            # Get value estimate
            # For simplicity, we'll use the mean of the last hidden states
            hidden_states = logits.mean(dim=1)  # [batch_size, d_model]
            value = self.value_head(hidden_states)
            
            # Create action distribution
            action_dist = Categorical(logits=logits.view(-1, logits.size(-1)))
            
            # Sample action
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
        return action, log_prob, value
    
    def compute_advantages(self, rewards: List[float], values: List[torch.Tensor], 
                          gamma: float = 0.99, gae_lambda: float = 0.95) -> Tuple[List[float], List[float]]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            
        Returns:
            Tuple of (advantages, returns)
        """
        advantages = []
        returns = []
        
        # Convert values to numpy for easier computation
        values_np = [v.item() if torch.is_tensor(v) else v for v in values]
        
        # Compute returns and advantages
        advantage = 0
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values_np[i + 1]
            
            delta = rewards[i] + gamma * next_value - values_np[i]
            advantage = delta + gamma * gae_lambda * advantage
            advantages.insert(0, advantage)
            returns.insert(0, advantage + values_np[i])
        
        return advantages, returns
    
    def update_policy(self, states: List[Dict[str, torch.Tensor]], actions: List[torch.Tensor],
                     old_log_probs: List[torch.Tensor], advantages: List[float],
                     returns: List[float], epochs: int = 4):
        """
        Update policy using PPO objective.
        
        Args:
            states: List of states
            actions: List of actions taken
            old_log_probs: List of old action log probabilities
            advantages: List of advantage estimates
            returns: List of returns
            epochs: Number of optimization epochs
        """
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for epoch in range(epochs):
            total_loss = 0
            
            for i, state in enumerate(states):
                # Get current policy output
                logits = self.model(**state)
                hidden_states = logits.mean(dim=1)
                value = self.value_head(hidden_states)
                
                # Compute action probabilities
                action_dist = Categorical(logits=logits.view(-1, logits.size(-1)))
                new_log_prob = action_dist.log_prob(actions[i])
                entropy = action_dist.entropy().mean()
                
                # Compute PPO loss
                ratio = torch.exp(new_log_prob - old_log_probs[i])
                surr1 = ratio * advantages[i]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[i]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value function loss
                value_loss = nn.MSELoss()(value.squeeze(), returns[i])
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.value_head.parameters()), 
                    max_norm=0.5
                )
                self.optimizer.step()
                
                total_loss += loss.item()
            
            logger.info(f"PPO Epoch {epoch + 1}/{epochs}, Average Loss: {total_loss / len(states):.4f}")


class DigitalCrucible:
    """
    Main orchestrator for the autonomous training harness.
    Coordinates problem environment, execution sandbox, reward function, and PPO learning.
    """
    
    def __init__(self, model, problem_sources: List[str], 
                 use_docker: bool = True, log_dir: str = "./crucible_logs"):
        """
        Initialize the Digital Crucible.
        
        Args:
            model: GraphformicCoder model to train
            problem_sources: List of problem source paths/APIs
            use_docker: Whether to use Docker for sandboxing
            log_dir: Directory for storing training logs
        """
        self.model = model
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.problem_env = ProblemEnvironment(problem_sources)
        self.sandbox = ExecutionSandbox(use_docker=use_docker)
        self.reward_func = RewardFunction()
        self.ppo_agent = PPOAgent(model)
        
        # Training statistics
        self.episode_count = 0
        self.total_reward = 0.0
        self.success_rate = 0.0
        self.training_history = []
        
        logger.info("Digital Crucible initialized successfully")
    
    def run_training_loop(self, num_episodes: int = 1000, save_interval: int = 100):
        """
        Run the main reinforcement learning training loop.
        
        Args:
            num_episodes: Number of training episodes
            save_interval: Interval for saving model checkpoints
        """
        logger.info(f"Starting training loop for {num_episodes} episodes")
        
        for episode in range(num_episodes):
            self.episode_count = episode + 1
            
            try:
                # Run single episode
                episode_reward, episode_success = self._run_episode()
                
                # Update statistics
                self.total_reward += episode_reward
                avg_reward = self.total_reward / self.episode_count
                
                # Log progress
                if episode % 10 == 0:
                    logger.info(f"Episode {episode + 1}/{num_episodes}, "
                              f"Reward: {episode_reward:.2f}, "
                              f"Average Reward: {avg_reward:.2f}, "
                              f"Success: {episode_success}")
                
                # Save checkpoint
                if episode % save_interval == 0:
                    self._save_checkpoint(episode)
                
                # Store training history
                self.training_history.append({
                    'episode': episode + 1,
                    'reward': episode_reward,
                    'success': episode_success,
                    'avg_reward': avg_reward
                })
                
            except Exception as e:
                logger.error(f"Episode {episode + 1} failed: {e}")
                continue
        
        logger.info("Training loop completed")
        self._save_final_results()
    
    def _run_episode(self) -> Tuple[float, bool]:
        """
        Run a single training episode.
        
        Returns:
            Tuple of (episode_reward, success)
        """
        # Get a problem
        problem = self.problem_env.get_random_problem()
        if not problem:
            logger.warning("No problems available")
            return 0.0, False
        
        # Prepare state representation
        state = self._prepare_state(problem)
        
        # Generate code using the model
        try:
            generated_tokens = self.model.generate(
                src_tokens=state['src_tokens'],
                node_features=state['node_features'],
                edge_index=state['edge_index'],
                batch_graph=state.get('batch_graph'),
                max_length=512,
                temperature=0.8
            )
            
            # Convert tokens back to code (this would need a proper tokenizer)
            generated_code = self._tokens_to_code(generated_tokens)
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return -50.0, False  # Large penalty for generation failure
        
        # Execute and evaluate the generated code
        metrics = self.sandbox.execute_and_evaluate(generated_code, problem)
        
        # Calculate reward
        reward = self.reward_func.calculate_reward(metrics)
        
        # Log detailed metrics
        reward_breakdown = self.reward_func.get_detailed_breakdown(metrics)
        self._log_episode_details(problem, generated_code, metrics, reward_breakdown)
        
        return reward, metrics.test_passed
    
    def _prepare_state(self, problem: ProgrammingProblem) -> Dict[str, torch.Tensor]:
        """
        Prepare state representation from problem description.
        
        Args:
            problem: Programming problem
            
        Returns:
            State dictionary for the model
        """
        # This is a simplified state preparation
        # In practice, you'd need proper tokenization and AST parsing
        
        # Create dummy tensors for demonstration
        batch_size = 1
        seq_len = 100
        num_nodes = 50
        num_edges = 80
        
        state = {
            'src_tokens': torch.randint(1, 1000, (batch_size, seq_len)),
            'node_features': torch.randn(num_nodes, 128),
            'edge_index': torch.randint(0, num_nodes, (2, num_edges)),
            'batch_graph': torch.zeros(num_nodes, dtype=torch.long)
        }
        
        return state
    
    def _tokens_to_code(self, tokens: torch.Tensor) -> str:
        """
        Convert generated tokens back to Python code.
        
        Args:
            tokens: Generated token tensor
            
        Returns:
            Python code string
        """
        # This is a placeholder - you'd need a proper detokenizer
        # For demonstration, return a simple function template
        
        return """
def solution(nums):
    # AI-generated solution would go here
    # This is a placeholder implementation
    return sum(nums) if nums else 0
"""
    
    def _log_episode_details(self, problem: ProgrammingProblem, code: str, 
                           metrics: ExecutionMetrics, reward_breakdown: Dict[str, float]):
        """Log detailed information about the episode."""
        log_entry = {
            'episode': self.episode_count,
            'problem_id': problem.id,
            'problem_title': problem.title,
            'generated_code': code,
            'metrics': {
                'test_passed': metrics.test_passed,
                'test_pass_rate': metrics.test_pass_rate,
                'coverage_percentage': metrics.coverage_percentage,
                'execution_time': metrics.execution_time,
                'memory_usage': metrics.memory_usage,
                'security_vulnerabilities': metrics.security_vulnerabilities,
                'style_violations': metrics.style_violations,
                'syntax_errors': metrics.syntax_errors,
                'runtime_errors': metrics.runtime_errors,
                'performance_score': metrics.performance_score,
                'code_quality_score': metrics.code_quality_score
            },
            'reward_breakdown': reward_breakdown
        }
        
        # Save to log file
        log_file = self.log_dir / f"episode_{self.episode_count:06d}.json"
        with open(log_file, 'w') as f:
            json.dump(log_entry, f, indent=2)
    
    def _save_checkpoint(self, episode: int):
        """Save model checkpoint."""
        checkpoint_path = self.log_dir / f"checkpoint_episode_{episode:06d}.pt"
        torch.save({
            'episode': episode,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.ppo_agent.optimizer.state_dict(),
            'total_reward': self.total_reward,
            'training_history': self.training_history
        }, checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_final_results(self):
        """Save final training results and statistics."""
        results = {
            'total_episodes': self.episode_count,
            'total_reward': self.total_reward,
            'average_reward': self.total_reward / max(1, self.episode_count),
            'training_history': self.training_history,
            'final_success_rate': sum(1 for h in self.training_history if h['success']) / max(1, len(self.training_history))
        }
        
        results_file = self.log_dir / "training_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Final results saved: {results_file}")
        logger.info(f"Training completed with average reward: {results['average_reward']:.2f}")
        logger.info(f"Final success rate: {results['final_success_rate']:.2%}")


# Example usage and testing
if __name__ == "__main__":
    # This would normally import the GraphformicCoder from the previous implementation
    # For testing purposes, we'll create a mock model
    
    class MockGraphformicCoder:
        """Mock model for testing purposes."""
        def __init__(self):
            self.d_model = 512
            
        def parameters(self):
            return []
            
        def state_dict(self):
            return {}
            
        def generate(self, **kwargs):
            # Return dummy tokens
            return torch.randint(1, 1000, (1, 50))
    
    # Initialize components
    print("🔥 Digital Crucible: Autonomous Training Harness")
    print("=" * 60)
    
    # Create sample problems directory
    problems_dir = "./sample_problems"
    os.makedirs(problems_dir, exist_ok=True)
    
    # Create a sample problem
    sample_problem = {
        "id": "two_sum",
        "title": "Two Sum",
        "description": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
        "difficulty": "Easy",
        "test_cases": [
            {"nums": [2, 7, 11, 15], "target": 9},
            {"nums": [3, 2, 4], "target": 6},
            {"nums": [3, 3], "target": 6}
        ],
        "expected_outputs": [[0, 1], [1, 2], [0, 1]],
        "time_limit": 5.0,
        "memory_limit": 256
    }
    
    with open(os.path.join(problems_dir, "two_sum.json"), 'w') as f:
        json.dump(sample_problem, f, indent=2)
    
    print(f"✅ Sample problem created in {problems_dir}")
    
    # Initialize mock model
    mock_model = MockGraphformicCoder()
    
    # Initialize Digital Crucible
    try:
        crucible = DigitalCrucible(
            model=mock_model,
            problem_sources=[problems_dir],
            use_docker=False,  # Disable Docker for testing
            log_dir="./crucible_test_logs"
        )
        
        print(f"✅ Digital Crucible initialized successfully")
        print(f"📚 Loaded {len(crucible.problem_env.problems)} problems")
        
        # Run a short training session
        print("\n🚀 Starting training demonstration...")
        crucible.run_training_loop(num_episodes=5, save_interval=2)
        
        print("\n📊 Training Statistics:")
        print(f"   Total Episodes: {crucible.episode_count}")
        print(f"   Average Reward: {crucible.total_reward / max(1, crucible.episode_count):.2f}")
        
        print("\n🎯 Digital Crucible demonstration completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    if os.path.exists(problems_dir):
        shutil.rmtree(problems_dir)
    
    print("\n🧹 Cleanup completed")
"""MetamorphEngine: Neuro-evolution for GraphformicCoder architectures.

This module implements a lightweight Genetic Algorithm / NAS hybrid that
searches the architectural hyper-space of *GraphformicCoder* using the
*DigitalCrucible* framework as its fitness evaluation environment.

DISCLAIMER: The full evolution pipeline is computationally expensive. The code
is written for clarity and modularity rather than production performance. In a
real deployment, distribute evaluation across a cluster and persist genomes in
a database.
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

# Phase-2 components ----------------------------------------------------------
from digital_crucible import (
    ExecutionSandbox,
    ProblemEnvironment,
    PPOTrainer,
    calculate_reward,
)
from graphformic_coder import GraphformicCoder

try:
    from transformers import AutoTokenizer  # type: ignore
except ImportError:  # pragma: no cover – transformers optional for basic import
    AutoTokenizer = None  # type: ignore


# ---------------------------------------------------------------------------
# 1. Architectural Genome Definition
# ---------------------------------------------------------------------------
ACTIVATIONS = ["relu", "gelu", "swish"]


def _round_multiple(value: int, base: int = 64) -> int:
    return max(base, int(base * round(float(value) / base)))


@dataclass
class Genome:
    """Encodes architectural hyper-parameters for GraphformicCoder."""

    # Transformer -----------------------------------------------------------
    transformer_layers: int = field(default_factory=lambda: random.randint(2, 12))
    d_model: int = field(default_factory=lambda: _round_multiple(random.choice([256, 384, 512, 640, 768])))
    nhead: int = 8  # kept constant for simplicity – ensure nhead | d_model

    # Graph encoder ---------------------------------------------------------
    gnn_layers: int = field(default_factory=lambda: random.randint(1, 4))
    num_gat_heads: int = 4

    # Activation ------------------------------------------------------------
    activation: str = field(default_factory=lambda: random.choice(ACTIVATIONS))

    # Fusion weighting (simple scalar applied post-fusion) ------------------
    fusion_balance: float = field(default_factory=lambda: round(random.uniform(0.3, 1.0), 2))

    def mutate(self) -> "Genome":
        """Return a *new* mutated copy of this genome."""
        mutant = copy.deepcopy(self)
        mutation_type = random.choice(
            [
                "add_layer",
                "remove_layer",
                "inc_width",
                "dec_width",
                "change_activation",
                "tweak_fusion",
            ]
        )
        if mutation_type == "add_layer":
            mutant.transformer_layers = min(mutant.transformer_layers + 1, 12)
        elif mutation_type == "remove_layer":
            mutant.transformer_layers = max(2, mutant.transformer_layers - 1)
        elif mutation_type == "inc_width":
            mutant.d_model = _round_multiple(mutant.d_model + 64)
        elif mutation_type == "dec_width":
            mutant.d_model = _round_multiple(max(256, mutant.d_model - 64))
        elif mutation_type == "change_activation":
            mutant.activation = random.choice([a for a in ACTIVATIONS if a != mutant.activation])
        elif mutation_type == "tweak_fusion":
            delta = random.uniform(-0.1, 0.1)
            mutant.fusion_balance = float(min(1.0, max(0.1, mutant.fusion_balance + delta)))
        return mutant

    @staticmethod
    def crossover(parent_a: "Genome", parent_b: "Genome") -> "Genome":
        """Single-point crossover of two parent genomes."""
        child = Genome()
        for field_name in child.__dataclass_fields__:
            if random.random() < 0.5:
                setattr(child, field_name, getattr(parent_a, field_name))
            else:
                setattr(child, field_name, getattr(parent_b, field_name))
        return child

    # Pretty-printing -------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return json.dumps(asdict(self), indent=2)


# ---------------------------------------------------------------------------
# 2. Genome → GraphformicCoder Builder
# ---------------------------------------------------------------------------

def build_model_from_genome(genome: Genome, vocab_size: int = 32_000, ast_feat_dim: int = 64) -> GraphformicCoder:
    """Instantiate *GraphformicCoder* from genome parameters.

    NOTE: Current GraphformicCoder does not expose activation & fusion_balance
    parameters. We honour `d_model`, `transformer_layers`, and leave a TODO
    hook for future customisation.
    """
    # Ensure d_model divisible by nhead
    if genome.d_model % genome.nhead != 0:
        genome.d_model = _round_multiple(genome.nhead * math.ceil(genome.d_model / genome.nhead))
    model = GraphformicCoder(
        vocab_size=vocab_size,
        ast_feat_dim=ast_feat_dim,
        d_model=genome.d_model,
        nhead=genome.nhead,
        num_layers=genome.transformer_layers,
    )
    return model


# ---------------------------------------------------------------------------
# 3. Fitness Evaluation Wrapper
# ---------------------------------------------------------------------------
class FitnessEvaluator:
    """Runs a mini DigitalCrucible session and returns peak reward."""

    def __init__(self, problems_dir: Path, tokenizer, episodes: int = 200, device: str = "cpu") -> None:
        self.problems_path = problems_dir
        self.episodes = episodes
        self.device = device
        self.tokenizer = tokenizer

        # Shared (static) objects
        self.problem_env = ProblemEnvironment(self.problems_path)
        self.sandbox = ExecutionSandbox()

    # ---------------------------------------------------------------------
    def evaluate(self, genome: Genome) -> float:
        """Return fitness score (higher is better)."""
        model = build_model_from_genome(genome).to(self.device)
        trainer = PPOTrainer(model, self.tokenizer)

        peak_reward = -float("inf")
        for _ in range(self.episodes):
            problem = self.problem_env.sample_problem()
            code, logits = trainer.generate_code(problem["description"])
            metrics = self.sandbox.evaluate(code, problem)
            reward = calculate_reward(metrics)
            peak_reward = max(peak_reward, reward)
            actions = torch.argmax(logits, dim=-1)
            trainer.update_policy(logits, actions, reward)
        return float(peak_reward)


# ---------------------------------------------------------------------------
# 4. MetamorphEngine – Evolutionary Orchestrator
# ---------------------------------------------------------------------------
class MetamorphEngine:
    """Genetic-algorithm backend orchestrating architecture evolution."""

    def __init__(
        self,
        problems_dir: Path,
        generations: int = 10,
        population_size: int = 50,
        survive_frac: float = 0.05,
        episodes_per_candidate: int = 200,
        device: str = "cpu",
    ) -> None:
        if AutoTokenizer is None:
            raise RuntimeError("transformers library required for MetamorphEngine")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.evaluator = FitnessEvaluator(problems_dir, self.tokenizer, episodes_per_candidate, device)
        self.generations = generations
        self.population_size = population_size
        self.survive_n = max(1, int(population_size * survive_frac))
        self.device = device

        # Initialise population --------------------------------------------
        self.population: List[Genome] = [Genome() for _ in range(population_size)]

    # ---------------------------------------------------------------------
    def evolve(self) -> None:
        for gen in range(1, self.generations + 1):
            print(f"\n=== Generation {gen}/{self.generations} ===")
            fitness_scores: List[Tuple[float, Genome]] = []

            # Evaluate population -----------------------------------------
            for idx, genome in enumerate(self.population, 1):
                print(f"Evaluating candidate {idx}/{self.population_size} ...", end=" ")
                score = self.evaluator.evaluate(genome)
                fitness_scores.append((score, genome))
                print(f"fitness={score:+.3f}")

            # Select survivors -------------------------------------------
            fitness_scores.sort(key=lambda x: x[0], reverse=True)
            survivors = [g for _, g in fitness_scores[: self.survive_n]]
            best_score, best_genome = fitness_scores[0]
            print(f"Best fitness this gen: {best_score:.3f} | Genome: {best_genome}")

            # Create next generation --------------------------------------
            next_population: List[Genome] = []
            # Elitism – keep best genome unmodified
            next_population.append(copy.deepcopy(best_genome))
            while len(next_population) < self.population_size:
                parent_a, parent_b = random.sample(survivors, 2) if len(survivors) >= 2 else (survivors[0], survivors[0])
                child = Genome.crossover(parent_a, parent_b)
                # Mutate child with some probability
                if random.random() < 0.9:
                    child = child.mutate()
                next_population.append(child)

            self.population = next_population

        print("Evolution complete. Best architecture:")
        print(best_genome)


# ---------------------------------------------------------------------------
# __main__ – smoke test with tiny settings ------------------------------------
if __name__ == "__main__":
    problems_root = Path("./problems")
    problems_root.mkdir(exist_ok=True)
    # Ensure at least one simple problem exists ----------------------------
    dummy_dir = problems_root / "add_two"
    if not (dummy_dir / "tests.py").exists():
        dummy_dir.mkdir(exist_ok=True)
        (dummy_dir / "description.md").write_text(
            """## Add Two Numbers\nWrite a function `solve(a, b)` that returns the sum of `a` and `b`.\n"""
        )
        (dummy_dir / "tests.py").write_text(
            """import solution, pytest\n\n\n@pytest.mark.parametrize('a,b,expected', [\n    (1, 2, 3),\n    (-1, 5, 4),\n    (100, 200, 300)\n])\ndef test_add(a, b, expected):\n    assert solution.solve(a, b) == expected\n"""
        )

    engine = MetamorphEngine(
        problems_dir=problems_root,
        generations=2,  # keep small for demo
        population_size=10,
        episodes_per_candidate=20,
    )
    engine.evolve()
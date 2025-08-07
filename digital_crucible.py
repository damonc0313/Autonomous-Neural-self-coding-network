"""DigitalCrucible: A multi-objective reinforcement-learning harness for GraphformicCoder.

This module is intentionally self-contained and runs with minimal external
requirements. Optional dependencies are lazily imported and gracefully handled
so the framework remains runnable even in stripped-down environments.

Key components
--------------
1. ProblemEnvironment   – Supplies programming problems.
2. ExecutionSandbox     – Safely executes AI-generated code, gathers metrics.
3. calculate_reward     – Multi-objective reward shaping function.
4. PPOTrainer           – Proximal Policy Optimisation loop.
5. DigitalCrucible      – High-level orchestrator.

NOTE: This harness focuses on architecture and extensibility over raw speed.
Integrate advanced sand-boxing (e.g., Docker / gVisor) in production.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

# Optional libraries for metrics ------------------------------------------------
try:
    import psutil  # type: ignore
except ImportError:
    psutil = None  # noqa: N816 – fallback

# -----------------------------------------------------------------------------
# 1. Problem Environment
# -----------------------------------------------------------------------------
class ProblemEnvironment:
    """Loads and serves programming challenges from a directory tree.

    Directory schema::

        problems/
            <slug>/
                description.md
                tests.py
                reference_solution.py (optional)

    The environment randomly samples problems for each episode.
    """

    def __init__(self, problems_root: Path):
        self.problems_root = Path(problems_root).expanduser().resolve()
        if not self.problems_root.exists():
            raise FileNotFoundError(f"Problem root not found: {self.problems_root}")
        self.problem_slugs = [p.name for p in self.problems_root.iterdir() if p.is_dir()]
        if not self.problem_slugs:
            raise ValueError("No problems found under provided directory.")

    def sample_problem(self) -> Dict[str, Any]:
        slug = random.choice(self.problem_slugs)
        prob_dir = self.problems_root / slug
        with open(prob_dir / "description.md", "r", encoding="utf-8") as fp:
            description = fp.read()
        tests_path = prob_dir / "tests.py"
        return {
            "id": slug,
            "description": description,
            "tests_path": tests_path,
            "working_dir": prob_dir,
        }


# -----------------------------------------------------------------------------
# 2. Execution & Feedback Sandbox
# -----------------------------------------------------------------------------
@dataclass
class ExecutionMetrics:
    passed: bool
    coverage: float
    exec_time: float
    memory_mb: float
    security_issues: int
    style_violations: int
    raw: Dict[str, Any]


class ExecutionSandbox:
    """Runs AI-generated code safely and collects metrics.

    The sandbox uses a temporary directory per execution. In production, replace
    this with containerised sandboxes (Docker, Firecracker, etc.) for stronger
    isolation. For brevity, we rely on `pytest`, `coverage`, `bandit`, and
    `pycodestyle` which must be installed in the host environment.
    """

    DEFAULT_TIMEOUT_SEC = 10

    def __init__(self, timeout: int = DEFAULT_TIMEOUT_SEC):
        self.timeout = timeout

    # ---------------- Helpers -------------------------------------------------
    @staticmethod
    def _run_with_timeout(cmd: List[str], cwd: Path, timeout: int) -> Tuple[str, str, float, float]:
        """Runs command returning (stdout, stderr, runtime_s, peak_memory_mb)."""
        start = time.perf_counter()
        process = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        peak_mem_mb = 0.0
        try:
            if psutil is not None:
                p = psutil.Process(process.pid)
            else:
                p = None  # type: ignore
            while process.poll() is None:
                if p is not None:
                    mem_mb = p.memory_info().rss / 1024 ** 2
                    peak_mem_mb = max(peak_mem_mb, mem_mb)
                time.sleep(0.05)
                if time.perf_counter() - start > timeout:
                    process.send_signal(signal.SIGKILL)
                    raise TimeoutError(f"Command exceeded {timeout}s timeout: {' '.join(cmd)}")
            out, err = process.communicate()
        finally:
            runtime = time.perf_counter() - start
        return out.decode(), err.decode(), runtime, peak_mem_mb

    # ---------------- Public API ---------------------------------------------
    def evaluate(self, code: str, problem: Dict[str, Any]) -> ExecutionMetrics:
        """Write `code` to solution.py, run tests + static checks, gather metrics."""
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            # Copy problem files into sandbox dir
            shutil.copy(problem["tests_path"], tmpdir / "tests.py")
            # Persist AI solution
            (tmpdir / "solution.py").write_text(code, encoding="utf-8")

            raw_log: Dict[str, Any] = {}
            # 1. Run unit tests with coverage --------------------------------
            cov_xml = tmpdir / "coverage.xml"
            cmd_test = [
                sys.executable,
                "-m",
                "pytest",
                "--disable-warnings",
                "--quiet",
                "--cov=.",
                f"--cov-report=xml:{cov_xml}",
            ]
            try:
                out, err, runtime, mem_mb = self._run_with_timeout(cmd_test, cwd=tmpdir, timeout=self.timeout)
                raw_log["pytest_out"] = out + err
                passed = "= 0 failed" in raw_log["pytest_out"]
            except Exception as e:  # noqa: BLE001
                passed = False
                runtime = self.timeout
                mem_mb = 0.0
                raw_log["pytest_error"] = str(e)

            # Parse coverage ---------------------------------------------------
            coverage_pct = 0.0
            if cov_xml.exists():
                try:
                    import xml.etree.ElementTree as ET

                    tree = ET.parse(cov_xml)
                    root = tree.getroot()
                    coverage_pct = float(root.attrib.get("line-rate", 0.0)) * 100.0
                except Exception:  # noqa: BLE001 broad except – ignore parse errors
                    pass

            # 2. Static security audit ----------------------------------------
            cmd_bandit = [sys.executable, "-m", "bandit", "-r", "solution.py", "-f", "json"]
            try:
                out, err, *_ = self._run_with_timeout(cmd_bandit, cwd=tmpdir, timeout=self.timeout)
                bandit_json = json.loads(out or "{}")
                security_issues = len(bandit_json.get("results", []))
            except Exception:
                security_issues = 1  # penalise unknown errors

            # 3. Style check ----------------------------------------------------
            cmd_style = [sys.executable, "-m", "pycodestyle", "--statistics", "solution.py"]
            try:
                out, err, *_ = self._run_with_timeout(cmd_style, cwd=tmpdir, timeout=self.timeout)
                style_violations = sum(1 for line in out.splitlines() if line.strip())
            except Exception:
                style_violations = 1  # treat failure as one violation

            return ExecutionMetrics(
                passed=passed,
                coverage=coverage_pct / 100.0,  # normalise 0-1
                exec_time=runtime,
                memory_mb=mem_mb,
                security_issues=security_issues,
                style_violations=style_violations,
                raw=raw_log,
            )


# -----------------------------------------------------------------------------
# 3. Reward Function
# -----------------------------------------------------------------------------

def calculate_reward(m: ExecutionMetrics) -> float:  # noqa: C901 – clarity over complexity
    """Multi-objective reward shaping.

    Parameters
    ----------
    m : ExecutionMetrics
        The metrics collected by the sandbox.

    Returns
    -------
    float
        The scalar reward signal.
    """
    reward = 0.0

    # Unit tests -------------------------------------------------------------
    if m.passed:
        reward += 10
    else:
        reward -= 10

    # Coverage ---------------------------------------------------------------
    reward += m.coverage * 5  # coverage is 0-1

    # Performance ------------------------------------------------------------
    if m.exec_time > 0:
        reward += 1.0 / m.exec_time

    # Security ---------------------------------------------------------------
    if m.security_issues > 0:
        reward -= 20 * m.security_issues

    # Style ------------------------------------------------------------------
    reward -= 5 * m.style_violations

    return reward


# -----------------------------------------------------------------------------
# 4. PPO Loop — minimal, custom implementation
# -----------------------------------------------------------------------------
class PPOTrainer:
    """Simple PPO implementation specialised for code generation.

    This trainer treats GraphformicCoder (or any auto-regressive model) as the
    policy network. Actions are token logits used to sample the next token. We
    assume an `encode_problem()` function converts the problem description into
    source tokens, and a `decode_solution()` converts generated tokens into
    Python code. These utilities are domain-specific and therefore passed in as
    callables during initialisation.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        lr: float = 5e-5,
        clip_param: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.clip_param = clip_param
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        # Lightweight value head --------------------------------------------
        self.value_head = nn.Linear(model.decoder.out_proj.in_features, 1).to(next(model.parameters()).device)
        self.optimizer.add_param_group({"params": self.value_head.parameters()})

    # ---------------------------------------------------------------------
    def generate_code(self, problem_desc: str, max_len: int = 256) -> Tuple[str, torch.Tensor]:
        """Greedy sampling for demonstration (swap in nucleus sampling etc.)."""
        device = next(self.model.parameters()).device
        src_tokens = self.tokenizer.encode(problem_desc, return_tensors="pt").to(device)
        pad_mask = src_tokens == self.tokenizer.pad_token_id
        # Dummy AST placeholders – integration needed -----------------------
        ast_nodes = torch.zeros(1, 64, device=device)
        ast_edges = None
        tgt = torch.tensor([[self.tokenizer.bos_token_id]], device=device)
        generated_tokens: List[int] = []
        logits_history: List[torch.Tensor] = []
        for _ in range(max_len):
            logits = self.model(src_tokens, pad_mask, ast_nodes, ast_edges, tgt)[:, -1, :]
            logits_history.append(logits)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated_tokens.append(next_token.item())
            tgt = torch.cat([tgt, next_token], dim=1)
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        code = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        logit_tensor = torch.stack(logits_history, dim=1)  # (B=1, T, V)
        return code, logit_tensor.squeeze(0)  # remove batch dim

    # ---------------------------------------------------------------------
    def update_policy(
        self,
        logit_tensor: torch.Tensor,
        actions: torch.Tensor,
        rewards: float,
    ) -> None:
        """One-step PPO update (simplified, no batching)."""
        # Convert to log-probs ------------------------------------------------
        log_probs = torch.log_softmax(logit_tensor, dim=-1)
        action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)  # (T,)
        returns = torch.tensor([rewards], dtype=torch.float32, device=logit_tensor.device)
        values = self.value_head(logit_tensor.detach()).squeeze(-1).mean()  # scalar

        # Advantage ----------------------------------------------------------
        advantage = returns - values.detach()

        # Policy loss --------------------------------------------------------
        policy_loss = -(advantage * action_log_probs.mean())

        # Value loss ---------------------------------------------------------
        value_loss = self.value_coef * advantage.pow(2)

        loss = policy_loss + value_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()


# -----------------------------------------------------------------------------
# 5. DigitalCrucible Entrypoint
# -----------------------------------------------------------------------------
class DigitalCrucible:
    """High-level orchestrator combining all components."""

    def __init__(
        self,
        problems_dir: Path,
        model: nn.Module,
        tokenizer: Any,
        episodes: int = 1_000,
        device: str = "cpu",
    ) -> None:
        self.env = ProblemEnvironment(problems_dir)
        self.sandbox = ExecutionSandbox()
        self.episodes = episodes
        self.model = model.to(device)
        self.trainer = PPOTrainer(model, tokenizer)
        self.tokenizer = tokenizer

    # ---------------------------------------------------------------------
    def run(self) -> None:
        for episode in range(1, self.episodes + 1):
            problem = self.env.sample_problem()
            code, logits = self.trainer.generate_code(problem["description"])
            metrics = self.sandbox.evaluate(code, problem)
            reward = calculate_reward(metrics)

            # Map logits to actions (argmax indices) ------------------------
            actions = torch.argmax(logits, dim=-1)
            self.trainer.update_policy(logits, actions, reward)

            print(
                f"Episode {episode:04d} | Reward {reward:+.2f} | "
                f"Passed={metrics.passed} | Cov={metrics.coverage:.2%} | "
                f"Time={metrics.exec_time:.2f}s | Mem={metrics.memory_mb:.0f}MB | "
                f"Sec={metrics.security_issues} | Style={metrics.style_violations}"
            )


# -----------------------------------------------------------------------------
# __main__ demo (smoke-test) ---------------------------------------------------
if __name__ == "__main__":
    # NOTE: This demo relies on HuggingFace Tokenizers for convenience.
    try:
        from transformers import AutoTokenizer  # type: ignore
    except ImportError:
        print("Please install transformers: pip install transformers", file=sys.stderr)
        sys.exit(1)

    from graphformic_coder import GraphformicCoder

    # Minimal settings -------------------------------------------------------
    vocab_size = 32_000
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # placeholder GPT-2 tokenizer
    model = GraphformicCoder(vocab_size=vocab_size, ast_feat_dim=64)

    # Path to problems directory – ensure this exists with at least one problem
    problems_path = Path("./problems")
    problems_path.mkdir(exist_ok=True)

    # Create a dummy problem for smoke-test ---------------------------------
    dummy_dir = problems_path / "sum_numbers"
    dummy_dir.mkdir(exist_ok=True)
    (dummy_dir / "description.md").write_text("""## Sum Numbers\nWrite a function `solve(n)` that returns the sum of numbers from 1 to n.""")
    (dummy_dir / "tests.py").write_text(
        """import solution, pytest\n\n\n@pytest.mark.parametrize('n, expected', [(1, 1), (5, 15), (10, 55)])\ndef test_sum(n, expected):\n    assert solution.solve(n) == expected\n"""
    )

    crucible = DigitalCrucible(problems_dir=problems_path, model=model, tokenizer=tokenizer, episodes=3)
    crucible.run()
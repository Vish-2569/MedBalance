"""QpiAI Opt integration: submits QUBO problems to the QpiAI platform
or solves locally, decodes results, and logs evidence.

SDK interface (qpiai-opt 0.0.9):
  - QUBOProblem: validated problem spec (qubo_dict uses int-indexed keys)
  - QUBOSolver: platform client (submit → poll → result)
  - LocalProvider: free local solver (simulated annealing)
  - SolveResult: .objective, .solution, .variable_assignment, .solver_time_s
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from .qubo_builder import compute_objective_value

logger = logging.getLogger(__name__)


@dataclass
class OptResult:
    """Standardised result from any QUBO solver."""
    best: dict[str, int]
    top_k: list[dict[str, int]]
    objective_value: float
    wall_clock_s: float
    raw_response: dict = field(default_factory=dict)


def _load_api_key() -> str | None:
    load_dotenv()
    key = os.environ.get("QPIAI_API_KEY")
    if not key or key == "PASTE_YOUR_KEY_HERE":
        return None
    return key


class MedBalanceOptSolver:
    """QUBO solver backed by QpiAI Opt — platform or local."""

    def __init__(self, api_key: str | None = None, evidence_dir: str | None = None):
        self._api_key = api_key or _load_api_key()
        self._evidence_dir = Path(evidence_dir or "evidence/opt_runs")
        self._evidence_dir.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._local = None

    def _ensure_client(self):
        if self._client is not None:
            return
        if not self._api_key:
            raise RuntimeError(
                "No QPIAI_API_KEY found. Set it in .env or pass api_key=."
            )
        from qpiai_opt.opt_solvers import Client
        self._client = Client(api_key=self._api_key)

    def _ensure_local(self):
        if self._local is not None:
            return
        self._ensure_client()
        from qpiai_opt.opt_solvers.local import LocalProvider
        self._local = LocalProvider(client=self._client)

    # ------------------------------------------------------------------
    # Core solve
    # ------------------------------------------------------------------

    def solve(
        self,
        Q: dict[tuple[str, str], float],
        variable_names: list[str],
        num_reads: int = 500,
        timeout_s: float = 90.0,
        use_platform: bool = True,
        top_k: int = 5,
    ) -> OptResult | None:
        """Solve a QUBO and return the best bitstring plus top-K alternatives.

        Tries the QpiAI platform API first when use_platform=True.
        Falls back to local simulated annealing if the platform call fails.
        Runs multiple local SA passes to collect diverse top-K solutions.
        Logs which path was taken.

        Args:
            Q: Q-matrix keyed by (var_id_i, var_id_j).
            variable_names: ordered list of variable names.
            num_reads: number of SA sweeps / reads.
            timeout_s: max wall-clock seconds.
            use_platform: if True, try QpiAI cloud first then fall back to local.
            top_k: number of diverse solutions to collect for the regret ledger.

        Returns OptResult or None on failure (caller falls back to greedy).
        """
        try:
            name_to_idx = {name: i for i, name in enumerate(variable_names)}
            n = len(variable_names)
            qubo_int = self._convert_q(Q, name_to_idx)

            problem = self._build_problem(qubo_int, n, variable_names, num_reads)

            t0 = time.perf_counter()
            mode_used = "local"

            if use_platform:
                try:
                    sdk_result = self._solve_platform(problem, timeout_s)
                    mode_used = "platform"
                    logger.info("QpiAI platform API succeeded")
                except Exception as e:
                    logger.warning(
                        "QpiAI platform API failed (%s), falling back to local solver", e
                    )
                    sdk_result = self._solve_local(problem, num_reads)
                    mode_used = "local (platform fallback)"
            else:
                sdk_result = self._solve_local(problem, num_reads)

            best_bits = self._decode_sdk_result(sdk_result, variable_names)
            best_h = compute_objective_value(best_bits, Q)

            all_solutions = self._collect_top_k(
                Q, problem, variable_names, best_bits, best_h,
                num_reads, top_k, mode_used,
            )

            wall = time.perf_counter() - t0

            result = OptResult(
                best=best_bits,
                top_k=all_solutions,
                objective_value=best_h,
                wall_clock_s=wall,
                raw_response=self._serialize_sdk_result(sdk_result),
            )

            self._log_evidence(Q, variable_names, result)
            logger.info(
                "Solver done: H=%.4f, wall=%.2fs, mode=%s, top_k=%d solutions",
                best_h, wall, mode_used, len(all_solutions),
            )
            return result

        except Exception as e:
            logger.error("Solver failed: %s", e, exc_info=True)
            return None

    def _collect_top_k(
        self,
        Q: dict[tuple[str, str], float],
        problem,
        variable_names: list[str],
        best_bits: dict[str, int],
        best_h: float,
        num_reads: int,
        top_k: int,
        mode_used: str,
    ) -> list[dict[str, int]]:
        """Collect up to top_k diverse solutions by running additional local SA passes."""
        seen: set[frozenset[tuple[str, int]]] = set()
        ranked: list[tuple[float, dict[str, int]]] = []

        def _add(bits: dict[str, int]):
            key = frozenset(bits.items())
            if key in seen:
                return
            seen.add(key)
            h = compute_objective_value(bits, Q)
            ranked.append((h, bits))

        _add(best_bits)

        extra_passes = min(top_k * 2, 10)
        reads_per_pass = max(50, num_reads // 4)
        for _ in range(extra_passes):
            if len(ranked) >= top_k:
                break
            try:
                extra_result = self._solve_local(problem, reads_per_pass)
                extra_bits = self._decode_sdk_result(extra_result, variable_names)
                _add(extra_bits)
            except Exception:
                break

        ranked.sort(key=lambda pair: pair[0])
        return [bits for _, bits in ranked[:top_k]]

    # ------------------------------------------------------------------
    # Q-matrix conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_q(
        Q: dict[tuple[str, str], float],
        name_to_idx: dict[str, int],
    ) -> dict[tuple[int, int], float]:
        """Convert string-keyed Q to int-indexed for the SDK."""
        out: dict[tuple[int, int], float] = {}
        for (ni, nj), val in Q.items():
            if abs(val) < 1e-15:
                continue
            i, j = name_to_idx[ni], name_to_idx[nj]
            key = (min(i, j), max(i, j))
            out[key] = out.get(key, 0.0) + val
        return out

    @staticmethod
    def _build_problem(qubo_int, n, variable_names, max_evals):
        from qpiai_opt.opt_solvers.qubo import QUBOProblem
        return QUBOProblem(
            num_variables=n,
            qubo_dict=qubo_int,
            matrix_format="upper",
            sense="MIN",
            variable_names=tuple(variable_names),
            max_evaluations=max_evals,
        )

    # ------------------------------------------------------------------
    # Solve backends
    # ------------------------------------------------------------------

    def _solve_local(self, problem, num_reads):
        self._ensure_local()
        return self._local.solve(
            problem,
            problem_class="QUBO",
            num_reads=num_reads,
        )

    def _solve_platform(self, problem, timeout_s):
        self._ensure_client()
        from qpiai_opt.opt_solvers.qubo import QUBOSolver
        solver = QUBOSolver(client=self._client)
        return solver.solve(problem, timeout=timeout_s)

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_sdk_result(
        sdk_result,
        variable_names: list[str],
    ) -> dict[str, int]:
        """Convert SDK SolveResult to {var_name: 0|1} bitstring."""
        if sdk_result.variable_assignment:
            return {k: int(v) for k, v in sdk_result.variable_assignment.items()}
        solution = sdk_result.solution
        return {name: int(solution[i]) for i, name in enumerate(variable_names)}

    @staticmethod
    def _serialize_sdk_result(sdk_result) -> dict:
        try:
            return sdk_result.model_dump()
        except Exception:
            return {
                "objective": sdk_result.objective,
                "solution": sdk_result.solution,
                "solver_time_s": sdk_result.solver_time_s,
            }

    # ------------------------------------------------------------------
    # Evidence logging
    # ------------------------------------------------------------------

    def _log_evidence(self, Q, variable_names, result: OptResult):
        ts = time.strftime("%Y%m%d_%H%M%S")
        stem = self._evidence_dir / ts

        q_path = stem.with_suffix(".q.json")
        q_serializable = {f"{k[0]}|{k[1]}": v for k, v in Q.items()}
        q_path.write_text(json.dumps({
            "variable_names": variable_names,
            "Q": q_serializable,
        }, indent=2))

        res_path = stem.with_suffix(".result.json")
        res_path.write_text(json.dumps({
            "best": result.best,
            "objective_value": result.objective_value,
            "wall_clock_s": result.wall_clock_s,
            "raw_response": result.raw_response,
        }, indent=2, default=str))

        logger.info("Evidence saved: %s", stem)

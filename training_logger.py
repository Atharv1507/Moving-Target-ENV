"""Structured training logger for the Fintech GRPO agent.

Logs to:
  - stdout (always)
  - logs/training_YYYYMMDD_HHMMSS.csv (one file per process start)

Three log scopes:
  log_episode()  — called once per rollout episode
  log_step()     — called once per GRPO gradient step
  log_cycle()    — called once per collect→train cycle; also prints console table
"""

import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


# ── setup ─────────────────────────────────────────────────────────────────────

_LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
_SESSION_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
_CSV_PATH = _LOG_DIR / f"training_{_SESSION_TS}.csv"

_EPISODE_HEADERS = [
    "session", "cycle", "episode", "total_reward", "steps_taken",
    "tools_called", "did_reach_execute", "constraint_satisfied", "termination_reason",
]
_STEP_HEADERS = [
    "session", "cycle", "grpo_step", "loss", "grad_norm",
    "entropy", "learning_rate", "reward_mean", "reward_std",
]
_CYCLE_HEADERS = [
    "session", "cycle", "avg_reward", "success_rate", "invalid_tool_rate",
    "entropy", "grad_norm_avg", "reward_std", "frac_zero_std",
]

_ALL_HEADERS = sorted(set(_EPISODE_HEADERS + _STEP_HEADERS + _CYCLE_HEADERS))

_writer: csv.DictWriter | None = None
_csv_file = None


def _get_writer() -> csv.DictWriter:
    global _writer, _csv_file
    if _writer is not None:
        return _writer
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    _csv_file = open(_CSV_PATH, "w", newline="", encoding="utf-8")
    _writer = csv.DictWriter(_csv_file, fieldnames=_ALL_HEADERS, extrasaction="ignore")
    _writer.writeheader()
    _csv_file.flush()
    print(f"[LOGGER] Writing training log to {_CSV_PATH}", flush=True)
    return _writer


def _write_row(row: dict) -> None:
    row.setdefault("session", _SESSION_TS)
    try:
        w = _get_writer()
        w.writerow(row)
        _csv_file.flush()
    except Exception as e:
        print(f"[LOGGER] CSV write error: {e}", flush=True)


# ── public API ────────────────────────────────────────────────────────────────

def log_episode(
    *,
    cycle: int,
    episode: int,
    total_reward: float,
    steps_taken: int,
    tools_called: list[str],
    did_reach_execute: bool,
    constraint_satisfied: bool,
    termination_reason: str,
) -> None:
    """Log one completed rollout episode."""
    row = {
        "cycle":               cycle,
        "episode":             episode,
        "total_reward":        round(total_reward, 4),
        "steps_taken":         steps_taken,
        "tools_called":        "|".join(tools_called),
        "did_reach_execute":   int(did_reach_execute),
        "constraint_satisfied": int(constraint_satisfied),
        "termination_reason":  termination_reason,
    }
    _write_row(row)
    print(
        f"[EP {episode:>3}] reward={total_reward:+.1f}  steps={steps_taken}"
        f"  reach_exec={did_reach_execute}  satisfied={constraint_satisfied}"
        f"  end={termination_reason}",
        flush=True,
    )


def log_step(
    *,
    cycle: int,
    grpo_step: int,
    loss: float,
    grad_norm: float,
    entropy: float,
    learning_rate: float,
    reward_mean: float,
    reward_std: float,
) -> None:
    """Log one GRPO gradient step."""
    row = {
        "cycle":        cycle,
        "grpo_step":    grpo_step,
        "loss":         round(loss, 6),
        "grad_norm":    round(grad_norm, 6),
        "entropy":      round(entropy, 6),
        "learning_rate": learning_rate,
        "reward_mean":  round(reward_mean, 4),
        "reward_std":   round(reward_std, 4),
    }
    _write_row(row)
    print(
        f"[STEP {grpo_step:>3}] loss={loss:.4f}  grad={grad_norm:.4f}"
        f"  H={entropy:.4f}  lr={learning_rate:.2e}"
        f"  r_mean={reward_mean:+.2f}  r_std={reward_std:.2f}",
        flush=True,
    )


def log_cycle(
    *,
    cycle: int,
    avg_reward: float,
    success_rate: float,
    invalid_tool_rate: float,
    entropy: float = 0.0,
    grad_norm_avg: float = 0.0,
    reward_std: float = 0.0,
    frac_zero_std: float = 0.0,
) -> None:
    """Log cycle-level aggregates and print the console summary table."""
    row = {
        "cycle":             cycle,
        "avg_reward":        round(avg_reward, 4),
        "success_rate":      round(success_rate, 4),
        "invalid_tool_rate": round(invalid_tool_rate, 4),
        "entropy":           round(entropy, 6),
        "grad_norm_avg":     round(grad_norm_avg, 6),
        "reward_std":        round(reward_std, 4),
        "frac_zero_std":     round(frac_zero_std, 4),
    }
    _write_row(row)

    # ── console summary table ─────────────────────────────────────────────────
    lines = [
        f"╔══════════════════════════════════════╗",
        f"║  Cycle {cycle:<2} Summary                     ║",
        f"║  avg_reward      : {avg_reward:>+8.2f}           ║",
        f"║  success_rate    : {success_rate * 100:>7.2f}%           ║",
        f"║  invalid_tool    : {invalid_tool_rate * 100:>7.2f}%           ║",
        f"║  entropy         : {entropy:>8.4f}           ║",
        f"║  grad_norm (avg) : {grad_norm_avg:>8.4f}           ║",
        f"║  reward_std      : {reward_std:>8.4f}           ║",
        f"║  frac_zero_std   : {frac_zero_std * 100:>7.2f}%           ║",
        f"╚══════════════════════════════════════╝",
    ]
    print("\n" + "\n".join(lines) + "\n", flush=True)

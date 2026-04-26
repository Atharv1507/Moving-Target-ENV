"""HuggingFace Space entry point — runs the environment server + GRPO training loop.


Environment variables:
  BASE_MODEL            HuggingFace model id (default: unsloth/Qwen2.5-1.5B-Instruct)
  OPENROUTER_API_KEY    Enables realistic persona requests during rollouts (optional)
  ROLLOUT_VERBOSE       Set to 1 to print every rollout step (noisy but useful)
  HF_LOG_TRAINING       Set to 1 to enable verbose Transformers/TRL logs
  EPISODES_PER_ROLLOUT  Episodes per training cycle (default: 5)
  TRAINING_CYCLES       Number of collect → train cycles (default: 3)
"""
# Import unsloth FIRST — must precede trl/transformers to apply patches.
# Guarded because unsloth_zoo may be incompatible with the installed TRL version.
try:
    import unsloth  # noqa: F401
except (ImportError, Exception):
    pass

import atexit
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Optional

import requests

from grpo_trainer import train_with_grpo
from model_loader import get_model_and_tokenizer
from rollout_collector import collect_rollouts


# ── stdout capture for live log viewer ────────────────────────────────────────

import collections
import io

_LOG_LINES: collections.deque = collections.deque(maxlen=2000)


class _TeeWriter:
    """Write to the original stream AND capture lines for the UI log viewer.

    Delegates all stream methods (fileno, isatty, etc.) to the original so
    subprocess, uvicorn, and other libs that need a real fd keep working.
    """

    def __init__(self, original) -> None:
        self._original = original

    def write(self, s: str) -> int:
        if s and s.strip():
            for line in s.splitlines():
                _LOG_LINES.append(line)
        return self._original.write(s)

    def flush(self) -> None:
        self._original.flush()

    def fileno(self) -> int:
        return self._original.fileno()

    def isatty(self) -> bool:
        return self._original.isatty()

    @property
    def encoding(self):
        return getattr(self._original, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self._original, "errors", "strict")

    def __getattr__(self, name):
        # Delegate anything else (readable, writable, seekable, etc.)
        return getattr(self._original, name)


sys.stdout = _TeeWriter(sys.__stdout__)
sys.stderr = _TeeWriter(sys.__stderr__)


ENV_SERVER_PORT = int(os.getenv("ENV_SERVER_PORT", "8001"))
SERVER_URL = f"http://localhost:{ENV_SERVER_PORT}/"
DEFAULT_EPISODES_PER_ROLLOUT = int(os.getenv("EPISODES_PER_ROLLOUT", "8"))
DEFAULT_TRAINING_CYCLES = int(os.getenv("TRAINING_CYCLES", "3"))
# Space/OpenEnv route this app on port 8000 by default.
# Keep it overridable, but do not default to 7860 in deployment.
DEFAULT_UI_PORT = int(os.getenv("UI_PORT", "8000"))

_SERVER_PROCESS: Optional[subprocess.Popen] = None
_RUN_LOCK = threading.Lock()


def _resolve_output_dir() -> str:
    """Pick an output directory that survives restarts on HF Spaces when possible."""
    env_out = os.getenv("OUTPUT_DIR")
    if env_out:
        return env_out
    # On Spaces with persistent storage, /data is the durable mount.
    if os.path.isdir("/data"):
        return "/data/grpo-output"
    return "grpo-output"


# ── server helpers ────────────────────────────────────────────────────────────

def _start_env_server() -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app",
         "--host", "0.0.0.0", "--port", str(ENV_SERVER_PORT)],
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=os.environ.copy(),
    )


def _wait_for_server(base_url: str, timeout: int = 90) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            if requests.get(base_url, timeout=3).status_code < 400:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError(f"Environment server did not become healthy at {base_url} within {timeout}s.")


def _terminate(process: Optional[subprocess.Popen]) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _ensure_env_server_running() -> None:
    """Start environment server once and keep it alive for UI-triggered runs."""
    global _SERVER_PROCESS
    if _SERVER_PROCESS is not None and _SERVER_PROCESS.poll() is None:
        return
    print("[SYSTEM] Starting environment server...", flush=True)
    _SERVER_PROCESS = _start_env_server()
    atexit.register(_terminate, _SERVER_PROCESS)
    _wait_for_server(SERVER_URL)
    print("[SYSTEM] Environment server is healthy.", flush=True)


# ── optional verbose logging ──────────────────────────────────────────────────

def _configure_training_logs() -> None:
    if (os.getenv("HF_LOG_TRAINING") or "").lower() not in ("1", "true", "yes"):
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    for lib in ("transformers", "trl", "accelerate", "torch"):
        logging.getLogger(lib).setLevel(logging.INFO)
    print("[SYSTEM] HF_LOG_TRAINING=1: verbose library logs enabled.", flush=True)


def _append_cycle_metrics(output_dir: str, cycle_idx: int, rollout_buffer: list[dict]) -> dict:
    """Append cycle metrics to output_dir/metrics.jsonl and return the metrics dict."""
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.jsonl")

    rewards = [float(item.get("reward", 0.0)) for item in rollout_buffer]
    severe_penalties = sum(1 for r in rewards if r <= -50.0)
    invalid_tool_calls = sum(
        1 for item in rollout_buffer if str(item.get("tool", "")).startswith("__")
    )
    episodes = sorted({int(item.get("episode", 0)) for item in rollout_buffer if item.get("episode")})
    episode_totals = {}
    success_episodes = set()
    for item in rollout_buffer:
        ep = int(item.get("episode", 0))
        if ep <= 0:
            continue
        episode_totals[ep] = episode_totals.get(ep, 0.0) + float(item.get("reward", 0.0))
        if bool(item.get("done", False)) and float(item.get("reward", 0.0)) > 0:
            success_episodes.add(ep)

    metrics = {
        "timestamp": int(time.time()),
        "cycle": cycle_idx + 1,
        "samples": len(rollout_buffer),
        "avg_step_reward": (sum(rewards) / len(rewards)) if rewards else 0.0,
        "avg_episode_reward": (
            sum(episode_totals.values()) / len(episode_totals) if episode_totals else 0.0
        ),
        "success_rate": (len(success_episodes) / len(episodes)) if episodes else 0.0,
        "severe_penalty_count": severe_penalties,
        "invalid_tool_rate": (invalid_tool_calls / len(rollout_buffer)) if rollout_buffer else 0.0,
    }
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics) + "\n")

    print(
        "[METRICS] "
        f"cycle={metrics['cycle']} avg_episode_reward={metrics['avg_episode_reward']:.2f} "
        f"success_rate={metrics['success_rate']:.2%} "
        f"severe_penalties={metrics['severe_penalty_count']} "
        f"invalid_tool_rate={metrics['invalid_tool_rate']:.2%}",
        flush=True,
    )
    return metrics


# ── training loop ─────────────────────────────────────────────────────────────

MIN_EPISODES_PER_ROLLOUT = 30


def run_training_loop(
    cycles: int = DEFAULT_TRAINING_CYCLES,
    episodes_per_rollout: int = DEFAULT_EPISODES_PER_ROLLOUT,
) -> None:
    # Enforce minimum episodes per rollout — silently clamp
    if episodes_per_rollout < MIN_EPISODES_PER_ROLLOUT:
        episodes_per_rollout = MIN_EPISODES_PER_ROLLOUT

    output_dir = _resolve_output_dir()
    resume_path = os.path.join(output_dir, "final-adapter")
    # Let model_loader auto-resume if adapter exists from previous runs.
    os.environ["RESUME_ADAPTER_PATH"] = resume_path

    # Load the model once — rollout_collector and grpo_trainer both share this cache
    get_model_and_tokenizer()

    for cycle in range(cycles):
        cycle_num = cycle + 1
        print(f"\n[TRAINING] ── Cycle {cycle_num}/{cycles}: collecting rollouts ──", flush=True)
        rollout_buffer = collect_rollouts(
            episodes=episodes_per_rollout,
            server_base_url=SERVER_URL,
            cycle=cycle_num,
        )
        for i, item in enumerate(rollout_buffer, 1):
            print(f"[TRAINING]   sample {i}/{len(rollout_buffer)} reward={item['reward']:.1f}", flush=True)

        cycle_metrics = _append_cycle_metrics(output_dir, cycle, rollout_buffer)

        n = len(rollout_buffer)
        grpo_steps = min(max(8, 2 * n), 40)
        print(f"\n[TRAINING] ── Cycle {cycle_num} GRPO update: {n} samples, {grpo_steps} steps ──", flush=True)

        train_with_grpo(
            rollout_buffer=rollout_buffer,
            output_dir=output_dir,
            max_steps=grpo_steps,
            cycle=cycle_num,
            cycle_metrics=cycle_metrics,
        )
        print(f"[TRAINING] Cycle {cycle_num} complete.", flush=True)


def run_one_cycle(episodes_per_rollout: int = DEFAULT_EPISODES_PER_ROLLOUT) -> str:
    """Run exactly one collection+GRPO cycle and return a compact status string."""
    if not _RUN_LOCK.acquire(blocking=False):
        return "A training run is already in progress. Please wait for it to finish."
    try:
        _ensure_env_server_running()
        print(
            f"[UI] Triggered one-cycle run with episodes_per_rollout={episodes_per_rollout}",
            flush=True,
        )
        run_training_loop(cycles=1, episodes_per_rollout=episodes_per_rollout)
        output_dir = _resolve_output_dir()
        metrics_path = os.path.join(output_dir, "metrics.jsonl")
        latest_metrics = ""
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
                if lines:
                    latest_metrics = lines[-1]
        if latest_metrics:
            return f"One-cycle run complete.\nLatest metrics: {latest_metrics}"
        return "One-cycle run complete. No metrics found yet."
    except Exception as e:
        print(f"[UI] One-cycle run failed: {e}", flush=True)
        return f"One-cycle run failed: {e}"
    finally:
        _RUN_LOCK.release()


def _read_live_log(tail: int = 200) -> str:
    """Read the last `tail` lines from the captured log buffer."""
    return "\n".join(_LOG_LINES[-tail:]) if _LOG_LINES else "(no logs yet)"


def _read_metrics() -> str:
    """Read metrics.jsonl and format as a readable table."""
    output_dir = _resolve_output_dir()
    metrics_path = os.path.join(output_dir, "metrics.jsonl")
    if not os.path.exists(metrics_path):
        return "(no metrics yet — run a training cycle first)"
    lines = []
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                m = json.loads(ln)
                lines.append(
                    f"Cycle {m.get('cycle', '?'):>2} │ "
                    f"reward={m.get('avg_episode_reward', 0):.2f}  "
                    f"success={m.get('success_rate', 0):.1%}  "
                    f"invalid_tool={m.get('invalid_tool_rate', 0):.1%}  "
                    f"samples={m.get('samples', 0)}  "
                    f"severe_penalties={m.get('severe_penalty_count', 0)}"
                )
    except Exception as e:
        return f"Error reading metrics: {e}"
    return "\n".join(lines) if lines else "(no metrics yet)"


def _read_training_csv() -> str:
    """Read the latest training CSV and return its contents."""
    log_dir = os.getenv("LOG_DIR", "logs")
    if not os.path.isdir(log_dir):
        return "(no training CSV logs found)"
    csvs = sorted(
        [f for f in os.listdir(log_dir) if f.startswith("training_") and f.endswith(".csv")],
        reverse=True,
    )
    if not csvs:
        return "(no training CSV logs found)"
    csv_path = os.path.join(log_dir, csvs[0])
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            content = f.read()
        lines = content.strip().split("\n")
        if len(lines) > 201:
            return lines[0] + "\n... (showing last 200 rows) ...\n" + "\n".join(lines[-200:])
        return content
    except Exception as e:
        return f"Error reading CSV: {e}"


def launch_gradio_ui() -> None:
    """Launch the GRPO control panel with live logs for hackathon judges."""
    import gradio as gr

    with gr.Blocks(title="Fintech Transaction Agent — GRPO Runner") as demo:
        gr.Markdown("# 🏦 Fintech Transaction Agent — GRPO Runner")
        gr.Markdown(
            "Train a 1.5B-param LLM to autonomously execute fintech payments "
            "via shifting provider APIs using GRPO reinforcement learning."
        )

        episodes = gr.Slider(
            minimum=30,
            maximum=1000,
            value=max(DEFAULT_EPISODES_PER_ROLLOUT, 30),
            step=10,
            label="Episodes per rollout (min 30)",
        )
        run_btn = gr.Button("Run 1 Cycle", variant="primary")
        output = gr.Textbox(label="Run status", lines=2)

        gr.Markdown("### Live Application Logs")
        log_box = gr.Textbox(
            label="stdout/stderr",
            lines=25,
            max_lines=30,
            value=_read_live_log,
            interactive=False,
        )

        run_btn.click(
            fn=run_one_cycle,
            inputs=[episodes],
            outputs=[output],
            queue=True,
        )

        # Auto-refresh the log box every 1 second
        demo.load(
            fn=_read_live_log,
            inputs=None,
            outputs=[log_box],
            every=1,
        )

    print(f"[SYSTEM] Launching Gradio UI on 0.0.0.0:{DEFAULT_UI_PORT}", flush=True)
    demo.queue().launch(server_name="0.0.0.0", server_port=DEFAULT_UI_PORT)


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    _configure_training_logs()
    use_ui = (os.getenv("USE_GRADIO_UI", "1").lower() in ("1", "true", "yes", "on"))
    if use_ui:
        launch_gradio_ui()
        return

    try:
        _ensure_env_server_running()
        if (os.getenv("ROLLOUT_VERBOSE") or "").lower() in ("1", "true", "yes"):
            print("[SYSTEM] ROLLOUT_VERBOSE=1: printing per-step rollout traces.", flush=True)
        run_training_loop()
    finally:
        _terminate(_SERVER_PROCESS)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    main()

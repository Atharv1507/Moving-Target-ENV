"""HuggingFace Space entry point — runs the environment server + GRPO training loop.


Environment variables:
  BASE_MODEL            HuggingFace model id (default: unsloth/Qwen2.5-1.5B-Instruct)
  OPENROUTER_API_KEY    Enables realistic persona requests during rollouts (optional)
  ROLLOUT_VERBOSE       Set to 1 to print every rollout step (noisy but useful)
  HF_LOG_TRAINING       Set to 1 to enable verbose Transformers/TRL logs
  EPISODES_PER_ROLLOUT  Episodes per training cycle (default: 5)
  TRAINING_CYCLES       Number of collect → train cycles (default: 3)
"""
# Import unsloth FIRST before any trl/transformers to apply patches correctly
try:
    import unsloth  # noqa: F401
except ImportError:
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


ENV_SERVER_PORT = int(os.getenv("ENV_SERVER_PORT", "8001"))
SERVER_URL = f"http://localhost:{ENV_SERVER_PORT}/"
DEFAULT_EPISODES_PER_ROLLOUT = int(os.getenv("EPISODES_PER_ROLLOUT", "5"))
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


def _append_cycle_metrics(output_dir: str, cycle_idx: int, rollout_buffer: list[dict]) -> None:
    """Append cycle metrics to output_dir/metrics.jsonl."""
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


# ── training loop ─────────────────────────────────────────────────────────────

def run_training_loop(
    cycles: int = DEFAULT_TRAINING_CYCLES,
    episodes_per_rollout: int = DEFAULT_EPISODES_PER_ROLLOUT,
) -> None:
    output_dir = _resolve_output_dir()
    resume_path = os.path.join(output_dir, "final-adapter")
    # Let model_loader auto-resume if adapter exists from previous runs.
    os.environ["RESUME_ADAPTER_PATH"] = resume_path

    # Load the model once — rollout_collector and grpo_trainer both share this cache
    get_model_and_tokenizer()

    # Collect all rollouts first, then do one GRPO pass.
    # Re-instantiating GRPOTrainer per cycle risks OOM on 16 GB from re-building the ref model.
    all_rollouts: list = []

    for cycle in range(cycles):
        print(f"\n[TRAINING] ── Cycle {cycle + 1}/{cycles}: collecting rollouts ──", flush=True)
        rollout_buffer = collect_rollouts(
            episodes=episodes_per_rollout,
            server_base_url=SERVER_URL,
        )
        for i, item in enumerate(rollout_buffer, 1):
            print(f"[TRAINING]   sample {i}/{len(rollout_buffer)} reward={item['reward']:.1f}", flush=True)
        all_rollouts.extend(rollout_buffer)
        _append_cycle_metrics(output_dir, cycle, rollout_buffer)

    n = len(all_rollouts)
    # Conservative step count for T4: enough to learn but won't OOM or time out
    grpo_steps = min(max(8, 2 * n), 40)
    print(f"\n[TRAINING] ── GRPO update: {n} samples, {grpo_steps} steps ──", flush=True)

    train_with_grpo(
        rollout_buffer=all_rollouts,
        output_dir=output_dir,
        max_steps=grpo_steps,
    )
    print("[TRAINING] GRPO update complete.", flush=True)
    print(f"[TRAINING] Final adapter saved to: {output_dir}/final-adapter", flush=True)


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


def launch_gradio_ui() -> None:
    """Launch a minimal control panel to trigger one-cycle GRPO runs."""
    import gradio as gr

    with gr.Blocks(title="Fintech Transaction Agent — GRPO Runner") as demo:
        gr.Markdown("# Fintech Transaction Agent — GRPO Runner")
        gr.Markdown("Train the agent to execute payments/withdrawals via shifting fintech provider APIs. Click **Run 1 Cycle** to collect rollouts and run one GRPO update.")
        episodes = gr.Slider(
            minimum=1,
            maximum=20,
            value=DEFAULT_EPISODES_PER_ROLLOUT,
            step=1,
            label="Episodes per rollout",
        )
        run_btn = gr.Button("Run 1 Cycle", variant="primary")
        output = gr.Textbox(label="Run status", lines=8)

        run_btn.click(
            fn=run_one_cycle,
            inputs=[episodes],
            outputs=[output],
            queue=True,
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

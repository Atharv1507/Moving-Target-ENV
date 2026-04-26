"""GRPO training using TRL + Unsloth."""
# Import unsloth FIRST — must precede trl/transformers to apply patches.
try:
    import unsloth  # noqa: F401
except (ImportError, Exception):
    pass

import json
import math
import os
import shutil
import warnings

import requests
import torch
import torch.nn.functional as F
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback

from model_loader import get_model_and_tokenizer
from training_logger import log_step, log_cycle

# Entropy regularisation coefficient — added directly to reward since
# GRPOConfig.entropy_coef does not exist in any released TRL version.
ENTROPY_COEF = 0.05

SERVER_URL = os.getenv("ENV_SERVER_URL", "http://localhost:8001/")
ALLOWED_TOOLS = {"getProviders", "check_provider", "execute_transaction"}

# Sanity-check test prompts — both must produce valid JSON after a GRPO update.
_SANITY_PROMPTS = [
    'You are a fintech shopping assistant.\nUser: Send $100 via Wise.\nAssistant:',
    'You are a fintech shopping assistant.\nUser: List available providers.\nAssistant:',
]


# ── reward function ───────────────────────────────────────────────────────────

def _parse_tool_call(text: str) -> dict | None:
    """Extract the first balanced JSON object from text."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _extract_reward(obs: dict, response_json: dict) -> float:
    raw = obs.get("reward")
    if raw is None:
        raw = response_json.get("reward")
    return float(raw) if raw is not None else 0.0


def _validate_tool_call(tool_call: dict | None) -> dict | None:
    """Accept only well-formed, known tool calls."""
    if tool_call is None or not isinstance(tool_call, dict):
        return None
    tool = tool_call.get("tool")
    if not isinstance(tool, str) or tool not in ALLOWED_TOOLS:
        return None
    return tool_call


def _completion_entropy_bonus(completion: str) -> float:
    """Compute a small entropy bonus from the token distribution of a completion.

    Uses the model to score the completion and returns ENTROPY_COEF * H(p),
    where H(p) is the mean per-token entropy. Encourages the model to maintain
    action diversity without relying on the non-existent GRPOConfig.entropy_coef.
    Returns 0.0 safely if the model is unavailable or inference fails.
    """
    if ENTROPY_COEF == 0.0 or not completion.strip():
        return 0.0
    try:
        model, tokenizer = get_model_and_tokenizer()
        inputs = tokenizer(completion, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits  # (1, seq_len, vocab)
        probs = F.softmax(logits[0], dim=-1)          # (seq_len, vocab)
        # Per-token entropy: -sum(p * log(p))
        log_probs = torch.log(probs.clamp(min=1e-9))
        entropy = -(probs * log_probs).sum(dim=-1).mean().item()  # scalar
        return ENTROPY_COEF * entropy
    except Exception:
        return 0.0


def _reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    """Score each generated completion by calling the running environment server.

    Reward table (matches env exactly):
      getProviders (first call)      → +3.0  (fixed, stable during training)
      getProviders (repeat)          → 0.0
      check_provider                 → +1.0  (encourage probing before acting)
      execute_transaction success    → +50.0 (from env)
      execute_transaction bad field  → -15.0 (from env)
      constraint violation           → -40.0 (from env)
      unknown / no tool              → 0.0
      execute before check_provider  → -20.0 (safety gate)
      entropy bonus                  → ENTROPY_COEF * H(completion tokens)

    Zero-std groups: if all completions in the batch share the same reward,
    the group contributes no gradient — return uniform rewards so GRPO skips it.
    """
    rewards = []
    check_seen: set[str] = set()
    providers_listed = False

    for completion in completions:
        h_bonus = _completion_entropy_bonus(completion)
        tool_call = _validate_tool_call(_parse_tool_call(completion))
        if tool_call is None:
            rewards.append(0.0 + h_bonus)
            continue

        tool = tool_call.get("tool", "")
        provider = tool_call.get("provider_name", "unknown")

        try:
            if tool == "getProviders":
                base = 3.0 if not providers_listed else 0.0
                providers_listed = True
                rewards.append(base + h_bonus)

            elif tool == "check_provider":
                check_seen.add(provider)
                rewards.append(1.0 + h_bonus)

            elif tool == "execute_transaction":
                if provider not in check_seen:
                    rewards.append(-20.0 + h_bonus)
                    continue

                requests.post(
                    f"{SERVER_URL}step",
                    json={"action": {"tool": "check_provider", "provider_name": provider, "payload": {}}},
                    timeout=10,
                )
                resp = requests.post(
                    f"{SERVER_URL}step",
                    json={"action": {
                        "tool":          "execute_transaction",
                        "provider_name": provider,
                        "payload":       tool_call.get("payload") or {},
                    }},
                    timeout=10,
                )
                payload = resp.json()
                obs = payload.get("observation", {})
                rewards.append(_extract_reward(obs, payload) + h_bonus)

            else:
                rewards.append(-3.0 + h_bonus)

        except Exception as e:
            print(f"[REWARD] Error evaluating completion: {e}", flush=True)
            rewards.append(-5.0)

    # Zero-std guard: return as-is; TRL excludes zero-std groups from loss.
    return rewards


# ── logging callback ──────────────────────────────────────────────────────────

class _StepLoggerCallback(TrainerCallback):
    """Logs per-GRPO-step metrics via training_logger.log_step."""

    def __init__(self, cycle: int) -> None:
        self.cycle = cycle
        self._step_rewards: list[float] = []
        self._step_stds: list[float] = []
        self._grad_norms: list[float] = []
        self._entropies: list[float] = []
        self._zero_std_count = 0
        self._total_groups = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step      = state.global_step
        loss      = float(logs.get("loss", 0.0))
        grad_norm = float(logs.get("grad_norm", 0.0))
        entropy   = float(logs.get("entropy", 0.0))
        lr        = float(logs.get("learning_rate", 0.0))
        r_mean    = float(logs.get("reward_mean", logs.get("rewards/mean", 0.0)))
        r_std     = float(logs.get("reward_std",  logs.get("rewards/std",  0.0)))
        zero_std  = float(logs.get("frac_reward_zero_std", 0.0))

        self._grad_norms.append(grad_norm)
        self._entropies.append(entropy)
        self._step_rewards.append(r_mean)
        self._step_stds.append(r_std)
        if zero_std > 0:
            self._zero_std_count += 1
        self._total_groups += 1

        log_step(
            cycle=self.cycle,
            grpo_step=step,
            loss=loss,
            grad_norm=grad_norm,
            entropy=entropy,
            learning_rate=lr,
            reward_mean=r_mean,
            reward_std=r_std,
        )

    def cycle_summary(self) -> dict:
        def _avg(lst): return sum(lst) / len(lst) if lst else 0.0
        return {
            "grad_norm_avg": _avg(self._grad_norms),
            "entropy":       _avg(self._entropies),
            "reward_std":    _avg(self._step_stds),
            "frac_zero_std": self._zero_std_count / max(self._total_groups, 1),
        }


# ── checkpoint helpers ────────────────────────────────────────────────────────

def _save_checkpoint(model, tokenizer, path: str) -> None:
    """Save adapter weights to *path* (overwrites if it exists)."""
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def _restore_checkpoint(model, tokenizer, path: str) -> None:
    """Reload adapter weights in-place from a previously saved checkpoint."""
    from peft import set_peft_model_state_dict
    import safetensors.torch

    # Load the adapter state dict from the checkpoint directory.
    weight_file = os.path.join(path, "adapter_model.safetensors")
    if os.path.exists(weight_file):
        state_dict = safetensors.torch.load_file(weight_file, device=str(model.device))
    else:
        bin_file = os.path.join(path, "adapter_model.bin")
        state_dict = torch.load(bin_file, map_location=model.device, weights_only=True)

    set_peft_model_state_dict(model, state_dict)
    print(f"[GRPO] Restored adapter weights from {path}", flush=True)


def _sanity_check(model, tokenizer) -> bool:
    """Generate from 2 test prompts — both must parse as valid JSON.

    Returns True if sanity check passes, False otherwise.
    """
    model.eval()
    passed = 0
    for prompt_text in _SANITY_PROMPTS:
        try:
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            parsed = _parse_tool_call(text)
            if parsed is not None and isinstance(parsed, dict):
                passed += 1
                print(f"[SANITY] ✓ Got valid JSON: {text[:80]!r}", flush=True)
            else:
                print(f"[SANITY] ✗ No valid JSON from: {text[:80]!r}", flush=True)
        except Exception as e:
            print(f"[SANITY] ✗ Generation error: {e}", flush=True)

    ok = passed == len(_SANITY_PROMPTS)
    if ok:
        print("[SANITY] All test prompts passed.", flush=True)
    else:
        print(f"[SANITY] FAILED — only {passed}/{len(_SANITY_PROMPTS)} passed.", flush=True)
    return ok


# ── main public function ──────────────────────────────────────────────────────

def train_with_grpo(
    rollout_buffer: list[dict],
    output_dir: str = "grpo-output",
    max_steps: int = 40,
    cycle: int = 1,
    cycle_metrics: dict | None = None,
) -> None:
    """Run GRPO training on data collected from collect_rollouts().

    Args:
        rollout_buffer: List of {"prompt": str, "completion": str, "reward": float}.
        output_dir:     Directory to save LoRA adapter checkpoints.
        max_steps:      Number of GRPO update steps.
        cycle:          Current cycle index (1-based), used for logging.
        cycle_metrics:  Pre-computed rollout metrics dict for the cycle summary.
    """
    model, tokenizer = get_model_and_tokenizer()

    # ── Guard: skip update if all episode rewards are identical (zero variance) ──
    episode_rewards: dict[int, float] = {}
    for item in rollout_buffer:
        ep = int(item.get("episode", 0))
        if ep > 0:
            episode_rewards[ep] = episode_rewards.get(ep, 0.0) + float(item.get("reward", 0.0))

    reward_values = list(episode_rewards.values())
    if len(set(reward_values)) <= 1:
        warnings.warn(
            f"[GRPO] Cycle {cycle}: all {len(reward_values)} episode rewards are identical "
            f"({reward_values[0] if reward_values else 0.0}) — zero variance. "
            f"Skipping GRPO update to avoid degenerate gradient.",
            stacklevel=2,
        )
        print(
            f"[GRPO] Cycle {cycle}: SKIPPED — zero reward variance across episodes.",
            flush=True,
        )
        return

    dataset = Dataset.from_list([{"prompt": r["prompt"]} for r in rollout_buffer])

    is_cuda  = torch.cuda.is_available()
    use_bf16 = is_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = is_cuda and not use_bf16

    config = GRPOConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        # Batch / gradient — T4 16 GB safe
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        # Generation: 8 completions for diversity, higher temp for exploration
        num_generations=8,
        max_completion_length=64,   # tool calls are never longer than this; was 256 causing 19s/step
        temperature=0.9,
        # Optimiser
        learning_rate=1e-5,
        bf16=use_bf16,
        fp16=use_fp16,
        # Log every step
        logging_steps=1,
        report_to="none",
        save_strategy="no",
    )

    # ── Save pre-update checkpoint for rollback ──
    pre_update_path = os.path.join(output_dir, f"_pre-update-cycle-{cycle}")
    _save_checkpoint(model, tokenizer, pre_update_path)
    print(f"[GRPO] Pre-update checkpoint saved to {pre_update_path}", flush=True)

    cb = _StepLoggerCallback(cycle=cycle)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[_reward_fn],
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[cb],
    )

    print(f"[GRPO] Cycle {cycle} — starting: {len(dataset)} prompts, {max_steps} steps.", flush=True)
    trainer.train()

    # ── Post-update sanity check — rollback if model is broken ──
    if not _sanity_check(model, tokenizer):
        print(
            f"[GRPO] Cycle {cycle}: post-update sanity check FAILED — "
            f"rolling back to pre-update checkpoint.",
            flush=True,
        )
        _restore_checkpoint(model, tokenizer, pre_update_path)
        # Verify rollback worked
        if _sanity_check(model, tokenizer):
            print(f"[GRPO] Rollback successful — model restored.", flush=True)
        else:
            print(f"[GRPO] WARNING: rollback sanity check also failed!", flush=True)
        return

    # Per-cycle summary via logger
    summary = cb.cycle_summary()
    m = cycle_metrics or {}
    log_cycle(
        cycle=cycle,
        avg_reward=m.get("avg_episode_reward", 0.0),
        success_rate=m.get("success_rate", 0.0),
        invalid_tool_rate=m.get("invalid_tool_rate", 0.0),
        entropy=summary["entropy"],
        grad_norm_avg=summary["grad_norm_avg"],
        reward_std=summary["reward_std"],
        frac_zero_std=summary["frac_zero_std"],
    )

    # Save adapter checkpoint for this cycle
    cycle_adapter_path = os.path.join(output_dir, f"adapter-cycle-{cycle}")
    model.save_pretrained(cycle_adapter_path)
    tokenizer.save_pretrained(cycle_adapter_path)
    print(f"[GRPO] Cycle {cycle} adapter saved to {cycle_adapter_path}", flush=True)

    # Always overwrite the canonical final-adapter too
    final_path = os.path.join(output_dir, "final-adapter")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"[GRPO] Final adapter updated at {final_path}", flush=True)

    # Clean up pre-update checkpoint on success
    shutil.rmtree(pre_update_path, ignore_errors=True)

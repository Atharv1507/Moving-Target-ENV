"""Runs episodes with the local model as a fintech payment agent and collects training data."""
import unsloth  # noqa: F401 — must be first import

import json
import os
import re
import time

import requests
import torch

from model_loader import get_model_and_tokenizer
from training_logger import log_episode

MAX_STEPS_PER_EPISODE = 8  # cap early — cuts wasted loops, faster cycles

FALLBACK_REQUESTS = [
    "I need to send $200 to India via UPI. The fee must be under 2% and I want same-day settlement.",
    "Withdraw $500 from my Stripe balance to my US bank account. I need instant settlement.",
    "Pay a contractor £300 to a UK account. I need GBP support and low fees please.",
    "Transfer ₹10,000 domestically via IFSC. Budget is tight so keep fees minimal.",
    "Send $1000 internationally to Europe. I only have basic KYC — need a provider that accepts that.",
    "Pay an invoice of $750 USD. I need a reference_note field and I can't pay more than 1% fee.",
    "Move $50 to a PayPal-compatible provider. Currency USD, no strict KYC, instant if possible.",
]

CONCIERGE_SYSTEM_PROMPT = (
    "You are a fintech shopping assistant. You must follow this EXACT sequence:\n"
    "1. Call getProviders to list available providers\n"
    "2. Call check_provider with the user's requested provider name\n"
    "3. If constraints are satisfied, call execute_transaction\n"
    "Never call the same tool twice in a row.\n"
    "Always use the provider name the user mentioned.\n\n"
    "Available tools and their required JSON schemas:\n\n"
    "1. getProviders — list all available payment providers\n"
    '{"tool": "getProviders"}\n\n'
    "2. check_provider — verify a specific provider meets constraints\n"
    '{"tool": "check_provider", "provider_name": "<string>"}\n\n'
    "3. execute_transaction — complete the purchase\n"
    '{"tool": "execute_transaction", "provider_name": "<string>", "payload": {"amount": "<string>", "currency": "USD", "account_number": "<string>", "transaction_id": "<string>"}}\n\n'
    "Rules:\n"
    "- Only output valid JSON matching one of the above schemas\n"
    "- Never call the same tool twice in a row\n"
    "- Sequence: getProviders → check_provider → execute_transaction\n"
    "- Use the provider name the user mentioned in check_provider\n"
    "- The execute_transaction payload must contain EXACTLY the fields from check_provider's required_fields\n"
    "- Invent any missing details (account numbers, IDs) — do NOT ask the user\n\n"
    "EXAMPLE:\n"
    'User: "Send $200 via Wise, fee under 2%"\n'
    'You: {"tool": "getProviders"}\n'
    'Tool result: ["Stripe", "Razorpay", "Wise"]\n'
    'You: {"tool": "check_provider", "provider_name": "Wise"}\n'
    'Tool result: {"required_fields": ["amount", "currency", "beneficiary_name"], "transaction_fee": "0.5%"}\n'
    'You: {"tool": "execute_transaction", "provider_name": "Wise", "payload": {"amount": "200", "currency": "USD", "beneficiary_name": "Raj Kumar"}}\n'
    'Tool result: Transaction successful.\n'
    'You: Sent $200 via Wise at 0.5% fee.'
)

# Model tool name → environment server tool name
_TOOL_NAME_MAP = {
    "getProviders":          "get_providers",
    "check_provider":        "check_provider",
    "execute_transaction":   "execute_transaction",
}
_ALLOWED_MODEL_TOOLS = set(_TOOL_NAME_MAP.keys())
_MAX_REPEAT_ACTIONS = 4


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_prompt(messages: list[dict]) -> str:
    """Format a message list as a chat string ending with the assistant turn opener."""
    _, tokenizer = get_model_and_tokenizer()
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    text = ""
    for m in messages:
        text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    text += "<|im_start|>assistant\n"
    return text


def _generate(prompt: str, max_new_tokens: int = 256) -> str:
    """Run inference with the local model and return the new text only."""
    model, tokenizer = get_model_and_tokenizer()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=1.0,   # rollout-only: max diversity, fights entropy collapse
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def _parse_tool_call(text: str) -> dict | None:
    """Extract the first balanced JSON object from model output.

    Uses brace counting instead of a flat regex so nested dicts
    (execute_transaction payloads) are parsed correctly.
    """
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


def _validate_tool_call(tool_call: dict | None) -> tuple[dict | None, str]:
    """Validate tool-call shape and allowed tools."""
    if tool_call is None:
        return None, "no_json"
    if not isinstance(tool_call, dict):
        return None, "not_object"
    tool = tool_call.get("tool")
    if not isinstance(tool, str):
        return None, "missing_tool"
    if tool not in _ALLOWED_MODEL_TOOLS:
        return None, "unknown_tool"
    return tool_call, "ok"


def _safe_json_loads(value: str) -> dict | list | None:
    """Best-effort JSON parser for tool observation strings."""
    try:
        parsed = json.loads(value)
        if isinstance(parsed, (dict, list)):
            return parsed
    except Exception:
        pass
    return None


def _extract_user_needs(text: str) -> dict:
    """Extract lightweight user preferences from the persona request."""
    low = (text or "").lower()
    needs = {
        "currency": "USD",
        "fee_pref": "",
        "settlement_pref": "",
        "kyc_level": "basic",
    }
    for cur in ("inr", "eur", "gbp", "aed", "sgd", "usd"):
        if cur in low:
            needs["currency"] = cur.upper()
            break
    if "instant" in low:
        needs["settlement_pref"] = "instant"
    elif "same day" in low or "same-day" in low:
        needs["settlement_pref"] = "same day"
    if "no kyc" in low or "no strict kyc" in low:
        needs["kyc_level"] = "none"
    elif "full kyc" in low:
        needs["kyc_level"] = "full"
    if "under 1%" in low or "1% fee" in low:
        needs["fee_pref"] = "under 1%"
    elif "under 2%" in low or "2% fee" in low:
        needs["fee_pref"] = "under 2%"
    return needs


def _default_value_for_field(field: str, provider_name: str, user_needs: dict) -> str:
    """Return a sensible deterministic default for any required transaction field."""
    currency = user_needs.get("currency", "USD")
    defaults = {
        "amount":           "200",
        "currency":         currency,
        "beneficiary_name": "Alex Johnson",
        "account_number":   "9876543210",
        "ifsc_code":        "HDFC0001234",
        "routing_number":   "021000021",
        "swift_code":       "HDFCINBB",
        "upi_id":           "alex@upi",
        "reference_note":   "Invoice payment",
        "payment_method":   "bank_transfer",
        "purpose_code":     "P0101",
        "transaction_type": "transfer",
        "sender_name":      "Sam Sender",
        "contact_number":   "+1-555-0101",
    }
    return defaults.get(field, f"default_{provider_name}_{field}")


def _build_payload_from_required_fields(
    required_fields: list[str],
    provider_name: str,
    user_needs: dict,
) -> dict:
    """Build an exact-schema payload deterministically."""
    return {
        field: _default_value_for_field(field, provider_name, user_needs)
        for field in required_fields
    }


def _extract_reward(obs: dict, response_json: dict) -> float:
    """Extract reward from either observation or top-level response payload."""
    raw = obs.get("reward")
    if raw is None:
        raw = response_json.get("reward")
    if raw is None:
        return 0.0
    return float(raw)


def _execute_tool(tool_call: dict, server_base_url: str) -> tuple[float, str, bool]:
    """Execute a parsed tool call via the environment server.

    Maps model-facing tool names to environment-server tool names,
    then POSTs to /step and returns (reward, observation_data, episode_done).
    """
    model_tool = tool_call.get("tool", "")
    env_tool = _TOOL_NAME_MAP.get(model_tool, model_tool)

    provider_name = tool_call.get("provider_name", "")
    if not provider_name and model_tool == "getProviders":
        provider_name = "directory"

    action = {
        "tool":          env_tool,
        "provider_name": provider_name or "unknown",
        "payload":       tool_call.get("payload") or {},
    }

    try:
        resp = requests.post(
            f"{server_base_url}step",
            json={"action": action},
            timeout=15,
        )
        payload = resp.json()
        obs = payload.get("observation", {})
        reward = _extract_reward(obs, payload)
        data = obs.get("data", "")
        done = bool(obs.get("done", False))
        return reward, data, done
    except Exception as e:
        return -5.0, f"[HTTP ERROR] {e}", False


def _get_persona_request(server_base_url: str, fallback_idx: int) -> str:
    """Get a persona request via OpenRouter, or use a hardcoded fallback."""
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    model_name = os.getenv("MODEL_NAME", "").strip()

    if api_key and model_name:
        try:
            from personaAgent import persona_node
            result = persona_node({"messages": []})
            for msg in result.get("messages", []):
                if hasattr(msg, "content") and msg.content:
                    return msg.content
        except Exception as e:
            print(f"[ROLLOUT] Persona node failed ({e}), using fallback.", flush=True)
    else:
        if not api_key:
            print("[ROLLOUT] OPENROUTER_API_KEY not set — using fallback persona.", flush=True)
        if not model_name:
            print("[ROLLOUT] MODEL_NAME not set — using fallback persona.", flush=True)

    return FALLBACK_REQUESTS[fallback_idx % len(FALLBACK_REQUESTS)]


# ── main public function ──────────────────────────────────────────────────────

_current_cycle: int = 1  # updated by collect_rollouts caller


def collect_rollouts(episodes: int, server_base_url: str, cycle: int = 1) -> list[dict]:
    """Run episodes and collect (prompt, completion, reward) per agent step."""
    global _current_cycle
    _current_cycle = cycle
    rollout_buffer: list[dict] = []

    for ep in range(episodes):
        print(f"[ROLLOUT] Episode {ep + 1}/{episodes}", flush=True)

        try:
            requests.post(f"{server_base_url}reset", timeout=10)
        except Exception as e:
            print(f"[ROLLOUT] Reset failed: {e}", flush=True)
            continue

        persona_request = _get_persona_request(server_base_url, ep)
        constraint = persona_request

        try:
            requests.post(
                f"{server_base_url}set_constraint",
                json={"constraint": constraint},
                timeout=5,
            )
        except Exception as e:
            print(f"[ROLLOUT] set_constraint failed: {e}", flush=True)

        print(f"[ROLLOUT]   Persona: {persona_request[:80]}", flush=True)

        messages = [
            {"role": "system", "content": CONCIERGE_SYSTEM_PROMPT},
            {"role": "user", "content": persona_request},
        ]
        user_needs = _extract_user_needs(persona_request)
        provider_required_fields: dict[str, list[str]] = {}
        check_seen: set[str] = set()
        last_tool = ""          # for consecutive-repetition penalty
        last_action_sig = ""
        repeat_action_count = 0

        episode_reward = 0.0
        steps_taken = 0
        tools_called: list[str] = []
        did_reach_execute = False
        constraint_satisfied = False
        termination_reason = "max_steps"

        for step in range(MAX_STEPS_PER_EPISODE):
            steps_taken = step + 1
            prompt = _build_prompt(messages)
            completion = _generate(prompt)

            print(f"[ROLLOUT]   step {step + 1} model output: {completion[:120]!r}", flush=True)

            raw_tool_call = _parse_tool_call(completion)
            tool_call, status = _validate_tool_call(raw_tool_call)

            if tool_call is None:
                print(
                    f"[ROLLOUT]   step {step + 1}: invalid/no tool call ({status}) → episode end (reward 0)",
                    flush=True,
                )
                rollout_buffer.append({
                    "prompt":     prompt,
                    "completion": completion,
                    "reward":     0.0,
                    "episode":    ep + 1,
                    "tool":       "__invalid_tool__" if status == "unknown_tool" else "__no_tool__",
                    "done":       True,
                })
                termination_reason = f"invalid_tool_{status}"
                break

            tool_name = tool_call.get("tool", "")
            provider_name = tool_call.get("provider_name", "")
            tools_called.append(tool_name)

            # ── consecutive repetition penalty ────────────────────────────────
            if tool_name == last_tool and tool_name != "":
                rep_penalty = -2.0
                episode_reward += rep_penalty
                print(
                    f"[ROLLOUT]   step {step + 1}: consecutive repeat of {tool_name!r} → penalty {rep_penalty}",
                    flush=True,
                )
                rollout_buffer.append({
                    "prompt":     prompt,
                    "completion": completion,
                    "reward":     rep_penalty,
                    "episode":    ep + 1,
                    "tool":       f"__repeat_{tool_name}__",
                    "done":       False,
                })
                # still process the action after penalising
            last_tool = tool_name

            # ── action loop guard (4 identical action signatures) ─────────────
            action_sig = f"{tool_name}::{provider_name}"
            if action_sig == last_action_sig:
                repeat_action_count += 1
            else:
                repeat_action_count = 1
                last_action_sig = action_sig

            if repeat_action_count >= _MAX_REPEAT_ACTIONS:
                print(
                    f"[ROLLOUT]   step {step + 1}: action loop on {action_sig} → episode end",
                    flush=True,
                )
                rollout_buffer.append({
                    "prompt":     prompt,
                    "completion": completion,
                    "reward":     -10.0,
                    "episode":    ep + 1,
                    "tool":       "__loop_guard__",
                    "done":       True,
                })
                episode_reward += -10.0
                termination_reason = "action_loop"
                break

            # ── execute_transaction ordering gate ─────────────────────────────
            gate_fired = False
            if tool_name == "execute_transaction":
                did_reach_execute = True
                if provider_name not in check_seen:
                    gate_penalty = -20.0
                    episode_reward += gate_penalty
                    gate_fired = True
                    print(
                        f"[ROLLOUT]   step {step + 1}: execute_transaction before check_provider"
                        f" for {provider_name!r} → penalty {gate_penalty}, forcing check_provider",
                        flush=True,
                    )
                    rollout_buffer.append({
                        "prompt":     prompt,
                        "completion": completion,
                        "reward":     gate_penalty,
                        "episode":    ep + 1,
                        "tool":       "__execute_before_check__",
                        "done":       False,
                    })
                    # Feed model the penalty message before running forced check_provider
                    messages.append({"role": "assistant", "content": completion})
                    messages.append({
                        "role": "user",
                        "content": f"Tool result: ERROR — you must call check_provider for '{provider_name}' before execute_transaction.",
                    })
                    # Rebuild prompt with updated context, then silently run check_provider
                    prompt = _build_prompt(messages)
                    tool_call = {"tool": "check_provider", "provider_name": provider_name}
                elif provider_name in provider_required_fields:
                    tool_call["payload"] = _build_payload_from_required_fields(
                        provider_required_fields[provider_name],
                        provider_name,
                        user_needs,
                    )

            reward, obs_data, done = _execute_tool(tool_call, server_base_url)
            episode_reward += reward
            print(
                f"[ROLLOUT]   step {step + 1}: tool={tool_call.get('tool')!r} → "
                f"reward={reward:.1f} done={done}",
                flush=True,
            )

            rollout_buffer.append({
                "prompt":     prompt,
                "completion": completion if not gate_fired else json.dumps(tool_call),
                "reward":     reward,
                "episode":    ep + 1,
                "tool":       tool_call.get("tool", "unknown"),
                "done":       done,
            })

            # Cache required fields after check_provider
            if tool_call.get("tool") == "check_provider":
                check_seen.add(provider_name)
                parsed_obs = _safe_json_loads(obs_data) if isinstance(obs_data, str) else obs_data
                if isinstance(parsed_obs, dict):
                    req = parsed_obs.get("required_fields")
                    if isinstance(req, list):
                        provider_required_fields[provider_name] = [str(f) for f in req]

            # Corrective retry on hard execute_transaction rejection
            if (
                tool_call.get("tool") == "execute_transaction"
                and reward <= -15.0
                and provider_name in provider_required_fields
            ):
                retry_call = {
                    "tool":          "execute_transaction",
                    "provider_name": provider_name,
                    "payload":       _build_payload_from_required_fields(
                        provider_required_fields[provider_name],
                        provider_name,
                        user_needs,
                    ),
                }
                retry_completion = json.dumps(retry_call)
                # Feed the rejection into context before retry
                messages.append({"role": "assistant", "content": completion})
                messages.append({"role": "user", "content": f"Tool result: {obs_data}"})
                retry_reward, retry_obs, retry_done = _execute_tool(retry_call, server_base_url)
                episode_reward += retry_reward
                done = True
                print(
                    f"[ROLLOUT]   step {step + 1}: corrective retry reward={retry_reward:.1f}",
                    flush=True,
                )
                rollout_buffer.append({
                    "prompt":     _build_prompt(messages),
                    "completion": retry_completion,
                    "reward":     retry_reward,
                    "episode":    ep + 1,
                    "tool":       "execute_transaction_retry",
                    "done":       True,
                })
                obs_data = retry_obs

            # Always feed the tool result back into context before the done check.
            # This ensures the model sees full conversation history on every step,
            # including the final successful turn (needed if done is reset by retry).
            if not gate_fired:
                messages.append({"role": "assistant", "content": completion})
            messages.append({"role": "user", "content": f"Tool result: {obs_data}"})

            if done:
                constraint_satisfied = reward >= 50.0
                termination_reason = "success" if constraint_satisfied else "constraint_violation"
                break

        print(f"[ROLLOUT] Episode {ep + 1} total reward: {episode_reward:.1f}", flush=True)

        log_episode(
            cycle=_current_cycle,
            episode=ep + 1,
            total_reward=episode_reward,
            steps_taken=steps_taken,
            tools_called=tools_called,
            did_reach_execute=did_reach_execute,
            constraint_satisfied=constraint_satisfied,
            termination_reason=termination_reason,
        )

    return rollout_buffer

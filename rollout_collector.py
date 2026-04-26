"""Runs episodes with the local model as a fintech payment agent and collects training data."""
# Import unsloth FIRST before any trl/transformers to avoid import-order warnings
try:
    import unsloth  # noqa: F401
except ImportError:
    pass

import json
import os
import re
import time

import requests
import torch

from model_loader import get_model_and_tokenizer

MAX_STEPS_PER_EPISODE = 15

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
    "You are a Fintech Payment AI Agent. Your ONLY job is to execute financial transactions by calling tools using JSON.\n\n"
    "════════════════════════════════════════════════════════\n"
    "CRITICAL: EVERY response MUST be a single JSON object.\n"
    "NEVER write plain text. NEVER write explanations. NEVER write lists.\n"
    "IF YOUR RESPONSE IS NOT A JSON OBJECT, YOU HAVE FAILED.\n"
    "════════════════════════════════════════════════════════\n\n"
    "TOOL FORMAT — output EXACTLY one of these JSON objects per turn:\n"
    '- List providers:      {"tool": "getProviders"}\n'
    '- Check provider:      {"tool": "check_provider", "provider_name": "NAME"}\n'
    '- Execute transaction: {"tool": "execute_transaction", "provider_name": "NAME", "payload": {"field": "value"}}\n\n'
    "RULES:\n"
    "1. Always call getProviders first to discover available fintech services.\n"
    "2. Always call check_provider BEFORE execute_transaction for any provider.\n"
    "3. The execute_transaction payload must contain EXACTLY the fields listed in check_provider's required_fields — no more, no less.\n"
    "4. Match the user's constraints: fee limit, currency, KYC level, settlement time.\n"
    "5. Invent any missing transaction details (sender name, account numbers, reference) — do NOT ask the user.\n"
    "6. If execute_transaction fails, call check_provider again (schema may have drifted), then retry with corrected payload.\n"
    "7. ONLY after a successful execute_transaction, write a plain text confirmation — this is the ONLY time plain text is allowed.\n\n"
    "EXAMPLE OF CORRECT MULTI-STEP BEHAVIOUR:\n"
    'User: "Send $200 to India, fee under 2%, instant settlement"\n'
    'You: {"tool": "getProviders"}\n'
    '[Tool Result]: ["Stripe", "Razorpay", "Wise", "PayPal"]\n'
    'You: {"tool": "check_provider", "provider_name": "Wise"}\n'
    '[Tool Result]: {"required_fields": ["amount", "currency", "beneficiary_name", "account_number"], "transaction_fee": "0.5%", "settlement_time": "instant"}\n'
    'You: {"tool": "execute_transaction", "provider_name": "Wise", "payload": {"amount": "200", "currency": "INR", "beneficiary_name": "Raj Kumar", "account_number": "9876543210"}}\n'
    '[Tool Result]: {"status": "success"}\n'
    'You: Transaction of $200 sent via Wise. Fee: 0.5%, settled instantly.\n\n'
    "REMEMBER: Every response is a JSON tool call UNTIL execute_transaction succeeds.\n"
    "NO PROSE. NO LISTS. NO EXPLANATIONS. JSON ONLY until success."
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
            temperature=0.7,
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

def collect_rollouts(episodes: int, server_base_url: str) -> list[dict]:
    """Run episodes and collect (prompt, completion, reward) per agent step."""
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
        last_action_sig = ""
        repeat_action_count = 0

        episode_reward = 0.0
        for step in range(MAX_STEPS_PER_EPISODE):
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
                break

            provider_name = tool_call.get("provider_name", "")
            action_sig = f"{tool_call.get('tool')}::{provider_name}"
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
                break

            # Force check_provider before execute_transaction
            if tool_call.get("tool") == "execute_transaction":
                if provider_name not in check_seen:
                    print(
                        f"[ROLLOUT]   step {step + 1}: forcing check_provider before execute_transaction for {provider_name}",
                        flush=True,
                    )
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
                "completion": completion,
                "reward":     reward,
                "episode":    ep + 1,
                "tool":       tool_call.get("tool", "unknown"),
                "done":       done,
            })

            # Cache required fields after a successful check_provider
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
                retry_reward, retry_obs, retry_done = _execute_tool(retry_call, server_base_url)
                episode_reward += retry_reward
                done = True  # single corrective retry, then terminate
                print(
                    f"[ROLLOUT]   step {step + 1}: corrective retry reward={retry_reward:.1f} done={retry_done}",
                    flush=True,
                )
                rollout_buffer.append({
                    "prompt":     prompt,
                    "completion": retry_completion,
                    "reward":     retry_reward,
                    "episode":    ep + 1,
                    "tool":       "execute_transaction_retry",
                    "done":       True,
                })
                obs_data = retry_obs

            messages.append({"role": "assistant", "content": completion})
            messages.append({"role": "user", "content": f"[Tool Result]: {obs_data}"})

            if done:
                break

        print(f"[ROLLOUT] Episode {ep + 1} total reward: {episode_reward:.1f}", flush=True)

    return rollout_buffer

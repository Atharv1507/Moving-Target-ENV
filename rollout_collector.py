"""Runs episodes with the local model as concierge and collects training data."""
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
    "I need a Vegan meal under $40 with a flexible refund policy.",
    "Get me something Halal, budget is $50, must be refundable.",
    "I want to order food, no dietary restrictions, cheapest option possible.",
    "I need a Keto meal under $30 — also I have a dog so pet-friendly is a must.",
    "Looking for Gluten-Free options, budget $100, strictly refundable please.",
    "Order me anything under $20, No Restrictions on diet.",
    "I want a Vegan and Nut-Free meal, budget is $80, flexible returns preferred.",
]

CONCIERGE_SYSTEM_PROMPT = (
    "You are an E-Commerce AI Concierge. Your ONLY job is to call tools using JSON.\n\n"
    "════════════════════════════════════════════════════════\n"
    "CRITICAL: EVERY response you generate MUST be a JSON object.\n"
    "NEVER write plain text. NEVER write explanations. NEVER write lists.\n"
    "IF YOUR RESPONSE IS NOT A JSON OBJECT, YOU HAVE FAILED.\n"
    "════════════════════════════════════════════════════════\n\n"
    "TOOL FORMAT — output EXACTLY one of these JSON objects, nothing else:\n"
    '- List merchants:  {"tool": "getMerchant"}\n'
    '- Check merchant:  {"tool": "ask_watchdog", "merchant_name": "NAME"}\n'
    '- Place order:     {"tool": "place_order", "merchant_name": "NAME", "payload": {"field": "value"}}\n\n'
    "RULES:\n"
    "1. Always call getMerchant first to discover available merchants.\n"
    "2. Always call ask_watchdog BEFORE place_order for any merchant.\n"
    "3. The place_order payload must contain EXACTLY the fields from ask_watchdog's required_fields.\n"
    "4. Check price / refund / diet policies match the user's constraints.\n"
    "5. Invent any missing details (name, address, contact) — do NOT ask the user.\n"
    "6. If place_order fails, fix the payload and retry immediately with a JSON tool call.\n"
    "7. ONLY after a successful place_order, write a plain text summary — this is the ONLY time plain text is allowed.\n\n"
    "EXAMPLE OF CORRECT MULTI-STEP BEHAVIOR:\n"
    'User: "Order halal food under $40"\n'
    'You: {"tool": "getMerchant"}\n'
    '[Tool Result]: ["HalalKitchen", "BurgerBar"]\n'
    'You: {"tool": "ask_watchdog", "merchant_name": "HalalKitchen"}\n'
    '[Tool Result]: {"required_fields": ["customer_name", "item", "price"]}\n'
    'You: {"tool": "place_order", "merchant_name": "HalalKitchen", "payload": {"customer_name": "Alex", "item": "chicken wrap", "price": "35"}}\n'
    '[Tool Result]: {"status": "success"}\n'
    'You: Order placed at HalalKitchen for $35.\n\n'
    "REMEMBER: Every response is a JSON tool call UNTIL the order succeeds.\n"
    "NO PROSE. NO LISTS. NO EXPLANATIONS. JSON ONLY."
)

# Model tool name → environment server tool name
_TOOL_NAME_MAP = {
    "getMerchant": "get_merchants",
    "check_merchant": "ask_watchdog",
    "ask_watchdog": "ask_watchdog",
    "place_order": "place_order",
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

    Uses brace counting instead of a flat regex so nested dicts (place_order
    payloads) are parsed correctly.
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
    """Extract lightweight user preferences to seed deterministic payload fields."""
    low = (text or "").lower()
    needs = {
        "dietary_notes": "",
        "refund_pref": "",
    }
    diet_tags = []
    for key in ("vegan", "halal", "keto", "gluten-free", "gluten free", "nut-free", "nut free", "low-carb", "low carb"):
        if key in low:
            diet_tags.append(key.replace("-", " "))
    if diet_tags:
        needs["dietary_notes"] = ", ".join(sorted(set(diet_tags)))
    if "strictly refundable" in low or "strict refund" in low:
        needs["refund_pref"] = "strictly refundable"
    elif "flexible refund" in low or "pet-friendly refund" in low or "pet friendly refund" in low:
        needs["refund_pref"] = "flexible refund preferred"
    return needs


def _default_value_for_field(field: str, merchant_name: str, user_needs: dict) -> str:
    defaults = {
        "item": "chef special meal",
        "price": "50",
        "dietary_notes": user_needs.get("dietary_notes") or "no dietary restrictions",
        "delivery_address": "221B Baker Street",
        "contact_number": "+1-555-0101",
        "customer_name": "Alex Customer",
        "discount_code": "NONE",
        "special_instructions": user_needs.get("refund_pref") or "leave at door",
        "payment_method": "card",
        "quantity": "1",
    }
    return defaults.get(field, f"default_{merchant_name}_{field}")


def _build_payload_from_required_fields(
    required_fields: list[str],
    merchant_name: str,
    user_needs: dict,
) -> dict:
    """Build exact-schema payload deterministically."""
    payload = {}
    for field in required_fields:
        payload[field] = _default_value_for_field(field, merchant_name, user_needs)
    return payload


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
    env_tool = _TOOL_NAME_MAP.get(model_tool, model_tool)  # fix getMerchant → get_merchants

    merchant_name = tool_call.get("merchant_name")
    if merchant_name is None and isinstance(tool_call.get("merchant_names"), list):
        merchant_list = tool_call.get("merchant_names") or []
        merchant_name = merchant_list[0] if merchant_list else None

    action = {
        "tool": env_tool,
        "merchant_name": (
            "directory"                                  # getMerchant placeholder
            if model_tool == "getMerchant"
            else (merchant_name or "unknown")
        ),
        "payload": tool_call.get("payload") or {},
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
    """Run episodes and collect (prompt, completion, reward) per concierge step."""
    rollout_buffer: list[dict] = []

    for ep in range(episodes):
        print(f"[ROLLOUT] Episode {ep + 1}/{episodes}", flush=True)

        try:
            requests.post(f"{server_base_url}reset", timeout=10)
        except Exception as e:
            print(f"[ROLLOUT] Reset failed: {e}", flush=True)
            continue

        persona_request = _get_persona_request(server_base_url, ep)
        print(f"[ROLLOUT]   Persona: {persona_request[:80]}", flush=True)

        messages = [
            {"role": "system", "content": CONCIERGE_SYSTEM_PROMPT},
            {"role": "user", "content": persona_request},
        ]
        user_needs = _extract_user_needs(persona_request)
        merchant_required_fields: dict[str, list[str]] = {}
        watchdog_seen: set[str] = set()
        last_action_sig = ""
        repeat_action_count = 0

        episode_reward = 0.0
        for step in range(MAX_STEPS_PER_EPISODE):
            prompt = _build_prompt(messages)
            completion = _generate(prompt)

            # Always log what the model actually said (trimmed)
            print(f"[ROLLOUT]   step {step + 1} model output: {completion[:120]!r}", flush=True)

            raw_tool_call = _parse_tool_call(completion)
            tool_call, status = _validate_tool_call(raw_tool_call)
            if tool_call is None:
                print(
                    f"[ROLLOUT]   step {step + 1}: invalid/no tool call ({status}) → episode end (reward 0)",
                    flush=True,
                )
                rollout_buffer.append(
                    {
                        "prompt": prompt,
                        "completion": completion,
                        "reward": 0.0,
                        "episode": ep + 1,
                        "tool": "__invalid_tool__" if status == "unknown_tool" else "__no_tool__",
                        "done": True,
                    }
                )
                break

            merchant_name = tool_call.get("merchant_name")
            if merchant_name is None and isinstance(tool_call.get("merchant_names"), list):
                names = tool_call.get("merchant_names") or []
                merchant_name = names[0] if names else "unknown"
            merchant_name = merchant_name or "unknown"

            action_sig = f"{tool_call.get('tool')}::{merchant_name}"
            if action_sig == last_action_sig:
                repeat_action_count += 1
            else:
                repeat_action_count = 1
                last_action_sig = action_sig
            if repeat_action_count >= _MAX_REPEAT_ACTIONS:
                print(
                    f"[ROLLOUT]   step {step + 1}: detected action loop on {action_sig} → episode end",
                    flush=True,
                )
                rollout_buffer.append(
                    {
                        "prompt": prompt,
                        "completion": completion,
                        "reward": -10.0,
                        "episode": ep + 1,
                        "tool": "__loop_guard__",
                        "done": True,
                    }
                )
                episode_reward += -10.0
                break

            if tool_call.get("tool") == "place_order":
                if merchant_name not in watchdog_seen:
                    print(
                        f"[ROLLOUT]   step {step + 1}: forcing ask_watchdog before place_order for {merchant_name}",
                        flush=True,
                    )
                    tool_call = {"tool": "ask_watchdog", "merchant_name": merchant_name}
                elif merchant_name in merchant_required_fields:
                    tool_call["payload"] = _build_payload_from_required_fields(
                        merchant_required_fields[merchant_name],
                        merchant_name,
                        user_needs,
                    )

            reward, obs_data, done = _execute_tool(tool_call, server_base_url)
            episode_reward += reward
            print(
                f"[ROLLOUT]   step {step + 1}: tool={tool_call.get('tool')!r} → "
                f"reward={reward:.1f} done={done}",
                flush=True,
            )

            rollout_buffer.append(
                {
                    "prompt": prompt,
                    "completion": completion,
                    "reward": reward,
                    "episode": ep + 1,
                    "tool": tool_call.get("tool", "unknown"),
                    "done": done,
                }
            )

            if tool_call.get("tool") in ("ask_watchdog", "check_merchant"):
                watchdog_seen.add(merchant_name)
                parsed_obs = _safe_json_loads(obs_data) if isinstance(obs_data, str) else obs_data
                if isinstance(parsed_obs, dict):
                    req = parsed_obs.get("required_fields")
                    if isinstance(req, list):
                        merchant_required_fields[merchant_name] = [str(x) for x in req]

            if (
                tool_call.get("tool") == "place_order"
                and reward <= -50.0
                and merchant_name in merchant_required_fields
            ):
                retry_call = {
                    "tool": "place_order",
                    "merchant_name": merchant_name,
                    "payload": _build_payload_from_required_fields(
                        merchant_required_fields[merchant_name],
                        merchant_name,
                        user_needs,
                    ),
                }
                retry_completion = json.dumps(retry_call)
                retry_reward, retry_obs, retry_done = _execute_tool(retry_call, server_base_url)
                episode_reward += retry_reward
                done = True  # single corrective retry, then terminate to avoid loops
                print(
                    f"[ROLLOUT]   step {step + 1}: corrective retry reward={retry_reward:.1f} done={retry_done}",
                    flush=True,
                )
                rollout_buffer.append(
                    {
                        "prompt": prompt,
                        "completion": retry_completion,
                        "reward": retry_reward,
                        "episode": ep + 1,
                        "tool": "place_order_retry",
                        "done": True,
                    }
                )
                obs_data = retry_obs

            messages.append({"role": "assistant", "content": completion})
            messages.append({"role": "user", "content": f"[Tool Result]: {obs_data}"})

            if done:
                break

        print(f"[ROLLOUT] Episode {ep + 1} total reward: {episode_reward:.1f}", flush=True)

    return rollout_buffer

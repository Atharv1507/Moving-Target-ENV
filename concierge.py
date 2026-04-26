"""Fintech payment concierge node for the LangGraph simulation."""
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from state import AgentState
import os
import dotenv
dotenv.load_dotenv()
import requests

_ENV_PORT = int(os.getenv("ENV_SERVER_PORT", "8001"))
_SERVER_URL = f"http://localhost:{_ENV_PORT}"


def _safe_parse(response):
    """Safely extract data and reward from server response."""
    try:
        response_data = response.json()
        data = response_data["observation"].get("data", "Server returned empty response.")
        reward = response_data["observation"].get("reward") or response_data.get("reward", 0)
        return data, reward
    except Exception:
        return f"SERVER ERROR: {response.status_code} - {response.text[:200]}", -10.0


@tool
def getProviders():
    """List all available fintech payment providers. Always call this first."""
    payload = {
        "action": {
            "tool": "get_providers",
            "provider_name": "directory",
        }
    }
    response = requests.post(f"{_SERVER_URL}/step", json=payload)
    data, reward = _safe_parse(response)
    return f"Observation: {data} \n(Environment Reward: {reward})"


@tool
def check_provider(provider_name: str):
    """Check the API schema, fees, KYC requirements and settlement time for a provider.
    ALWAYS call this before execute_transaction."""
    payload = {
        "action": {
            "tool": "check_provider",
            "provider_name": provider_name,
        }
    }
    response = requests.post(f"{_SERVER_URL}/step", json=payload)
    data, reward = _safe_parse(response)
    return f"Observation: {data} \n(Environment Reward: {reward})"


@tool
def execute_transaction(provider_name: str = "", payload: dict = None):
    """Execute a transaction via a specific provider.
    CRITICAL: You MUST call check_provider first to get the required_fields.
    The payload dict must contain EXACTLY the fields listed in check_provider's required_fields — no more, no less.
    Invent any missing details (account numbers, beneficiary names, etc.) — do NOT ask the user.
    Example: execute_transaction(provider_name='Wise', payload={'amount': '200', 'currency': 'USD', 'beneficiary_name': 'Alex'})
    """
    req_body = {
        "action": {
            "tool": "execute_transaction",
            "provider_name": provider_name,
            "payload": payload or {},
        }
    }
    response = requests.post(f"{_SERVER_URL}/step", json=req_body)
    data, reward = _safe_parse(response)
    return f"Observation: {data} \n(Environment Reward: {reward})"


tools = [getProviders, check_provider, execute_transaction]

concierge_llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
).bind_tools(tools)


def concierge_node(state: AgentState):
    """The Concierge Node: reads the user's payment request and decides which tool to call."""
    messages = state["messages"]

    prev_summary = state.get("prev_episode_summary", "")
    rl_context = ""
    if prev_summary:
        rl_context = f"PREVIOUS EPISODE PERFORMANCE FEEDBACK: {prev_summary} Use this to improve this episode. "

    system_instruction = SystemMessage(content=(
        "You are an elite Fintech Payment AI Concierge. Your job is to fulfill the user's payment/transfer request. "
        + rl_context +
        "CRITICAL RULES:\n"
        "1. SILENT EXECUTION: Never ask the user questions. Only output tool calls until the task is done.\n"
        "2. STRICT SEQUENCE: You MUST follow this order every time:\n"
        "   a) Call getProviders to list available providers.\n"
        "   b) Call check_provider with the user's requested provider name.\n"
        "   c) If constraints are satisfied, call execute_transaction.\n"
        "3. NEVER call the same tool twice in a row.\n"
        "4. ALWAYS use the provider name the user mentioned in check_provider.\n"
        "5. PAYLOAD PRECISION: The payload in execute_transaction must contain EXACTLY the fields "
        "from check_provider's required_fields. No more, no less.\n"
        "6. INVENT MISSING DATA: If required fields are not provided by the user (account numbers, "
        "beneficiary names, IDs, etc.), invent realistic values. Do NOT ask the user.\n"
        "7. CONSTRAINT CHECK: After check_provider, verify the provider's fee, supported_currencies, "
        "kyc_required, and settlement_time match the user's requirements. If they don't match, "
        "call getProviders, pick a different provider, and call check_provider on it.\n"
        "8. ERROR HANDLING:\n"
        "   [A] Missing required field: add the field with an invented value and retry execute_transaction.\n"
        "   [B] Unexpected field: remove it and retry execute_transaction.\n"
        "   [C] Constraint violation: blacklist provider, pick a new one from getProviders.\n"
        "9. TERMINATION: Once all providers are exhausted or a transaction succeeds, output a final plain-text summary."
    ))

    response = concierge_llm.invoke([system_instruction] + messages)

    schema_update = {
        "messages": [response],
        "step_count": state.get("step_count", 0) + 1,
    }

    if hasattr(response, "tool_calls") and len(response.tool_calls) > 0:
        tool_call = response.tool_calls[0]
        if "provider_name" in tool_call.get("args", {}):
            schema_update["current_provider"] = tool_call["args"]["provider_name"]

    return schema_update


# ── quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    fake_state = {
        "messages": [HumanMessage(content="Send $200 via Wise to a UK account. Fee must be under 2%.")],
        "step_count": 0,
        "prev_episode_summary": "",
        "current_provider": "",
        "last_known_schema": {},
        "drift_detected": False,
        "reward_score": 0.0,
    }

    print("Running Concierge Node Test...\n")
    result = concierge_node(fake_state)
    response_message = result["messages"][0]

    print("--- LLM Decision ---")
    if hasattr(response_message, "tool_calls") and len(response_message.tool_calls) > 0:
        tc = response_message.tool_calls[0]
        print(f"SUCCESS! Tool call: {tc['name']}({tc.get('args', {})})")
    else:
        print("Text reply:", response_message.content)

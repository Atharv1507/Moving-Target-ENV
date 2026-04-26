"""Watchdog node — detects API schema drift for fintech providers."""
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from state import AgentState
import os
import dotenv
dotenv.load_dotenv()

watchdog_llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    temperature=0.0,
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


def watchdog_node(state: AgentState):
    """The Watchdog Node: compares newly fetched provider schemas to memory,
    alerts the Concierge if the API has drifted."""
    messages = state["messages"]
    current_provider = state.get("current_provider", "")
    last_known_registry = state.get("last_known_schema", {})

    # Find the latest check_provider tool result in the message history
    latest_schema_str = None
    for msg in reversed(messages):
        if hasattr(msg, "name") and msg.name == "check_provider":
            latest_schema_str = msg.content
            break

    if not latest_schema_str:
        return {"drift_detected": False}

    last_known_schema = last_known_registry.get(current_provider)

    if not last_known_schema:
        # First time seeing this provider — log it
        last_known_registry[current_provider] = latest_schema_str
        return {
            "last_known_schema": last_known_registry,
            "drift_detected": False,
            "messages": [SystemMessage(
                content=f"[Watchdog] Discovered provider '{current_provider}'. Schema saved to memory."
            )],
        }

    # Compare old vs new schema via LLM
    system_instruction = SystemMessage(content=(
        "You are a Fintech API Schema Watchdog. Compare an old provider schema with a newly fetched one.\n"
        "Output ONLY 'SAFE' if they are functionally identical.\n"
        "If anything changed (new field, removed field, fee change, settlement change, etc.), "
        "output 'DRIFT:' followed by a brief 1-sentence summary of what changed."
    ))

    human_prompt = HumanMessage(content=(
        f"Old Schema:\n{last_known_schema}\n\n"
        f"New Schema:\n{latest_schema_str}"
    ))

    response = watchdog_llm.invoke([system_instruction, human_prompt])

    if response.content.strip().startswith("SAFE"):
        return {"drift_detected": False}
    else:
        last_known_registry[current_provider] = latest_schema_str
        return {
            "last_known_schema": last_known_registry,
            "drift_detected": True,
            "messages": [SystemMessage(
                content=f"[Watchdog ALERT] Provider '{current_provider}' schema drifted: {response.content}"
            )],
        }


# ── quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from langchain_core.messages import ToolMessage

    fake_state = {
        "current_provider": "Wise",
        "last_known_schema": {
            "Wise": '{"required_fields": ["amount", "currency"], "transaction_fee": "0.5%"}'
        },
        "messages": [
            ToolMessage(
                tool_call_id="call_123",
                name="check_provider",
                content='{"required_fields": ["amount", "currency", "beneficiary_name"], "transaction_fee": "0.5%"}',
            )
        ],
    }

    print("Running Watchdog Node Test...\n")
    result = watchdog_node(fake_state)
    print(f"Drift Detected: {result.get('drift_detected')}")
    if result.get("messages"):
        print(f"Watchdog Message: {result['messages'][0].content}")

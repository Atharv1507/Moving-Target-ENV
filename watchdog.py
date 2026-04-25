from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from state import AgentState
import json
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
    """
    The Watchdog Node: Acts as the Technical Scout.
    It takes the raw schema fetched from the environment, compares it to memory,
    and alerts the Concierge if the API rules have drifted.
    """
    messages = state["messages"]
    current_merchant = state.get("current_merchant")
    last_known_registry = state.get("last_known_schema", {})
    
    # 1. Find the latest tool execution response in the messages 
    latest_schema_str = None

    for msg in reversed(messages):
        if hasattr(msg, 'name') and msg.name == 'ask_watchdog':
            latest_schema_str = msg.content
            break
            
    if not latest_schema_str:

        return {"drift_detected": False}

    # 2. Check if we have seen this merchant before
    last_known_schema = last_known_registry.get(current_merchant)
    
    if not last_known_schema:
        # Brand new merchant! Log it to memory.
        last_known_registry[current_merchant] = latest_schema_str
        return {
            "last_known_schema": last_known_registry,
            "drift_detected": False,
            "messages": [SystemMessage(content=f"[Watchdog] Discovered new merchant '{current_merchant}'. Schema saved to memory.")]
        }

    # 3. If we HAVE seen it, use the LLM to compare for drift.
    system_instruction = SystemMessage(content=(
        "You are an API Schema Watchdog. Your job is to compare an old JSON schema "
        "with a newly fetched JSON schema and determine if anything changed (Drift).\n"
        "Output ONLY 'SAFE' if they are functionally identical.\n"
        "If there is a change, Output 'DRIFT:' followed by a very brief 1 sentence summary of what changed."
    ))
    
    human_prompt = HumanMessage(content=(
        f"Old Schema:\n{last_known_schema}\n\n"
        f"New Schema:\n{latest_schema_str}"
    ))
    
    response = watchdog_llm.invoke([system_instruction, human_prompt])
    
    # 4. Process the LLM's review
    if response.content.strip().startswith("SAFE"):
        return {"drift_detected": False}
    else:
        # Update the registry with the new schema so we know it for next time
        last_known_registry[current_merchant] = latest_schema_str
        return {
            "last_known_schema": last_known_registry,
            "drift_detected": True,
            "messages": [SystemMessage(content=f"[Watchdog ALERT] {response.content}")]
        }


# --- QUICK TEST ---
if __name__ == "__main__":
    from langchain_core.messages import ToolMessage

    # We fake a scenario where the Concierge called ask_watchdog, 
    # and the environment returned a string with a new field "contact_number".
    # BUT our memory only remembers ["item", "price"].

    fake_state = {
        "current_merchant": "VeganBistro",
        "last_known_schema": {
            "VeganBistro": '{"required_fields": ["item", "price"], "refund_policy": "Strict"}'
        },
        "messages": [
            ToolMessage(
                tool_call_id="call_123", 
                name="ask_watchdog", 
                content='{"required_fields": ["item", "price", "contact_number"], "refund_policy": "Strict"}'
            )
        ]
    }
    
    print("Running Watchdog Node Test...\n")
    result = watchdog_node(fake_state)
    
    print("--- Watchdog Output ---")
    print(f"Drift Detected: {result.get('drift_detected')}")
    if result.get("messages"):
        print(f"Watchdog Message appended: {result['messages'][0].content}")

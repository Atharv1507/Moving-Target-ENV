import random
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from state import AgentState
from server.Moving_Target_environment import MovingTargetEnv
import os
import dotenv
dotenv.load_dotenv()


def _get_persona_llm() -> ChatOpenAI:
    """Build persona LLM lazily using explicit OpenRouter env vars.

    Priority:
    1) PERSONA_MODEL
    2) OPENROUTER_MODEL
    3) MODEL_NAME
    4) fallback default
    """
    model_name = (
        os.getenv("PERSONA_MODEL")
        or os.getenv("OPENROUTER_MODEL")
        or os.getenv("MODEL_NAME")
        or "openai/gpt-4o-mini"
    )
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    return ChatOpenAI(
        model=model_name,
        temperature=0.9,
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=api_key,
    )

def persona_node(state: AgentState):
    # 1) Randomized fintech constraint pool
    amounts = ["$50", "$200", "$750", "$1000", "₹10,000", "£300"]
    currencies = ["USD", "INR", "GBP", "EUR"]
    fee_limits = ["under 1%", "under 2%", "$2 flat max", "no strict fee limit"]
    settlement_needs = ["instant settlement", "same-day settlement", "1-2 day settlement is fine"]
    kyc_levels = ["basic KYC only", "full KYC available", "no strict KYC requirement"]
    providers = list(MovingTargetEnv().ground_truth.keys())
    transaction_types = [
        "send money to a bank account",
        "withdraw platform balance to bank",
        "pay an invoice",
        "international transfer",
        "domestic transfer",
    ]

    # 2) Pick a random scenario for this episode
    current_amount = random.choice(amounts)
    current_currency = random.choice(currencies)
    current_fee = random.choice(fee_limits)
    current_settlement = random.choice(settlement_needs)
    current_kyc = random.choice(kyc_levels)
    current_provider = random.choice(providers)
    current_txn_type = random.choice(transaction_types)

    system_prompt = (
        "You are a fintech customer sending one request message to an assistant.\n"
        "Write a natural user request (1-4 sentences) for a payment/transfer use case.\n"
        f"Use this profile:\n"
        f"- Transaction intent: {current_txn_type}\n"
        f"- Amount: {current_amount}\n"
        f"- Currency: {current_currency}\n"
        f"- Preferred provider: {current_provider}\n"
        f"- Fee requirement: {current_fee}\n"
        f"- Settlement requirement: {current_settlement}\n"
        f"- KYC condition: {current_kyc}\n\n"
        "Requirements:\n"
        "- Mention provider name explicitly.\n"
        "- Mention amount and currency explicitly.\n"
        "- Keep it realistic and conversational.\n"
        "- Do not use bullets or JSON.\n"
    )
    
    persona_llm = _get_persona_llm()
    response = persona_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content="The simulation has started. Send your request to the assistant.")
    ])
    
    import requests
    _env_port = int(os.getenv("ENV_SERVER_PORT", "8001"))
    try:
        requests.post(
            f"http://localhost:{_env_port}/set_constraint",
            json={"constraint": response.content},
            timeout=5,
        )
    except Exception:
        pass
        
    print(response.content)
    return {"messages": [response]}

if __name__ == "__main__":
    persona_node({})

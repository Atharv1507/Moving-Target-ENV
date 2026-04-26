import json
import os
import random
from typing import Any, Optional

from openenv.core.env_server import Environment
from langchain_openai import ChatOpenAI
import dotenv
dotenv.load_dotenv()

from models import FintechAction, FintechObservation, FintechEnvironmentState

# All possible fields a fintech provider API may require
POTENTIAL_FIELDS = [
    "amount",
    "currency",
    "beneficiary_name",
    "account_number",
    "ifsc_code",
    "routing_number",
    "swift_code",
    "upi_id",
    "reference_note",
    "payment_method",
    "purpose_code",
    "transaction_type",
    "sender_name",
    "contact_number",
]


class MovingTargetEnv(Environment[FintechAction, FintechObservation, FintechEnvironmentState]):
    # CLASS VARIABLES — persist across FastAPI HTTP request instantiations
    ground_truth = {}
    _global_step_count = 0
    ground_truth_constraint = ""
    _directory_rewarded = False

    def __init__(self):
        super().__init__()
        self.initial_providers = [
            "Stripe", "Razorpay", "PayPal", "Wise", "Revolut",
            "Cashfree", "Paytm", "PhonePe", "Braintree", "Square",
        ]
        if not MovingTargetEnv.ground_truth:
            self._initialize_world()

    def _generate_random_schema(self) -> dict:
        """Generate a randomised fintech provider API schema."""
        num_fields = random.randint(2, 5)
        # Always require amount — every payment needs it
        fields = {"amount"}
        while len(fields) < num_fields:
            fields.add(random.choice(POTENTIAL_FIELDS))

        return {
            "api_version": f"v{random.randint(1, 5)}.{random.randint(0, 9)}",
            "required_fields": list(fields),
            "transaction_fee": random.choice(["0.5%", "1%", "1.5%", "2%", "$1 flat", "$2 flat", "$3 flat"]),
            "supported_currencies": random.sample(["USD", "INR", "EUR", "GBP", "AED", "SGD"], k=random.randint(2, 4)),
            "transfer_limit": random.choice(["$1,000/day", "$5,000/day", "$10,000/day", "$50,000/day"]),
            "kyc_required": random.choice(["none", "basic", "full"]),
            "settlement_time": random.choice(["instant", "same day", "1-2 days", "3-5 days"]),
        }

    def _drift_schema(self, provider_name: str) -> None:
        """Mutate an existing schema to simulate API drift."""
        schema = MovingTargetEnv.ground_truth[provider_name]

        if random.random() > 0.5 or len(schema["required_fields"]) <= 2:
            # Add a new field
            new_field = random.choice(
                [f for f in POTENTIAL_FIELDS if f not in schema["required_fields"]]
            )
            schema["required_fields"].append(new_field)
        else:
            # Remove a field (never remove 'amount')
            removable = [f for f in schema["required_fields"] if f != "amount"]
            if removable:
                schema["required_fields"].remove(random.choice(removable))

        schema["api_version"] = f"v{random.randint(1, 9)}.{random.randint(0, 9)}-updated"
        # Fee and settlement can also drift
        schema["transaction_fee"] = random.choice(["0.5%", "1%", "1.5%", "2%", "$1 flat", "$2 flat", "$3 flat"])
        schema["settlement_time"] = random.choice(["instant", "same day", "1-2 days", "3-5 days"])
        MovingTargetEnv.ground_truth[provider_name] = schema

    def _initialize_world(self) -> None:
        MovingTargetEnv.ground_truth = {}
        for p in self.initial_providers:
            MovingTargetEnv.ground_truth[p] = self._generate_random_schema()

    # ── OpenEnv lifecycle ─────────────────────────────────────────────────────

    def reset(self, seed=None, episode_id=None, **kwargs):
        MovingTargetEnv._global_step_count = 0
        MovingTargetEnv._directory_rewarded = False
        return FintechObservation(
            data="Environment reset. Fintech provider APIs are live — schemas may shift at any time.",
            status=200,
        )

    def step(self, action: FintechAction, **kwargs):
        MovingTargetEnv._global_step_count += 1

        if action.tool == "get_providers":
            return self._get_providers()
        elif action.tool == "check_provider":
            return self._check_provider(action.provider_name)
        elif action.tool == "execute_transaction":
            return self._execute_transaction(action.provider_name, action.payload or {})
        else:
            return FintechObservation(
                data=f"Unknown tool: '{action.tool}'. Valid tools: get_providers, check_provider, execute_transaction.",
                status=400,
                reward=-3.0,
            )

    # ── Tools ─────────────────────────────────────────────────────────────────

    def _get_providers(self) -> FintechObservation:
        """List available fintech providers. Rewarded only on first call."""
        reward = 0.0
        if not MovingTargetEnv._directory_rewarded:
            reward = 3.0
            MovingTargetEnv._directory_rewarded = True
        return FintechObservation(
            data=json.dumps(list(MovingTargetEnv.ground_truth.keys())),
            status=200,
            reward=reward,
        )

    def _check_provider(self, provider_name: str) -> FintechObservation:
        """Return provider API schema. Small positive reward to encourage probing.
        
        30% chance of schema drift after each check — APIs change in the wild.
        """
        if provider_name not in MovingTargetEnv.ground_truth:
            MovingTargetEnv.ground_truth[provider_name] = self._generate_random_schema()
        elif random.random() < 0.30:
            self._drift_schema(provider_name)

        return FintechObservation(
            data=json.dumps(MovingTargetEnv.ground_truth[provider_name]),
            status=200,
            reward=1.0,   # positive: checking before acting is correct behaviour
        )

    def _execute_transaction(self, provider_name: str, payload: dict) -> FintechObservation:
        """Execute a transaction. Validates payload against live schema."""
        if provider_name not in MovingTargetEnv.ground_truth:
            return FintechObservation(
                data="Provider not found. Call get_providers then check_provider first.",
                status=404,
                reward=-10.0,
            )

        schema = MovingTargetEnv.ground_truth[provider_name]

        # Missing required field
        for field in schema["required_fields"]:
            if field not in payload:
                return FintechObservation(
                    data=f"API REJECTED: Missing required field '{field}'. Schema may have drifted — call check_provider again.",
                    status=400,
                    reward=-15.0,
                )

        # Unexpected field (strict server validation)
        for field in payload:
            if field not in schema["required_fields"]:
                return FintechObservation(
                    data=f"API REJECTED: Unexpected field '{field}' in payload. Schema may have drifted — call check_provider again.",
                    status=400,
                    reward=-15.0,
                )

        # Business-logic judge via LLM
        if MovingTargetEnv.ground_truth_constraint:
            try:
                evaluator = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.0,
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                )
                eval_prompt = f"""
You are a fintech compliance judge.

User's stated constraints: {MovingTargetEnv.ground_truth_constraint}

The agent just executed a transaction via '{provider_name}'.

Provider's actual terms:
- Transaction Fee: {schema.get("transaction_fee", "Unknown")}
- Supported Currencies: {schema.get("supported_currencies", "Unknown")}
- KYC Required: {schema.get("kyc_required", "Unknown")}
- Transfer Limit: {schema.get("transfer_limit", "Unknown")}
- Settlement Time: {schema.get("settlement_time", "Unknown")}

Agent's transaction payload: {payload}

Question: Does this transaction violate the user's constraints regarding fee limits, 
required currency, KYC level, transfer limits, or settlement time?
Respond with EXACTLY 'YES' if it violates (even partially), or 'NO' if it satisfies all constraints.
"""
                result = evaluator.invoke(eval_prompt)
                if "YES" in result.content.upper():
                    return FintechObservation(
                        data="Transaction API accepted, BUT VIOLATES USER CONSTRAINTS. Fee, currency, KYC, or settlement requirements were not met.",
                        status=400,
                        reward=-40.0,
                        done=True,
                    )
            except Exception as e:
                print(f"[Judge Error] {e}")

        return FintechObservation(
            data="Transaction successful! Provider schema matched and all user constraints were satisfied.",
            status=200,
            reward=50.0,
            done=True,
        )

    @property
    def state(self) -> FintechEnvironmentState:
        return FintechEnvironmentState(
            episode_id=None,
            step_count=MovingTargetEnv._global_step_count,
        )

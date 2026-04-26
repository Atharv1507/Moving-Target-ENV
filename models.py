from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class FintechAction(BaseModel):
    """Action for the Fintech Transaction environment."""
    tool: str = Field(description="Tool to call: 'get_providers', 'check_provider', or 'execute_transaction'")
    provider_name: str = Field(default="", description="Name of the fintech provider")
    payload: Optional[Dict[str, Any]] = Field(default=None, description="Transaction payload (for execute_transaction)")


class FintechObservation(BaseModel):
    """Observation returned by the Fintech environment."""
    data: str = Field(default="", description="Response data from the environment")
    status: int = Field(default=200, description="HTTP-like status code")
    reward: Optional[float] = Field(default=None, description="Reward scalar for RL.")
    done: bool = Field(default=False, description="Flag indicating if the episode is finished.")


class FintechEnvironmentState(BaseModel):
    """The literal state of the environment server."""
    episode_id: Optional[str] = Field(default=None)
    step_count: int = Field(default=0)


# Legacy aliases so any remaining imports don't break during migration
MovingTargetAction = FintechAction
MovingTargetObservation = FintechObservation
MovingTargetEnvironmentState = FintechEnvironmentState

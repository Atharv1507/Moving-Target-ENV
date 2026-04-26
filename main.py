"""LangGraph simulation entry point for the Fintech Payment Agent."""
import os
import re
import requests
import dotenv
dotenv.load_dotenv()

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from state import AgentState
from personaAgent import persona_node
from concierge import concierge_node, tools
from watchdog import watchdog_node

_ENV_PORT = int(os.getenv("ENV_SERVER_PORT", "8001"))
_SERVER_URL = f"http://localhost:{_ENV_PORT}"


def route_concierge_output(state: AgentState):
    """Route to tool executor if the LLM issued a tool call, else end the episode."""
    messages = state.get("messages", [])
    if not messages:
        return END
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tools"
    return END


# ── Build graph ───────────────────────────────────────────────────────────────

workflow = StateGraph(AgentState)

workflow.add_node("persona",    persona_node)
workflow.add_node("concierge",  concierge_node)
workflow.add_node("tools",      ToolNode(tools))
workflow.add_node("watchdog",   watchdog_node)

workflow.add_edge(START, "persona")
workflow.add_edge("persona", "concierge")

workflow.add_conditional_edges(
    "concierge",
    route_concierge_output,
    {"tools": "tools", END: END},
)

# After tool execution, pass through watchdog to detect schema drift, then loop back
workflow.add_edge("tools", "watchdog")
workflow.add_edge("watchdog", "concierge")

app = workflow.compile()


# ── Main simulation loop ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n[SYSTEM] Starting Fintech Payment Agent LangGraph Simulation...")
    print(f"[SYSTEM] Environment server expected at {_SERVER_URL}\n")

    EPISODES = 5
    lifelong_memory: dict = {}
    total_lifetime_score = 0.0
    prev_episode_summary = ""

    for episode in range(EPISODES):
        print(f"\n{'=' * 55}")
        print(f"               STARTING EPISODE {episode + 1}/{EPISODES}")
        print(f"{'=' * 55}\n")

        print(f"[SYSTEM] Resetting environment at {_SERVER_URL}/reset ...")
        try:
            requests.post(f"{_SERVER_URL}/reset", timeout=10)
        except Exception as e:
            print(f"[SYSTEM ERROR] Could not reach server: {e}")
            break

        initial_state: AgentState = {
            "messages":             [],
            "current_provider":     "",
            "last_known_schema":    lifelong_memory,
            "drift_detected":       False,
            "reward_score":         0.0,
            "prev_episode_summary": prev_episode_summary,
            "step_count":           0,
        }

        episode_score = 0.0
        context_insight: list[str] = []

        for output in app.stream(initial_state, stream_mode="updates"):
            for node_name, state_update in output.items():
                print(f"\n--- Output from Node: {node_name} ---")

                if "messages" in state_update:
                    last_msg = state_update["messages"][-1]

                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        tc = last_msg.tool_calls[0]
                        print(f"[Tool Call]: {tc['name']}({tc.get('args', {})})")

                    elif hasattr(last_msg, "name") and getattr(last_msg, "name") in [t.name for t in tools]:
                        content_str = last_msg.content.strip()
                        print(f"[Tool Result]: {content_str}")

                        match = re.search(r"\(Environment Reward: ([-\d.]+)\)", content_str)
                        if match:
                            curr_score = float(match.group(1))
                            episode_score += curr_score
                            if curr_score <= -15.0:
                                reason = (
                                    content_str
                                    .split("\n(Environment")[0]
                                    .replace("Observation: ", "")
                                    .strip()
                                )
                                context_insight.append(
                                    f"Tool '{last_msg.name}' cost {abs(curr_score):.0f} pts: {reason}"
                                )
                    else:
                        print(f"[{node_name.capitalize()}]: {last_msg.content}")

                if state_update.get("drift_detected"):
                    print("--> WARNING: Watchdog flagged schema drift!")

                if "last_known_schema" in state_update:
                    lifelong_memory = state_update["last_known_schema"]

        print(f"\n[SYSTEM] Episode {episode + 1} complete. Score: {episode_score:.1f}")
        print(f"[SYSTEM] Insights: {context_insight or ['No major mistakes.']}")
        total_lifetime_score += episode_score

        rl_insights = "\n- ".join(context_insight) if context_insight else "No major mistakes."
        prev_episode_summary = (
            f"In episode {episode + 1} you scored {episode_score:.1f} points.\n"
            f"Mistakes:\n- {rl_insights}\n"
            "Avoid repeating these mistakes next episode."
        )

    print(f"\n{'=' * 55}")
    print(f"[SYSTEM] ALL EPISODES COMPLETE. TOTAL SCORE: {total_lifetime_score:.1f}")
    print(f"{'=' * 55}\n")

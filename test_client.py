import sys
import asyncio
from Moving_Target.client import MovingTargetEnv
from Moving_Target.models import MovingTargetAction

async def main():
    print("Connecting to OpenEnv Server on localhost:8000...")
    
    # We initialize the client to point to the server we started
    client = MovingTargetEnv(base_url="http://localhost:8000")
    
    try:
        # 1. Always reset the environment at the beginning of a session
        print("\n--- Resetting Environment ---")
        reset_result = await client.reset()
        print(f"Observation Data: {reset_result.observation.data}")
        print(f"Observation Status: {reset_result.observation.status}")

        # 2. Let's use the Scout Tool (ask_watchdog)
        print("\n--- Testing Scout Tool: ask_watchdog ---")
        scout_action = MovingTargetAction(tool="ask_watchdog", merchant_name="VeganBistro")
        scout_result = await client.step(scout_action)
        print(f"Observation Data: {scout_result.observation.data}")
        print(f"Observation Status: {scout_result.observation.status}")

        # 3. Let's test the Executioner Tool with a bad payload
        print("\n--- Testing Executioner Tool: place_order (BAD payload) ---")
        bad_order_action = MovingTargetAction(
            tool="place_order", 
            merchant_name="VeganBistro", 
            payload={"item": "Salad"} # missing price
        )
        bad_result = await client.step(bad_order_action)
        print(f"Observation Data: {bad_result.observation.data}")
        print(f"Observation Status: {bad_result.observation.status}")
        
    except Exception as e:
        print(f"Connection Error: {e}")
        print("Make sure your server is running via `python -m server.app`")
    finally:
        # Keep things tidy by closing out the connection
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())

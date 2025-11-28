import logging
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    #RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel
import json, os
from datetime import datetime

logger = logging.getLogger("agent")

load_dotenv(".env.local")

LOG_FILE = "wellness_log.json"
FRAUD_DB = "fraud_cases.json"



def load_sessions():
    if not os.path.exists(LOG_FILE):
        return {"sessions": []}
    with open(LOG_FILE, "r") as f:
        return json.load(f)


def save_session(entry):
    data = load_sessions()
    data["sessions"].append(entry)
    with open(LOG_FILE, "w") as f:
        json.dump(data, f, indent=2)


class Assistant(Agent):
    previous = load_sessions()["sessions"][-1] if load_sessions()["sessions"] else None
    previous_note = ""
    if previous:
        previous_note = f"""
          Here is the previous session data you may reference:
 
        - Previous mood: {previous["mood"]}
        - Previous energy: {previous["energy"]}
        - Previous goals: {', '.join(previous["goals"])}
        - Previous summary: {previous["summary"]}
         Please reference 1 small item from the previous day when greeting the user."""
    def __init__(self) -> None:
        super().__init__(
            instructions = """
You are GroceryGenie, a friendly food & grocery ordering voice assistant for the fictional store FreshBasket.

Your responsibilities:
1. Load the product catalog from `catalog.json`.
2. Maintain an in-memory cart during a session.
3. Understand user intents:
   - Add items to cart (with quantity, size, brand).
   - Remove or update items.
   - List cart contents.
   - Add multiple items for "ingredients for X" requests.
4. Place the order when the user is done.
5. Save all orders to `orders.json` (append to list).
6. Respond conversationally, confirm actions clearly.

-----------------------------
CATALOG HANDLING
-----------------------------
• On startup, read `catalog.json` and store items internally.
• Each catalog item includes:
  - id, name, category, price
  - optional attributes: brand, size, tags

If the user requests an item not in the catalog:
    → Ask clarifying questions or offer alternatives.

-----------------------------
CART RULES
-----------------------------
• Cart is a Python dict:
  {
    "items": [
      { "id": ..., "name": ..., "qty": ..., "price": ..., "notes": ... }
    ],
    "total": ...
  }

• When adding items:
    → Confirm: “Added 2 cartons of milk to your cart.”

• When removing:
    → Confirm: “Removed eggs from your cart.”

• When listing:
    → If empty: “Your cart is empty.”
    → Else list all items + total.

-----------------------------
INGREDIENT INTELLIGENCE
-----------------------------
You support high-level requests like:
  “I need ingredients for a peanut butter sandwich.”
  “Add ingredients for pasta for two.”

Use the following built-in recipe mapping:

RECIPES = {
    "peanut butter sandwich": ["bread", "peanut butter"],
    "grilled cheese": ["bread", "cheese"],
    "pasta": ["pasta", "pasta sauce"],
    "salad": ["lettuce", "tomatoes", "olive oil"],
}

• When user asks for ingredients:
     → Identify the recipe.
     → Add all mapped catalog items.
     → Quantities default to 1 unless user specifies servings.
     → Confirm verbally.

-----------------------------
ORDER PLACEMENT
-----------------------------
When user says: “place my order”, “that’s all”, “I’m done”, etc.:

1. Read current cart.
2. Summarize items + total to user.
3. Create an order object:

{
  "order_id": <generated integer>,
  "timestamp": <ISO string>,
  "items": [...],
  "total": <float>,
  "status": "received"
}

4. Save it to `orders.json`:
   • If file exists → append.
   • Else → create with a list containing the order.

5. Clear the cart and confirm:
   “Your order has been placed! You’ll receive updates soon.”

-----------------------------
PERSONALITY & ASSISTANT BEHAVIOR
-----------------------------
• Friendly, clear, concise.
• Ask follow-up clarifying questions.
• When uncertain about item size/brand → ask.
• Never hallucinate items not in catalog.
• Use conversational tone suitable for voice.

-----------------------------
ERROR HANDLING
-----------------------------
• If user asks for an unavailable item:
      → Suggest similar items.
• If user asks to remove something not in cart:
      → Say: “I don’t see that in your cart.”

-----------------------------
END OF SYSTEM BEHAVIOR
-----------------------------
"""

)
    
    
    
   
    #Use this tool to look up current weather information in the given location.
    #
    #     If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.
    #
    #     Args:
    #         location: The location to look up weather information for (e.g. city name)
    #     """
    #
    #     logger.info(f"Looking up weather for {location}")
    #
    #     return "sunny with a temperature of 70 degrees."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


@function_tool
def load_fraud_case(username: str) -> dict:
    """Load the user fraud profile from JSON DB."""
    import json
    if not os.path.exists(FRAUD_DB):
        return {"error": "DB_NOT_FOUND"}

    with open(FRAUD_DB, "r") as f:
        db = json.load(f)

    for user in db["users"]:
        if user["username"].lower() == username.lower():
            return user

    return {"error": "USER_NOT_FOUND"}


@function_tool
def update_fraud_case(username: str, status: str, notes: str) -> dict:
    """Update the fraud case outcome in JSON DB."""
    import json
    if not os.path.exists(FRAUD_DB):
        return {"error": "DB_NOT_FOUND"}

    with open(FRAUD_DB, "r") as f:
        db = json.load(f)

    updated = False
    for user in db["users"]:
        if user["username"].lower() == username.lower():
            user["fraud_case"]["status"] = status
            user["fraud_case"]["notes"] = notes
            updated = True

    if not updated:
        return {"error": "USER_NOT_FOUND"}

    with open(FRAUD_DB, "w") as f:
        json.dump(db, f, indent=2)

    return {"status": "SUCCESS"}

async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=deepgram.STT(model="nova-3"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=google.LLM(
                model="gemini-2.5-flash",
            ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=murf.TTS(
                voice="en-US-matthew", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )
   
    

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()
    
    

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

import logging
import uuid
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

ORDERS_FILE = "orders.json"



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
        

try:
    with open(ORDERS_FILE, "r") as f:
        ORDERS = json.load(f)
except:
     ORDERS = []        
        

      


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
You are a voice-driven shopping assistant.  
Your job is to help the user browse products, understand their shopping intent, 
and create orders using the provided merchant functions.

You MUST follow these rules:

1. **Never invent products.**  
   Only reference products returned by the `list_products` tool.

2. **All catalog and order logic MUST be done through tools**  
   – list_products  
   – create_order  
   – get_last_order  
   You should NOT fabricate catalog data or compute totals manually.

3. **Interpret natural language shopping intent.**
   Examples:
   - “Show me mugs under 900.”
   - “Do you have black hoodies?”
   - “I’ll take the second one.”
   - “What did I just buy?”

4. **When needed, automatically translate user requests into tool calls.**  
   Always explain your tool calls to the user in natural language after you receive the tool result.

5. **Product references must be grounded.**
   - If the user says “the second hoodie”, resolve it from the last list of products you showed.
   - If unclear, ask politely for clarification.

6. **For orders:**
   - Resolve the exact product ID(s)
   - Include quantity (default 1 if user doesn’t specify)
   - Create the order using the `create_order` tool
   - Confirm the order details to the user

7. **For last order queries:**
   - Use `get_last_order`
   - Summarize the order in simple terms

8. **Be conversational, friendly, and concise.**
   - This agent is voice-first.
   - Keep responses short and clear.

9. **Never output JSON unless returning a tool call.**
   When talking to the user, speak naturally.

Your goal:
Provide a smooth, voice-friendly shopping flow that mirrors an ACP-inspired commerce workflow.

tools=[
    {
        "name": "list_products",
        "description": "List products with optional filters",
        "parameters": {
            "type": "object",
            "properties": {
                "filters": {
                    "type": "object",
                    "nullable": True
                }
            }
        }
    },
    {
        "name": "create_order",
        "description": "Create an order from line items",
        "parameters": {
            "type": "object",
            "properties": {
                "line_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "product_id": {"type": "string"},
                            "quantity": {"type": "number"}
                        },
                        "required": ["product_id"]
                    }
                }
            },
            "required": ["line_items"]
        }
    },
    {
        "name": "get_last_order",
        "description": "Retrieve the last order created",
        "parameters": {"type": "object"}
    }
],

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

CATALOG = [
    {
        "id": "mug-001",
        "name": "Stoneware Coffee Mug",
        "description": "Handmade ceramic mug",
        "price": 800,
        "currency": "INR",
        "category": "mug",
        "color": "white",
    },
    {
        "id": "mug-002",
        "name": "Blue Ceramic Mug",
        "description": "Glossy blue finish",
        "price": 950,
        "currency": "INR",
        "category": "mug",
        "color": "blue",
    },
    {
        "id": "hoodie-001",
        "name": "Black Cotton Hoodie",
        "description": "Unisex, soft cotton",
        "price": 1600,
        "currency": "INR",
        "category": "hoodie",
        "color": "black",
        "sizes": ["S", "M", "L", "XL"]
    },
    {
        "id": "tshirt-001",
        "name": "Graphic T-Shirt",
        "description": "Printed tee",
        "price": 700,
        "currency": "INR",
        "category": "tshirt",
        "color": "white",
        "sizes": ["M", "L", "XL"]
    },
]  
@function_tool
def list_products(filters=None):
    results = CATALOG
    if filters:
        for key, value in filters.items():
            if key == "max_price":
                results = [p for p in results if p["price"] <= value]
            else:
                results = [p for p in results if p.get(key) == value]
    return results


@function_tool
def create_order(line_items):
    total = 0
    resolved_items = []

    for item in line_items:
        p = next((x for x in CATALOG if x["id"] == item["product_id"]), None)
        if not p:
            raise ValueError(f"Product {item['product_id']} not found")

        qty = item.get("quantity", 1)
        resolved_items.append({
            "product_id": p["id"],
            "name": p["name"],
            "quantity": qty,
            "unit_amount": p["price"],
            "currency": p["currency"]
        })
        total += p["price"] * qty

    order = {
        "id": str(uuid.uuid4()),
        "items": resolved_items,
        "total": total,
        "currency": "INR",
        "created_at": datetime.utcnow().isoformat()
    }

    ORDERS.append(order)
    with open(ORDERS_FILE, "w") as f:
        json.dump(ORDERS, f, indent=2)

    return order

@function_tool
def get_last_order():
    if not ORDERS:
        return None
    return ORDERS[-1]

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

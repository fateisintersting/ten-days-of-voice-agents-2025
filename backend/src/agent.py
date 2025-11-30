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
            instructions  = """
You are the AI host of a high-energy TV improv game show called **“Improv Battle!”**  
You speak directly to the player over voice.  
Your job is to run a structured improv game with 3–5 rounds, providing scenarios, listening to the player's improv, and reacting realistically.

===========================
## 🔥 HOST PERSONALITY
===========================
- High-energy, witty, charismatic.
- You tease lightly, praise when earned, critique when needed.
- You are never abusive or insulting.
- You treat this like a real TV improv competition.

Your reactions should:
- Sometimes be supportive.
- Sometimes be neutral or unimpressed.
- Sometimes be mildly critical.
Randomly vary your tone each round.

===========================
## 🎮 GAME STATE RULES
===========================
The backend maintains a Python dict named `improv_state` with fields:
{
  "player_name": str or None,
  "current_round": int,
  "max_rounds": int,
  "rounds": [],            # each: {"scenario": str, "host_reaction": str}
  "phase": str             # "intro" | "awaiting_improv" | "reacting" | "done"
}

Your behavior depends on `phase` :

---------------------------
### 🟦 phase="intro"
---------------------------
- Introduce the show.
- Ask for the player's name **if not provided**.
- Explain the rules:
  - There will be several improv rounds.
  - You will give a scenario.
  - The player performs the improv.
  - You react!
- Once intro is done, set:
    improv_state["phase"] = "awaiting_improv"
    improv_state["current_round"] = 0
- Then immediately launch Round 1 by giving the player the first scenario.

---------------------------
### 🟨 phase="awaiting_improv"
---------------------------
- You have already given the scenario.
- Wait for the player's improv.
- When the player finishes OR says a clear stop phrase like “end scene”, “okay I'm done”, etc., then:
    improv_state["phase"] = "reacting"
- Do **not** give another scenario yet.
- Do **not** critique yet.

---------------------------
### 🟥 phase="reacting"
---------------------------
- Deliver a realistic reaction to the player's performance.
- Mention specific things they did.
- Mix praise + critique with varied tone.
- Store your reaction in improv_state["rounds"].

Then:
    improv_state["current_round"] += 1

If current_round < max_rounds:
    - Start the next scenario.
    - Set phase back to "awaiting_improv"
Else:
    - Move to final summary.
    improv_state["phase"] = "done"

---------------------------
### 🟩 phase="done"
---------------------------
- Give a short but lively closing monologue summarizing:
  - The player's overall improv style.
  - Memorable moments from previous scenes.
- Thank them for playing Improv Battle.
- Do not start another round.

===========================
## 🎭 SCENARIO GENERATION
===========================
Each scenario must:
- Give the player a character or role.
- Provide a conflict or tension.
- Invite emotional or comedic acting.

Examples:
- “You are a restaurant server who must confess that the customer’s soup has achieved sentience.”
- “You are a medieval knight trying to convince a dragon to become your life coach.”

Generate scenarios yourself. Make each one unique.

After giving a scenario say:  
**“Alright, take it away!”**

===========================
## ⏹ EARLY EXIT HANDLING
===========================
If the user says something like:
- “stop game”
- “end show”
- “I want to quit”
Then:
- Confirm they want to exit.
- If yes, gracefully end with a short farewell.
- Set improv_state["phase"] = "done".

===========================
## 🔊 GENERAL SPEAKING RULES
===========================
- Keep host lines 1–3 sentences.
- No long monologues except the final summary.
- Always speak in a TV-host tone.
- Make the player feel like they are on a real improv stage.

===========================
## 🧠 STATE INTERPRETATION
===========================
You **must** respond according to the current improv_state.  
Do not start new rounds or scenarios unless the phase requires it.  
Do not react before the player performs.  
Never contradict the stored state.

===========================
## 🚀 READY TO HOST
===========================
Begin hosting **only when the session starts**.  
Honor the state machine.  
Stay in character as the host of **IMPROV BATTLE!**
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

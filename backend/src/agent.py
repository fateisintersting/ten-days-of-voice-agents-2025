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
You are a Fraud Alert Voice Agent for a fictional bank called **Aurora Bank**.

Your job is to handle a suspicious-transaction review for a customer.  
All data is FAKE.  
Never ask for sensitive information such as full card numbers, PINs, passwords, SSNs, or anything confidential.  
Use only the security question stored inside the database entry.

===========================================================
                 🎯  AGENT BEHAVIOR RULES
===========================================================

1. INTRODUCTION
   - When the session starts, greet the customer calmly and professionally.
   - Say you are calling from Aurora Bank’s Fraud Prevention Department.
   - Explain you detected an unusual transaction and need to verify the account holder.

2. DATABASE INTERACTION
   - When the session begins:
       -> Ask the user for their username.
       -> Load the matching fraud case from the database (provided by backend).
       -> Store the loaded case in memory for the entire call.
   - Never ask for or handle real card data.

3. VERIFICATION FLOW
   - Ask only ONE non-sensitive verification question pulled from the fraud case:
       -> (e.g., “What is the name of the city you were born in?”)
   - If the customer answers correctly:
         -> Proceed to suspicious-transaction explanation.
   - If verification fails:
         -> Inform them politely that verification did not succeed.
         -> Mark the database case as "verification_failed".
         -> End the session.

4. SUSPICIOUS TRANSACTION DETAILS
   After verification:
     - Read aloud the merchant name, transaction amount, masked card number,
       location, and timestamp from the loaded fraud case.
     - Ask: “Did you make this transaction? Yes or no?”

5. DECISION HANDLING
   - If user says YES (they made the transaction):
         -> Mark case as “confirmed_safe”.
         -> Add note: “Customer confirmed the transaction.”
   - If user says NO (they did not make the transaction):
         -> Mark case as “confirmed_fraud”.
         -> Add note: “Customer denied the transaction; card blocked and dispute started (mock).”

6. DATABASE UPDATE
   - At the end of the call:
       -> Call the backend to save the updated fraud case.
       -> Always log the final status and summary.

7. TONE & SAFETY
   - Use professional, calm, reassuring language.
   - Never request:
         * Full card number
         * PIN
         * Passwords
         * SSN
         * CVV
         * Secret credentials
   - Only use data that already exists in the preloaded fraud case.

8. END OF CALL
   - Clearly summarize what action was taken.
   - Thank the customer.
   - End the session smoothly.

===========================================================
                  🎤  CALL FLOW SUMMARY
===========================================================
1. Intro  
2. Ask for username  
3. Load fraud case  
4. Verification question  
5. If fail → update DB → end  
6. If pass → read suspicious transaction  
7. Ask: “Did you make this transaction?”  
8. Update DB based on yes/no  
9. Confirm action  
10. End call

Follow this workflow exactly unless instructed otherwise by the developer.
tools:[
    {
  "tool": "update_fraud_case",
  "arguments": {
    "username": "John",
    "status": "confirmed_safe",
    "notes": "Customer confirmed the transaction."
  }
},
{
  "tool": "update_fraud_case",
  "arguments": {
    "username": "John",
    "status": "confirmed_fraud",
    "notes": "Customer denied the transaction; dispute started (mock)."
  }
}
]

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

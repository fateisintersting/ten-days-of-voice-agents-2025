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
            instructions= """
You are Zomato’s SDR (Sales Development Representative) voice agent.  
Your role is to greet users warmly, understand their needs, answer questions strictly using provided company content, collect lead information, and produce a summary at the end of the call.

========================
COMPANY OVERVIEW (Zomato)
========================
Zomato is an Indian multinational food-tech company specializing in:
- Online food ordering & delivery  
- Restaurant discovery, menus, reviews  
- Table reservations  
- Contactless dining  
- B2B services such as “Zomato for Enterprise”

Founded in 2008 as Foodiebay by Deepinder Goyal and Pankaj Chaddah, rebranded as Zomato in 2010.  
Mission: “Better food for more people.”  
Active across India and multiple global markets.

========================
BASIC FAQ DATA
========================
Q: What does Zomato do?  
A: Zomato connects customers with restaurants and delivery partners through food delivery, restaurant discovery, and dining services.

Q: Who is Zomato for?  
A: Anyone looking to order food online, discover restaurants, book tables, or use enterprise dining solutions.

Q: Do you have a free tier?  
A: Using the Zomato app for browsing restaurants and placing orders is free. Charges apply for orders, delivery, and premium programs if selected.

Q: Do restaurants pay to list?  
A: Basic listing is free; additional promotional services may be available (only mention if asked).

Q: Do you operate outside India?  
A: Yes, Zomato has had an international presence in several countries.

========================
AGENT BEHAVIOR RULES
========================
1. Greet warmly and ask what brought them here.
2. Keep the conversation focused on understanding their needs.
3. When user asks about the company:
   - Search within the supplied FAQ text (simple keyword match).
   - Answer ONLY from this content; do NOT invent details.
4. Gradually collect lead details in a natural conversation:
   - Name
   - Company
   - Email
   - Role
   - Use case (why they’re interested)
   - Team size
   - Timeline (Now / Soon / Later)
5. Store these fields internally in JSON form as user replies:
   {
     "name": "",
     "company": "",
     "email": "",
     "role": "",
     "use_case": "",
     "team_size": "",
     "timeline": ""
   }
6. When the user indicates the call is ending ("that’s all", "done", "thanks"):
   - Give a short verbal summary of who they are, what they need, and timeline.
   - Output the final collected JSON.

========================
TONE & STYLE
========================
Friendly, concise, professional.  
Never guess or fabricate information outside the provided company details.  
Always aim to understand user goals and move toward lead qualification.

You are now ready to act as Zomato’s SDR.
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

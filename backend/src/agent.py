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


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""Role & Persona
You are BrewBuddy, a friendly, warm, conversational barista for the coffee brand Moonbeam Coffee Roasters.
Your tone is friendly, upbeat, and helpful—like a real barista taking a customer’s order while keeping things efficient.

Core Responsibilities

Maintain and update an order state object with this structure:

{
  "drinkType": "string",
  "size": "string",
  "milk": "string",
  "extras": ["string"],
  "name": "string"
}


Guide the user through the entire order, asking clarifying questions only for missing or unclear fields.

Never assume details unless the user clearly states them.

Confirm final order before saving.

Once complete, call the Python function save_order_to_json(order: dict) containing the final order.

After saving, present a neat text summary of the order for the customer.

Conversation Flow Rules
1. Start

Greet the customer.

Ask for the first missing field (drink type).

2. State Completion Logic

For each field in the order:

If missing → ask a clarifying question.

If ambiguous → ask for confirmation.

If user gives multiple details at once → update all relevant fields.

3. Order Field Requirements

drinkType: e.g., latte, cappuccino, cold brew, mocha, americano

size: small, medium, large

milk: whole, oat, soy, almond, 2%, no milk

extras: syrups, sweeteners, toppings, extra shots (array)

name: customer’s preferred name

4. At All Times

Speak as a friendly barista.

Keep responses concise and natural, like actual order-taking.

If the user changes a detail mid-order, update the order state.

5. Order Completion

When all fields are filled:

Read back the final order.

Ask for confirmation: “Does everything look right?”

On confirmation, call:

save_order_to_json(order_state)
after that
generate_drink_html(order_state)

Then send the customer a warm thank-you message and a clean summary.

tools = [
    {
        "name": "save_order_to_json",
        "description": "Save the completed coffee order to a JSON file.",
        "parameters": {
            "type": "object",
            "properties": {
                "order": {"type": "object"}
            },
            "required": ["order"]
        }
    },
    {
        "name": "generate_drink_html",
        "description": "Create the Html Img fo Coffe.",
        "parameters": {
            "type": "object",
            "properties": {
                "order": {"type": "object"}
            },
            "required": ["order"]
        }
    }
]
""",
        )
        order_state = {
        "drinkType": "",
        "size": "",
        "milk": "",
        "extras": [],
        "name": ""
        }
        
    @function_tool
    async def save_order_to_json(self, order: dict):
        """Save a completed coffee order to a timestamped JSON file."""
        os.makedirs("orders", exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"orders/order_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(order, f, indent=2)
             
        return {
            "status": "success",
            "file": filename
        }
    @function_tool   
    async def generate_drink_html(self,order: dict):
     """Generate a simple HTML visualization of the customer's drink order."""
     size_map = {
        "small": "80px",
        "medium": "120px",
        "large": "160px"
    }

     cup_height = size_map.get(order["size"], "120px")
     show_whipped = "whipped cream" in order.get("extras", [])

     whipped_html = """
    <div class="whipped"></div>
    """ if show_whipped else ""

     html = f"""
    <html>
    <head>
    <style>
        body {{
            background: #f4f1ea;
            font-family: sans-serif;
            text-align: center;
            padding-top: 40px;
        }}
        .cup {{
            width: 60px;
            height: {cup_height};
            background: #c17f43;
            margin: 0 auto;
            border-radius: 0 0 12px 12px;
        }}
        .whipped {{
            width: 0;
            height: 0;
            border-left: 25px solid transparent;
            border-right: 25px solid transparent;
            border-bottom: 40px solid white;
            margin: 0 auto;
        }}
    </style>
    </head>
    <body>
        <h2>{order["size"].capitalize()} {order["drinkType"].capitalize()}</h2>
        {whipped_html}
        <div class="cup"></div>
    </body>
    </html>
    """

    # Save to disk
     with open("drink_preview.html", "w") as f:
        f.write(html)

     return {"status": "success", "file": "drink_preview.html"}


    # To add tools, use the @function_tool decorator.
    # Here's an example that adds a simple weather tool.
    # You also have to add `from livekit.agents import function_tool, RunContext` to the top of this file
    # @function_tool
    # async def lookup_weather(self, context: RunContext, location: str):
    #     """Use this tool to look up current weather information in the given location.
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

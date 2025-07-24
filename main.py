import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel
import chainlit as cl

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-2.0-flash"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Set up model
client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=BASE_URL)
model = OpenAIChatCompletionsModel(model=MODEL, openai_client=client)

# Specialized agents
mobile_agent = Agent(
    name="mobile_agent",
    instructions="You're a mobile phone expert. Answer questions about phones and help users choose.",
    model=model,
)

laptop_agent = Agent(
    name="laptop_agent",
    instructions="You're a laptop expert. Help users pick the best laptops based on their needs.",
    model=model,
)

electronics_agent = Agent(
    name="electronics_agent",
    instructions="You are an expert in electronic items. Help users find gadgets like phones, laptops, headphones, etc.",
    model=model
)

clothing_agent = Agent(
    name="clothing_agent",
    instructions="You are a clothing expert. Help users find dresses, jeans, shirts, etc., based on their style and needs.",
    model=model
)

groceries_agent = Agent(
    name="groceries_agent",
    instructions="You are a grocery shopping expert. Suggest grocery items based on the user's needs.",
    model=model
)


# Main router agent
shopping_agent = Agent(
    name="shopping_agent",
    instructions="""
    Greet the user and ask what product they are looking for.
    Route the question to the right sub-agent: mobile_agent, laptop_agent, groceries_agent, electronics_agent, clothing_agent.
    If the product doesn't match, politely inform the user.
    """,
    model=model,
    handoffs=[mobile_agent, laptop_agent, groceries_agent, electronics_agent, clothing_agent],
)

# Show greeting on chat start
@cl.on_chat_start
async def on_chat_start():
     await cl.Message(content="🛍️ Welcome! I am your Shopping Assistant!\n\n🤔 What product are you looking for? (e.g., mobile, laptop, groceries, electronics or clothings)").send()


# Handle user questions
@cl.on_message
async def handle_message(message: cl.Message):
    result = await Runner.run(starting_agent=shopping_agent, input=message.content)
    await cl.Message(content=result.final_output).send()

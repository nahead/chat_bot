from agents import OpenAIChatCompletionsModel , Agent,Runner
from openai import AsyncOpenAI
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import os
load_dotenv()
base_url=os.getenv('base_url')
gemini_api = os.getenv('gemini_api')

client=AsyncOpenAI(
    api_key=gemini_api,
    base_url=base_url
)
model=OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client 
)

agent=Agent(
    name='assistant',
    instructions='your helpfull assistant',
    model=model
)


app = FastAPI()

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)   

@app.get("/")
def home():
    return {"message": "Hello from Nahead jokhio"}

class Message(BaseModel):
    message: str

@app.post("/chat")
async def chat(message: Message):
    result=await Runner.run(
        agent,
        message.message
    )
    return {
        "response": result.final_output
    }
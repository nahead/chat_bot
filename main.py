# main.py (Render ke liye updated)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agents import OpenAIChatCompletionsModel, Agent, Runner
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

base_url = os.getenv('base_url')
gemini_api = os.getenv('gemini_api')

client = AsyncOpenAI(
    api_key=gemini_api,
    base_url=base_url
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

agent = Agent(
    name='assistant',
    instructions='You are a helpful AI assistant. Provide clear, concise, and helpful responses.',
    model=model
)

app = FastAPI(title="AI Chat API", version="1.0.0")

# Render deployment ke liye CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000", 
        "https://your-frontend-domain.vercel.app",  # Aapka frontend URL
        "*"  # Temporary - production mein specific URL daalna
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    message: str

@app.get("/")
def home():
    return {"message": "AI Chat API is running successfully!", "status": "active"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "AI Chat API"}

@app.post("/chat")
async def chat(message: Message):
    try:
        print(f"🤖 Processing message: {message.message}")
        result = await Runner.run(agent, message.message)
        response_text = result.final_output
        
        print(f"✅ Response generated: {response_text[:100]}...")
        return {"response": response_text}
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {"response": f"Sorry, I encountered an error: {str(e)}"}

# For Render deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
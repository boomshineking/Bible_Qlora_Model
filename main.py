from fastapi import FastAPI
from pydantic import BaseModel
from model import tokenizer, model
import torch

app = FastAPI()

# ✅ Pydantic input schema
class ChatRequest(BaseModel):
    prompt: str

@app.get("/")
def root():
    return {"message": "🚀 Bible chatbot is up and running!"}

@app.post("/chat")
async def chat(request: ChatRequest):
    prompt = request.prompt

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return {"response": response}

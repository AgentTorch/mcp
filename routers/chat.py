# routers/chat.py
from fastapi import APIRouter, HTTPException, Body
from typing import List
from models.nlp import ChatRequest, Message
from controller import Controller

router = APIRouter(
    prefix="/api/chat",
    tags=["chat"],
)

controller = Controller()

@router.post("/")
async def chat(request: ChatRequest):
    """Process a chat request and return a response"""
    try:
        # Extract the latest message and history
        messages = request.messages
        latest_message = messages[-1].content if messages else ""
        history = [{"role": msg.role, "content": msg.content} for msg in messages[:-1]]
        
        # Process the chat request
        response = await controller.process_chat_request(latest_message, history)
        
        return {
            "response": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
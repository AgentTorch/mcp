# routers/chat.py
from fastapi import APIRouter, HTTPException, Body
from typing import List, Dict, Any
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
        response, simulation_data = await controller.process_chat_request(latest_message, history)
        
        # If simulation data is available, return it as well
        if simulation_data:
            return {
                "response": response,
                "simulationData": simulation_data
            }
        else:
            return {
                "response": response
            }
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        # Return a friendly error message
        return {
            "response": "I'm sorry, but I'm having trouble processing your request right now. Please try again in a moment."
        }
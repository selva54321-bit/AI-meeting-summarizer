from fastapi import FastAPI, HTTPException, WebSocket, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import torch
import sounddevice as sd
import numpy as np
import soundfile as sf
from scipy.io.wavfile import write
import whisper
import os
import asyncio
import json
from datetime import datetime, timedelta
import base64
import io
from langchain import hub
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_openai_functions_agent, AgentExecutor
import threading
import queue
from fastapi.responses import RedirectResponse
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
import json
#hello

# CLIENT_SECRET_FILE = "credentials.json"
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLIENT_SECRET_FILE = os.path.join(BASE_DIR, "credentials.json")

SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/calendar.events"
]

REDIRECT_URI = "http://localhost:8000/auth/callback"

user_tokens = {}  # later move to Mongo

"""Mongo DB"""
from pymongo import MongoClient


# MongoDB Connection
# MONGO_URI = "mongodb://localhost:27017"
MONGO_URI = os.environ.get("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["ai_meeting_db"]
meetings_collection = db["meetings"]
google_tokens_collection = db["google_tokens"]
# google_tokens_collection = db["google_tokens"]


# Import all the required modules from app2.py
from app2 import (
    model,
    build_rag_chain,
    make_tools,
    summerize,
    SCOPES
)

app = FastAPI(title="AI Meeting Assistant API")
# 1️⃣ Serve all files directly from pyfiles
base_path = os.path.dirname(__file__)
app.mount("/pyfiles", StaticFiles(directory=base_path), name="pyfiles")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class TranscriptionRequest(BaseModel):
    audio_data: str  # Base64 encoded audio data

class TranscriptionResponse(BaseModel):
    text: str

class SummaryRequest(BaseModel):
    transcript: str

class SummaryResponse(BaseModel): 
    summary: str

class AgentRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None

class EmailRequest(BaseModel):
    to: str
    subject: str
    body: str

class ReminderRequest(BaseModel):
    task: str
    minutes_from_now: Optional[int] = None
    date: Optional[str] = None
    time: Optional[str] = None

class AgentResponse(BaseModel):
    response: str
    thought_process: Optional[str] = None

class RenameMeeting(BaseModel):
    title: str

# Global state
agent_instances = {}
recordings = {}

# Recording state
recording_state = {
    "is_recording": False,
    "stream": None,
    "audio_data": [],
    "sample_rate": 44100
}

from fastapi.responses import RedirectResponse

@app.get("/")
def root():
    token_doc = google_tokens_collection.find_one({"user": "default"})
    
    if not token_doc:
        return RedirectResponse("/auth/login")
    
    return {"status": "App ready"}

@app.get("/auth/status")
def auth_status():
    token_doc = google_tokens_collection.find_one({"user": "default"})
    if token_doc:
        return {"logged_in": True}
    return {"logged_in": False}

@app.get("/auth/login")
def google_login():
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRET_FILE,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI,
    )

    auth_url, state = flow.authorization_url(
        access_type="offline",
        prompt="consent"
    )

    return RedirectResponse(auth_url)



@app.get("/auth/callback")
def google_callback(code: str):
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRET_FILE,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI,
    )

    flow.fetch_token(code=code)
    creds = flow.credentials

    google_tokens_collection.update_one(
        {"user": "default"},
        {"$set": {"token": creds.to_json()}},
        upsert=True
    )

    # redirect to frontend app
    return RedirectResponse("http://localhost:5173")

@app.post("/api/record/start", response_model=dict)
async def start_recording():
    """
    Start recording system audio using Stereo Mix (if available).
    """
    try:
        if recording_state["is_recording"]:
            raise HTTPException(status_code=400, detail="Recording already in progress")

        # Search for Stereo Mix device
        device_idx = None 
        for i, d in enumerate(sd.query_devices()):
            if "stereo mix" in d["name"].lower():
                device_idx = i
                print(f"✅ Found Stereo Mix at index {i}: {d['name']}")
                break

        if device_idx is None:
            raise HTTPException(status_code=404, detail="Stereo Mix device not found")

        # Recording parameters
        SR = 44100  # Sample Rate
        
        def audio_callback(indata, frames, time, status):
            """Callback function to receive audio data"""
            if status:
                print(f"Audio callback status: {status}")
            recording_state["audio_data"].append(indata.copy())

        # Start recording
        recording_state["stream"] = sd.InputStream(
            samplerate=SR,
            channels=2,
            device=device_idx,
            callback=audio_callback,
            blocksize=1024
        )
        
        recording_state["stream"].start()
        recording_state["is_recording"] = True
        recording_state["audio_data"] = []  # Clear previous data
        
        print("🎙️ Recording started from Stereo Mix...")
        
        return JSONResponse(content={
            "message": "Recording started successfully",
            "status": "recording"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/record/stop", response_model=dict)
async def stop_recording():
    """
    Stop recording and save the audio file.
    """
    try:
        if not recording_state["is_recording"] or not recording_state["stream"]:
            raise HTTPException(status_code=400, detail="No recording in progress")

        # Stop recording
        recording_state["stream"].stop()
        recording_state["stream"].close()
        recording_state["is_recording"] = False
        
        # Save the recorded data
        if recording_state["audio_data"]:
            audio_array = np.concatenate(recording_state["audio_data"], axis=0)
            
            # Normalize to prevent clipping
            if np.max(np.abs(audio_array)) > 0:
                audio_array = audio_array / np.max(np.abs(audio_array))
            
            # Save as WAV file
            OUTPUT_FILE = "system_plus_mic.wav"
            sf.write(OUTPUT_FILE, audio_array, recording_state["sample_rate"])
            print(f"💾 Saved recording as {OUTPUT_FILE}")
            
            duration = len(audio_array) / recording_state["sample_rate"]
            
            return JSONResponse(content={
                "message": "Recording stopped and saved successfully",
                "file_path": os.path.abspath(OUTPUT_FILE),
                "duration_seconds": round(duration, 2),
                "status": "stopped"
            })
        else:
            raise HTTPException(status_code=400, detail="No audio data recorded")

    except Exception as e:
        recording_state["is_recording"] = False
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/record/status", response_model=dict)
async def get_recording_status():
    """
    Get current recording status.
    """
    return {
        "is_recording": recording_state["is_recording"],
        "duration": len(recording_state["audio_data"]) / recording_state["sample_rate"] if recording_state["audio_data"] else 0
    }

@app.post("/api/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe an audio file"""
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{datetime.now().timestamp()}.wav"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Load model and transcribe
        model_w = whisper.load_model("medium", device="cuda" if torch.cuda.is_available() else "cpu")
        result = model_w.transcribe(temp_path,task="translate", language="en")
        
        # Cleanup
        os.remove(temp_path)
        
        return TranscriptionResponse(text=result["text"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/summarize", response_model=SummaryResponse)
async def create_summary(request: SummaryRequest):
    """Generate meeting summary from transcript"""
    try:
        summary = summerize(request.transcript)
        return SummaryResponse(summary=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agent/initialize", response_model=dict)
async def initialize_agent(summary: str, transcript: str):
    """Initialize a new agent instance"""
    try:
        # Create unique ID for this agent instance
        agent_id = str(datetime.now().timestamp())
        
        # Set up the agent
        qa_chain = build_rag_chain(summary, transcript)
        tools = make_tools(qa_chain)
        
        prompt = hub.pull("hwchase17/openai-functions-agent")
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        agent = create_openai_functions_agent(model, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            memory=memory
        )  
        
        # Store the agent instance
        agent_instances[agent_id] = {
            "executor": agent_executor,
            "summary": summary,
            "transcript": transcript
        }

        meeting_data = {
        "meeting_id": agent_id,
        "title": f"Meeting {datetime.now().strftime('%d %b %H:%M')}",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "summary": summary,
        "transcript": transcript,
        "messages": []
        }

        meetings_collection.insert_one(meeting_data)
        return {
            "agent_id": agent_id,
            "status": "initialized"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agent/query", response_model=AgentResponse)
async def query_agent(request: AgentRequest):
    """Query an initialized agent"""
    try:
        agent_id = request.conversation_id
        if not agent_id or agent_id not in agent_instances:
            raise HTTPException(status_code=404, detail="Agent not found")
            
        agent_executor = agent_instances[agent_id]["executor"]
        response = await agent_executor.ainvoke({"input": request.query})
        ai_output=response["output"]

        # Update chat history in MongoDB
        meetings_collection.update_one(
            {"meeting_id": agent_id},
            {
                "$push": {
                    "messages": {
                        "role": "user",
                        "text": request.query,
                        "time": datetime.utcnow()
                    }
                },
                "$set": {"updated_at": datetime.utcnow()}
            }
        )

        meetings_collection.update_one(
            {"meeting_id": agent_id},
            {
                "$push": {
                    "messages": {
                        "role": "assistant",
                        "text": ai_output,
                        "time": datetime.utcnow()
                    }
                },
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        
        return AgentResponse(
            response=response["output"],
            thought_process=response.get("intermediate_steps", None)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#for the left side meeting list
@app.get("/api/meetings")
def get_meetings():
    meetings = meetings_collection.find(
        {},
        {
            "_id": 0,
            "meeting_id": 1,
            "title": 1,
            "created_at": 1,
            "updated_at": 1
        }
    ).sort("updated_at", -1)

    return list(meetings)

@app.get("/api/meetings/{meeting_id}")
def get_meeting(meeting_id: str):
    meeting = meetings_collection.find_one(
        {"meeting_id": meeting_id},
        {"_id": 0}
    )

    if not meeting:
        raise HTTPException(404, "Meeting not found")

    return meeting

@app.put("/api/meetings/{meeting_id}/title")
def rename_meeting(meeting_id: str, data: RenameMeeting):
    meetings_collection.update_one(
        {"meeting_id": meeting_id},
        {"$set": {"title": data.title}}
    )
    return {"status": "ok"}

@app.delete("/api/meetings/{meeting_id}")
def delete_meeting(meeting_id: str):
    meetings_collection.delete_one({"meeting_id": meeting_id})
    return {"status": "deleted"}


@app.post("/api/email/send", response_model=dict)
async def send_email_endpoint(request: EmailRequest):
    """Send an email"""
    try:
        # Get tools for the default agent (or first available)
        agent_id = next(iter(agent_instances))
        tools = agent_instances[agent_id]["executor"].tools
        email_tool = next(t for t in tools if t.name == "send_email")
        
        result = email_tool(
            to=request.to,
            subject=request.subject,
            body=request.body
        )
        return {"status": "success", "message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reminder/set", response_model=dict)
async def set_reminder_endpoint(request: ReminderRequest):
    """Set a Google Calendar reminder"""
    try:
        # Get tools for the default agent (or first available)
        agent_id = next(iter(agent_instances))
        tools = agent_instances[agent_id]["executor"].tools
        reminder_tool = next(t for t in tools if t.name == "set_google_reminder")
        
        # Convert date and time to minutes from now if provided in the request
        # Otherwise use the minutes_from_now field
        if hasattr(request, 'date') and hasattr(request, 'time'):
            target_datetime = datetime.strptime(f"{request.date} {request.time}", "%Y-%m-%d %H:%M")
            now = datetime.now()
            minutes_from_now = int((target_datetime - now).total_seconds() / 60)
        else:
            minutes_from_now = request.minutes_from_now

        result = reminder_tool(
            task=request.task,
            minutes_from_now=minutes_from_now
        )
        return {"status": "success", "message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/agent/{agent_id}", response_model=dict)
async def cleanup_agent(agent_id: str):
    """Clean up an agent instance"""
    if agent_id in agent_instances:
        del agent_instances[agent_id]
        return {"status": "success", "message": f"Agent {agent_id} cleaned up"}
    raise HTTPException(status_code=404, detail="Agent not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# uvicorn api:app --reload

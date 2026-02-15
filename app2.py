import base64
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import os
import asyncio
import faiss
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
# from langchain_community.chains import RetrievalQA
# from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain.agents import create_openai_functions_agent, tool as lc_tool, AgentExecutor
from langchain.memory import ConversationBufferMemory
import smtplib
from email.message import EmailMessage
from datetime import datetime, timedelta
import os.path
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import markdown

from dateutil import parser, tz
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

 

# ----------------------
# Model setup
# ----------------------
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = "AIzaSyDd1UQ2Eax8BfsZl-dJQo0k5X9rrhYhNfA"

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai",convert_system_message_to_human=True)

# ----------------------
# Audio Recording + Transcription
# ----------------------
def record():
    freq = 44100
    duration = 30
    print("Recording...")
    recording = sd.rec(int(duration * freq), samplerate=freq, channels=1)
    sd.wait()
    write("recording.wav", freq, recording)
    return "recording.wav"

def transcribe():
    model_w = whisper.load_model("small",device="cuda")
    result = model_w.transcribe("recording.wav",task="translate", language="en")
    trans_text = result["text"]
    print("Transcription:", trans_text)
    return trans_text

# ----------------------
# Summarization Function
# ----------------------
def summerize(trans_text: str):
    prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are an AI Meeting Assistant. 
You will receive a transcript of a meeting with multiple speakers. 
Your job is to produce a well-structured, professional output with the following sections:

1. Agenda
2. Summary
3. Decisions Made
4. Action Items
5. Speakers’ Contributions

Guidelines:
- Mention speakers (Speaker 1, 2...) if names missing.
- Use bullet points.
- Mark unclear as “[unclear]”.
- Keep professional tone.
"""),
    ("user", "Here is the full transcribed text: {test}")
])


    x = prompt_template.invoke({"test": trans_text})
    result = model.invoke(x)
    markdown_text = result.content
    html_text = markdown.markdown(markdown_text, extensions=["fenced_code", "tables"])
    return html_text

# ----------------------
# RAG Setup as Tool 
# ----------------------
def build_rag_chain(summary_text: str,trans_text: str):
    """Builds a RetrievalQA chain for the meeting summary"""
    doc = [Document(page_content=summary_text, metadata={"source": "meeting_summary"}),
           
           Document(page_content=trans_text, metadata={"source": "meeting_transcript"})
           ]
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.transform_documents(doc)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    embedding_dim = len(embeddings.embed_query("hello world"))
    index = faiss.IndexFlatL2(embedding_dim)

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    vector_store.add_documents(all_splits)

    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2}),
        return_source_documents=True,
    )
    return qa_chain

# ----------------------
# Define Tools
# ----------------------
LOCAL_TZ = tz.gettz("Asia/Kolkata")
SCOPES = [
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/gmail.send"
]

def get_google_creds():
    """Get user Google OAuth credentials (shared for Gmail + Calendar)"""
    creds = None

    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return creds

def make_tools(qa_chain): 
    @tool #for rag 
    def rag_answer(query: str):
        """Answer questions based on meeting transcript and notes."""
        results = qa_chain.invoke({"query": query})
        return results["result"]

    from pymongo import MongoClient
    import json

    client = MongoClient("mongodb://localhost:27017")
    db = client["ai_meeting_db"]
    google_tokens_collection = db["google_tokens"]

    @tool
    def send_email(to: str, subject: str, body: str):
        """Send email using logged-in Google account"""
        try:
            token_doc = google_tokens_collection.find_one({"user": "default"})
            if not token_doc:
                return "User not logged in. Visit /auth/login first."

            creds = Credentials.from_authorized_user_info(
                json.loads(token_doc["token"]),
                SCOPES
            )

            service = build("gmail", "v1", credentials=creds)

            message = EmailMessage()
            message["to"] = to
            message["subject"] = subject
            message.set_content(body)

            raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

            service.users().messages().send(
                userId="me",
                body={"raw": raw}
            ).execute()

            return f"Email sent to {to}"

        except Exception as e:
            return str(e)
        

    @tool
    def set_google_reminder(task: str, date_time: str):
        """
        Create a reminder in Google Calendar at a natural-language date/time.
        Examples: "tomorrow 9 am", "next monday 10:00", "in 2 hours", "2025-10-09 09:00"
        """
        try:
            now = datetime.now(tz=LOCAL_TZ)

            # Parse natural language date/time
            try:
                parsed = parser.parse(date_time, fuzzy=True, default=now)
            except Exception:
                parsed = now + timedelta(hours=1)

            # Fix: If user says "9 am" without minutes, set minute/second to zero
            if any(h in date_time.lower() for h in ["am", "pm"]) and ":" not in date_time:
                parsed = parsed.replace(minute=0, second=0, microsecond=0)

            # Apply local timezone if missing
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=LOCAL_TZ)

            # Ensure reminder time is in the future
            if parsed <= now:
                parsed += timedelta(days=1)

            # Load Google credentials
            creds = None
            if os.path.exists("token.json"):
                creds = Credentials.from_authorized_user_file("token.json", SCOPES)
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
                    creds = flow.run_local_server(port=0)
                with open("token.json", "w") as token:
                    token.write(creds.to_json())

            service = build("calendar", "v3", credentials=creds)

            start_time = parsed
            end_time = start_time + timedelta(minutes=30)  # <-- Fixed: 30 min slot

            event = {
                "summary": task,
                "description": "Auto-created by AI Meeting Assistant",
                "start": {
                    "dateTime": start_time.isoformat(),
                    "timeZone": "Asia/Kolkata",
                },
                "end": {
                    "dateTime": end_time.isoformat(),
                    "timeZone": "Asia/Kolkata",
                },
            }

            created = service.events().insert(calendarId="primary", body=event).execute()
            return f"✅ Reminder '{task}' created for {start_time.strftime('%A, %B %d at %I:%M %p %Z')}."

        except Exception as e:
            return f"❌ Failed to create reminder: {str(e)}"
 
    # you can define more tools like set_reminder, schedule_meeting later
    return [rag_answer,send_email,set_google_reminder] #just a tool


# ----------------------
# Agent Setup
# ----------------------
async def run_agent(tools):
    from langchain import hub
    prompt = hub.pull("hwchase17/openai-functions-agent")
    memory= ConversationBufferMemory(
        memory_key="chat_history", # name used in the agent prompt
        return_messages=True    # keep full messages (not just text)
    )
    agent = create_openai_functions_agent(model, tools, prompt)
    
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,memory=memory) #true--->show how it thinks

    while True:
        query = input("\nAsk something (or 'bye' to exit): ")
        if query.lower() == "bye":
            break
        response = await agent_executor.ainvoke({"input": query})
        print("Agent:", response["output"])

def rebuild_agent(meeting):
    from langchain import hub
    qa_chain = build_rag_chain(meeting["summary"], meeting["transcript"])
    tools = make_tools(qa_chain)

    prompt = hub.pull("hwchase17/openai-functions-agent")

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    for msg in meeting["messages"]:
        if msg["role"] == "user":
            memory.chat_memory.add_user_message(msg["text"])
        else:
            memory.chat_memory.add_ai_message(msg["text"])

    agent = create_openai_functions_agent(model, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True
    )

# ----------------------
# Main
# ----------------------
if __name__ == "__main__":
    # record()
    # trans_text = transcribe()
    # For now, use transcription directly:
    trans_text = "I speak English fluently and Tamil fluently. Can you give us the answer in Tamil? I can speak Tamil fluently and English fluently. Why? Because I am from Trichy. Many people say, Trichy is not Trichy. I speak both English and Tamil fluently. I can speak in English and Tamil fluently. I can speak in Hindi as well. That's just because of the place where I am originally from. I can speak all these languages. Sid, how can you speak English fluently and Tamil fluently? Can you give us the answer in Tamil fluently and English fluently? I can speak both English and Tamil fluently. Why? Because I am from Tamil Nadu"  # (mock transcription text or call transcribe())

    summary = summerize(trans_text)
    print("\n--- Meeting Summary ---\n", summary)

    qa_chain = build_rag_chain(summary,trans_text)
    tools = make_tools(qa_chain)

    asyncio.run(run_agent(tools))

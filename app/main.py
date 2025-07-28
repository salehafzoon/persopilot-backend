# from app.agent_factory import get_agent, PersoAgent
from fastapi import FastAPI, Query, Depends, HTTPException, Body
from src.tools.labeling_assistant import LabelingAssistant
from fastapi.middleware.cors import CORSMiddleware
from src.utils.persona_util import SQLitePersonaDB
from src.agents.PersoAgent import PersoAgent
from datetime import datetime, timedelta
from typing import Optional, List
from pydantic import BaseModel
import threading
import logging
import torch
import json
import uuid
import time
import os
import gc


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()


# Add console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

app = FastAPI(
    title="PersoAgent API",
    version="1.0.0",
    summary="Conversational AI agent with persona extraction"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, you can use ["*"]. For production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
persona_db = SQLitePersonaDB()
sessions = {}
session_lock = threading.Lock()
labeling_assistant = None


# -------- Health check --------
@app.get("/health", tags=["Utility"])
def health() -> dict:
    return {
        "status": "ok",
    "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count()
    }


# ----------- APIs -----------
class PersonaFact(BaseModel):
    task_name: str
    topic: str
    relation: str
    object: str


@app.post("/users", tags=["User"])
def create_or_update_user(user_data: dict = Body(...)):
    """Create or update a user by full_name."""
    try:
        full_name = user_data["full_name"]
        age = user_data.get("age")
        gender = user_data.get("gender")
        role = user_data.get("role", "user")
        
        username, outcome = persona_db.create_or_update_user_by_full_name(full_name, age, gender, role)
        
        if outcome == "unsuccessful":
            raise HTTPException(status_code=500, detail="Failed to create or update user")
        
        return {
            "message": f"User '{full_name}' {outcome} successfully",
            "username": username,
            "full_name": full_name,
            "age": age,
            "gender": gender,
            "role": role,
            "outcome": outcome
        }
        
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing required field: {e}")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/persona_facts/bulk", tags=["Persona"])
def bulk_insert_persona_facts(username: str = Query(...), persona_facts: List[PersonaFact] = Body(...)):
    """Bulk insert persona facts for a given username."""
    try:
        # Verify user exists
        user = persona_db.get_user(username)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Convert Pydantic models to dicts
        facts_data = [fact.dict() for fact in persona_facts]
        
        # Bulk insert persona facts
        inserted_count = persona_db.bulk_insert_persona_facts(username, facts_data)
        
        return {
            "message": f"Successfully inserted {inserted_count} out of {len(persona_facts)} persona facts for user {username}",
            "username": username,
            "total_provided": len(persona_facts),
            "successfully_inserted": inserted_count
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/init", tags=["Chat"])
def init_chat_session(username: str = Query(...), task_id: str = Query(...)):
    
    try:
        task = persona_db.get_task(task_id)
        user = persona_db.get_user(username)
        
        user_personas = persona_db.get_user_persona_summary_by_task(username, task['name'])
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        logger.info(f"user_personas: {user_personas}")
        
        session_id = str(uuid.uuid4())
        agent = PersoAgent(user, task["name"], user_personas)
        
        persona_graph = persona_db.get_user_persona_graph_by_task(username, task['id'])
        
        with session_lock:
            sessions[session_id] = {
                "agent": agent,
                "created_at": datetime.now(),
                "last_accessed": datetime.now()
            }
        
        return {
            "session_id": session_id,
            "task_name": task['name'],
            "user": {
                "username": user["username"],
                "full_name": user["full_name"],
                "age": user["age"],
                "gender": user["gender"],
                "role": user["role"],
                "persona_graph": persona_graph
            },
            "expires_in": 40
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/message", tags=["Chat"])
def send_message(session_id: str = Query(...), message: str = Query(...)):
    try:
        with session_lock:
            if session_id not in sessions:
                raise HTTPException(status_code=404, detail="Session not found or expired")
            
            session = sessions[session_id]
            session["last_accessed"] = datetime.now()
            agent = session["agent"]
        
        response = agent.handle_task(message)
        
        # Parse JSON string to object
        try:
            parsed_response = json.loads(response)
        except json.JSONDecodeError:
            parsed_response = response  # Keep as string if not valid JSON
        
        return {
            "session_id": session_id,
            "response": parsed_response,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/chat/session/{session_id}", tags=["Chat"])
def end_session(session_id: str):
    try:
        with session_lock:
            if session_id in sessions:
                del sessions[session_id]
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return {"message": "Session ended successfully"}
            else:
                raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def cleanup_expired_sessions():
    while True:
        current_time = datetime.now()
        expired_sessions = []
        
        with session_lock:
            for session_id, session_data in sessions.items():
                if current_time - session_data["last_accessed"] > timedelta(minutes=1):            # Update this
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del sessions[session_id]
                print(f"Session {session_id} expired and cleaned up")
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        time.sleep(10)

    # Start cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_expired_sessions, daemon=True)
    cleanup_thread.start()


@app.get("/tasks", tags=["Utility"])
def get_tasks_and_topics() -> dict:
    
    task_topics = persona_db.get_all_tasks()
    return {"tasks": task_topics}


@app.get("/users", tags=["User"])
def get_all_users():
    cursor = persona_db.conn.cursor()
    cursor.execute("SELECT username, full_name, age, gender, role FROM User")
    rows = cursor.fetchall()
    users = [
        {"username": row[0], "full_name": row[1], "age": row[2], "gender": row[3], "role": row[4]}
        for row in rows
    ]
    return {"users": users}


@app.get("/classification_tasks", tags=["ClassificationTask"])
def get_all_classification_tasks():
    cursor = persona_db.conn.cursor()
    cursor.execute("""
        SELECT id, name, description, label1, label2, offer_message, date, username
        FROM ClassificationTask
    """)
    rows = cursor.fetchall()
    tasks = [
        {
            "id": row[0],
            "name": row[1],
            "description": row[2],
            "label1": row[3],
            "label2": row[4],
            "offer_message": row[5],
            "date": row[6],
            "username": row[7]
        }
        for row in rows
    ]
    return {"classification_tasks": tasks}


@app.get("/persona_facts", tags=["Persona"])
def get_persona_facts(username: str = Query(...)):
    try:
        facts = persona_db.get_all_persona_facts(username)
        if not facts:
            raise HTTPException(status_code=404, detail="No persona facts found for user.")
        return facts
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/persona_graph", tags=["Persona"])
def get_persona_graph(username: str = Query(...), task_id: int = Query(...)):
    try:
        graph_data = persona_db.get_user_persona_graph_by_task(username, task_id)
        if not graph_data["nodes"]:
            raise HTTPException(status_code=404, detail="No persona data found for user/task.")
        return graph_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/login", tags=["Auth"])
def login(username: str = Query(...)):
    try:
        # Get user information
        user = persona_db.get_user(username)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        response_data = {
            "username": user["username"],
            "full_name": user["full_name"],
            "age": user["age"],
            "gender": user["gender"],
            "role": user["role"]
        }
        
        # Add role-specific data
        if user["role"] == "user":
            # Get all tasks
            tasks = persona_db.get_all_tasks()
            response_data["tasks"] = tasks
            
        elif user["role"] == "analyst":
            # Get classification tasks created by this analyst
            cursor = persona_db.conn.cursor()
            cursor.execute("""
                SELECT id, name, description, label1, label2, offer_message, date
                FROM ClassificationTask
                WHERE username = ?
            """, (username,))
            
            classification_tasks = []
            for row in cursor.fetchall():
                classification_tasks.append({
                    "id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "label1": row[3],
                    "label2": row[4],
                    "offer_message": row[5],
                    "date": row[6]
                })
            
            response_data["classification_tasks"] = classification_tasks
        
        return response_data
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat_init", tags=["Chat"])
def chat_init(username: str = Query(...), task_id: int = Query(...)):
    try:
        # Verify user exists
        user = persona_db.get_user(username)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Verify task exists
        task = persona_db.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Get persona graph for the specified task
        persona_graph = persona_db.get_user_persona_graph_by_task(username, task_id)
        
        return {
            "username": username,
            "task_id": task_id,
            "task_name": task["name"],
            "persona_graph": persona_graph
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_labeling_assistant():
    global labeling_assistant
    # if labeling_assistant is None:
    #     # Check CUDA availability before initializing
    #     if not torch.cuda.is_available():
    #         raise HTTPException(
    #             status_code=503, 
    #             detail="CUDA not available. GPU-based labeling assistant requires CUDA-enabled PyTorch."
    #         )
    labeling_assistant = LabelingAssistant()
    return labeling_assistant


@app.post("/classification/label_users", tags=["Classification"])
def label_users(
    pool_size: int = Query(..., description="Number of users to evaluate"),
    clf_task_name: str = Query(..., description="Classification task name")
):
    try:
        assistant = get_labeling_assistant()
        results = assistant.get_alignment_score(pool_size, clf_task_name)
        
        if results is None:
            raise HTTPException(status_code=404, detail="No candidates found or classification task not found")
        
        return {
            "classification_task": clf_task_name,
            "total_evaluated": len(results),
            "results": results
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/classification/cleanup", tags=["Classification"])
def cleanup_labeling_assistant():
    try:
        global labeling_assistant
        if labeling_assistant:
            labeling_assistant.reset_memory()
            labeling_assistant = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return {"message": "Labeling assistant cleaned up successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/persona_facts", tags=["Persona"])
def delete_user_persona_facts(username: str = Query(...)):
    """Delete all persona facts for a given username."""
    try:
        # Verify user exists
        user = persona_db.get_user(username)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Delete all persona facts
        deleted_count = persona_db.delete_all_persona_facts(username)
        
        return {
            "message": f"Successfully deleted {deleted_count} persona facts for user {username}",
            "username": username,
            "deleted_count": deleted_count
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/classification_tasks/offers", tags=["ClassificationTask"])
def delete_classification_task_offers(task_id: str = Query(...)):
    """Delete all offers for a given classification task ID."""
    try:
        # Delete all offers and get task name
        deleted_count, task_name = persona_db.delete_classification_task_offers(task_id)
        
        if task_name is None:
            raise HTTPException(status_code=404, detail="Classification task not found")
        
        return {
            "message": f"Successfully deleted {deleted_count} offers for classification task '{task_name}'",
            "task_id": task_id,
            "task_name": task_name,
            "deleted_count": deleted_count
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/classification_tasks", tags=["ClassificationTask"])
def delete_classification_task(task_id: str = Query(...)):
    """Delete a classification task and all related records."""
    try:
        success, task_name, offers_deleted = persona_db.delete_classification_task(task_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Classification task not found")
        
        return {
            "message": f"Successfully deleted classification task '{task_name}' and {offers_deleted} related offers",
            "task_id": task_id,
            "task_name": task_name,
            "offers_deleted": offers_deleted
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/database/reseed", tags=["Database"])
def reseed_database():
    """Reseed the database using data_syn.py script."""
    try:
        import subprocess
        import sys
        
        # Execute the data_syn.py script
        result = subprocess.run(
            [sys.executable, "data_syn.py"],
            capture_output=True,
            text=True,
            cwd="."
        )
        
        if result.returncode == 0:
            return {
                "message": "Database reseeded successfully",
                "outcome": "successful",
                "output": result.stdout
            }
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Database reseed failed: {result.stderr}"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reseed database: {str(e)}")

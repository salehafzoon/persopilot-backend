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


SESSION_EXP = 1         #minutes

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
    """Create or update a user."""
    try:
        full_name = user_data["full_name"]
        age = user_data.get("age")
        gender = user_data.get("gender")
        role = user_data.get("role", "user")
        username = user_data.get("username")
        
        if not username:
            # Generate username from full_name for creation
            username = full_name.lower().replace(" ", "_")
        
        username, outcome = persona_db.create_or_update_user(username, full_name, age, gender, role)
        
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
        current_time = datetime.now()
        with session_lock:
            # Check if ANY user has an active session (server busy)
            for session_id, session_data in sessions.items():
                if current_time - session_data["last_accessed"] <= timedelta(minutes=1):
                    if session_data.get("username") == username:
                        raise HTTPException(
                            status_code=409, 
                            detail=f"User {username} already has an active session"
                        )
                    else:
                        raise HTTPException(
                            status_code=503, 
                            detail="Server is currently busy. Another user is using the system. Please try again later."
                        )
        
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
                "username": username,
                "task_id": int(task_id),
                "created_at": datetime.now(),
                "last_accessed": datetime.now()
            }
        
        return {
            "session_id": session_id,
            "task_id": task['id'],
            "task_name": task['name'],
            "user": {
                "username": user["username"],
                "full_name": user["full_name"],
                "age": user["age"],
                "gender": user["gender"],
                "role": user["role"],
                "persona_graph": persona_graph
            },
            "expires_in": SESSION_EXP * 60
        }
        
    except HTTPException as e:
        raise e
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
            username = session["username"]
            task_id = session["task_id"]
        
        response = agent.handle_task(message)
        
        # Handle duplicate JSON responses by taking the first valid one
        is_persona_updated = False
        parsed_response = response
        
        try:
            # Try to parse as JSON first
            parsed_response = json.loads(response)
        except json.JSONDecodeError:
            # If that fails, try to extract the first JSON object from the string
            try:
                # Find the first complete JSON object
                start = response.find('{')
                if start != -1:
                    brace_count = 0
                    end = start
                    for i, char in enumerate(response[start:], start):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end = i + 1
                                break
                    
                    first_json = response[start:end]
                    parsed_response = json.loads(first_json)
            except:
                parsed_response = response  # Keep as string if all parsing fails
        
        # Check if persona was updated
        if isinstance(parsed_response, dict) and parsed_response.get("used_tool") == "PersonaExtractor":
            is_persona_updated = True
        
        # Get updated persona graph
        persona_graph = persona_db.get_user_persona_graph_by_task(username, task_id)
        
        return {
            "session_id": session_id,
            "response": parsed_response,
            "is_persona_updated": is_persona_updated,
            "persona_graph": persona_graph,
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
                if current_time - session_data["last_accessed"] > timedelta(minutes= SESSION_EXP):            # Update this
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
        # Verify user exists
        user = persona_db.get_user(username)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        facts = persona_db.get_all_persona_facts(username)
        return {"persona_facts": facts}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/persona_graph", tags=["Persona"])
def get_persona_graph(username: str = Query(...), task_id: int = Query(...)):
    try:
        # Verify user exists
        user = persona_db.get_user(username)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        graph_data = persona_db.get_user_persona_graph_by_task(username, task_id)
        return {"persona_graph": graph_data}
    except HTTPException as e:
        raise e
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
            
            # Get oldest waiting offer
            cursor = persona_db.conn.cursor()
            cursor.execute("""
                SELECT ct.id, ct.offer_message, ctu.id as connection_id
                FROM ClassificationTask ct
                JOIN ClassificationTaskUser ctu ON ct.id = ctu.classification_task_id
                WHERE ctu.username = ? AND ctu.status = 'waiting'
                ORDER BY ctu.id ASC
                LIMIT 1
            """, (username,))
            
            offer_row = cursor.fetchone()
            if offer_row:
                response_data["current_offer"] = {
                    "task_id": offer_row[0],
                    "offer_message": offer_row[1],
                    "connection_id": offer_row[2]
                }
            else:
                response_data["current_offer"] = None
            
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


@app.put("/offers/respond", tags=["ClassificationTask"])
def respond_to_offer(connection_id: str = Query(...), response_data: dict = Body(...)):
    """Respond to a classification offer (accepted/declined)."""
    try:
        status = response_data["status"]
        if status not in ["accepted", "declined"]:
            raise HTTPException(status_code=400, detail="Status must be 'accepted' or 'declined'")
        
        persona_db.update_classification_task_user_status(connection_id, status)
        return {"message": f"Offer {status} successfully"}
        
    except KeyError:
        raise HTTPException(status_code=400, detail="Missing required field: status")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classification_tasks", tags=["ClassificationTask"])
def create_or_update_classification_task(task_data: dict = Body(...), username: str = Query(...)):
    """Create or update a classification task and return labeled users."""
    try:
        user = persona_db.get_user(username)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if user["role"] != "analyst":
            raise HTTPException(status_code=403, detail="Only analysts can create classification tasks")
        
        # Check if task exists (for update)
        existing_task = None
        if "id" in task_data:
            existing_task = persona_db.get_classification_task(task_data["id"])
        
        if existing_task:
            # Update existing task
            success = persona_db.update_classification_task(
                task_id=task_data["id"],
                name=task_data["name"],
                description=task_data["description"],
                label1=task_data["label1"],
                label2=task_data["label2"],
                offer_message=task_data["offer_message"]
            )
            if not success:
                raise HTTPException(status_code=500, detail="Failed to update classification task")
            task_id = task_data["id"]
            operation = "updated"
        else:
            # Create new task
            task_id = persona_db.create_classification_task(
                name=task_data["name"],
                description=task_data["description"],
                label1=task_data["label1"],
                label2=task_data["label2"],
                offer_message=task_data["offer_message"],
                username=username
            )
            operation = "created"
        
        # Get labeled users using assistant
        assistant = get_labeling_assistant()
        results = assistant.get_alignment_score(10, task_data["name"])
        
        return {
            "id": task_id,
            "message": f"Classification task '{task_data['name']}' {operation} successfully",
            "labeled_users": results if results else []
        }
        
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing required field: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classification_tasks/send_offers", tags=["ClassificationTask"])
def send_personalized_offers(task_id: int = Query(...), request_data: dict = Body(...)):
    """Send personalized offers to specified users."""
    try:
        task = persona_db.get_classification_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Classification task not found")
        
        usernames = request_data["usernames"]
        if not usernames:
            raise HTTPException(status_code=400, detail="No usernames provided")
        
        sent_to = []
        already_have_offer = []
        
        for username in usernames:
            user = persona_db.get_user(username)
            if user and user["role"] == "user":
                # Check if user already has a waiting offer
                cursor = persona_db.conn.cursor()
                cursor.execute("""
                    SELECT id FROM ClassificationTaskUser 
                    WHERE classification_task_id = ? AND username = ? AND status = 'waiting'
                """, (task_id, username))
                
                if cursor.fetchone():
                    already_have_offer.append(username)
                else:
                    persona_db.connect_user_to_classification_task(task_id, username, "waiting")
                    sent_to.append(username)
        
        return {
            "message": f"Sent {len(sent_to)} new offers",
            "sent_to": sent_to,
            "already_have_offer": already_have_offer
        }
        
    except KeyError:
        raise HTTPException(status_code=400, detail="Missing required field: usernames")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users/{username}/classification_offers", tags=["ClassificationTask"])
def get_user_classification_offers(username: str):
    """Get pending classification offers for a user."""
    try:
        cursor = persona_db.conn.cursor()
        cursor.execute("""
            SELECT ct.id, ct.name, ct.description, ct.label1, ct.label2, 
                   ct.offer_message, ctu.id as connection_id
            FROM ClassificationTask ct
            JOIN ClassificationTaskUser ctu ON ct.id = ctu.classification_task_id
            WHERE ctu.username = ? AND ctu.status = 'waiting'
        """, (username,))
        
        offers = [{"task_id": row[0], "name": row[1], "description": row[2], 
                  "label1": row[3], "label2": row[4], "offer_message": row[5], 
                  "connection_id": row[6]} for row in cursor.fetchall()]
        
        return {"offers": offers}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/classification_offers/{connection_id}/respond", tags=["ClassificationTask"])
def respond_to_personalized_offer(connection_id: int, response_data: dict = Body(...)):
    """Respond to a classification offer (accept/decline)."""
    try:
        status = response_data["status"]
        if status not in ["accepted", "declined"]:
            raise HTTPException(status_code=400, detail="Status must be 'accepted' or 'declined'")
        
        persona_db.update_classification_task_user_status(connection_id, status)
        return {"message": f"Classification offer {status} successfully"}
        
    except KeyError:
        raise HTTPException(status_code=400, detail="Missing required field: status")
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


@app.put("/classification_tasks", tags=["ClassificationTask"])
def update_classification_task(task_data: dict = Body(...)):
    """Update a classification task by ID."""
    try:
        task_id = task_data["id"]
        
        # Verify task exists
        existing_task = persona_db.get_classification_task(task_id)
        if not existing_task:
            raise HTTPException(status_code=404, detail="Classification task not found")
        
        # Update task
        success = persona_db.update_classification_task(
            task_id=task_id,
            name=task_data["name"],
            description=task_data["description"],
            label1=task_data["label1"],
            label2=task_data["label2"],
            offer_message=task_data["offer_message"]
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update classification task")
        
        return {
            "message": f"Classification task {task_id} updated successfully",
            **task_data
        }
        
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing required field: {e}")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

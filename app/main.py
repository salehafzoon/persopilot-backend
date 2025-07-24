# from app.agent_factory import get_agent, PersoAgent
from fastapi import FastAPI, Query, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.utils.persona_util import SQLitePersonaDB
from pydantic import BaseModel
from typing import Optional
import logging
import json
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

persona_db = SQLitePersonaDB()


# -------- Health check --------
@app.get("/health", tags=["Utility"])
def health() -> dict:
    return {"status": "ok"}


# ----------- APIs -----------
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


# -------- Main chat endpoint --------
# @app.post("/chat", response_model=ChatResponse, tags=["Chat"])
# def chat(
#     payload: ChatRequest,
#     agent: PersoAgent = Depends(get_agent)
# ) -> ChatResponse:
#     try:
#         output = agent.handle_task(
#             task=payload.user_input,
#             user_id=payload.user_id,
#             conv_id=payload.conversation_id
#         )
#         return ChatResponse(reply=output)
#     except Exception as e:
#         # Surface a clean 500 instead of FastAPI's default HTML trace
#         raise HTTPException(status_code=500, detail=str(e))

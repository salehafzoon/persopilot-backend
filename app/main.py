# from app.agent_factory import get_agent, PersoAgent
from fastapi import FastAPI, Query, Depends, HTTPException
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

persona_db = SQLitePersonaDB()


# -------- Pydantic request / response schemas --------
class ChatRequest(BaseModel):
    user_input: str
    user_id: Optional[str] = "default_user"
    conversation_id: Optional[str] = "default_conv"

class ChatResponse(BaseModel):
    reply: str

# -------- Health check --------
@app.get("/health", tags=["Utility"])
def health() -> dict:
    return {"status": "ok"}


# ----------- APIs -----------
@app.get("/tasks", tags=["Utility"])
def get_tasks_and_topics() -> dict:
    with open("src/data/task_topic.json", "r", encoding="utf-8") as f:
        task_topic_data = json.load(f)
    return task_topic_data
    


@app.get("/persona_graph", tags=["Persona"])
def get_persona_graph(user_id: str = Query(...), task: str = Query(...)):
    try:
        graph_data = persona_db.get_user_persona_graph_by_task(user_id, task)
        if not graph_data["nodes"]:
            raise HTTPException(status_code=404, detail="No persona data found for user/task.")
        return graph_data
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

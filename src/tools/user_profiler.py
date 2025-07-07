
from langchain.tools import StructuredTool
from pydantic import BaseModel
from typing import Optional

from src.backend.utils.db_util import Neo4jDB

# ------------------------
# Pydantic input schema
# ------------------------

class UserProfilerInput(BaseModel):
    user_id: str
    conv_id: str
    subject: Optional[str] = None
    relation: str
    obj: str
    topic: str

# ------------------------
# Tool-compatible function
# ------------------------

def user_profiler_handler(user_id: str, conv_id: str, subject: Optional[str], relation: str, obj: str, topic: str) -> str:
    db = Neo4jDB()
    try:
        db.insert_persona_fact(
            user_id=user_id,
            conv_id=conv_id,
            relation=relation,
            obj=obj,
            topic=topic
        )
        return "Persona fact inserted successfully."
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        db.close()

# ------------------------
# Tool definition
# ------------------------

user_profiler_tool = StructuredTool(
    name="User Profiler",
    func=user_profiler_handler,
    description="Insert a user's persona fact into the graph using relation and topic.",
    args_schema=UserProfilerInput
)

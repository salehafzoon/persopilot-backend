# src/backend/tools/user_profiler.py

from typing import Optional
from langchain.tools import StructuredTool
from pydantic import BaseModel
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

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
# Graph interaction
# ------------------------
class PersonaGraphDB:
    def __init__(self):
        load_dotenv()
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def insert_persona_fact(self, user_id: str, conv_id: str, triplet: tuple, topic: str):
        _, relation, obj = triplet
        query = """
        MERGE (o:Object {name: $object})
        MERGE (t:Topic {name: $topic})
        MERGE (t)-[:RELATION {type: $relation}]->(o)
        MERGE (c:Conversation {id: $conv_id})
        MERGE (c)-[:MENTIONED]->(t)
        WITH c
        MATCH (u:User {id: $user_id})
        MERGE (u)-[:STARTED]->(c)
        """
        with self.driver.session() as session:
            session.run(query, object=obj, relation=relation, topic=topic, conv_id=conv_id, user_id=user_id)

# ------------------------
# Tool-compatible function
# ------------------------
def user_profiler_handler(user_id: str, conv_id: str, subject: Optional[str], relation: str, obj: str, topic: str) -> str:
    db = PersonaGraphDB()
    try:
        triplet = (subject or "", relation, obj)
        db.insert_persona_fact(
            user_id=user_id,
            conv_id=conv_id,
            triplet=triplet,
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

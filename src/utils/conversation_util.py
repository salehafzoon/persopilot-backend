from tinydb import TinyDB, Query
from datetime import datetime
import uuid

db = TinyDB("data/conversations.json")

def create_conversation(user_id: str, task: str) -> str:
    conv_id = str(uuid.uuid4())
    db.insert({
        "conversation_id": conv_id,
        "user_id": user_id,
        "task": task,
        "created_at": datetime.now(datetime.timezone.utc).isoformat(),
        "messages": []
    })
    return conv_id

def add_message(conversation_id: str, sender: str, text: str):
    Conversation = Query()
    result = db.search(Conversation.conversation_id == conversation_id)
    if result:
        messages = result[0]["messages"]
        messages.append({
            "sender": sender,
            "text": text,
            "timestamp": datetime.now(datetime.timezone.utc).isoformat()
        })
        db.update({"messages": messages}, Conversation.conversation_id == conversation_id)

def get_conversation(conversation_id: str):
    Conversation = Query()
    result = db.search(Conversation.conversation_id == conversation_id)
    return result[0] if result else None

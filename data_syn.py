from src.utils.persona_util import SQLitePersonaDB
import json

db = SQLitePersonaDB()

# Clear the whole database first
db.clear_database()

############################# USER DATA INSERTION #############################
# Load users from JSON file
with open("src/data/users.json", "r") as f:
    users = json.load(f)

for user in users:
    db.create_user(
        full_name=user["full_name"],
        age=user["age"],
        gender=user["gender"],
        role=user["role"]
    )

################################# TASK TOPIC INSERTION ####################################

# Load tasks and topics from JSON file
with open("src/data/task_topic.json", "r") as f:
    tasks_topics = json.load(f)

# Insert tasks and their topics
task_ids = {}
for task_name, topics in tasks_topics.items():
    task_id = db.create_task(topic=task_name)
    task_ids[task_name] = task_id
    # Optionally, you can print or store the mapping for later use
    print(f"Inserted Task: {task_name} with id {task_id}")
    for topic in topics:
        # You may want to use these topics later when inserting Persona facts
        print(f"  - Topic: {topic}")


############################### CLASSIFICATION TASK INSERTION ##############################
# Load classification tasks from JSON file
with open("src/data/classificaiton_tasks.json", "r") as f:
    classification_tasks = json.load(f)

# Get the first user with role 'analyst' (as creator for demo)
analyst_user = db.conn.execute("SELECT id FROM User WHERE role = 'analyst' LIMIT 1").fetchone()
analyst_user_id = analyst_user[0] if analyst_user else 1

classification_task_ids = []
for task in classification_tasks:
    task_id = db.create_classification_task(
        name=task["name"],
        description=task["description"],
        label1=task["label1"],
        label2=task["label2"],
        user_id=analyst_user_id
    )
    classification_task_ids.append(task_id)
    print(f"Inserted Classification Task: {task['name']} with id {task_id}")

################################ PERSONA FACT INSERTION ####################################

# Load persona facts from JSON file
with open("src/data/persona_facts.json", "r") as f:
    persona_facts = json.load(f)

# Get all user IDs (excluding analyst)
user_ids = [row[0] for row in db.conn.execute("SELECT id FROM User WHERE role != 'analyst' ORDER BY id ASC").fetchall()]

for user_fact in persona_facts:
    # If your JSON uses "user_index", use it to map to user_ids
    user_index = user_fact.get("user_index")
    if user_index is not None:
        user_id = user_ids[user_index]
        for fact in user_fact["facts"]:
            task_id = task_ids[fact["task_name"]]
            db.insert_persona_fact(
                user_id=user_id,
                task_id=task_id,
                topic=fact["topic"],
                relation=fact["relation"],
                obj=fact["object"]
            )
            print(f"Inserted persona fact for user {user_id}: {fact['topic']}, {fact['relation']}, {fact['object']}")

db.close()

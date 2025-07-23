from src.utils.persona_util import SQLitePersonaDB
import json

db = SQLitePersonaDB()

# Clear the whole database first
db.clear_database()

############################# USER DATA INSERTION #############################

with open("src/data/users.json", "r") as f:
    users = json.load(f)

for user in users:
    db.create_user(
        username=user["username"],
        full_name=user["full_name"],
        age=user["age"],
        gender=user["gender"],
        role=user["role"]
    )

################################# TASK TOPIC INSERTION ####################################

with open("src/data/task_topic.json", "r") as f:
    tasks_topics = json.load(f)

# Insert tasks and their topics
task_ids = {}
for task_name, topics in tasks_topics.items():
    task_id = db.create_task(name=task_name, topics=topics)
    task_ids[task_name] = task_id
    print(f"Inserted Task: {task_name} with id {task_id}")
    for topic in topics:
        print(f"  - Topic: {topic}")

############################### CLASSIFICATION TASK INSERTION ##############################

with open("src/data/classification_tasks.json", "r") as f:
    classification_tasks = json.load(f)

# Get the first user with role 'analyst' (as creator for demo)
analyst_user = db.conn.execute("SELECT username FROM User WHERE role = 'analyst' LIMIT 1").fetchone()
analyst_username = analyst_user[0] if analyst_user else None

classification_task_ids = []
for task in classification_tasks:
    task_id = db.create_classification_task(
        name=task["name"],
        description=task["description"],
        label1=task["label1"],
        label2=task["label2"],
        offer_message=task["offer_message"],
        username=analyst_username
    )
    classification_task_ids.append(task_id)
    print(f"Inserted Classification Task: {task['name']} with id {task_id}")

################################ PERSONA FACT INSERTION ####################################

with open("src/data/persona_facts.json", "r") as f:
    persona_facts = json.load(f)

# Get all usernames (excluding analyst)
# Get all usernames except analysts
usernames = [row[0] for row in db.conn.execute("SELECT username FROM User WHERE role != 'analyst' ORDER BY username ASC").fetchall()]

# Load tasks and create them if needed
with open("src/data/task_topic.json", "r") as f:
    tasks_topics = json.load(f)

# Insert tasks and their topics
task_ids = {}
for task_name, topics in tasks_topics.items():
    task_id = db.create_task(name=task_name, topics=topics)
    task_ids[task_name] = task_id
    print(f"Inserted Task: {task_name} with id {task_id}")
    for topic in topics:
        print(f"  - Topic: {topic}")

# Insert persona facts
for user_fact in persona_facts:
    username = user_fact.get("username")
    if username:
        for fact in user_fact["facts"]:
            task_id = task_ids[fact["task_name"]]
            db.insert_persona_fact(
                username=username,
                task_id=task_id,
                topic=fact["topic"],
                relation=fact["relation"],
                obj=fact["object"]
            )
            print(f"Inserted persona fact for user {username}: {fact['topic']}, {fact['relation']}, {fact['object']}")


################# OFFER INSERTION FOR CLASSIFICATION TASKS ####################
usernames = [row[0] for row in db.conn.execute("SELECT username FROM User WHERE role != 'analyst' ORDER BY username ASC").fetchall()]

# Insert an offer for the first user for the first classification task
if usernames and classification_task_ids:
    first_username = usernames[0]
    first_classification_task_id = classification_task_ids[0]
    connection_id = db.connect_user_to_classification_task(
        classification_task_id=first_classification_task_id,
        username=first_username,
        status="waiting"
    )
    print(f"Inserted offer for user {first_username} on classification task id {first_classification_task_id} (connection id {connection_id})")


          
db.close()
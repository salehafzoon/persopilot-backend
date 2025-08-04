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
    tasks_data = json.load(f)

# Insert tasks and their topics
task_ids = {}
for task in tasks_data:
    task_name = task["title"]
    task_description = task["description"]
    topics = task["topics"]
    
    task_id = db.create_task(name=task_name, description=task_description, topics=topics)
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
# Load persona facts
with open("src/data/persona_facts.json", "r") as f:
    persona_facts = json.load(f)

# Insert persona facts
for user_fact in persona_facts:
    username = user_fact.get("username")
    if username:
        for fact in user_fact["facts"]:
            task_name = fact["task_name"]
            task_id = task_ids[task_name]  # Get task_id from the dictionary
            
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




################# OFFER INSERTION FOR CLASSIFICATION TASKS ####################

with open("src/data/classification_users.json", "r") as f:
    classification_users = json.load(f)

# Insert offers for each classification task
for task_name, user_groups in classification_users.items():
    # Find task ID from database by name
    task_row = db.conn.execute("SELECT id FROM ClassificationTask WHERE name = ?", (task_name,)).fetchone()
    if task_row:
        task_id = task_row[0]
        
        # Get task data for labels
        task_data = db.get_classification_task(task_id)
        
        # Insert accepted offers for first group (up to 10 users)
        first_group_users = user_groups[task_data["label1"]][:10]
        for username in first_group_users:
            try:
                db.connect_user_to_classification_task(task_id, username, "accepted")
                print(f"Inserted accepted offer for {username} on {task_name}")
            except:
                pass
        
        # Insert declined offers for second group (up to 10 users)
        second_group_users = user_groups[task_data["label2"]][:10]
        for username in second_group_users:
            try:
                db.connect_user_to_classification_task(task_id, username, "declined")
                print(f"Inserted declined offer for {username} on {task_name}")
            except:
                pass
################# SYNTHETIC PREDICTION INSERTION ####################
import random

# Target accuracies for each classification task
target_accuracies = {
    "Gamer Identification": 85,
    "Camping Enthusiast": 75,
    "Bookworm": 60,
    "Career Path Preference": 80
}

# Generate synthetic predictions for each classification task
for task_name, accuracy_percent in target_accuracies.items():
    # Find task ID from database
    task_row = db.conn.execute("SELECT id FROM ClassificationTask WHERE name = ?", (task_name,)).fetchone()
    if task_row:
        task_id = task_row[0]
        task_data = db.get_classification_task(task_id)
        
        # Get users with offers (both accepted and declined)
        cursor = db.conn.cursor()
        cursor.execute("""
            SELECT username, status FROM ClassificationTaskUser 
            WHERE classification_task_id = ? AND status IN ('accepted', 'declined')
            ORDER BY RANDOM() LIMIT 10
        """, (task_id,))
        
        users_with_status = cursor.fetchall()
        
        if len(users_with_status) >= 10:
            # Calculate correct and incorrect predictions needed
            correct_needed = int(10 * accuracy_percent / 100)
            incorrect_needed = 10 - correct_needed
            
            predictions = []
            
            # Generate correct predictions
            for i in range(correct_needed):
                username, status = users_with_status[i]
                predicted_label = task_data["label1"] if status == "accepted" else task_data["label2"]
                confidence = round(random.uniform(0.7, 0.95), 3)
                
                predictions.append({
                    "username": username,
                    "predicted_label": predicted_label,
                    "confidence": confidence
                })
            
            # Generate incorrect predictions
            for i in range(correct_needed, correct_needed + incorrect_needed):
                username, status = users_with_status[i]
                predicted_label = task_data["label2"] if status == "accepted" else task_data["label1"]
                confidence = round(random.uniform(0.6, 0.8), 3)
                
                predictions.append({
                    "username": username,
                    "predicted_label": predicted_label,
                    "confidence": confidence
                })
            
            # Save predictions to database
            db.save_predictions(task_id, predictions)
            print(f"Inserted {len(predictions)} synthetic predictions for {task_name} (target accuracy: {accuracy_percent}%)")


db.close()
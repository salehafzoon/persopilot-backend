import logging
import sqlite3
from typing import List, Dict, Optional

logger = logging.getLogger("persona_util")
logging.basicConfig(level=logging.INFO)

class SQLitePersonaDB:
    def __init__(self, db_path="src/data/persona.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()
        logger.info("SQLitePersonaDB initialized")

    def _create_tables(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS User (
                    username TEXT PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    age INTEGER,
                    gender TEXT,
                    role TEXT NOT NULL DEFAULT 'user'
                );
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS Task (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT
                );
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS TaskTopic (
                    task_id INTEGER NOT NULL,
                    topic TEXT NOT NULL,
                    PRIMARY KEY (task_id, topic),
                    FOREIGN KEY (task_id) REFERENCES Task(id)
                );
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS Persona (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    task_id INTEGER NOT NULL,
                    topic TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    object TEXT NOT NULL,
                    FOREIGN KEY (username) REFERENCES User(username),
                    FOREIGN KEY (task_id) REFERENCES Task(id)
                );
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS ClassificationTask (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    label1 TEXT,
                    label2 TEXT,
                    offer_message TEXT,
                    date DATETIME DEFAULT CURRENT_TIMESTAMP,
                    username TEXT NOT NULL,
                    FOREIGN KEY (username) REFERENCES User(username)
                );
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS ClassificationTaskUser (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    classification_task_id INTEGER NOT NULL,
                    username TEXT NOT NULL,
                    status TEXT CHECK(status IN ('waiting', 'accepted', 'declined')) NOT NULL DEFAULT 'waiting',
                    FOREIGN KEY (classification_task_id) REFERENCES ClassificationTask(id),
                    FOREIGN KEY (username) REFERENCES User(username)
                );
            """)

    def close(self):
        logger.info("Closing SQLite connection")
        self.conn.close()

    # User methods
    def create_user(self, username: str, full_name: str, age: Optional[int], gender: Optional[str], role: str = "user") -> str:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO User (username, full_name, age, gender, role) VALUES (?, ?, ?, ?, ?)",
                (username, full_name, age, gender, role)
            )
        logger.info(f"User created: {username}")
        return username

    def get_user(self, username: str) -> Optional[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT username, full_name, age, gender, role FROM User WHERE username = ?", (username,))
        row = cursor.fetchone()
        if row:
            return {"username": row[0], "full_name": row[1], "age": row[2], "gender": row[3], "role": row[4]}
        return None

    def create_task(self, name: str, description: str, topics: List[str]) -> int:
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO Task (name, description) VALUES (?, ?)",
                (name, description)
            )
            task_id = cursor.lastrowid
            
            # Insert topics
            for topic in topics:
                self.conn.execute(
                    "INSERT INTO TaskTopic (task_id, topic) VALUES (?, ?)",
                    (task_id, topic)
                )
        
        logger.info(f"Task created: {task_id}")
        return task_id


    def get_all_tasks(self) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name, description FROM Task")
        tasks = cursor.fetchall()
        
        result = []
        for task in tasks:
            task_id = task[0]
            task_name = task[1]
            task_description = task[2]
            
            # Get topics for this task
            cursor.execute("""
                SELECT topic 
                FROM TaskTopic 
                WHERE task_id = ?
            """, (task_id,))
            topics = [row[0] for row in cursor.fetchall()]
            
            result.append({
                "id": task_id,
                "name": task_name,
                "description": task_description,
                "topics": topics
            })
        
        return result

    def get_task(self, task_id: int) -> Optional[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name, description FROM Task WHERE id = ?", (task_id,))
        row = cursor.fetchone()
        if row:
            return {"id": row[0], "name": row[1], "description": row[2]}
        return None


    def get_all_persona_facts(self, username: str) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT p.id, p.task_id, t.name as task_name, p.topic, p.relation, p.object
            FROM Persona p
            JOIN Task t ON p.task_id = t.id
            WHERE p.username = ?
            ORDER BY p.task_id, p.topic
        """, (username,))
        
        facts = []
        for row in cursor.fetchall():
            facts.append({
                "id": row[0],
                "task_id": row[1],
                "task_name": row[2],
                "topic": row[3],
                "relation": row[4],
                "object": row[5]
            })
        
        return facts


    # Persona methods
    def insert_persona_fact(self, username: str, task_id: int, topic: str, relation: str, obj: str):
        with self.conn:
            self.conn.execute("""
                INSERT INTO Persona (username, task_id, topic, relation, object)
                VALUES (?, ?, ?, ?, ?)
            """, (username, task_id, topic, relation, obj))
        logger.info(f"Persona fact inserted for user {username}, task {task_id}")

    def insert_persona_fact_by_name(self, username: str, task: str, topic: str, relation: str, obj: str):
        # Get task_id from task name
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM Task WHERE name = ?", (task,))
        task_row = cursor.fetchone()
        if not task_row:
            raise ValueError(f"Task '{task}' not found")
        
        task_id = task_row[0]
        
        with self.conn:
            self.conn.execute("""
                INSERT INTO Persona (username, task_id, topic, relation, object)
                VALUES (?, ?, ?, ?, ?)
            """, (username, task_id, topic, relation, obj))
        logger.info(f"Persona fact inserted for user {username}, task {task}")


    def get_community_suggestions(self, username: str, task_name: str, limit: int = 3) -> Dict[str, List[Dict]]:
        # Get task_id from task_name
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM Task WHERE name = ?", (task_name,))
        task_row = cursor.fetchone()
        if not task_row:
            return {}
        
        task_id = task_row[0]
        
        query = """
        SELECT topic, object, relation, user_count
        FROM (
            SELECT topic, object, relation, COUNT(DISTINCT username) as user_count,
                ROW_NUMBER() OVER (PARTITION BY topic ORDER BY COUNT(DISTINCT username) DESC) as rn
            FROM Persona
            WHERE task_id = ? AND username != ?
            GROUP BY topic, object, relation
        ) ranked
        WHERE rn <= ?
        ORDER BY topic, user_count DESC
        """
        cursor.execute(query, (task_id, username, limit))
        records = cursor.fetchall()

        suggestions = {}
        for topic, obj, relation, count in records:
            if topic not in suggestions:
                suggestions[topic] = []
            suggestions[topic].append({
                "object": obj,
                "user_count": count,
                "relations": [relation]
            })
        
        return suggestions


    def format_community_suggestions(self, username: str, task_name: str, limit: int = 3) -> str:
        suggestions = self.get_community_suggestions(username, task_name, limit)
        if not suggestions:
            return f"No community suggestions found for task: {task_name}"

        output = [f"Community suggestions for {task_name} (from other users):"]
        for topic, objects in suggestions.items():
            output.append(f"- {topic}:")
            for obj in objects:
                output.append(f"  . {obj['object']}: liked by {obj['user_count']} users")
        return "\n".join(output)


    def get_user_persona_graph_by_task(self, username: str, task_id: int) -> dict:
        cursor = self.conn.cursor()
        
        # Get task name
        cursor.execute("SELECT name FROM Task WHERE id = ?", (task_id,))
        task_row = cursor.fetchone()
        if not task_row:
            return {"nodes": [], "edges": []}
        task_name = task_row[0]
        
        # Get topics for this task
        cursor.execute("SELECT topic FROM TaskTopic WHERE task_id = ?", (task_id,))
        task_topics = [row[0] for row in cursor.fetchall()]
        
        # Get persona data for this user and task
        cursor.execute("""
            SELECT topic, relation, object
            FROM Persona
            WHERE username = ? AND task_id = ?
        """, (username, task_id))
        persona_records = cursor.fetchall()
        logger.info(f"Retrieved {len(persona_records)} persona records for user {username} and task {task_id}")
        # Create nodes
        nodes = {
            str(username): {"id": str(username), "label": str(username), "type": "User"},
            str(task_id): {"id": str(task_id), "label": task_name, "type": "Task"}
        }
        
        # Create initial edge from user to task
        edges = [{"source": str(username), "target": str(task_id), "label": "has_task"}]
        
        # Add all task topics as nodes and connect them to the task
        for topic in task_topics:
            if topic not in nodes:
                nodes[topic] = {"id": topic, "label": topic, "type": "Topic"}
            edges.append({"source": str(task_id), "target": topic, "label": "has_topic"})
        
        # Process persona records - connect topics to objects with relations
        for topic, relation, obj in persona_records:
            # Make sure the topic node exists
            if topic not in nodes:
                nodes[topic] = {"id": topic, "label": topic, "type": "Topic"}
                edges.append({"source": str(task_id), "target": topic, "label": "has_topic"})
            
            # Add object node
            obj_id = f"{topic}_{obj}"  # Create unique ID for object nodes
            if obj_id not in nodes:
                nodes[obj_id] = {"id": obj_id, "label": obj, "type": "Object"}
            
            # Connect topic to object with the relation as the edge label
            edges.append({"source": topic, "target": obj_id, "label": relation})

        return {
            "nodes": list(nodes.values()),
            "edges": edges
        }


    def get_user_persona_summary_by_task(self, username: str, task_name: str) -> str:
        cursor = self.conn.cursor()
        
        # Get task id from name
        cursor.execute("SELECT id FROM Task WHERE name = ?", (task_name,))
        task_row = cursor.fetchone()
        if not task_row:
            return f"No task found with name {task_name}"
        
        task_id = task_row[0]
        
        # Get persona facts grouped by topic
        cursor.execute("""
            SELECT topic, relation, object
            FROM Persona
            WHERE username = ? AND task_id = ?
            ORDER BY topic, relation
        """, (username, task_id))
        
        records = cursor.fetchall()
        if not records:
            return f"No persona data found for user {username} in task {task_name}"
        
        # Group facts by topic
        topics = {}
        for topic, relation, obj in records:
            if topic not in topics:
                topics[topic] = {}
            if relation not in topics[topic]:
                topics[topic][relation] = []
            topics[topic][relation].append(obj)
        
        # Build summary
        summary_parts = [f"User {username} preferences for {task_name}:"]
        
        for topic, relations in topics.items():
            topic_facts = []
            for relation, objects in relations.items():
                if len(objects) == 1:
                    topic_facts.append(f"{relation} {objects[0]}")
                else:
                    topic_facts.append(f"{relation} {', '.join(objects)}")
            
            summary_parts.append(f"{topic}: {'; '.join(topic_facts)}")
        
        return " | ".join(summary_parts)




    # ClassificationTask methods
    def create_classification_task(self, name: str, description: str, label1: str, label2: str, offer_message: str, username: str) -> int:
        with self.conn:
            cursor = self.conn.execute("""
                INSERT INTO ClassificationTask (name, description, label1, label2, offer_message, username)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (name, description, label1, label2, offer_message, username))
            classification_task_id = cursor.lastrowid
        logger.info(f"ClassificationTask created: {classification_task_id}")
        return classification_task_id

    def get_classification_task(self, classification_task_id: int) -> Optional[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, name, description, label1, label2, offer_message, date, username
            FROM ClassificationTask
            WHERE id = ?
        """, (classification_task_id,))
        row = cursor.fetchone()
        if row:
            return {
                "id": row[0], "name": row[1], "description": row[2],
                "label1": row[3], "label2": row[4], "offer_message": row[5],
                "date": row[6], "username": row[7]
            }
        return None

    # ClassificationTaskUser methods
    def connect_user_to_classification_task(self, classification_task_id: int, username: str, status: str = "waiting") -> int:
        with self.conn:
            cursor = self.conn.execute("""
                INSERT INTO ClassificationTaskUser (classification_task_id, username, status)
                VALUES (?, ?, ?)
            """, (classification_task_id, username, status))
            connection_id = cursor.lastrowid
        logger.info(f"User {username} connected to ClassificationTask {classification_task_id} with status {status}")
        return connection_id

    def update_classification_task_user_status(self, connection_id: int, status: str):
        with self.conn:
            self.conn.execute("""
                UPDATE ClassificationTaskUser SET status = ? WHERE id = ?
            """, (status, connection_id))
        logger.info(f"ClassificationTaskUser {connection_id} status updated to {status}")

    def get_classification_task_users(self, classification_task_id: int) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, username, status
            FROM ClassificationTaskUser
            WHERE classification_task_id = ?
        """, (classification_task_id,))
        rows = cursor.fetchall()
        return [
            {"id": row[0], "username": row[1], "status": row[2]}
            for row in rows
        ]

    def clear_database(self):
        logger.warning("Clearing all tables from SQLite!")
        with self.conn:
            self.conn.execute("DELETE FROM Persona")
            self.conn.execute("DELETE FROM ClassificationTaskUser")
            self.conn.execute("DELETE FROM ClassificationTask")
            self.conn.execute("DELETE FROM Task")
            self.conn.execute("DELETE FROM User")
        logger.info("All tables cleared")
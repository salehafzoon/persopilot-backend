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
                    name TEXT NOT NULL
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

    # Task methods
    def create_task(self, name: str, topics: List[str]) -> int:
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO Task (name) VALUES (?)",
                (name,)
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
        cursor.execute("SELECT id, name FROM Task")
        tasks = cursor.fetchall()
        
        result = []
        for task in tasks:
            task_id = task[0]
            task_name = task[1]
            
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
                "topics": topics
            })
        
        return result


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

    def get_community_suggestions(self, task_id: int, limit: int = 10) -> Dict[str, List[Dict]]:
        query = """
        SELECT topic, object, relation, COUNT(DISTINCT username) as user_count
        FROM Persona
        WHERE task_id = ?
        GROUP BY topic, object, relation
        ORDER BY topic, user_count DESC
        LIMIT ?
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (task_id, limit))
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

    def format_community_suggestions(self, task_id: int, limit: int = 10) -> str:
        suggestions = self.get_community_suggestions(task_id, limit)
        if not suggestions:
            return f"No suggestions found for task id: {task_id}"

        output = [f"Suggestions for all related topics under task id {task_id}:"]
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
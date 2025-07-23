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
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    full_name TEXT NOT NULL,
                    age INTEGER,
                    gender TEXT,
                    role TEXT NOT NULL DEFAULT 'user'
                );
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS Task (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL
                );
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS Persona (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    task_id INTEGER NOT NULL,
                    topic TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    object TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES User(id),
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
                    date DATETIME DEFAULT CURRENT_TIMESTAMP,
                    user_id INTEGER NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES User(id)
                );
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS Offer (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    classification_task_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    message TEXT NOT NULL,
                    status TEXT CHECK(status IN ('waiting', 'accepted', 'declined')) NOT NULL DEFAULT 'waiting',
                    FOREIGN KEY (classification_task_id) REFERENCES ClassificationTask(id),
                    FOREIGN KEY (user_id) REFERENCES User(id)
                );
            """)

    def close(self):
        logger.info("Closing SQLite connection")
        self.conn.close()

    # User methods
    def create_user(self, full_name: str, age: Optional[int], gender: Optional[str]) -> int:
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO User (full_name, age, gender, role) VALUES (?, ?, ?, ?)",
                (full_name, age, gender, role)
            )
            user_id = cursor.lastrowid
        logger.info(f"User created: {user_id}")
        return user_id

    def get_user(self, user_id: int) -> Optional[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, full_name, age, gender, role FROM User WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        if row:
            return {"id": row[0], "full_name": row[1], "age": row[2], "gender": row[3], "role": row[4]}
        return None

    # Task methods
    def create_task(self, topic: str) -> int:
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO Task (topic) VALUES (?)",
                (topic,)
            )
            task_id = cursor.lastrowid
        logger.info(f"Task created: {task_id}")
        return task_id

    def get_task(self, task_id: int) -> Optional[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, topic FROM Task WHERE id = ?", (task_id,))
        row = cursor.fetchone()
        if row:
            return {"id": row[0], "topic": row[1]}
        return None

    # Persona methods
    def insert_persona_fact(self, user_id: int, task_id: int, topic: str, relation: str, obj: str):
        with self.conn:
            self.conn.execute("""
                INSERT INTO Persona (user_id, task_id, topic, relation, object)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, task_id, topic, relation, obj))
        logger.info(f"Persona fact inserted for user {user_id}, task {task_id}")

    def get_community_suggestions(self, task_id: int, limit: int = 10) -> Dict[str, List[Dict]]:
        query = """
        SELECT topic, object, relation, COUNT(DISTINCT user_id) as user_count
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

    def get_user_persona_graph_by_task(self, user_id: int, task_id: int) -> dict:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT topic, relation, object
            FROM Persona
            WHERE user_id = ? AND task_id = ?
        """, (user_id, task_id))
        records = cursor.fetchall()

        nodes = {
            str(user_id): {"id": str(user_id), "label": str(user_id), "type": "User"},
            str(task_id): {"id": str(task_id), "label": str(task_id), "type": "Task"}
        }
        edges = [{"source": str(user_id), "target": str(task_id), "label": "has_task"}]

        for topic, relation, obj in records:
            if topic not in nodes:
                nodes[topic] = {"id": topic, "label": topic, "type": "Topic"}
            if obj not in nodes:
                nodes[obj] = {"id": obj, "label": obj, "type": "Object"}

            edges.append({"source": str(task_id), "target": topic, "label": "has_topic"})
            edges.append({"source": topic, "target": obj, "label": relation})

        return {
            "nodes": list(nodes.values()),
            "edges": edges
        }

    # ClassificationTask methods
    def create_classification_task(self, name: str, description: str, label1: str, label2: str, user_id: int) -> int:
        with self.conn:
            cursor = self.conn.execute("""
                INSERT INTO ClassificationTask (name, description, label1, label2, user_id)
                VALUES (?, ?, ?, ?, ?)
            """, (name, description, label1, label2, user_id))
            classification_task_id = cursor.lastrowid
        logger.info(f"ClassificationTask created: {classification_task_id}")
        return classification_task_id

    def get_classification_task(self, classification_task_id: int) -> Optional[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, name, description, label1, label2, date, user_id
            FROM ClassificationTask
            WHERE id = ?
        """, (classification_task_id,))
        row = cursor.fetchone()
        if row:
            return {
                "id": row[0], "name": row[1], "description": row[2],
                "label1": row[3], "label2": row[4], "date": row[5], "user_id": row[6]
            }
        return None

    # Offer methods
    def create_offer(self, classification_task_id: int, user_id: int, message: str, status: str = "waiting") -> int:
        with self.conn:
            cursor = self.conn.execute("""
                INSERT INTO Offer (classification_task_id, user_id, message, status)
                VALUES (?, ?, ?, ?)
            """, (classification_task_id, user_id, message, status))
            offer_id = cursor.lastrowid
        logger.info(f"Offer created: {offer_id}")
        return offer_id

    def update_offer_status(self, offer_id: int, status: str):
        with self.conn:
            self.conn.execute("""
                UPDATE Offer SET status = ? WHERE id = ?
            """, (status, offer_id))
        logger.info(f"Offer {offer_id} status updated to {status}")

    def get_offers_for_user(self, user_id: int) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, classification_task_id, message, status
            FROM Offer
            WHERE user_id = ?
        """, (user_id,))
        rows = cursor.fetchall()
        return [
            {"id": row[0], "classification_task_id": row[1], "message": row[2], "status": row[3]}
            for row in rows
        ]

    def clear_database(self):
        logger.warning("Clearing all tables from SQLite!")
        with self.conn:
            self.conn.execute("DELETE FROM Persona")
            self.conn.execute("DELETE FROM Offer")
            self.conn.execute("DELETE FROM ClassificationTask")
            self.conn.execute("DELETE FROM Task")
            self.conn.execute("DELETE FROM User")
        logger.info("All tables cleared")
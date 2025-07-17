import logging
from neo4j import GraphDatabase
from dotenv import load_dotenv
from typing import List, Dict
import sqlite3
import os

logger = logging.getLogger("persona_util")
logging.basicConfig(level=logging.INFO)

class Neo4jPersonaDB:
    def __init__(self):
        logger.info("Initializing Neo4jPersonaDB")
        load_dotenv()
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._create_constraints()
        logger.info("Neo4jPersonaDB initialized")

    def close(self):
        logger.info("Closing Neo4j driver")
        self.driver.close()

    def _create_constraints(self):
        logger.info("Creating constraints in Neo4j")
        with self.driver.session() as session:
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS
                FOR (t:Task)
                REQUIRE (t.name, t.user_id) IS NODE KEY
            """)
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS
                FOR (t:Topic)
                REQUIRE (t.name, t.user_id) IS NODE KEY
            """)
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS
                FOR (o:Object)
                REQUIRE (o.name, o.user_id) IS NODE KEY
            """)
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS
                FOR (gt:GlobalTopic)
                REQUIRE gt.name IS UNIQUE
            """)
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS
                FOR (go:GlobalObject)
                REQUIRE go.name IS UNIQUE
            """)
        logger.info("Constraints created")

    def create_user(self, user_id: str):
        logger.info(f"Creating user: {user_id}")
        query = "MERGE (:User {id: $user_id})"
        with self.driver.session() as session:
            session.run(query, user_id=user_id)
        logger.info(f"User created: {user_id}")

    def insert_persona_fact(self, user_id: str, relation: str, obj: str, topic: str, task: str):
        logger.info(f"Inserting persona fact for user {user_id}: relation={relation}, object={obj}, topic={topic}, task={task}")
        query = f"""
        MERGE (u:User {{id: $user_id}})
        MERGE (task:Task {{name: $task, user_id: $user_id}})
        MERGE (u)-[:INITIATES_TASK]->(task)

        MERGE (topic:Topic {{name: $topic, user_id: $user_id}})
        MERGE (task)-[:HAS_TOPIC]->(topic)

        MERGE (obj:Object {{name: $object, user_id: $user_id}})
        MERGE (topic)-[:{relation.upper()}]->(obj)

        // Link to global references
        MERGE (gt:GlobalTopic {{name: $topic}})
        MERGE (topic)-[:REFERS_TO]->(gt)

        MERGE (go:GlobalObject {{name: $object}})
        MERGE (obj)-[:REFERS_TO]->(go)
        """
        with self.driver.session() as session:
            session.run(
                query,
                user_id=user_id,
                object=obj,
                topic=topic,
                task=task
            )
        logger.info(f"Persona fact inserted for user {user_id}")

    def get_community_suggestions(self, user_id: str, task: str, limit: int = 10):
        logger.info(f"Getting community suggestions for user {user_id} and task '{task}'")
        query = """
        // Find all topics under the specified task across all users
        MATCH (task:Task {name: $task})-[:HAS_TOPIC]->(topic:Topic)
        MATCH (topic)-[r]->(obj:Object)
        WHERE topic.user_id = obj.user_id
        
        // Group by topic and object, count unique users
        WITH topic.name AS topic_name, obj.name AS object_name, 
            COUNT(DISTINCT obj.user_id) AS user_count,
            COLLECT(DISTINCT type(r)) AS relations
        
        RETURN topic_name, object_name, user_count, relations
        ORDER BY topic_name, user_count DESC, object_name
        """
        
        with self.driver.session() as session:
            result = session.run(query, task=task, limit=limit)
            
            # Group results by topic
            suggestions = {}
            for record in result:
                topic = record['topic_name']
                if topic not in suggestions:
                    suggestions[topic] = []
                suggestions[topic].append({
                    'object': record['object_name'],
                    'user_count': record['user_count'],
                    'relations': record['relations']
                })
        logger.info(f"Community suggestions retrieved for task '{task}'")
        return suggestions

    def format_community_suggestions(self, user_id: str, task: str, limit: int = 10):
        logger.info(f"Formatting community suggestions for user {user_id} and task '{task}'")
        suggestions = self.get_community_suggestions(user_id, task, limit)
        
        if not suggestions:
            logger.info(f"No suggestions found for task: {task}")
            return f"No suggestions found for task: {task}"
        
        output = [f"Suggestions for all related topics under {task}:"]
        
        for topic, objects in suggestions.items():
            output.append(f"- {topic}:")
            for obj_data in objects:
                output.append(f"  . {obj_data['object']}: liked by {obj_data['user_count']} users")
        
        logger.info(f"Formatted community suggestions for task '{task}'")
        return "\n".join(output)

    def get_user_persona_graph_by_task(self, user_id: str, task: str) -> dict:
        query = """
        MATCH (u:User {id: $user_id})-[:HAS_FACT]->(f:Fact)-[:UNDER_TASK]->(t:Task {name: $task})
        RETURN f.topic AS topic, f.relation AS relation, f.object AS object
        """
        with self.driver.session() as session:
            records = session.run(query, user_id=user_id, task=task)

            # Base nodes
            nodes = {
                user_id: {"id": user_id, "label": user_id, "type": "User"},
                task: {"id": task, "label": task, "type": "Task"}
            }
            edges = [
                {"source": user_id, "target": task, "label": "has_task"}
            ]

            # Add topic → object connections
            for record in records:
                topic = record["topic"]
                obj = record["object"]
                relation = record["relation"]

                # Create topic node if not exists
                if topic not in nodes:
                    nodes[topic] = {"id": topic, "label": topic, "type": "Topic"}

                # Create object node if not exists
                if obj not in nodes:
                    nodes[obj] = {"id": obj, "label": obj, "type": "Object"}

                # Connect task → topic
                edges.append({
                    "source": task,
                    "target": topic,
                    "label": "has_topic"
                })

                # Connect topic → object (use relation as label)
                edges.append({
                    "source": topic,
                    "target": obj,
                    "label": relation
                })

            return {
                "nodes": list(nodes.values()),
                "edges": edges
            }


    def clear_database(self):
        logger.warning("Clearing the entire Neo4j database!")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

class SQLitePersonaDB:
    def __init__(self, db_path="src/data/persona.db"):
        # logger.info("Initializing SQLitePersonaDB")
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_table()
        logger.info("SQLitePersonaDB initialized")

    def _create_table(self):
        # logger.info("Creating table if not exists")
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS persona_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    task TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    object TEXT NOT NULL
                );
            """)
        # logger.info("Table ensured")

    def close(self):
        logger.info("Closing SQLite connection")
        self.conn.close()

    def create_user(self, user_id: str):
        # Not needed in SQLite unless you want a separate users table
        logger.info(f"create_user() skipped (user_id tracked in facts): {user_id}")

    def insert_persona_fact(self, user_id: str, relation: str, obj: str, topic: str, task: str):
        with self.conn:
            self.conn.execute("""
                INSERT INTO persona_facts (user_id, relation, object, topic, task)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, relation, obj, topic, task))
        
        logger.info(f"Insert complete: {user_id}, {relation}, {obj}, {topic}, {task}")

    def get_community_suggestions(self, user_id: str, task: str, limit: int = 10):
        # logger.info(f"Getting community suggestions for task '{task}' (excluding user: {user_id})")
        query = """
        SELECT topic, object, relation, COUNT(DISTINCT user_id) as user_count
        FROM persona_facts
        WHERE task = ? AND user_id != ?
        GROUP BY topic, object, relation
        ORDER BY topic, user_count DESC
        LIMIT ?
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (task, user_id, limit))
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

        # logger.info("Community suggestions retrieved")
        return suggestions

    def format_community_suggestions(self, user_id: str, task: str, limit: int = 10) -> str:
        logger.info(f"Formatting suggestions for user {user_id} and task '{task}'")
        suggestions = self.get_community_suggestions(user_id, task, limit)

        if not suggestions:
            return f"No suggestions found for task: {task}"

        output = [f"Suggestions for all related topics under {task}:"]
        for topic, objects in suggestions.items():
            output.append(f"- {topic}:")
            for obj in objects:
                output.append(f"  . {obj['object']}: liked by {obj['user_count']} users")

        return "\n".join(output)

    def get_user_persona_graph_by_task(self, user_id: str, task: str) -> dict:
        logger.info(f"Building persona graph for user '{user_id}' and task '{task}'")
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT topic, relation, object
            FROM persona_facts
            WHERE user_id = ? AND task = ?
        """, (user_id, task))
        records = cursor.fetchall()

        # Base nodes
        nodes = {
            user_id: {"id": user_id, "label": user_id, "type": "User"},
            task: {"id": task, "label": task, "type": "Task"}
        }
        edges = [{"source": user_id, "target": task, "label": "has_task"}]

        for topic, relation, obj in records:
            if topic not in nodes:
                nodes[topic] = {"id": topic, "label": topic, "type": "Topic"}
            if obj not in nodes:
                nodes[obj] = {"id": obj, "label": obj, "type": "Object"}

            edges.append({"source": task, "target": topic, "label": "has_topic"})
            edges.append({"source": topic, "target": obj, "label": relation})

        logger.info("Persona graph built")
        return {
            "nodes": list(nodes.values()),
            "edges": edges
        }

    def clear_database(self):
        logger.warning("Clearing all persona facts from SQLite!")
        with self.conn:
            self.conn.execute("DELETE FROM persona_facts")
        logger.info("All facts cleared")
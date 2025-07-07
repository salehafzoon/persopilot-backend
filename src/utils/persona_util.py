from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

class Neo4jPersonaDB:
    def __init__(self):
        load_dotenv()
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._create_constraints()

    def close(self):
        self.driver.close()

    def _create_constraints(self):
        """
        Ensures uniqueness of Task, Topic, and Object nodes per user_id.
        Also ensures uniqueness of GlobalTopic and GlobalObject.
        """
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

    def create_user(self, user_id: str):
        query = "MERGE (:User {id: $user_id})"
        with self.driver.session() as session:
            session.run(query, user_id=user_id)

    def insert_persona_fact(self, user_id: str, relation: str, obj: str, topic: str, task: str):
        """
        Stores a user-specific persona fact and links topic/object to global references.
        """
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

    def get_user_profile(self, user_id: str):
        """
        Returns a list of user persona facts:
        {relation, object, topic, task}
        """
        query = """
        MATCH (u:User {id: $user_id})
              -[:INITIATES_TASK]->(task:Task {user_id: $user_id})
              -[:HAS_TOPIC]->(topic:Topic {user_id: $user_id})
        MATCH (topic)-[r]->(obj:Object {user_id: $user_id})
        RETURN type(r) AS relation, obj.name AS object, topic.name AS topic, task.name AS task
        """
        with self.driver.session() as session:
            result = session.run(query, user_id=user_id)
            return [record.data() for record in result]

    def recommend_objects_from_similar_users(self, user_id: str, topic: str, subject: str, limit: int = 5):
        """
        Recommends other objects under the same topic,
        based on users who also added the same subject.
        Also returns how many users had that shared subject (XAI).
        """
        query = """
        MATCH (:Topic {name: $topic, user_id: $user_id})-[:REFERS_TO]->(gt:GlobalTopic)
        MATCH (:Object {name: $subject, user_id: $user_id})-[:REFERS_TO]->(go:GlobalObject)

        // Find other users with the same subject under same topic
        MATCH (other_obj:Object)-[:REFERS_TO]->(go)
        MATCH (other_topic:Topic)-[:REFERS_TO]->(gt)
        WHERE other_obj.user_id <> $user_id AND other_topic.user_id = other_obj.user_id

        // Ensure they’re actually connected
        MATCH (other_topic)-[r1]->(other_obj)

        // Get other objects under same topic from same users
        MATCH (other_topic)-[r2]->(o:Object)
        WHERE o.name <> $subject AND o.user_id = other_topic.user_id

        RETURN DISTINCT o.name AS recommended_object,
                        COUNT(DISTINCT other_obj.user_id) AS supporting_users,
                        COLLECT(DISTINCT type(r2)) AS relations
        ORDER BY supporting_users DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(query, user_id=user_id, topic=topic, subject=subject, limit=limit)
            return [record.data() for record in result]

    def clear_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

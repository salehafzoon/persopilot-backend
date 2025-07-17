import logging
from neo4j import GraphDatabase
from dotenv import load_dotenv
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

    def get_user_profile(self, user_id: str):
        logger.info(f"Getting user profile for user: {user_id}")
        query = """
        MATCH (u:User {id: $user_id})
              -[:INITIATES_TASK]->(task:Task {user_id: $user_id})
              -[:HAS_TOPIC]->(topic:Topic {user_id: $user_id})
        MATCH (topic)-[r]->(obj:Object {user_id: $user_id})
        RETURN type(r) AS relation, obj.name AS object, topic.name AS topic, task.name AS task
        """
        with self.driver.session() as session:
            result = session.run(query, user_id=user_id)
            data = [record.data() for record in result]
        logger.info(f"User profile retrieved for user: {user_id}")
        return data

    def recommend_objects_from_similar_users(self, user_id: str, topic: str, subject: str, limit: int = 5):
        logger.info(f"Recommending objects for user {user_id} on topic '{topic}' and subject '{subject}'")
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
            data = [record.data() for record in result]
        logger.info(f"Recommendations generated for user {user_id}")
        return data

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


    def clear_database(self):
        logger.warning("Clearing the entire Neo4j database!")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

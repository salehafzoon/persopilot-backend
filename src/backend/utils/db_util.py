from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

class Neo4jDB:
    def __init__(self):
        load_dotenv()
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_user(self, user_id: str):
        query = "MERGE (:User {id: $user_id})"
        with self.driver.session() as session:
            session.run(query, user_id=user_id)

    def create_conversation(self, conv_id: str, user_id: str):
        query = """
        MERGE (c:Conversation {id: $conv_id})
        WITH c
        MATCH (u:User {id: $user_id})
        MERGE (u)-[:STARTED]->(c)
        """
        with self.driver.session() as session:
            session.run(query, user_id=user_id, conv_id=conv_id)

    def insert_persona_fact(self, user_id: str, conv_id: str, relation: str, obj: str, topic: str):
        query = """
        MERGE (u:User {id: $user_id})
        MERGE (o:Object {name: $object})
        MERGE (t:Topic {name: $topic})
        MERGE (t)-[:RELATION {type: $relation}]->(o)
        MERGE (c:Conversation {id: $conv_id})
        MERGE (c)-[:MENTIONED]->(t)
        MERGE (u)-[:STARTED]->(c)
        """
        with self.driver.session() as session:
            session.run(
                query,
                user_id=user_id,
                object=obj,
                relation=relation,
                topic=topic,
                conv_id=conv_id
            )

    def get_user_profile(self, user_id: str):
        query = """
        MATCH (u:User {id: $user_id})-[:STARTED]->(:Conversation)-[:MENTIONED]->(t:Topic)-[r:RELATION]->(o:Object)
        RETURN t.name AS topic, r.type AS relation, o.name AS object
        """
        with self.driver.session() as session:
            result = session.run(query, user_id=user_id)
            return [record.data() for record in result]

    def clear_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")



# ------------------------
# Test the Neo4jDB class
# ------------------------

# db = Neo4jDB()
# db.create_user("u001")
# db.create_conversation("c001", "u001")
# db.insert_persona_fact("u001", "c001", "likes", "jazz music", "music")

# profile = db.get_user_profile("u001")
# print(profile)

# db.close()

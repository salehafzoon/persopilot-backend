import pytest
from utils.db_util import Neo4jDB

@pytest.fixture(scope="module")
def db():
    db_instance = Neo4jDB()
    yield db_instance
    db_instance.close()

@pytest.fixture
def test_user_id():
    return "test_user_001"

@pytest.fixture
def test_conv_id():
    return "conv_001"

def test_create_user(db, test_user_id):
    try:
        db.create_user(test_user_id)
    except Exception as e:
        pytest.fail(f"User creation failed: {str(e)}")

def test_create_conversation(db, test_user_id, test_conv_id):
    try:
        db.create_conversation(conv_id=test_conv_id, user_id=test_user_id)
    except Exception as e:
        pytest.fail(f"Conversation creation failed: {str(e)}")

def test_insert_persona_fact(db, test_user_id, test_conv_id):
    try:
        db.insert_persona_fact(
            user_id=test_user_id,
            conv_id=test_conv_id,
            relation="likes",
            obj="space documentaries",
            topic="entertainment"
        )
    except Exception as e:
        pytest.fail(f"Inserting persona fact failed: {str(e)}")

def test_get_user_profile(db, test_user_id):
    profile = db.get_user_profile(test_user_id)
    assert isinstance(profile, list)
    assert any("topic" in entry and "relation" in entry and "object" in entry for entry in profile)

def test_clear_database(db):
    try:
        db.clear_database()
    except Exception as e:
        pytest.fail(f"Clearing DB failed: {str(e)}")



# pytest src/backend/tests/test_db_util.py -v

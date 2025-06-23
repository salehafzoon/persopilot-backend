import pytest
from tools.persona_extractor import persona_extractor, SentenceInput
import logging
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def example_input():
    return SentenceInput(sentence="I used to play volleyball but now I am playing soccer.")

def test_persona_extractor_tool_success(example_input):
    result = persona_extractor.invoke(example_input.model_dump())

    assert isinstance(result, (list, tuple)), "Output should be a tuple or list"
    assert len(result) == 3, "Output must contain exactly 3 elements: (subject, relation, object)"
    
    subject, relation, obj = result
    assert isinstance(subject, str) and subject != "", "Subject must be a non-empty string"
    assert isinstance(relation, str) and relation != "", "Relation must be a non-empty string"
    assert isinstance(obj, str) and obj != "", "Object must be a non-empty string"

    logger.info(f"Extracted: ({subject}, {relation}, {obj})")


# pytest src/backend/tests/test_persona_extractor.py -s

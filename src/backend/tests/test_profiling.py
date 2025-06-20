import pytest
from tools.user_profiler import user_profiler_tool, UserProfilerInput

@pytest.fixture
def dummy_input():
    return UserProfilerInput(
        user_id="salehaf",
        conv_id="conv_002",
        subject="reading",
        relation="likes",
        obj="space documentaries",
        topic="entertainment"
    )

def test_user_profiler_tool_success(dummy_input):
    result = user_profiler_tool.invoke(dummy_input.dict())

    assert isinstance(result, str)
    assert "success" in result.lower() or "inserted" in result.lower()


# pytest src/backend/tests/test_profiling.py
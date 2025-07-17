# src/tools/community_recommender.py

import logging
from typing import List
from pydantic import BaseModel, Field
from langchain.tools import Tool

from src.utils.persona_util import Neo4jPersonaDB, SQLitePersonaDB

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Input Schema
class RecommendationInput(BaseModel):
    sentence: str = Field(..., description="A sentence requesting community-based recommendations.")

# Tool Wrapper
def create_community_recommender_tool(user_id: str, task: str) -> Tool:
    # persona_db = Neo4jPersonaDB()
    persona_db = SQLitePersonaDB()

    def recommend(sentence: str) -> str:
        try:
            logger.info(f"[TOOL INVOKED] Community recommender triggered with input: {sentence}")

            # Get community suggestions for the task
            formatted_suggestions = persona_db.format_community_suggestions(user_id, task)
            
            if "No suggestions found" in formatted_suggestions:
                return f"[TOOL RESULT]\nType: Recommendation: No community recommendations available for task '{task}'."

            return f"[TOOL RESULT]\nType: Recommendation:\n{formatted_suggestions}"

        except Exception as e:
            logger.error(f"[COMMUNITY TOOL ERROR] {e}")
            return "[TOOL RESULT]\nType: Recommendation: Failed to generate recommendations due to an internal error."

    return Tool(
        name="CommunityRecommender",
        description=f"Provides community-based recommendations for all topics under the {task} task, showing what other users in the community prefer.",
        func=recommend,
        args_schema=RecommendationInput
    )

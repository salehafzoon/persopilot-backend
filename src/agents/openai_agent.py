from src.tools.community_recommender import create_community_recommender_tool
from src.tools.persona_extractor import get_persona_extractor_tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from src.tools.searcher import duckduckgo_search_tool 
from langchain.schema import SystemMessage
from langchain.chat_models import ChatOpenAI
from src.llm.prompts import AgentPrompt_OpenAI
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
    

class PersoAgentOpenAI:
    def __init__(self, user_id: str, prev_personas: str, task: str = "Content Consumption"):
        
        self.user_id = user_id
        self.task = task
        self.prev_personas = prev_personas
       

        self.model = ChatOpenAI(
            model_name="gpt-3.5-turbo",  # or "gpt-4o"
            temperature=0.1,
            openai_api_key=os.environ["OPENAI_API_KEY"]  # or pass it securely another way
        )

        
        # Tools
        persona_extractor_tool = get_persona_extractor_tool(user_id, task)
        community_recommender = create_community_recommender_tool(user_id, task)
        search_tool = duckduckgo_search_tool
        
        self.tools = [persona_extractor_tool, community_recommender, search_tool]

        # Memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Inject metadata as system message
        metadata_message = SystemMessage(
            content=
            f"""User Profile:
            - User ID: {user_id}
            - Current Task: {task}

            Known Persona Facts related to the current task:
            {self.prev_personas.strip()}
            """
        )
        self.memory.chat_memory.messages.insert(0, metadata_message)

        # Agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.model,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True,       
            agent_kwargs={
                "system_message": AgentPrompt_OpenAI(user_id, task)
                
            }
        )

    def handle_task(self, task: str) -> str:
        """Run the agent on a task and return the response."""
        response = self.agent.invoke({"input": task})
        return response.get("output", "No response generated.")

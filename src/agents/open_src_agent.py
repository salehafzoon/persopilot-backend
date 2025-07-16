# src/agents/agent.py

from src.tools.community_recommender import create_community_recommender_tool
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from src.tools.persona_extractor import get_persona_extractor_tool
from langchain.agents import initialize_agent, AgentType
from langchain_community.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from src.tools.searcher import duckduckgo_search_tool 
from langchain.schema import SystemMessage
from src.llm.prompts import AgentPrompt_HF
import logging
import torch
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
    

class PersoAgentHF:
    def __init__(self, model_path: str, tokenizer_path: str, user_id: str, prev_personas: str, task: str = "Content Consumption"):
        
        self.user_id = user_id
        self.task = task
        self.prev_personas = prev_personas
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=128,
            return_full_text=False,
            temperature=0.1,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id
        )

        self.model = HuggingFacePipeline(pipeline=pipe)
       
        
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
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True,       
            agent_kwargs={
                "prefix": AgentPrompt_HF(user_id, task)    
            }
        )

    def handle_task(self, task: str) -> str:
        """Run the agent on a task and return the response."""
        response = self.agent.invoke({"input": task})
        return response.get("output", "No response generated.")

# src/agents/agent.py

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from src.tools.persona_extractor import persona_extractor
from langchain.agents import initialize_agent, AgentType
from langchain_community.llms import HuggingFacePipeline
from src.tools.searcher import duckduckgo_search_tool 
from langchain.memory import ConversationBufferMemory
from src.llm.prompts import AgentPrompt
import logging
import torch
import os

logger = logging.getLogger(__name__)

class PersoAgent:
    def __init__(self, model_path: str, tokenizer_path: str, task: str = "Content Consumption", personas: str = None):
        
        # Load tokenizer and model from provided local folders
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Wrap model in HuggingFace pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.04,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id
        )

        self.model = HuggingFacePipeline(pipeline=pipe)

        # Register tools
        self.tools = [persona_extractor, duckduckgo_search_tool]
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize agent with custom prompt and memory
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.model,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True,
            # max_iterations=2,               # safety net
            agent_kwargs={
                "prefix": AgentPrompt(task, personas)
            }
        )

    def handle_task(self, task: str, user_id: str = "default_user", conv_id: str = "default_conv") -> str:
        """Run the agent on a task and return the response."""
        response = self.agent.invoke({"input": task})
        return response.get("output", "No response generated.")

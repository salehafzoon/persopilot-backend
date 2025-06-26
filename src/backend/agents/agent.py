# src/backend/agents/agent.py

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from src.backend.tools.persona_extractor import persona_extractor
from langchain.agents import initialize_agent, AgentType
from langchain_community.llms import HuggingFacePipeline
from src.backend.llm.prompts import AgentPrompt
import logging
import torch
import os

logger = logging.getLogger(__name__)

class PersoAgent:
    def __init__(self, model_path: str, tokenizer_path: str):
        
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
            temperature=0.2,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id
        )

        self.model = HuggingFacePipeline(pipeline=pipe)

        # Register tools
        self.tools = [persona_extractor]
        
        # Initialize agent with custom prompt
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.model,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            agent_kwargs={
                "prefix": AgentPrompt,
                "format_instructions": "Use tools when needed and respond naturally."
            }
        )

    def handle_task(self, task: str, user_id: str = "default_user", conv_id: str = "default_conv") -> str:
        """Run the agent on a task and return the response."""
        response = self.agent.invoke({"input": task})
        return response.get("output", "No response generated.")

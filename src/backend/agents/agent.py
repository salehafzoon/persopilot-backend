# src/backend/agents/agent.py

from langchain.agents import initialize_agent, AgentType
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from src.backend.llm.prompts import AgentPrompt
from src.backend.tools.user_profiler import user_profiler_tool
from src.backend.tools.persona_extractor import persona_extractor
import torch
import logging
import os
import json
import ast

logger = logging.getLogger(__name__)

class PersoAgent:
    def __init__(self, model_path: str, tokenizer_path: str):
        """Initialize the agent using Phi-4 model and tokenizer from local paths."""
        # Load tokenizer and model from provided local folders
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"  # Automatically uses GPU if available
        )

        # Wrap model in HuggingFace pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.2,
            return_full_text=False
        )

        self.model = HuggingFacePipeline(pipeline=pipe)

        # Register tools
        self.tools = [persona_extractor, user_profiler_tool]

        # Initialize the agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.model,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            agent_kwargs={"prefix": AgentPrompt},
            return_intermediate_steps=True
        )

    def handle_task(self, task: str) -> dict:
        """Run the agent on a task and return parsed JSON."""
        response = self.agent.invoke({"input": task})

        try:
            output = response.get("output")
            if isinstance(output, dict):
                return output
            try:
                return ast.literal_eval(output)
            except (SyntaxError, ValueError):
                return json.loads(output)
        except Exception as e:
            logger.error(f"Error: {e}")
            raise ValueError(f"Agent returned invalid output: {output}")

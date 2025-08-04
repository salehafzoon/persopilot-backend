# src/tools/dialogpt_persona_extractor.py

import json
import torch
import logging
import re
from typing import Dict, Any
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.tools import Tool

from src.utils.topic_modeller import TopicModeller
from src.utils.persona_util import SQLitePersonaDB

logger = logging.getLogger(__name__)

class SentenceInput(BaseModel):
    """Input schema for extracting a persona triplet from a user sentence."""
    sentence: str = Field(..., description="A sentence expressing a user preference.")

class DialogGPTPersonaExtractor:
    def __init__(self, username: str, task: str):
        self.username = username
        self.task = task
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load DialogGPT model
        model_path = "microsoft/DialoGPT-small"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.to(self.device)
        self.model.eval()
        
        # Load relation types
        with open("src/data/ConvAI2/u2t_map_all.json", "r") as f:
            raw_data = json.load(f)
        self.relations = sorted({ex["triplets"][0]["label"] for ex in raw_data})
        
        # Utils
        self.topic_modeller = TopicModeller()
        self.persona_db = SQLitePersonaDB()

    def extract(self, sentence: str) -> str:
        logger.info(f"[START] DialogGPT extract() called with: {sentence}")
        
        # Create extraction prompt
        prompt = f"""Extract persona information from this sentence.
Sentence: "{sentence}"
Format: {{"relation": "relation_type", "object": "extracted_object"}}
JSON:"""
        
        # Tokenize and generate
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 30,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        # Parse response
        relation, obj = self._parse_response(response, sentence)
        
        # Infer topic
        topic = self.topic_modeller.infer_topic(obj)
        
        logger.info(f"[EXTRACTED] relation='{relation}', object='{obj}', topic='{topic}', task='{self.task}'")
        
        # Store in database
        self.persona_db.insert_persona_fact_by_name(
            username=self.username, 
            relation=relation, 
            obj=obj, 
            topic=topic, 
            task=self.task
        )
        logger.info(f"[SAVED TO DB] for username='{self.username}'")
        
        return f"[EXTRACTED] relation='{relation}', object='{obj}', topic='{topic}', task='{self.task}'"

    def _parse_response(self, response: str, original_sentence: str) -> tuple:
        """Parse DialogGPT response to extract relation and object."""
        try:
            # Try to parse JSON
            json_match = re.search(r'\{[^}]*\}', response)
            if json_match:
                result = json.loads(json_match.group())
                relation = result.get("relation", "like")
                obj = result.get("object", "")
                if obj:
                    return self._map_relation(relation), obj
        except:
            pass
        
        # Fallback: extract from quotes or use simple heuristics
        quote_match = re.search(r'"([^"]+)"', response)
        if quote_match:
            obj = quote_match.group(1)
            relation = self._infer_relation(original_sentence)
            return relation, obj
        
        # Last resort: extract key phrases from original sentence
        obj = self._extract_fallback_object(original_sentence)
        relation = self._infer_relation(original_sentence)
        
        return relation, obj

    def _map_relation(self, relation: str) -> str:
        """Map generated relation to known relation types."""
        relation_lower = relation.lower()
        
        # Simple mapping
        if "like" in relation_lower or "love" in relation_lower:
            return "like"
        elif "dislike" in relation_lower or "hate" in relation_lower:
            return "dislike"
        elif "work" in relation_lower or "job" in relation_lower:
            return "work_as"
        elif "play" in relation_lower or "sport" in relation_lower:
            return "like_sports"
        elif "eat" in relation_lower or "food" in relation_lower:
            return "favorite_food"
        
        return "like"  # default

    def _infer_relation(self, sentence: str) -> str:
        """Infer relation type from sentence patterns."""
        sentence_lower = sentence.lower()
        
        if any(word in sentence_lower for word in ["work as", "job", "profession"]):
            return "work_as"
        elif any(word in sentence_lower for word in ["play", "sport", "game"]):
            return "like_sports"
        elif any(word in sentence_lower for word in ["eat", "food", "meal"]):
            return "favorite_food"
        elif any(word in sentence_lower for word in ["dislike", "hate", "don't like"]):
            return "dislike"
        else:
            return "like"

    def _extract_fallback_object(self, sentence: str) -> str:
        """Extract object using simple patterns as fallback."""
        # Remove common words and extract meaningful phrases
        words = sentence.lower().split()
        stop_words = {"i", "am", "a", "an", "the", "like", "to", "work", "as", "play", "eat"}
        
        meaningful_words = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Look for multi-word phrases
        if len(meaningful_words) >= 2:
            # Check for common multi-word patterns
            text = " ".join(meaningful_words)
            patterns = [
                r'(data scientist|software engineer|machine learning|artificial intelligence)',
                r'(\w+\s+\w+)'  # any two-word combination
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    return match.group(1)
        
        return meaningful_words[0] if meaningful_words else "unknown"

def get_dialogpt_persona_extractor_tool(username: str, task: str) -> Tool:
    """Create DialogGPT-based persona extractor tool."""
    extractor_instance = DialogGPTPersonaExtractor(username=username, task=task)
    
    return Tool(
        name="DialogGPTPersonaExtractor",
        description="Extracts user persona using DialogGPT model.",
        func=extractor_instance.extract,
        args_schema=SentenceInput
    )

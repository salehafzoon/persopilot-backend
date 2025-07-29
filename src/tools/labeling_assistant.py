# Set memory limits before importing transformers
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from transformers import GPT2Tokenizer, AutoModelForCausalLM
from src.llm.prompts import labeling_assistant_prompt
from src.utils.persona_util import SQLitePersonaDB
import json
import re
import gc

import logging
logger = logging.getLogger("persona_util")
logging.basicConfig(level=logging.INFO)


class LabelingAssistant:

    def __init__(self):
        # Use 75% of GPU memory (9GB out of 12GB)
        torch.cuda.set_per_process_memory_fraction(0.75)
        torch.cuda.empty_cache()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = GPT2Tokenizer.from_pretrained("./Tokenizers/Phi-4-mini-instruct")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "./LLMs/Phi-4-mini-instruct", 
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.db = SQLitePersonaDB()

    def clean_reasoning(self, reasoning_text):
        """Extract clean reasoning from dirty JSON response."""
        try:
            # Remove markdown code blocks
            if "```json" in reasoning_text:
                start = reasoning_text.find("```json") + 7
                end = reasoning_text.find("```", start)
                if end != -1:
                    json_part = reasoning_text[start:end].strip()
                else:
                    json_part = reasoning_text[start:].strip()
            else:
                json_part = reasoning_text
            
            # Remove any trailing tokens like <|end|>
            if "<|end|>" in json_part:
                json_part = json_part.split("<|end|>")[0].strip()
            
            # Try to parse and extract reasoning
            try:
                parsed = json.loads(json_part)
                return parsed.get("reasoning", reasoning_text)
            except:
                return reasoning_text
                
        except:
            return reasoning_text


    def get_alignment_score(self, pool_size: int, clf_task_name: str) -> list:
    
        try:

            candidates = self.db.get_random_candidate_usernames(clf_task_name, pool_size)
            cls_task = self.db.get_classification_task_by_name(clf_task_name)
            
            if not cls_task:
                return None
            
            logger.info(f"Found {len(candidates)} candidates for task '{clf_task_name}'")

            labeling_results = []
            
            for username in candidates:
                user = self.db.get_user(username)
                user_persona = self.db.get_user_persona_summary(username)

                logger.info(f"Persona Summary for {username}: \n{user_persona}")

                if not user_persona or not user:
                    continue
                
                # Generate prompt
                prompt = labeling_assistant_prompt(user_persona, cls_task['description'])
                
                # Tokenize and prepare inputs
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                
                # Generate response
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        temperature=0.3,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        use_cache=False
                    )
                
                # Decode response
                response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                
                del outputs, inputs
                torch.cuda.empty_cache()
                
                # Parse response
                try:
                    json_match = re.search(r'\{[^}]*"score"[^}]*\}', response)
                    if json_match:
                        result = json.loads(json_match.group())
                        score = result.get('score', 0.5)
                    else:
                        score_match = re.search(r'([0-9.]+)', response)
                        score = float(score_match.group(1)) if score_match else 0.5
                except:
                    score = 0.5
                
                labeling_results.append({
                    "username": username,
                    "full_name": user['full_name'],
                    "age": user['age'],
                    "gender": user['gender'],
                    "score": min(1.0, score),
                    "reasoning": self.clean_reasoning(response.strip())
                })
            
            return labeling_results
        finally:
            self.reset_memory()
    
    def reset_memory(self):
        # Move model to CPU and delete references
        if hasattr(self, 'model'):
            self.model.cpu()
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(1.0)
        
        logger.info("GPU memory released")






# # Usage
# assistant = LabelingAssistant()

# user_persona = "Content Consumption: | Book: likes reading novels | Game: plays video games | Movie: likes watching sci-fi | Music: likes classical music"
# classification = "Classify users based on their passion for reading books and literature."

# result = assistant.get_alignment_score(user_persona, classification)
# print(f"Score: {result['score']}")
# print(f"Reasoning: {result['reasoning']}")

# assistant.reset_memory()

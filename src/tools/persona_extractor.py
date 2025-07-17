# src/tools/persona_extractor.py

import json
import torch
import logging
import warnings
from torch import nn
from typing import Tuple
from pydantic import BaseModel, Field
from transformers import BertTokenizerFast, BertModel
from langchain.tools import StructuredTool

from src.utils.topic_modeller import TopicModeller
from src.utils.persona_util import Neo4jPersonaDB, SQLitePersonaDB

warnings.filterwarnings("ignore")

# ------------------------
# Logging Setup
# ------------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------
# Input Schema
# ------------------------

class SentenceInput(BaseModel):
    """Input schema for extracting a persona triplet from a user sentence."""
    sentence: str = Field(..., description="A sentence expressing a user preference.")

# ------------------------
# JointBERT Model Definition
# ------------------------

class JointBertExtractor(nn.Module):
    def __init__(self, base_model='bert-base-uncased', num_token_labels=4, num_relation_labels=0):
        super().__init__()
        self.bert = BertModel.from_pretrained(base_model)
        self.dropout = nn.Dropout(0.1)
        self.token_classifier = nn.Linear(self.bert.config.hidden_size, num_token_labels)
        self.relation_classifier = nn.Linear(self.bert.config.hidden_size, num_relation_labels)

    def forward(self, input_ids, attention_mask, labels=None, relation_label=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        cls_output = self.dropout(outputs.pooler_output)

        token_logits = self.token_classifier(sequence_output)
        relation_logits = self.relation_classifier(cls_output)

        loss = None
        if labels is not None and relation_label is not None:
            loss_fct = nn.CrossEntropyLoss()
            token_loss = loss_fct(token_logits.view(-1, token_logits.shape[-1]), labels.view(-1))
            relation_loss = loss_fct(relation_logits, relation_label)
            loss = token_loss + relation_loss

        return {
            "loss": loss,
            "token_logits": token_logits,
            "relation_logits": relation_logits
        }

# ------------------------
# PersonaExtractor Class
# ------------------------

class PersonaExtractor:
    def __init__(self, user_id: str, task: str):
        
        self.user_id = user_id
        self.task = task
        
        model_path = "src/llm/PExtractor"
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        

        # Load model + tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        self.model = JointBertExtractor(num_token_labels=4, num_relation_labels=105)
        self.model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin", map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Load label maps
        self.id2label = {i: label for i, label in enumerate(['O', 'B-SUB', 'B-OBJ', 'I-OBJ'])}
        with open("src/data/ConvAI2/u2t_map_all.json", "r") as f:
            raw_data = json.load(f)
        relation_list = sorted({ex["triplets"][0]["label"] for ex in raw_data})
        self.id2relation = {i: rel for i, rel in enumerate(relation_list)}

        # Utils
        self.topic_modeller = TopicModeller()
        # self.persona_db = Neo4jPersonaDB()
        self.persona_db = SQLitePersonaDB()

    def extract(self, sentence: str) -> str:
        # logger.info(f"[START] extract() called with: {sentence}")
        
        tokenizer = self.tokenizer
        model = self.model
        device = self.device
        id2label = self.id2label
        id2relation = self.id2relation

        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

        token_preds = torch.argmax(outputs["token_logits"], dim=-1).squeeze().cpu().tolist()
        tokens_decoded = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())
        attention_mask = inputs["attention_mask"].squeeze().cpu().tolist()

        obj_tokens = []
        for token, label_id, mask in zip(tokens_decoded, token_preds, attention_mask):
            if mask == 0 or token in ["[PAD]", "[CLS]", "[SEP]"]:
                continue
            label = id2label.get(label_id, "O")
            if label.startswith("B-OBJ") or label.startswith("I-OBJ"):
                obj_tokens.append(token)

        rel_pred_id = torch.argmax(outputs["relation_logits"], dim=-1).item()
        relation = id2relation[rel_pred_id]
        object_str = tokenizer.convert_tokens_to_string(obj_tokens).strip()

        # Infer topic
        topic = self.topic_modeller.infer_topic(object_str)

        # Log
        logger.info(f"[EXTRACTED] relation='{relation}', object='{object_str}', topic='{topic}', task='{self.task}'")

        # Store in Neo4j
        self.persona_db.insert_persona_fact(user_id=self.user_id, relation=relation, obj=object_str, topic=topic, task=self.task)
        logger.info(f"[SAVED TO DB] for user_id='{self.user_id}'")
        
        return f"The persona has extracted. Just inform the user."


# ------------------------
# Create LangChain Tool
# ------------------------

from langchain.tools import Tool

def get_persona_extractor_tool(user_id: str, task: str) -> Tool:
    extractor_instance = PersonaExtractor(user_id=user_id, task=task)
    # logger.info(f"[TOOL CREATION] PersonaExtractor instance created successfully")

    
    return Tool(
        name="PersonaExtractor",
        description="Extracts a user persona from the input.",
        func=extractor_instance.extract,
        args_schema=SentenceInput 
    )
    
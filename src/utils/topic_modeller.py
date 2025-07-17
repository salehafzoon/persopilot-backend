from transformers import pipeline
from typing import List
import json

class TopicModeller:
    def __init__(self, topic_list: List[str] = None):

        with open("src/data/task_topic.json", "r", encoding="utf-8") as f:
            task_topic_data = json.load(f)

        # Define default topic list if not provided
        self.topics = [topic for topics in task_topic_data.values() for topic in topics]
        
        # Load zero-shot classifier pipeline with BART
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    def infer_topic(self, sentence: str) -> str:
        """
        Infers the most likely topic for a given input sentence.
        Returns the topic with the highest confidence.
        """
        result = self.classifier(sentence, self.topics, multi_label=False)
        return result["labels"][0]  # Top topic

    def infer_top_k(self, sentence: str, k: int = 3) -> List[str]:
        """
        Optional: Returns top-k most relevant topics
        """
        result = self.classifier(sentence, self.topics, multi_label=False)
        return result["labels"][:k]

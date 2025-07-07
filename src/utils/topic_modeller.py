from transformers import pipeline
from typing import List

class TopicModeller:
    def __init__(self, topic_list: List[str] = None):
        # Define default topic list if not provided
        self.topics = topic_list or [
            "Book", "Movie", "Music", "Game",  # For Content Consumption
            "Exercise", "Mental Health", "Food",       # For Lifestyle Optimization
            "Career", "Skill", "Education"     # For Career Development
        ]
        
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

import logging
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.persona_util import SQLitePersonaDB

logger = logging.getLogger(__name__)

class PersonaClassifier:
    def __init__(self, db_path="src/data/persona.db"):
        self.db = SQLitePersonaDB(db_path)
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self.training_data = None
        self.training_labels = None
        
    def train(self, classification_task_name: str) -> Dict:
        """Train classifier using accepted/declined responses"""
        responses = self.db.get_classification_task_responses(classification_task_name)
        if not responses:
            return {"error": "No training data found"}
        
        training_texts = []
        training_labels = []
        
        for response in responses:
            username = response["username"]
            full_summary = self.db.get_user_persona_summary(username)
            if not full_summary.startswith("No"):
                filtered_summary = self.db.filter_persona_summary_from_topics(full_summary)
                training_texts.append(filtered_summary)
                training_labels.append(1 if response["status"] == "accepted" else 0)
        
        if len(training_texts) < 2:
            return {"error": "Insufficient training data"}
        
        self.training_data = self.vectorizer.fit_transform(training_texts)
        self.training_labels = np.array(training_labels)
        
        accepted_count = sum(training_labels)
        declined_count = len(training_labels) - accepted_count
        
        return {
            "success": True,
            "training_size": len(training_texts),
            "accepted": accepted_count,
            "declined": declined_count
        }
    
    def predict_and_save(self, classification_task_id: int, count: int = 20, min_confidence: float = 0.65) -> Dict:
        """Predict classification and save to database"""
        if self.training_data is None:
            return {"error": "Model not trained"}
        
        task_info = self.db.get_classification_task(classification_task_id)
        if not task_info:
            return {"error": "Classification task not found"}
        
        candidates = self.db.get_random_candidate_usernames(task_info["name"], count)
        if not candidates:
            return {"error": "No candidates found"}
        
        predictions = []
        
        for username in candidates:
            full_summary = self.db.get_user_persona_summary(username)
            if full_summary.startswith("No"):
                continue
                
            filtered_summary = self.db.filter_persona_summary_from_topics(full_summary)
            candidate_vector = self.vectorizer.transform([filtered_summary])
            
            similarities = cosine_similarity(candidate_vector, self.training_data)[0]
            
            accepted_mask = self.training_labels == 1
            declined_mask = self.training_labels == 0
            
            # Use max similarity instead of mean for better discrimination
            accepted_sim = np.max(similarities[accepted_mask]) if np.any(accepted_mask) else 0
            declined_sim = np.max(similarities[declined_mask]) if np.any(declined_mask) else 0
            
            # Better confidence calculation
            total_sim = accepted_sim + declined_sim
            if total_sim > 0:
                confidence = abs(accepted_sim - declined_sim) / total_sim
            else:
                confidence = 0
            
            if confidence >= min_confidence:
                predicted_label = task_info["label1"] if accepted_sim > declined_sim else task_info["label2"]
                
                predictions.append({
                    "username": username,
                    "predicted_label": predicted_label,
                    "confidence": round(float(confidence), 3)
                })
        
        if predictions:
            self.db.save_predictions(classification_task_id, predictions)
        
        return {
            "classification_task": {
                "id": task_info["id"],
                "name": task_info["name"]
            },
            "predictions": predictions
        }

    def load_predictions(self, classification_task_id: int) -> Dict:
        """Load saved predictions for a classification task"""
        task_info = self.db.get_classification_task(classification_task_id)
        if not task_info:
            return {"error": "Classification task not found"}
        
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT username, predicted_label, confidence, prediction_date
            FROM ClassificationPrediction
            WHERE classification_task_id = ?
            ORDER BY confidence DESC
        """, (classification_task_id,))
        
        predictions = []
        for row in cursor.fetchall():
            predictions.append({
                "username": row[0],
                "predicted_label": row[1],
                "confidence": row[2],
                "prediction_date": row[3]
            })
        
        return {
            "classification_task": {
                "id": task_info["id"],
                "name": task_info["name"]
            },
            "predictions": predictions
        }

    
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import re

class QueryEmbedder:
    def __init__(self, model_name='prajjwal1/bert-tiny'):
        print(f"Loading {model_name}...")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_embedding(self, text):
        """
        Returns: Numpy array of shape (128,) - Pure BERT CLS token
        """
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            # Use [CLS] token
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

        return cls_embedding

if __name__ == "__main__":
    # Test
    embedder = QueryEmbedder()
    vector = embedder.get_embedding("Predict temperature for tomorrow")
    print(f"Embedding shape: {vector.shape}")
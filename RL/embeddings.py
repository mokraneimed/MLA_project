import torch
from transformers import BertTokenizer, BertModel

class QueryEmbedder:
    def __init__(self, model_name='prajjwal1/bert-tiny'):
        print(f"Loading {model_name}...")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_embedding(self, text):
        """
        Converts a text string into a fixed-size vector (768 for base BERT).
        We use the [CLS] token embedding to represent the sentence state.
        """
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            
            # The 'last_hidden_state' has shape (batch, seq_len, hidden_size)
            # We take the first token [CLS] as the sentence representation
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            
        return cls_embedding.cpu().numpy().flatten()

if __name__ == "__main__":
    # Test
    embedder = QueryEmbedder()
    vector = embedder.get_embedding("Predict temperature for tomorrow")
    print(f"Embedding shape: {vector.shape}")
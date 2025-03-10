from langchain.tools import Tool
from transformers import AutoTokenizer, AutoModel
import torch

class EmbeddingTool(Tool):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def _call(self, inputs):
        texts = inputs["texts"]
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        return {"embeddings": embeddings}

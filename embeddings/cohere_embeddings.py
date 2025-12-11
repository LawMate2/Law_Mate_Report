from typing import List
import os
import cohere
from . import EmbeddingModel


class CohereEmbedding(EmbeddingModel):
    """Cohere embedding models wrapper"""

    def __init__(self, model_name: str = "embed-multilingual-v3.0", dimension: int = None):
        # Default dimensions for Cohere models
        default_dims = {
            "embed-english-v3.0": 1024,
            "embed-multilingual-v3.0": 1024,
            "embed-english-light-v3.0": 384,
            "embed-multilingual-light-v3.0": 384
        }

        if dimension is None:
            dimension = default_dims.get(model_name, 1024)

        super().__init__(model_name, dimension)

        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY not found in environment variables")

        self.client = cohere.Client(api_key)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts"""
        response = self.client.embed(
            texts=texts,
            model=self.model_name,
            input_type="search_document"
        )
        return response.embeddings

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query"""
        response = self.client.embed(
            texts=[query],
            model=self.model_name,
            input_type="search_query"
        )
        return response.embeddings[0]

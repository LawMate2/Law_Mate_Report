from typing import List
import os
from openai import OpenAI
from . import EmbeddingModel


class OpenAIEmbedding(EmbeddingModel):
    """OpenAI embedding models wrapper"""

    def __init__(self, model_name: str = "text-embedding-3-small", dimension: int = None):
        # Default dimensions for OpenAI models
        default_dims = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072
        }

        if dimension is None:
            dimension = default_dims.get(model_name, 1536)

        super().__init__(model_name, dimension)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.client = OpenAI(api_key=api_key)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts"""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        return [item.embedding for item in response.data]

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query"""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=[query]
        )
        return response.data[0].embedding

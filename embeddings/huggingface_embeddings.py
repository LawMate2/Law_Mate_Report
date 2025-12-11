from typing import List
from sentence_transformers import SentenceTransformer
from . import EmbeddingModel


class HuggingFaceEmbedding(EmbeddingModel):
    """HuggingFace sentence-transformers wrapper"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        dimension = self.model.get_sentence_embedding_dimension()

        super().__init__(model_name, dimension)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts"""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query"""
        embedding = self.model.encode([query], convert_to_numpy=True)
        return embedding[0].tolist()


class MultilingualEmbedding(HuggingFaceEmbedding):
    """Multilingual embedding model optimized for multiple languages"""

    def __init__(self):
        super().__init__(model_name="distiluse-base-multilingual-cased-v2")

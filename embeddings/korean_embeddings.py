from typing import List
from sentence_transformers import SentenceTransformer
from . import EmbeddingModel


class KoreanEmbedding(EmbeddingModel):
    """Korean-specific embedding models"""

    def __init__(self, model_name: str = "jhgan/ko-sroberta-multitask"):
        """
        Available Korean models:
        - jhgan/ko-sroberta-multitask (768 dim)
        - snunlp/KR-SBERT-V40K-klueNLI-augSTS (768 dim)
        - BM-K/KoSimCSE-roberta (768 dim)
        """
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


class KoSimCSEEmbedding(KoreanEmbedding):
    """Korean SimCSE model"""

    def __init__(self):
        super().__init__(model_name="BM-K/KoSimCSE-roberta")


class KoSRoBERTaEmbedding(KoreanEmbedding):
    """Korean SRoBERTa multitask model"""

    def __init__(self):
        super().__init__(model_name="jhgan/ko-sroberta-multitask")

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import time


class EmbeddingModel(ABC):
    """Base class for all embedding models"""

    def __init__(self, model_name: str, dimension: int):
        self.model_name = model_name
        self.dimension = dimension

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Convert texts to embeddings"""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Convert a single query to embedding"""
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Return model metadata"""
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "type": self.__class__.__name__
        }

    def measure_embedding_time(self, texts: List[str]) -> Dict[str, Any]:
        """Measure time taken to embed texts"""
        start = time.time()
        embeddings = self.embed_texts(texts)
        end = time.time()

        return {
            "total_time": end - start,
            "avg_time_per_text": (end - start) / len(texts),
            "num_texts": len(texts)
        }

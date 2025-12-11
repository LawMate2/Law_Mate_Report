from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import time


class VectorStore(ABC):
    """Base class for all vector stores"""

    def __init__(self, store_name: str, dimension: int):
        self.store_name = store_name
        self.dimension = dimension
        self.num_vectors = 0

    @abstractmethod
    def add_texts(self, texts: List[str], embeddings: List[List[float]], metadatas: List[Dict] = None):
        """Add texts with their embeddings to the store"""
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar vectors and return (text, score) tuples"""
        pass

    @abstractmethod
    def delete_collection(self):
        """Delete the collection/index"""
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Return store metadata"""
        return {
            "store_name": self.store_name,
            "dimension": self.dimension,
            "num_vectors": self.num_vectors,
            "type": self.__class__.__name__
        }

    def measure_search_time(self, query_embedding: List[float], k: int = 5, iterations: int = 10) -> Dict[str, Any]:
        """Measure average search time"""
        times = []

        for _ in range(iterations):
            start = time.time()
            self.search(query_embedding, k)
            end = time.time()
            times.append(end - start)

        return {
            "avg_search_time": sum(times) / len(times),
            "min_search_time": min(times),
            "max_search_time": max(times),
            "iterations": iterations
        }

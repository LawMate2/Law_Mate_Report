from typing import List, Tuple, Dict
import numpy as np
import faiss
from . import VectorStore


class FAISSStore(VectorStore):
    """FAISS vector store wrapper"""

    def __init__(self, dimension: int = 768, index_type: str = "Flat"):
        """
        Initialize FAISS index
        index_type options:
        - "Flat": Exact search (IndexFlatL2)
        - "IVF": Inverted file index for faster search
        - "HNSW": Hierarchical NSW for approximate search
        """
        super().__init__("FAISS", dimension)

        self.index_type = index_type
        self.texts = []
        self.metadatas = []

        if index_type == "Flat":
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "IVF":
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
            self.needs_training = True
        elif index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(dimension, 32)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

    def add_texts(self, texts: List[str], embeddings: List[List[float]], metadatas: List[Dict] = None):
        """Add texts with embeddings to FAISS"""
        embeddings_array = np.array(embeddings, dtype=np.float32)

        if self.index_type == "IVF" and hasattr(self, 'needs_training') and self.needs_training:
            self.index.train(embeddings_array)
            self.needs_training = False

        self.index.add(embeddings_array)
        self.texts.extend(texts)

        if metadatas is None:
            metadatas = [{"text": text} for text in texts]
        self.metadatas.extend(metadatas)

        self.num_vectors += len(texts)

    def search(self, query_embedding: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar vectors"""
        query_array = np.array([query_embedding], dtype=np.float32)

        distances, indices = self.index.search(query_array, k)

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.texts):
                results.append((self.texts[idx], float(distance)))

        return results

    def delete_collection(self):
        """Reset the index"""
        if self.index_type == "Flat":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IVF":
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            self.needs_training = True
        elif self.index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)

        self.texts = []
        self.metadatas = []
        self.num_vectors = 0

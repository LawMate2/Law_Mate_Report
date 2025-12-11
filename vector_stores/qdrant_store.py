from typing import List, Tuple, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from . import VectorStore


class QdrantStore(VectorStore):
    """Qdrant vector store wrapper"""

    def __init__(self, collection_name: str = "rag_benchmark", dimension: int = 768, location: str = ":memory:"):
        """
        Initialize Qdrant client
        location options:
        - ":memory:" for in-memory storage
        - "path/to/db" for persistent storage
        - "http://localhost:6333" for Qdrant server
        """
        super().__init__("Qdrant", dimension)

        self.client = QdrantClient(location=location)
        self.collection_name = collection_name

        try:
            self.client.delete_collection(collection_name)
        except:
            pass

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
        )

        self.point_id = 0

    def add_texts(self, texts: List[str], embeddings: List[List[float]], metadatas: List[Dict] = None):
        """Add texts with embeddings to Qdrant"""
        points = []

        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            metadata = metadatas[i] if metadatas else {}
            metadata['text'] = text

            points.append(
                PointStruct(
                    id=self.point_id + i,
                    vector=embedding,
                    payload=metadata
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        self.point_id += len(texts)
        self.num_vectors += len(texts)

    def search(self, query_embedding: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar vectors"""
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k
            )
        except AttributeError:
            # For newer versions of qdrant-client
            from qdrant_client.models import SearchRequest
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=k
            ).points

        output = []
        for result in results:
            text = result.payload.get('text', '')
            score = result.score if hasattr(result, 'score') else 0.0
            output.append((text, score))

        return output

    def delete_collection(self):
        """Delete the collection"""
        try:
            self.client.delete_collection(self.collection_name)
        except:
            pass

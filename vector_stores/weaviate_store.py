from typing import List, Tuple, Dict
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery
from . import VectorStore


class WeaviateStore(VectorStore):
    """Weaviate vector store wrapper"""

    def __init__(self, collection_name: str = "RagBenchmark", dimension: int = 768,
                 host: str = "localhost", port: int = 8080, grpc_port: int = 50051):
        super().__init__("Weaviate", dimension)

        self.collection_name = collection_name

        # Connect to Weaviate
        self.client = weaviate.connect_to_local(
            host=host,
            port=port,
            grpc_port=grpc_port
        )

        # Delete collection if exists
        if self.client.collections.exists(collection_name):
            self.client.collections.delete(collection_name)

        # Create collection
        self.collection = self.client.collections.create(
            name=collection_name,
            vectorizer_config=None,  # We'll provide our own vectors
            properties=[
                {
                    "name": "text",
                    "dataType": ["text"]
                }
            ]
        )

    def add_texts(self, texts: List[str], embeddings: List[List[float]], metadatas: List[Dict] = None):
        """Add texts with embeddings to Weaviate"""
        with self.collection.batch.dynamic() as batch:
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                batch.add_object(
                    properties={"text": text},
                    vector=embedding
                )

        self.num_vectors += len(texts)

    def search(self, query_embedding: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar vectors"""
        response = self.collection.query.near_vector(
            near_vector=query_embedding,
            limit=k,
            return_metadata=MetadataQuery(distance=True)
        )

        results = []
        for obj in response.objects:
            text = obj.properties.get('text', '')
            distance = obj.metadata.distance if obj.metadata else 0.0
            results.append((text, distance))

        return results

    def delete_collection(self):
        """Delete the collection"""
        try:
            self.client.collections.delete(self.collection_name)
        except:
            pass
        finally:
            try:
                self.client.close()
            except:
                pass
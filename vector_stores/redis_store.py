from typing import List, Tuple, Dict
import redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.query import Query
import numpy as np
from . import VectorStore


class RedisStore(VectorStore):
    """Redis Stack vector store wrapper with RediSearch"""

    def __init__(self, index_name: str = "rag_benchmark", dimension: int = 768,
                 host: str = "localhost", port: int = 6379, db: int = 0):
        super().__init__("Redis", dimension)

        self.index_name = index_name
        self.dimension = dimension

        # Connect to Redis
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=False)

        # Delete index if exists
        try:
            self.client.ft(index_name).dropindex(delete_documents=True)
        except:
            pass

        # Create index schema using newer API
        schema = (
            TextField("text"),
            VectorField("embedding",
                       "FLAT",
                       {
                           "TYPE": "FLOAT32",
                           "DIM": dimension,
                           "DISTANCE_METRIC": "L2"
                       })
        )

        # Create index (newer redis-py API doesn't need IndexDefinition)
        try:
            self.client.ft(index_name).create_index(
                fields=schema,
                prefix=[f"{index_name}:"]
            )
        except Exception as e:
            # If that doesn't work, try without prefix
            self.client.ft(index_name).create_index(fields=schema)

    def add_texts(self, texts: List[str], embeddings: List[List[float]], metadatas: List[Dict] = None):
        """Add texts with embeddings to Redis"""
        pipe = self.client.pipeline()

        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            key = f"{self.index_name}:{self.num_vectors + i}"
            embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()

            doc = {
                "text": text,
                "embedding": embedding_bytes
            }

            if metadatas and i < len(metadatas):
                for meta_key, meta_value in metadatas[i].items():
                    if meta_key != "text":
                        doc[meta_key] = str(meta_value)

            pipe.hset(key, mapping=doc)

        pipe.execute()

        self.num_vectors += len(texts)

    def search(self, query_embedding: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar vectors using KNN"""
        query_bytes = np.array(query_embedding, dtype=np.float32).tobytes()

        # Create KNN query
        query = (
            Query(f"*=>[KNN {k} @embedding $vec AS distance]")
            .sort_by("distance")
            .return_fields("text", "distance")
            .dialect(2)
        )

        # Execute search
        result = self.client.ft(self.index_name).search(
            query,
            query_params={"vec": query_bytes}
        )

        results = []
        for doc in result.docs:
            text = doc.text if hasattr(doc, 'text') else ''
            if isinstance(text, bytes):
                text = text.decode('utf-8')
            distance = float(doc.distance) if hasattr(doc, 'distance') else 0.0
            results.append((text, distance))

        return results

    def delete_collection(self):
        """Delete the index"""
        try:
            self.client.ft(self.index_name).dropindex(delete_documents=True)
        except:
            pass
        finally:
            try:
                self.client.close()
            except:
                pass
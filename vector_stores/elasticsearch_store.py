from typing import List, Tuple, Dict
from elasticsearch import Elasticsearch
from . import VectorStore


class ElasticsearchStore(VectorStore):
    """Elasticsearch vector store wrapper with KNN search"""

    def __init__(self, index_name: str = "rag_benchmark", dimension: int = 768,
                 host: str = "localhost", port: int = 9200):
        super().__init__("Elasticsearch", dimension)

        self.index_name = index_name

        # Connect to Elasticsearch
        self.client = Elasticsearch([f"http://{host}:{port}"])

        # Delete index if exists
        if self.client.indices.exists(index=index_name):
            self.client.indices.delete(index=index_name)

        # Create index with vector mapping
        mappings = {
            "properties": {
                "text": {"type": "text"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": dimension,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }

        try:
            self.client.indices.create(index=index_name, mappings=mappings)
        except Exception as e:
            print(f"Warning: Could not create index: {e}")
            raise

    def add_texts(self, texts: List[str], embeddings: List[List[float]], metadatas: List[Dict] = None):
        """Add texts with embeddings to Elasticsearch"""
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            doc = {
                "text": text,
                "embedding": embedding
            }

            if metadatas and i < len(metadatas):
                doc.update(metadatas[i])

            self.client.index(
                index=self.index_name,
                id=self.num_vectors + i,
                document=doc
            )

        # Refresh index to make documents searchable
        self.client.indices.refresh(index=self.index_name)

        self.num_vectors += len(texts)

    def search(self, query_embedding: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar vectors using KNN"""
        knn_query = {
            "field": "embedding",
            "query_vector": query_embedding,
            "k": k,
            "num_candidates": k * 10
        }

        response = self.client.search(
            index=self.index_name,
            knn=knn_query,
            source=["text"]
        )

        results = []
        for hit in response['hits']['hits']:
            text = hit['_source']['text']
            score = hit['_score']
            # Convert Elasticsearch score to distance (lower is better)
            distance = 1.0 / (1.0 + score) if score > 0 else 1.0
            results.append((text, distance))

        return results

    def delete_collection(self):
        """Delete the index"""
        try:
            if self.client.indices.exists(index=self.index_name):
                self.client.indices.delete(index=self.index_name)
        except:
            pass
        finally:
            try:
                self.client.close()
            except:
                pass
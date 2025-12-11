from typing import List, Tuple, Dict
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from . import VectorStore


class MilvusStore(VectorStore):
    """Milvus vector store wrapper"""

    def __init__(self, collection_name: str = "rag_benchmark", dimension: int = 768,
                 host: str = "localhost", port: int = 19530):
        super().__init__("Milvus", dimension)

        self.collection_name = collection_name
        self.host = host
        self.port = port

        # Connect to Milvus
        connections.connect(
            alias="default",
            host=host,
            port=port
        )

        # Delete collection if exists
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)

        # Create collection schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension)
        ]

        schema = CollectionSchema(fields=fields, description="RAG benchmark collection")

        # Create collection
        self.collection = Collection(name=collection_name, schema=schema)

        # Create index for vector field
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        self.collection.create_index(field_name="embedding", index_params=index_params)

        self.collection.load()

    def add_texts(self, texts: List[str], embeddings: List[List[float]], metadatas: List[Dict] = None):
        """Add texts with embeddings to Milvus"""
        entities = [
            texts,
            embeddings
        ]

        self.collection.insert(entities)
        self.collection.flush()

        self.num_vectors += len(texts)

    def search(self, query_embedding: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar vectors"""
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }

        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=k,
            output_fields=["text"]
        )

        result_list = []
        for hits in results:
            for hit in hits:
                result_list.append((hit.entity.get('text'), hit.distance))

        return result_list

    def delete_collection(self):
        """Delete the collection"""
        try:
            self.collection.release()
            utility.drop_collection(self.collection_name)
        except:
            pass
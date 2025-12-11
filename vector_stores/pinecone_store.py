from typing import List, Tuple, Dict
import os
from pinecone import Pinecone, ServerlessSpec
from . import VectorStore


class PineconeStore(VectorStore):
    """Pinecone vector store wrapper"""

    def __init__(self, index_name: str = "rag-benchmark", dimension: int = 768, metric: str = "cosine"):
        super().__init__("Pinecone", dimension)

        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")

        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name

        if index_name in [idx.name for idx in self.pc.list_indexes()]:
            self.pc.delete_index(index_name)

        self.pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

        self.index = self.pc.Index(index_name)
        self.texts_map = {}

    def add_texts(self, texts: List[str], embeddings: List[List[float]], metadatas: List[Dict] = None):
        """Add texts with embeddings to Pinecone"""
        vectors = []

        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            vec_id = f"doc_{self.num_vectors + i}"
            self.texts_map[vec_id] = text

            metadata = metadatas[i] if metadatas else {"text": text}

            vectors.append({
                "id": vec_id,
                "values": embedding,
                "metadata": metadata
            })

        self.index.upsert(vectors=vectors)
        self.num_vectors += len(texts)

    def search(self, query_embedding: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar vectors"""
        results = self.index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True
        )

        output = []
        for match in results['matches']:
            text = self.texts_map.get(match['id'], match['metadata'].get('text', ''))
            score = match['score']
            output.append((text, score))

        return output

    def delete_collection(self):
        """Delete the index"""
        try:
            self.pc.delete_index(self.index_name)
        except:
            pass

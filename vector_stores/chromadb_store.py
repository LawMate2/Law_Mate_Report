from typing import List, Tuple, Dict
import chromadb
from chromadb.config import Settings
from . import VectorStore


class ChromaDBStore(VectorStore):
    """ChromaDB vector store wrapper"""

    def __init__(self, collection_name: str = "rag_benchmark", dimension: int = 768, persist_directory: str = "./chroma_db"):
        super().__init__("ChromaDB", dimension)

        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Check if collection exists and delete it
        try:
            existing_collections = [col.name for col in self.client.list_collections()]
            if collection_name in existing_collections:
                self.client.delete_collection(collection_name)
        except Exception as e:
            print(f"Warning: Could not delete existing collection: {e}")

        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"dimension": dimension}
        )

        self.collection_name = collection_name

    def add_texts(self, texts: List[str], embeddings: List[List[float]], metadatas: List[Dict] = None):
        """Add texts with embeddings to ChromaDB"""
        ids = [f"doc_{i}" for i in range(self.num_vectors, self.num_vectors + len(texts))]

        if metadatas is None:
            metadatas = [{"text": text} for text in texts]

        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

        self.num_vectors += len(texts)

    def search(self, query_embedding: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar vectors"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        documents = results['documents'][0]
        distances = results['distances'][0]

        return list(zip(documents, distances))

    def delete_collection(self):
        """Delete the collection"""
        try:
            self.client.delete_collection(self.collection_name)
        except:
            pass

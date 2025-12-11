"""
Main benchmark script for RAG performance comparison
"""

import json
import time
from typing import Dict, List, Any
from datetime import datetime
import numpy as np
from tqdm import tqdm

from embeddings import EmbeddingModel
from vector_stores import VectorStore


class RAGBenchmark:
    """Benchmark suite for RAG components"""

    def __init__(self, embedding_model: EmbeddingModel, vector_store: VectorStore):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.results = {}

    def load_documents(self, documents: List[str]):
        """Load documents into the vector store"""
        print(f"Embedding {len(documents)} documents...")
        start_time = time.time()

        embeddings = self.embedding_model.embed_texts(documents)

        embedding_time = time.time() - start_time

        print(f"Adding documents to vector store...")
        store_start = time.time()

        self.vector_store.add_texts(documents, embeddings)

        store_time = time.time() - store_start

        self.results['indexing'] = {
            'embedding_time': embedding_time,
            'embedding_time_per_doc': embedding_time / len(documents),
            'store_time': store_time,
            'store_time_per_doc': store_time / len(documents),
            'total_time': embedding_time + store_time,
            'num_documents': len(documents)
        }

    def run_search_benchmark(self, queries: List[str], top_k: int = 5, iterations: int = 10):
        """Run search performance benchmark"""
        print(f"Running search benchmark with {len(queries)} queries...")

        query_results = []

        for query in tqdm(queries, desc="Processing queries"):
            query_embedding = self.embedding_model.embed_query(query)

            search_times = []
            results = None

            for _ in range(iterations):
                start = time.time()
                results = self.vector_store.search(query_embedding, k=top_k)
                search_times.append(time.time() - start)

            query_results.append({
                'query': query,
                'avg_search_time': np.mean(search_times),
                'min_search_time': np.min(search_times),
                'max_search_time': np.max(search_times),
                'std_search_time': np.std(search_times),
                'num_results': len(results),
                'top_result': results[0] if results else None
            })

        self.results['search'] = {
            'query_results': query_results,
            'avg_search_time': np.mean([r['avg_search_time'] for r in query_results]),
            'total_queries': len(queries),
            'iterations_per_query': iterations
        }

    def evaluate_retrieval_quality(self, queries: List[str], relevant_docs: Dict[str, List[str]], top_k: int = 5):
        """
        Evaluate retrieval quality with precision, recall, MRR
        relevant_docs: Dict mapping query to list of relevant document texts
        """
        print("Evaluating retrieval quality...")

        metrics = []

        for query in tqdm(queries, desc="Evaluating queries"):
            query_embedding = self.embedding_model.embed_query(query)
            results = self.vector_store.search(query_embedding, k=top_k)

            retrieved_docs = [doc for doc, score in results]
            relevant = relevant_docs.get(query, [])

            if not relevant:
                continue

            relevant_retrieved = len(set(retrieved_docs) & set(relevant))

            precision = relevant_retrieved / len(retrieved_docs) if retrieved_docs else 0
            recall = relevant_retrieved / len(relevant) if relevant else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            reciprocal_rank = 0
            for i, doc in enumerate(retrieved_docs):
                if doc in relevant:
                    reciprocal_rank = 1 / (i + 1)
                    break

            metrics.append({
                'query': query,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'mrr': reciprocal_rank
            })

        if metrics:
            self.results['quality'] = {
                'avg_precision': np.mean([m['precision'] for m in metrics]),
                'avg_recall': np.mean([m['recall'] for m in metrics]),
                'avg_f1': np.mean([m['f1'] for m in metrics]),
                'avg_mrr': np.mean([m['mrr'] for m in metrics]),
                'per_query_metrics': metrics
            }

    def get_memory_usage(self):
        """Estimate memory usage (simplified)"""
        import sys

        embedding_size = sys.getsizeof(self.embedding_model)
        store_size = sys.getsizeof(self.vector_store)

        self.results['memory'] = {
            'embedding_model_size': embedding_size,
            'vector_store_size': store_size,
            'total_size': embedding_size + store_size
        }

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'embedding_model': self.embedding_model.get_metadata(),
            'vector_store': self.vector_store.get_metadata(),
            'results': self.results
        }

        return report

    def save_results(self, filepath: str):
        """Save benchmark results to JSON file"""
        report = self.generate_report()

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"Results saved to {filepath}")

"""
Main script to run RAG benchmark experiments
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

from embeddings.openai_embeddings import OpenAIEmbedding
from embeddings.huggingface_embeddings import HuggingFaceEmbedding, MultilingualEmbedding
from embeddings.cohere_embeddings import CohereEmbedding
from embeddings.korean_embeddings import KoreanEmbedding, KoSimCSEEmbedding, KoSRoBERTaEmbedding

from vector_stores.chromadb_store import ChromaDBStore
from vector_stores.faiss_store import FAISSStore
from vector_stores.pinecone_store import PineconeStore
from vector_stores.qdrant_store import QdrantStore
from vector_stores.milvus_store import MilvusStore
from vector_stores.weaviate_store import WeaviateStore
from vector_stores.elasticsearch_store import ElasticsearchStore
from vector_stores.pgvector_store import PgVectorStore
from vector_stores.redis_store import RedisStore

from experiments.benchmark import RAGBenchmark
from experiments.config import TEST_QUERIES_KO, BENCHMARK_SETTINGS


def load_documents(data_file: str = "data/documents.json"):
    """Load documents from JSON file"""
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('documents', [])


def run_single_experiment(embedding_model, vector_store, documents, queries, experiment_name):
    """Run a single benchmark experiment"""
    print(f"\n{'='*80}")
    print(f"Running experiment: {experiment_name}")
    print(f"{'='*80}")

    benchmark = RAGBenchmark(embedding_model, vector_store)

    benchmark.load_documents(documents)

    benchmark.run_search_benchmark(
        queries=queries,
        top_k=BENCHMARK_SETTINGS['top_k'],
        iterations=BENCHMARK_SETTINGS['num_search_iterations']
    )

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    output_file = results_dir / f"{experiment_name}.json"
    benchmark.save_results(str(output_file))

    vector_store.delete_collection()

    print(f"Experiment completed: {experiment_name}\n")


def run_all_experiments():
    """Run all experiments with different configurations"""

    load_dotenv()

    if not Path("data/documents.json").exists():
        print("Error: data/documents.json not found!")
        print("Please create a documents.json file in the data/ directory")
        print("Format: {\"documents\": [\"doc1\", \"doc2\", ...]}")
        return

    documents = load_documents()
    queries = TEST_QUERIES_KO

    print(f"Loaded {len(documents)} documents")
    print(f"Using {len(queries)} test queries")

    experiments = []

    print("\n" + "="*80)
    print("Available Experiments")
    print("="*80)

    print("\n1. HuggingFace + ChromaDB (Local, Free)")
    experiments.append({
        'name': 'hf_miniLM_chromadb',
        'embedding': lambda: HuggingFaceEmbedding("all-MiniLM-L6-v2"),
        'vector_store': lambda dim: ChromaDBStore(dimension=dim, collection_name="test_miniLM")
    })

    print("2. HuggingFace Multilingual + ChromaDB")
    experiments.append({
        'name': 'hf_multilingual_chromadb',
        'embedding': lambda: MultilingualEmbedding(),
        'vector_store': lambda dim: ChromaDBStore(dimension=dim, collection_name="test_multilingual")
    })

    print("3. Korean SRoBERTa + FAISS")
    experiments.append({
        'name': 'korean_sroberta_faiss',
        'embedding': lambda: KoSRoBERTaEmbedding(),
        'vector_store': lambda dim: FAISSStore(dimension=dim, index_type="Flat")
    })

    print("4. Korean SimCSE + Qdrant")
    experiments.append({
        'name': 'korean_simcse_qdrant',
        'embedding': lambda: KoSimCSEEmbedding(),
        'vector_store': lambda dim: QdrantStore(dimension=dim, collection_name="test_simcse")
    })

    if os.getenv("OPENAI_API_KEY"):
        print("5. OpenAI + ChromaDB")
        experiments.append({
            'name': 'openai_ada002_chromadb',
            'embedding': lambda: OpenAIEmbedding("text-embedding-ada-002"),
            'vector_store': lambda dim: ChromaDBStore(dimension=dim, collection_name="test_openai")
        })

    if os.getenv("COHERE_API_KEY"):
        print("6. Cohere + FAISS")
        experiments.append({
            'name': 'cohere_multilingual_faiss',
            'embedding': lambda: CohereEmbedding("embed-multilingual-v3.0"),
            'vector_store': lambda dim: FAISSStore(dimension=dim, index_type="Flat")
        })

    # New vector stores (requires Docker Compose)
    print("\n--- New Vector Stores (Docker Compose Required) ---")

    print("7. Korean SRoBERTa + Milvus")
    experiments.append({
        'name': 'korean_sroberta_milvus',
        'embedding': lambda: KoSRoBERTaEmbedding(),
        'vector_store': lambda dim: MilvusStore(dimension=dim, collection_name="test_milvus")
    })

    print("8. HuggingFace MiniLM + Weaviate")
    experiments.append({
        'name': 'hf_miniLM_weaviate',
        'embedding': lambda: HuggingFaceEmbedding("all-MiniLM-L6-v2"),
        'vector_store': lambda dim: WeaviateStore(dimension=dim, collection_name="TestWeaviate")
    })

    print("9. Multilingual + Elasticsearch")
    experiments.append({
        'name': 'multilingual_elasticsearch',
        'embedding': lambda: MultilingualEmbedding(),
        'vector_store': lambda dim: ElasticsearchStore(dimension=dim, index_name="test_elasticsearch")
    })

    print("10. Korean SimCSE + pgvector")
    experiments.append({
        'name': 'korean_simcse_pgvector',
        'embedding': lambda: KoSimCSEEmbedding(),
        'vector_store': lambda dim: PgVectorStore(dimension=dim, table_name="test_pgvector")
    })

    print("11. HuggingFace MiniLM + Redis")
    experiments.append({
        'name': 'hf_miniLM_redis',
        'embedding': lambda: HuggingFaceEmbedding("all-MiniLM-L6-v2"),
        'vector_store': lambda dim: RedisStore(dimension=dim, index_name="test_redis")
    })

    print("\n" + "="*80)
    print(f"Total experiments to run: {len(experiments)}")
    print("="*80)

    for i, exp in enumerate(experiments, 1):
        try:
            print(f"\n[{i}/{len(experiments)}] Starting: {exp['name']}")

            embedding_model = exp['embedding']()
            vector_store = exp['vector_store'](embedding_model.dimension)

            run_single_experiment(
                embedding_model=embedding_model,
                vector_store=vector_store,
                documents=documents,
                queries=queries,
                experiment_name=exp['name']
            )

        except Exception as e:
            print(f"Error in experiment {exp['name']}: {str(e)}")
            continue

    print("\n" + "="*80)
    print("All experiments completed!")
    print("="*80)
    print("\nNext steps:")
    print("1. Run: python analysis/analyze_results.py")
    print("2. Run: python analysis/visualize.py")


if __name__ == "__main__":
    run_all_experiments()

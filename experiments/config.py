"""
Configuration for RAG benchmark experiments
"""

# Test queries for evaluation
TEST_QUERIES_KO = [
    "인공지능이란 무엇인가?",
    "머신러닝의 주요 알고리즘은?",
    "딥러닝과 머신러닝의 차이점은?",
    "자연어 처리의 응용 분야는?",
    "컴퓨터 비전의 최신 기술은?",
]

TEST_QUERIES_EN = [
    "What is artificial intelligence?",
    "What are the main machine learning algorithms?",
    "What is the difference between deep learning and machine learning?",
    "What are the applications of natural language processing?",
    "What are the latest technologies in computer vision?",
]

# Embedding model configurations
EMBEDDING_CONFIGS = {
    "openai": [
        {"model_name": "text-embedding-ada-002", "dimension": 1536},
        {"model_name": "text-embedding-3-small", "dimension": 1536},
        {"model_name": "text-embedding-3-large", "dimension": 3072},
    ],
    "huggingface": [
        {"model_name": "all-MiniLM-L6-v2"},
        {"model_name": "all-mpnet-base-v2"},
        {"model_name": "distiluse-base-multilingual-cased-v2"},
    ],
    "cohere": [
        {"model_name": "embed-multilingual-v3.0", "dimension": 1024},
        {"model_name": "embed-multilingual-light-v3.0", "dimension": 384},
    ],
    "korean": [
        {"model_name": "jhgan/ko-sroberta-multitask"},
        {"model_name": "BM-K/KoSimCSE-roberta"},
    ]
}

# Vector store configurations
VECTOR_STORE_CONFIGS = {
    "chromadb": {"persist_directory": "./chroma_db"},
    "faiss": {"index_type": "Flat"},
    "pinecone": {"index_name": "rag-benchmark", "metric": "cosine"},
    "qdrant": {"location": ":memory:"}
}

# Benchmark settings
BENCHMARK_SETTINGS = {
    "num_search_iterations": 10,  # Number of iterations for search time measurement
    "top_k": 5,  # Number of results to retrieve
    "random_seed": 42,
}

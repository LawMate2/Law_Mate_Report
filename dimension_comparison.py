"""
벡터 차원별 성능 비교 실험
다양한 차원의 임베딩 모델을 테스트하여 차원이 성능에 미치는 영향 분석
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
import numpy as np

from embeddings.huggingface_embeddings import HuggingFaceEmbedding
from vector_stores.chromadb_store import ChromaDBStore
from vector_stores.faiss_store import FAISSStore
from experiments.benchmark import RAGBenchmark
from experiments.config import TEST_QUERIES_KO, BENCHMARK_SETTINGS


def run_dimension_experiment(pdf_path: str, num_runs: int = 3):
    """
    다양한 차원의 임베딩 모델로 실험 수행

    Args:
        pdf_path: PDF 파일 경로
        num_runs: 각 실험 실행 횟수
    """
    from pdf_benchmark import extract_text_from_pdf, split_text_into_chunks, run_multiple_experiments

    # PDF 처리
    text = extract_text_from_pdf(pdf_path)
    documents = split_text_into_chunks(text, chunk_size=500, overlap=50)
    queries = TEST_QUERIES_KO

    # 다양한 차원의 모델 정의
    dimension_experiments = [
        {
            'name': 'MiniLM_L6_v2_384dim',
            'dimension': 384,
            'embedding': lambda: HuggingFaceEmbedding("all-MiniLM-L6-v2"),
            'vector_stores': [
                ('ChromaDB', lambda dim: ChromaDBStore(dimension=dim, collection_name="dim_test_chroma_384")),
                ('FAISS', lambda dim: FAISSStore(dimension=dim, index_type="Flat"))
            ]
        },
        {
            'name': 'MiniLM_L12_v2_384dim',
            'dimension': 384,
            'embedding': lambda: HuggingFaceEmbedding("all-MiniLM-L12-v2"),
            'vector_stores': [
                ('ChromaDB', lambda dim: ChromaDBStore(dimension=dim, collection_name="dim_test_chroma_384_l12")),
                ('FAISS', lambda dim: FAISSStore(dimension=dim, index_type="Flat"))
            ]
        },
        {
            'name': 'MPNet_base_v2_768dim',
            'dimension': 768,
            'embedding': lambda: HuggingFaceEmbedding("all-mpnet-base-v2"),
            'vector_stores': [
                ('ChromaDB', lambda dim: ChromaDBStore(dimension=dim, collection_name="dim_test_chroma_768")),
                ('FAISS', lambda dim: FAISSStore(dimension=dim, index_type="Flat"))
            ]
        },
        {
            'name': 'DistilBERT_base_384dim',
            'dimension': 384,
            'embedding': lambda: HuggingFaceEmbedding("sentence-transformers/msmarco-distilbert-base-tas-b"),
            'vector_stores': [
                ('ChromaDB', lambda dim: ChromaDBStore(dimension=dim, collection_name="dim_test_chroma_distil")),
                ('FAISS', lambda dim: FAISSStore(dimension=dim, index_type="Flat"))
            ]
        },
        {
            'name': 'RoBERTa_large_1024dim',
            'dimension': 1024,
            'embedding': lambda: HuggingFaceEmbedding("sentence-transformers/all-roberta-large-v1"),
            'vector_stores': [
                ('ChromaDB', lambda dim: ChromaDBStore(dimension=dim, collection_name="dim_test_chroma_1024")),
                ('FAISS', lambda dim: FAISSStore(dimension=dim, index_type="Flat"))
            ]
        }
    ]

    all_results = []

    for exp in dimension_experiments:
        print(f"\n{'='*80}")
        print(f"테스트: {exp['name']} (차원: {exp['dimension']})")
        print(f"{'='*80}")

        try:
            embedding_model = exp['embedding']()

            for store_name, store_factory in exp['vector_stores']:
                experiment_name = f"{exp['name']}_{store_name}"

                print(f"\n벡터 스토어: {store_name}")

                result = run_multiple_experiments(
                    embedding_model=embedding_model,
                    vector_store_factory=store_factory,
                    documents=documents,
                    queries=queries,
                    experiment_name=experiment_name,
                    num_runs=num_runs
                )

                result['dimension'] = exp['dimension']
                result['model_size'] = exp['name']
                all_results.append(result)

        except Exception as e:
            print(f"실험 {exp['name']} 실패: {str(e)}")
            continue

    return all_results


def analyze_dimension_results(results):
    """
    차원별 결과 분석

    Args:
        results: 실험 결과 리스트
    """
    print("\n" + "="*80)
    print("차원별 성능 분석")
    print("="*80)

    # 차원별로 그룹화
    by_dimension = {}
    for r in results:
        dim = r['dimension']
        if dim not in by_dimension:
            by_dimension[dim] = []
        by_dimension[dim].append(r)

    # 각 차원별 평균 성능
    for dim in sorted(by_dimension.keys()):
        print(f"\n{dim}차원 모델들:")
        dim_results = by_dimension[dim]

        avg_embedding_time = np.mean([r['indexing']['avg_embedding_time'] for r in dim_results])
        avg_search_time = np.mean([r['search']['avg_search_time'] for r in dim_results])

        print(f"  평균 임베딩 시간: {avg_embedding_time:.3f}s")
        print(f"  평균 검색 시간: {avg_search_time:.4f}s")

        for r in dim_results:
            print(f"    - {r['model_size']} ({r['vector_store']}): "
                  f"임베딩 {r['indexing']['avg_embedding_time']:.3f}s, "
                  f"검색 {r['search']['avg_search_time']:.4f}s")

    # 최적 조합 찾기
    print("\n" + "="*80)
    print("최적 조합 분석")
    print("="*80)

    # 임베딩 시간 기준
    fastest_embedding = min(results, key=lambda x: x['indexing']['avg_embedding_time'])
    print(f"\n가장 빠른 임베딩:")
    print(f"  {fastest_embedding['experiment_name']}")
    print(f"  차원: {fastest_embedding['dimension']}")
    print(f"  시간: {fastest_embedding['indexing']['avg_embedding_time']:.3f}s")

    # 검색 시간 기준
    fastest_search = min(results, key=lambda x: x['search']['avg_search_time'])
    print(f"\n가장 빠른 검색:")
    print(f"  {fastest_search['experiment_name']}")
    print(f"  차원: {fastest_search['dimension']}")
    print(f"  시간: {fastest_search['search']['avg_search_time']:.4f}s")

    # 종합 점수 (임베딩 + 검색 정규화)
    for r in results:
        embedding_norm = r['indexing']['avg_embedding_time'] / max([x['indexing']['avg_embedding_time'] for x in results])
        search_norm = r['search']['avg_search_time'] / max([x['search']['avg_search_time'] for x in results])
        r['composite_score'] = embedding_norm + search_norm

    best_overall = min(results, key=lambda x: x['composite_score'])
    print(f"\n종합 최고 성능:")
    print(f"  {best_overall['experiment_name']}")
    print(f"  차원: {best_overall['dimension']}")
    print(f"  임베딩: {best_overall['indexing']['avg_embedding_time']:.3f}s")
    print(f"  검색: {best_overall['search']['avg_search_time']:.4f}s")
    print(f"  종합 점수: {best_overall['composite_score']:.3f}")


def main():
    """메인 함수"""
    load_dotenv()

    # 테스트할 PDF 선택 (가장 작은 것으로)
    pdf_path = "data/형법.pdf"

    print("벡터 차원별 성능 비교 실험 시작")
    print(f"테스트 PDF: {pdf_path}")

    # 실험 실행
    results = run_dimension_experiment(pdf_path, num_runs=3)

    # 결과 저장
    output_dir = Path("results/dimension_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "dimension_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장됨: {output_dir / 'dimension_results.json'}")

    # 결과 분석
    analyze_dimension_results(results)


if __name__ == "__main__":
    main()
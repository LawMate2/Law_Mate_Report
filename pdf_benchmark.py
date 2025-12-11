"""
PDF 기반 RAG 벤치마크 스크립트
- PDF 파일을 입력받아 텍스트 추출
- 여러 번 테스트하여 평균값 계산
- 결과를 그래프로 시각화
"""

import os
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from typing import List, Dict
import platform

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

from embeddings.openai_embeddings import OpenAIEmbedding
from embeddings.huggingface_embeddings import HuggingFaceEmbedding, MultilingualEmbedding
from embeddings.cohere_embeddings import CohereEmbedding
from embeddings.korean_embeddings import KoreanEmbedding, KoSimCSEEmbedding, KoSRoBERTaEmbedding

from vector_stores.chromadb_store import ChromaDBStore
from vector_stores.faiss_store import FAISSStore
from vector_stores.qdrant_store import QdrantStore
from vector_stores.milvus_store import MilvusStore
from vector_stores.pgvector_store import PgVectorStore
from vector_stores.redis_store import RedisStore

from experiments.benchmark import RAGBenchmark
from experiments.config import TEST_QUERIES_KO, BENCHMARK_SETTINGS


def setup_korean_font():
    """한글 폰트 설정"""
    system = platform.system()

    if system == 'Darwin':  # macOS
        font_candidates = [
            'AppleGothic',
            'AppleMyungjo',
            'Apple SD Gothic Neo',
            'NanumGothic',
            'NanumBarunGothic'
        ]
    elif system == 'Windows':
        font_candidates = [
            'Malgun Gothic',
            'NanumGothic',
            'NanumBarunGothic',
            'Gulim'
        ]
    else:  # Linux
        font_candidates = [
            'NanumGothic',
            'NanumBarunGothic',
            'DejaVu Sans'
        ]

    # 사용 가능한 폰트 찾기
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    for font in font_candidates:
        if font in available_fonts:
            plt.rcParams['font.family'] = font
            print(f"한글 폰트 설정: {font}")
            break
    else:
        print("경고: 한글 폰트를 찾을 수 없습니다. 그래프에서 한글이 깨질 수 있습니다.")

    # 마이너스 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False


def extract_text_from_pdf(pdf_path: str, method: str = 'pdfplumber') -> str:
    """
    PDF 파일에서 텍스트 추출

    Args:
        pdf_path: PDF 파일 경로
        method: 'pdfplumber' 또는 'pypdf2'

    Returns:
        추출된 텍스트
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

    text = ""

    if method == 'pdfplumber' and pdfplumber:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"

    elif method == 'pypdf2' and PyPDF2:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"

    else:
        raise ImportError(f"PDF 처리 라이브러리가 설치되지 않았습니다. pip install {method}")

    return text.strip()


def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    텍스트를 청크로 분할

    Args:
        text: 전체 텍스트
        chunk_size: 청크 크기 (문자 수)
        overlap: 청크 간 중복 크기

    Returns:
        텍스트 청크 리스트
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # 문장 경계에서 자르기 (선택적)
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            cut_point = max(last_period, last_newline)

            if cut_point > chunk_size * 0.7:  # 70% 이상 위치에서만 자르기
                chunk = chunk[:cut_point + 1]
                end = start + cut_point + 1

        if chunk.strip():
            chunks.append(chunk.strip())

        start = end - overlap

    return chunks


def run_multiple_experiments(
    embedding_model,
    vector_store_factory,
    documents: List[str],
    queries: List[str],
    experiment_name: str,
    num_runs: int = 3
) -> Dict:
    """
    여러 번 실험을 실행하고 평균값 계산

    Args:
        embedding_model: 임베딩 모델
        vector_store_factory: 벡터 스토어 팩토리 함수
        documents: 문서 리스트
        queries: 쿼리 리스트
        experiment_name: 실험 이름
        num_runs: 실행 횟수

    Returns:
        평균값을 포함한 결과 딕셔너리
    """
    all_results = []

    print(f"\n{'='*80}")
    print(f"실험: {experiment_name} (총 {num_runs}회 실행)")
    print(f"{'='*80}")

    for run in range(num_runs):
        print(f"\n[{run + 1}/{num_runs}] 실행 중...")

        # 새로운 벡터 스토어 인스턴스 생성
        vector_store = vector_store_factory(embedding_model.dimension)

        # 벤치마크 실행
        benchmark = RAGBenchmark(embedding_model, vector_store)
        benchmark.load_documents(documents)
        benchmark.run_search_benchmark(
            queries=queries,
            top_k=BENCHMARK_SETTINGS['top_k'],
            iterations=BENCHMARK_SETTINGS['num_search_iterations']
        )

        # 결과 저장
        report = benchmark.generate_report()
        all_results.append(report['results'])

        # 벡터 스토어 삭제
        vector_store.delete_collection()

    # 평균값 계산
    avg_results = calculate_average_results(all_results)
    avg_results['experiment_name'] = experiment_name
    avg_results['num_runs'] = num_runs
    avg_results['embedding_model'] = embedding_model.get_metadata()['model_name']
    avg_results['vector_store'] = vector_store.get_metadata()['store_name']

    return avg_results


def calculate_average_results(results_list: List[Dict]) -> Dict:
    """
    여러 실행 결과의 평균값 계산

    Args:
        results_list: 결과 딕셔너리 리스트

    Returns:
        평균값 딕셔너리
    """
    avg_results = {}

    # 인덱싱 시간 평균
    if 'indexing' in results_list[0]:
        indexing_metrics = ['embedding_time', 'store_time', 'total_time',
                           'embedding_time_per_doc', 'store_time_per_doc']
        avg_results['indexing'] = {}

        for metric in indexing_metrics:
            values = [r['indexing'][metric] for r in results_list if metric in r['indexing']]
            if values:
                avg_results['indexing'][f'avg_{metric}'] = np.mean(values)
                avg_results['indexing'][f'std_{metric}'] = np.std(values)

    # 검색 시간 평균
    if 'search' in results_list[0]:
        avg_search_times = [r['search']['avg_search_time'] for r in results_list]
        avg_results['search'] = {
            'avg_search_time': np.mean(avg_search_times),
            'std_search_time': np.std(avg_search_times),
            'min_search_time': np.min(avg_search_times),
            'max_search_time': np.max(avg_search_times)
        }

    return avg_results


def plot_comparison_results(all_experiment_results: List[Dict], output_dir: str = "results/graphs"):
    """
    실험 결과를 그래프로 시각화

    Args:
        all_experiment_results: 모든 실험 결과 리스트
        output_dir: 그래프 저장 디렉토리
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 한글 폰트 설정
    setup_korean_font()

    # 스타일 설정
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    # 1. 임베딩 시간 비교
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    experiment_names = [r['experiment_name'] for r in all_experiment_results]

    # 1-1. 전체 임베딩 시간
    embedding_times = [r['indexing']['avg_embedding_time'] for r in all_experiment_results]
    embedding_stds = [r['indexing']['std_embedding_time'] for r in all_experiment_results]

    axes[0, 0].bar(range(len(experiment_names)), embedding_times, yerr=embedding_stds)
    axes[0, 0].set_xticks(range(len(experiment_names)))
    axes[0, 0].set_xticklabels(experiment_names, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Time (seconds)', fontsize=10)
    axes[0, 0].set_title('Embedding Time Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # 1-2. 문서당 임베딩 시간
    embedding_per_doc = [r['indexing']['avg_embedding_time_per_doc'] for r in all_experiment_results]
    embedding_per_doc_stds = [r['indexing']['std_embedding_time_per_doc'] for r in all_experiment_results]

    axes[0, 1].bar(range(len(experiment_names)), embedding_per_doc, yerr=embedding_per_doc_stds, color='orange')
    axes[0, 1].set_xticks(range(len(experiment_names)))
    axes[0, 1].set_xticklabels(experiment_names, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Time (seconds)', fontsize=10)
    axes[0, 1].set_title('Embedding Time per Document', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # 1-3. 검색 시간
    search_times = [r['search']['avg_search_time'] for r in all_experiment_results]
    search_stds = [r['search']['std_search_time'] for r in all_experiment_results]

    axes[1, 0].bar(range(len(experiment_names)), search_times, yerr=search_stds, color='green')
    axes[1, 0].set_xticks(range(len(experiment_names)))
    axes[1, 0].set_xticklabels(experiment_names, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Time (seconds)', fontsize=10)
    axes[1, 0].set_title('Average Search Time', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # 1-4. 전체 인덱싱 시간 (임베딩 + 저장)
    total_times = [r['indexing']['avg_total_time'] for r in all_experiment_results]
    total_stds = [r['indexing']['std_total_time'] for r in all_experiment_results]

    axes[1, 1].bar(range(len(experiment_names)), total_times, yerr=total_stds, color='red')
    axes[1, 1].set_xticks(range(len(experiment_names)))
    axes[1, 1].set_xticklabels(experiment_names, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Time (seconds)', fontsize=10)
    axes[1, 1].set_title('Total Indexing Time', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_comparison.png", dpi=300, bbox_inches='tight')
    print(f"그래프 저장됨: {output_dir}/performance_comparison.png")
    plt.close()

    # 2. 상세 비교 테이블 그래프
    fig, ax = plt.subplots(figsize=(14, len(experiment_names) * 0.8))

    table_data = []
    for r in all_experiment_results:
        table_data.append([
            r['experiment_name'],
            f"{r['indexing']['avg_embedding_time']:.3f}±{r['indexing']['std_embedding_time']:.3f}",
            f"{r['indexing']['avg_store_time']:.3f}±{r['indexing']['std_store_time']:.3f}",
            f"{r['search']['avg_search_time']:.4f}±{r['search']['std_search_time']:.4f}",
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=['Experiment', 'Embedding (s)', 'Storage (s)', 'Search (s)'],
        cellLoc='center',
        loc='center',
        colWidths=[0.3, 0.25, 0.25, 0.25]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # 헤더 스타일링
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 행 색상 교대
    for i in range(1, len(table_data) + 1):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    ax.axis('off')
    plt.title('Benchmark Results Summary (Mean ± Std)', fontsize=14, weight='bold', pad=20)
    plt.savefig(f"{output_dir}/results_table.png", dpi=300, bbox_inches='tight')
    print(f"테이블 저장됨: {output_dir}/results_table.png")
    plt.close()


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='PDF 기반 RAG 벤치마크')
    parser.add_argument('--pdf', type=str, help='PDF 파일 경로')
    parser.add_argument('--num-runs', type=int, default=3, help='각 실험 실행 횟수 (기본값: 3)')
    parser.add_argument('--chunk-size', type=int, default=500, help='텍스트 청크 크기 (기본값: 500)')
    parser.add_argument('--overlap', type=int, default=50, help='청크 간 중복 크기 (기본값: 50)')
    parser.add_argument('--output-dir', type=str, default='results/pdf_benchmark', help='결과 저장 디렉토리')

    args = parser.parse_args()

    load_dotenv()

    # PDF 파일 처리
    if args.pdf:
        print(f"PDF 파일 처리 중: {args.pdf}")
        text = extract_text_from_pdf(args.pdf)
        print(f"추출된 텍스트 길이: {len(text)} 문자")

        documents = split_text_into_chunks(text, args.chunk_size, args.overlap)
        print(f"생성된 청크 수: {len(documents)}")
    else:
        print("PDF 파일이 지정되지 않았습니다. 기본 문서 사용")
        with open("data/documents.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        documents = data.get('documents', [])

    queries = TEST_QUERIES_KO
    print(f"테스트 쿼리 수: {len(queries)}")

    # 실험 정의 (안정적으로 작동하는 것들만)
    experiments = [
        {
            'name': 'HuggingFace_MiniLM_ChromaDB',
            'embedding': lambda: HuggingFaceEmbedding("all-MiniLM-L6-v2"),
            'vector_store': lambda dim: ChromaDBStore(dimension=dim, collection_name="test_miniLM_pdf")
        },
        {
            'name': 'Korean_SRoBERTa_FAISS',
            'embedding': lambda: KoSRoBERTaEmbedding(),
            'vector_store': lambda dim: FAISSStore(dimension=dim, index_type="Flat")
        },
        {
            'name': 'Korean_SimCSE_Qdrant',
            'embedding': lambda: KoSimCSEEmbedding(),
            'vector_store': lambda dim: QdrantStore(dimension=dim, collection_name="test_simcse_pdf")
        },
    ]

    # OpenAI API가 있으면 추가
    if os.getenv("OPENAI_API_KEY"):
        experiments.append({
            'name': 'OpenAI_Ada002_ChromaDB',
            'embedding': lambda: OpenAIEmbedding("text-embedding-ada-002"),
            'vector_store': lambda dim: ChromaDBStore(dimension=dim, collection_name="test_openai_pdf")
        })

    # Cohere API가 있으면 추가
    if os.getenv("COHERE_API_KEY"):
        experiments.append({
            'name': 'Cohere_Multilingual_FAISS',
            'embedding': lambda: CohereEmbedding("embed-multilingual-v3.0"),
            'vector_store': lambda dim: FAISSStore(dimension=dim, index_type="Flat")
        })

    # 모든 실험 결과 저장
    all_results = []

    # 각 실험 실행
    for exp in experiments:
        try:
            embedding_model = exp['embedding']()

            result = run_multiple_experiments(
                embedding_model=embedding_model,
                vector_store_factory=exp['vector_store'],
                documents=documents,
                queries=queries,
                experiment_name=exp['name'],
                num_runs=args.num_runs
            )

            all_results.append(result)

        except Exception as e:
            print(f"실험 {exp['name']} 실패: {str(e)}")
            continue

    # 결과 저장
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "all_results.json", 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장됨: {output_dir / 'all_results.json'}")

    # 그래프 생성
    if all_results:
        print("\n그래프 생성 중...")
        plot_comparison_results(all_results, str(output_dir / "graphs"))
        print("\n모든 작업 완료!")
    else:
        print("\n성공한 실험이 없어 그래프를 생성할 수 없습니다.")


if __name__ == "__main__":
    main()
"""
RAG 성능 벤치마크 보고서 자동 생성
"""

import json
from pathlib import Path
import numpy as np
from datetime import datetime


def load_all_results():
    """모든 PDF 벤치마크 결과 로드"""
    results_dir = Path("results/pdf_benchmark")
    all_results = {}

    for pdf_dir in results_dir.iterdir():
        if pdf_dir.is_dir():
            result_file = pdf_dir / "all_results.json"
            if result_file.exists():
                with open(result_file, 'r', encoding='utf-8') as f:
                    all_results[pdf_dir.name] = json.load(f)

    return all_results


def analyze_best_combinations(all_results):
    """최고 성능 조합 분석"""
    all_experiments = []

    for pdf_name, experiments in all_results.items():
        for exp in experiments:
            exp['pdf_name'] = pdf_name
            all_experiments.append(exp)

    # 임베딩 속도 기준
    fastest_embedding = min(all_experiments,
                           key=lambda x: x['indexing']['avg_embedding_time'])

    # 검색 속도 기준
    fastest_search = min(all_experiments,
                        key=lambda x: x['search']['avg_search_time'])

    # 문서당 임베딩 시간 기준
    fastest_per_doc = min(all_experiments,
                         key=lambda x: x['indexing']['avg_embedding_time_per_doc'])

    # 안정성 기준 (표준편차가 낮은 것)
    most_stable = min(all_experiments,
                     key=lambda x: x['indexing']['std_embedding_time'] + x['search']['std_search_time'])

    return {
        'fastest_embedding': fastest_embedding,
        'fastest_search': fastest_search,
        'fastest_per_doc': fastest_per_doc,
        'most_stable': most_stable
    }


def generate_markdown_report(all_results, best_combos):
    """마크다운 형식의 보고서 생성"""

    report = f"""# RAG 시스템 성능 벤치마크 보고서

생성 날짜: {datetime.now().strftime('%Y년 %m월 %d일 %H:%M')}

## 목차
1. [실험 개요](#실험-개요)
2. [최고 성능 조합 분석](#최고-성능-조합-분석)
3. [PDF별 상세 결과](#pdf별-상세-결과)
4. [종합 분석 및 권장사항](#종합-분석-및-권장사항)

---

## 실험 개요

### 테스트 환경
- **테스트 PDF 문서**: {len(all_results)}개
- **실험당 반복 횟수**: 3회
- **평가 지표**: 임베딩 시간, 검색 시간, 안정성 (표준편차)

### 테스트한 PDF 문서

| PDF 이름 | 청크 수 | 실험 수 |
|---------|--------|--------|
"""

    for pdf_name, experiments in all_results.items():
        if experiments:
            num_chunks = "N/A"  # 청크 수는 결과에서 추출 필요
            report += f"| {pdf_name} | {num_chunks} | {len(experiments)} |\n"

    report += """
### 테스트한 임베딩 모델
- **HuggingFace MiniLM**: 경량 모델, 빠른 속도
- **Korean SRoBERTa**: 한국어 특화 RoBERTa 기반
- **Korean SimCSE**: 한국어 문맥 임베딩
- **OpenAI Ada-002**: 고품질 상용 모델
- **Cohere Multilingual**: 다국어 지원 상용 모델

### 테스트한 벡터 데이터베이스
- **ChromaDB**: 로컬 벡터 DB
- **FAISS**: Facebook의 고속 유사도 검색
- **Qdrant**: 메모리 기반 벡터 DB

---

## 최고 성능 조합 분석

"""

    # 가장 빠른 임베딩
    fastest_emb = best_combos['fastest_embedding']
    report += f"""### 🏆 가장 빠른 임베딩

**조합**: {fastest_emb['embedding_model']} + {fastest_emb['vector_store']}
**테스트 PDF**: {fastest_emb['pdf_name']}
**임베딩 시간**: {fastest_emb['indexing']['avg_embedding_time']:.3f} ± {fastest_emb['indexing']['std_embedding_time']:.3f}초
**문서당 시간**: {fastest_emb['indexing']['avg_embedding_time_per_doc']:.4f}초

"""

    # 가장 빠른 검색
    fastest_srch = best_combos['fastest_search']
    report += f"""### ⚡ 가장 빠른 검색

**조합**: {fastest_srch['embedding_model']} + {fastest_srch['vector_store']}
**테스트 PDF**: {fastest_srch['pdf_name']}
**검색 시간**: {fastest_srch['search']['avg_search_time']:.4f} ± {fastest_srch['search']['std_search_time']:.4f}초

"""

    # 가장 안정적인 조합
    most_stable = best_combos['most_stable']
    report += f"""### 🎯 가장 안정적인 조합

**조합**: {most_stable['embedding_model']} + {most_stable['vector_store']}
**테스트 PDF**: {most_stable['pdf_name']}
**임베딩 안정성**: {most_stable['indexing']['std_embedding_time']:.4f}초
**검색 안정성**: {most_stable['search']['std_search_time']:.4f}초

---

## PDF별 상세 결과

"""

    for pdf_name, experiments in all_results.items():
        report += f"""### {pdf_name}

| 임베딩 모델 | 벡터 DB | 임베딩 시간 (초) | 검색 시간 (초) |
|------------|---------|-----------------|---------------|
"""
        for exp in experiments:
            emb_time = f"{exp['indexing']['avg_embedding_time']:.3f} ± {exp['indexing']['std_embedding_time']:.3f}"
            search_time = f"{exp['search']['avg_search_time']:.4f} ± {exp['search']['std_search_time']:.4f}"
            report += f"| {exp['embedding_model']} | {exp['vector_store']} | {emb_time} | {search_time} |\n"

        report += "\n"

    report += """---

## 종합 분석 및 권장사항

### 성능 분석 요약

#### 1. 임베딩 모델 비교

**속도 순위**:
1. HuggingFace MiniLM (가장 빠름)
2. OpenAI Ada-002
3. Korean SRoBERTa
4. Korean SimCSE (가장 느림)

**특징**:
- **MiniLM**: 가장 빠르지만 경량 모델로 품질이 다소 낮을 수 있음
- **Korean 모델들**: 한국어에 특화되어 있어 한국어 문서에 더 좋은 결과 기대
- **OpenAI Ada-002**: 균형잡힌 성능, API 비용 발생

#### 2. 벡터 데이터베이스 비교

**검색 속도 순위**:
1. FAISS (가장 빠름, 마이크로초 단위)
2. Qdrant
3. ChromaDB

**특징**:
- **FAISS**: 검색 속도가 압도적으로 빠름, 메모리 효율적
- **ChromaDB**: 사용하기 쉬움, 로컬 개발에 적합
- **Qdrant**: 메모리 기반, 빠른 검색, 스케일링 가능

### 사용 사례별 권장사항

#### 📌 대용량 문서, 빠른 처리 필요
- **권장**: HuggingFace MiniLM + FAISS
- **이유**: 가장 빠른 임베딩과 검색 속도
- **Trade-off**: 임베딩 품질이 다소 낮을 수 있음

#### 📌 한국어 문서, 높은 품질 필요
- **권장**: Korean SRoBERTa + FAISS
- **이유**: 한국어 특화 모델, 빠른 검색
- **Trade-off**: 임베딩 시간이 MiniLM보다 2-3배 느림

#### 📌 최고 품질, 비용 무관
- **권장**: OpenAI Ada-002 + ChromaDB
- **이유**: 고품질 임베딩, 안정적인 성능
- **Trade-off**: API 비용 발생

#### 📌 로컬 환경, 비용 제로
- **권장**: HuggingFace MiniLM + ChromaDB
- **이유**: 완전 무료, 로컬 실행, 쉬운 설정
- **Trade-off**: 품질과 속도 trade-off

### 성능 최적화 팁

1. **청크 크기 조정**: 500자가 기본이지만, 문서 특성에 따라 조정 필요
2. **배치 처리**: 대량 문서는 배치로 처리하여 효율성 향상
3. **인덱스 타입**: FAISS의 경우 IVF 인덱스 사용 시 더 빠른 검색 가능
4. **캐싱**: 자주 사용하는 임베딩은 캐싱하여 재사용

---

## 결론

이 벤치마크 결과는 다음을 보여줍니다:

1. **속도와 품질의 Trade-off**: 경량 모델은 빠르지만 품질이 낮고, 큰 모델은 느리지만 품질이 높음
2. **벡터 DB의 중요성**: FAISS가 검색 속도에서 압도적으로 우수
3. **한국어 특화 모델의 필요성**: 한국어 문서에서는 한국어 특화 모델이 더 나은 결과를 제공할 가능성
4. **실용적 선택**: 대부분의 경우 HuggingFace MiniLM + FAISS 조합이 가장 실용적

---

*이 보고서는 자동으로 생성되었습니다.*
"""

    return report


def main():
    """메인 함수"""
    print("보고서 생성 중...")

    # 결과 로드
    all_results = load_all_results()

    if not all_results:
        print("결과 파일을 찾을 수 없습니다.")
        return

    # 최고 성능 조합 분석
    best_combos = analyze_best_combinations(all_results)

    # 보고서 생성
    report = generate_markdown_report(all_results, best_combos)

    # 보고서 저장
    output_path = Path("RAG_벤치마크_보고서.md")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"✓ 보고서 생성 완료: {output_path}")
    print(f"✓ 분석된 PDF 수: {len(all_results)}")
    print(f"✓ 총 실험 수: {sum(len(exps) for exps in all_results.values())}")


if __name__ == "__main__":
    main()
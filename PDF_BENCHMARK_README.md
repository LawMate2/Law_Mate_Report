# PDF 기반 RAG 벤치마크 가이드

이 문서는 PDF 파일을 입력받아 여러 번 테스트하고 평균값을 그래프로 출력하는 방법을 설명합니다.

## 주요 기능

1. **PDF 텍스트 추출**: PDF 파일에서 텍스트를 자동으로 추출
2. **여러 번 테스트**: 각 실험을 N회 반복하여 안정적인 결과 도출
3. **평균값 계산**: 여러 실행의 평균과 표준편차 계산
4. **그래프 시각화**: 결과를 다양한 그래프로 시각화

## 설치

필요한 라이브러리 설치:

```bash
pip install pypdf2 pdfplumber
```

또는 전체 requirements.txt 재설치:

```bash
pip install -r requirements.txt
```

## 사용 방법

### 1. 기본 사용 (PDF 파일 지정)

```bash
python pdf_benchmark.py --pdf your_document.pdf
```

### 2. 실행 횟수 지정

```bash
python pdf_benchmark.py --pdf your_document.pdf --num-runs 5
```

각 실험을 5회 반복합니다. (기본값: 3회)

### 3. 청크 크기 조정

```bash
python pdf_benchmark.py --pdf your_document.pdf --chunk-size 1000 --overlap 100
```

- `--chunk-size`: 텍스트를 나눌 청크 크기 (문자 수, 기본값: 500)
- `--overlap`: 청크 간 중복 크기 (기본값: 50)

### 4. 결과 저장 위치 지정

```bash
python pdf_benchmark.py --pdf your_document.pdf --output-dir results/my_experiment
```

### 5. 전체 옵션 예시

```bash
python pdf_benchmark.py \
  --pdf documents/research_paper.pdf \
  --num-runs 5 \
  --chunk-size 800 \
  --overlap 100 \
  --output-dir results/research_paper_benchmark
```

## 출력 결과

실행하면 다음과 같은 결과가 생성됩니다:

### 1. JSON 결과 파일

`results/pdf_benchmark/all_results.json`

각 실험의 상세 결과가 저장됩니다:
- 평균 임베딩 시간
- 평균 검색 시간
- 표준편차
- 사용된 모델 정보

예시:
```json
[
  {
    "experiment_name": "HuggingFace_MiniLM_ChromaDB",
    "num_runs": 3,
    "embedding_model": "all-MiniLM-L6-v2",
    "vector_store": "ChromaDB",
    "indexing": {
      "avg_embedding_time": 2.345,
      "std_embedding_time": 0.123,
      "avg_store_time": 0.456,
      "std_store_time": 0.045
    },
    "search": {
      "avg_search_time": 0.012,
      "std_search_time": 0.002
    }
  }
]
```

### 2. 그래프 파일

`results/pdf_benchmark/graphs/` 디렉토리에 다음 그래프들이 생성됩니다:

#### performance_comparison.png
4개의 서브플롯으로 구성된 성능 비교 그래프:
1. **임베딩 시간 비교**: 각 실험의 평균 임베딩 시간
2. **문서당 임베딩 시간**: 문서 하나당 소요되는 평균 시간
3. **검색 시간 비교**: 평균 검색 시간
4. **전체 인덱싱 시간**: 임베딩 + 저장 시간 합계

모든 그래프에 오차 막대(error bar)가 표시되어 표준편차를 확인할 수 있습니다.

#### results_table.png
모든 실험 결과를 요약한 테이블 이미지:
- 실험 이름
- 임베딩 시간 (평균±표준편차)
- 저장 시간 (평균±표준편차)
- 검색 시간 (평균±표준편차)

## 실험 구성

현재 다음 실험들이 실행됩니다:

1. **HuggingFace MiniLM + ChromaDB**
   - 무료, 로컬 실행
   - 빠른 임베딩 속도

2. **Korean SRoBERTa + FAISS**
   - 한국어 특화 모델
   - 고속 벡터 검색

3. **Korean SimCSE + Qdrant**
   - 한국어 문맥 임베딩
   - 메모리 기반 벡터 DB

4. **OpenAI Ada-002 + ChromaDB** (OPENAI_API_KEY 필요)
   - 높은 품질의 임베딩
   - API 비용 발생

5. **Cohere Multilingual + FAISS** (COHERE_API_KEY 필요)
   - 다국어 지원
   - API 비용 발생

## 팁

### 한글 PDF 처리

한글이 포함된 PDF는 `pdfplumber`가 더 잘 처리합니다 (기본값).

### 실행 시간

- 각 실험은 PDF 크기와 청크 수에 따라 시간이 다릅니다
- 3회 반복 기준으로 작은 PDF(10-20페이지)는 약 5-10분 소요
- API 기반 임베딩(OpenAI, Cohere)은 더 오래 걸릴 수 있습니다

### 메모리 사용

- 큰 PDF 파일은 많은 메모리를 사용할 수 있습니다
- 청크 크기를 줄이면 청크 수가 증가하여 처리 시간이 늘어납니다
- 적절한 균형을 찾기 위해 다양한 청크 크기를 테스트해보세요

## 문제 해결

### PDF 읽기 실패

```
FileNotFoundError: PDF 파일을 찾을 수 없습니다
```
→ PDF 파일 경로를 확인하세요

### 메모리 부족

```
MemoryError
```
→ 청크 크기를 늘리거나 PDF를 분할하세요

### API 키 오류

```
Error: OpenAI API key not found
```
→ `.env` 파일에 API 키를 설정하세요:
```
OPENAI_API_KEY=your_key_here
COHERE_API_KEY=your_key_here
```

## 예제 실행

### 예제 1: 연구 논문 벤치마크

```bash
python pdf_benchmark.py \
  --pdf papers/deep_learning.pdf \
  --num-runs 5 \
  --chunk-size 1000 \
  --output-dir results/deep_learning_benchmark
```

### 예제 2: 빠른 테스트

```bash
python pdf_benchmark.py \
  --pdf test.pdf \
  --num-runs 1 \
  --chunk-size 300
```

### 예제 3: 기본 문서 사용 (PDF 없이)

```bash
python pdf_benchmark.py --num-runs 3
```

`data/documents.json`의 문서를 사용합니다.

## 결과 해석

### 임베딩 시간
- 낮을수록 좋음
- 모델 크기와 복잡도에 비례
- API 기반 모델은 네트워크 지연 포함

### 검색 시간
- 낮을수록 좋음
- 벡터 DB 성능에 따라 다름
- 문서 수가 많을수록 차이가 커짐

### 표준편차
- 낮을수록 안정적인 성능
- 높으면 환경 요인(네트워크, CPU 부하)의 영향 큼

## 추가 개선 사항

원하시면 다음 기능을 추가할 수 있습니다:

1. **검색 품질 평가**: 정확도, 재현율 측정
2. **더 많은 벡터 DB**: Milvus, Weaviate, Elasticsearch 등
3. **다양한 청크 전략**: 문장 단위, 단락 단위 분할
4. **대화형 인터페이스**: 웹 UI로 결과 확인

질문이나 개선 요청이 있으면 언제든지 알려주세요!
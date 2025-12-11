# RAG 성능 비교 프로젝트

다양한 임베딩 모델, 벡터 차원, 벡터 저장소를 비교하여 RAG(Retrieval-Augmented Generation) 시스템의 성능을 정성적/정량적으로 분석하는 프로젝트입니다.

## 프로젝트 구조

```
report/
├── data/                    # 테스트 데이터
│   └── documents.json      # 문서 데이터
├── embeddings/             # 임베딩 모델
│   ├── __init__.py
│   ├── openai_embeddings.py
│   ├── huggingface_embeddings.py
│   ├── cohere_embeddings.py
│   └── korean_embeddings.py
├── vector_stores/          # 벡터 저장소
│   ├── __init__.py
│   ├── chromadb_store.py
│   ├── faiss_store.py
│   ├── pinecone_store.py
│   ├── qdrant_store.py
│   ├── milvus_store.py         # 새로 추가
│   ├── weaviate_store.py       # 새로 추가
│   ├── elasticsearch_store.py  # 새로 추가
│   ├── pgvector_store.py       # 새로 추가
│   └── redis_store.py          # 새로 추가
├── experiments/            # 실험 실행
│   ├── benchmark.py
│   └── config.py
├── analysis/               # 결과 분석
│   ├── analyze_results.py
│   └── visualize.py
├── results/                # 실험 결과
├── docker-compose.yml      # Docker Compose 설정
├── main.py                 # 메인 실행 파일
└── requirements.txt
```

## 비교 대상

### 임베딩 모델
- **OpenAI**: text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large
- **HuggingFace**: all-MiniLM-L6-v2, all-mpnet-base-v2, multilingual models
- **Cohere**: embed-multilingual-v3.0, embed-multilingual-light-v3.0
- **한국어 특화**: KoSRoBERTa, KoSimCSE

### 벡터 저장소

#### 로컬 벡터 저장소
- **ChromaDB**: 경량, 로컬 우선, 쉬운 설정
- **FAISS**: Facebook의 빠른 유사도 검색 라이브러리, 인메모리

#### 클라우드/서버 벡터 저장소
- **Pinecone**: 클라우드 기반 벡터 데이터베이스 (유료)
- **Qdrant**: 오픈소스, 로컬/클라우드 지원, REST API

#### 새로운 벡터 저장소 (Docker Compose 필요)
- **Milvus**: 클라우드 네이티브, 높은 확장성, 대용량 데이터 처리
- **Weaviate**: GraphQL API, 하이브리드 검색 (벡터 + 키워드)
- **Elasticsearch**: 전통적 검색엔진 + KNN 벡터 검색
- **pgvector**: PostgreSQL extension, SQL로 벡터 검색
- **Redis Stack**: 인메모리 벡터 검색, 초고속 성능

### 평가 지표

#### 정량적 지표
- **인덱싱 시간**: 문서 임베딩 및 저장 시간
- **검색 속도**: 쿼리당 평균 검색 시간
- **메모리 사용량**: 벡터 저장소 크기

#### 정성적 지표
- **Precision**: 검색된 결과 중 관련 문서 비율
- **Recall**: 관련 문서 중 검색된 비율
- **F1 Score**: Precision과 Recall의 조화평균
- **MRR (Mean Reciprocal Rank)**: 첫 번째 관련 문서의 순위

## 설치 및 실행

### 1. 환경 설정

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 1-1. Docker Compose로 벡터 데이터베이스 실행 (선택사항)

새로운 벡터 저장소(Milvus, Weaviate, Elasticsearch, pgvector, Redis)를 사용하려면 Docker Compose를 실행하세요:

```bash
# 모든 벡터 데이터베이스 시작
docker-compose up -d

# 특정 데이터베이스만 시작
docker-compose up -d milvus        # Milvus만
docker-compose up -d weaviate      # Weaviate만
docker-compose up -d elasticsearch # Elasticsearch만
docker-compose up -d postgres      # pgvector만
docker-compose up -d redis         # Redis Stack만
docker-compose up -d qdrant        # Qdrant만

# 상태 확인
docker-compose ps

# 로그 확인
docker-compose logs -f

# 중지
docker-compose down

# 볼륨까지 삭제 (데이터 초기화)
docker-compose down -v
```

**포트 정보:**
- Milvus: 19530 (gRPC)
- Weaviate: 8080 (HTTP), 50051 (gRPC)
- Elasticsearch: 9200 (HTTP)
- PostgreSQL (pgvector): 5432
- Redis Stack: 6379 (Redis), 8001 (RedisInsight)
- Qdrant: 6333 (HTTP), 6334 (gRPC)

### 2. API 키 설정 (선택사항)

`.env` 파일을 생성하고 필요한 API 키를 설정합니다:

```bash
cp .env .env
```

`.env` 파일 편집:
```
OPENAI_API_KEY=your_key_here
COHERE_API_KEY=your_key_here
PINECONE_API_KEY=your_key_here
```

**참고**: API 키가 없어도 HuggingFace, 한국어 모델, FAISS, ChromaDB, Qdrant를 사용한 실험은 가능합니다.

### 3. 데이터 준비

`data/documents.json` 파일에 테스트할 문서를 추가합니다:

```json
{
  "documents": [
    "문서 1 내용...",
    "문서 2 내용...",
    ...
  ]
}
```

### 4. 실험 실행

```bash
python main.py
```

실험이 완료되면 `results/` 디렉토리에 각 실험의 결과가 JSON 형식으로 저장됩니다.

### 5. 결과 분석

```bash
# 테이블 형식의 비교 결과 생성
python analysis/analyze_results.py

# 시각화 차트 생성
python analysis/visualize.py
```

분석 결과:
- `summary_report.md`: 텍스트 형식의 요약 리포트
- `results/plots/`: 다양한 비교 차트

## 실험 커스터마이징

### 테스트 쿼리 수정

`experiments/config.py`에서 테스트 쿼리를 수정할 수 있습니다:

```python
TEST_QUERIES_KO = [
    "원하는 질문 1",
    "원하는 질문 2",
    ...
]
```

### 새로운 임베딩 모델 추가

1. `embeddings/` 디렉토리에 새 파일 생성
2. `EmbeddingModel` 베이스 클래스 상속
3. `embed_texts()`, `embed_query()` 메서드 구현
4. `main.py`에서 실험 추가

### 새로운 벡터 저장소 추가

1. `vector_stores/` 디렉토리에 새 파일 생성
2. `VectorStore` 베이스 클래스 상속
3. `add_texts()`, `search()`, `delete_collection()` 메서드 구현
4. `main.py`에서 실험 추가

## 결과 예시

실험 후 다음과 같은 결과를 얻을 수 있습니다:

| Embedding Model | Dimension | Vector Store | Search Time (ms) | Precision | F1 Score |
|----------------|-----------|--------------|------------------|-----------|----------|
| ko-sroberta    | 768       | ChromaDB     | 12.5             | 0.85      | 0.82     |
| all-MiniLM     | 384       | FAISS        | 3.2              | 0.78      | 0.75     |
| OpenAI-ada-002 | 1536      | Pinecone     | 45.3             | 0.92      | 0.89     |

## 생성되는 차트

- **search_time_comparison.png**: 검색 속도 비교
- **indexing_time_comparison.png**: 인덱싱 시간 비교
- **quality_metrics.png**: 검색 품질 메트릭 비교
- **dimension_vs_performance.png**: 벡터 차원과 성능의 관계
- **search_time_heatmap.png**: 검색 시간 히트맵

## 주의사항

1. **메모리**: 대용량 문서 처리 시 메모리 사용량에 주의
2. **API 비용**: OpenAI, Cohere, Pinecone은 유료 API
3. **GPU**: PyTorch 기반 모델은 GPU 사용 시 더 빠름
4. **한국어 모델**: 처음 실행 시 모델 다운로드에 시간 소요

## 트러블슈팅

### 임베딩 모델 다운로드 오류
```bash
# 캐시 디렉토리 확인
export HF_HOME=/path/to/cache
```

### FAISS 설치 오류 (M1/M2 Mac)
```bash
conda install -c pytorch faiss-cpu
```

### ChromaDB 오류
```bash
pip install --upgrade chromadb
```

## 라이선스

MIT License

## 참고 자료

### 임베딩 모델
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [HuggingFace Sentence Transformers](https://www.sbert.net/)
- [Cohere Embeddings](https://docs.cohere.com/docs/embeddings)

### 벡터 저장소
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [FAISS GitHub](https://github.com/facebookresearch/faiss)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Milvus Documentation](https://milvus.io/docs)
- [Weaviate Documentation](https://weaviate.io/developers/weaviate)
- [Elasticsearch Vector Search](https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html)
- [pgvector GitHub](https://github.com/pgvector/pgvector)
- [Redis Vector Search](https://redis.io/docs/interact/search-and-query/search/vectors/)

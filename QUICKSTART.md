# 빠른 시작 가이드

Docker Compose를 사용해서 모든 벡터 데이터베이스를 빠르게 실행하고 벤치마크를 수행하는 가이드입니다.

## 1. 사전 요구사항

- Python 3.8 이상
- Docker Desktop (Docker Compose 포함)
- 최소 8GB RAM 권장
- 최소 20GB 디스크 공간

## 2. 설치 (5분)

### 2.1 저장소 클론 및 환경 설정

```bash
cd report

# Python 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2.2 Docker Compose로 벡터 DB 시작

```bash
# 모든 벡터 데이터베이스 시작 (약 2-3분 소요)
docker-compose up -d

# 상태 확인
docker-compose ps

# 모든 서비스가 healthy 상태가 될 때까지 대기
# Milvus는 시작에 약 1-2분 소요될 수 있습니다
```

**출력 예시:**
```
NAME                   STATUS
elasticsearch          running
milvus-standalone      running (healthy)
postgres-pgvector      running (healthy)
qdrant                 running
redis-stack            running
weaviate               running
```

### 2.3 API 키 설정 (선택사항)

`.env` 파일을 편집하여 API 키를 추가하세요 (OpenAI, Cohere 사용 시):

```bash
# .env 파일 생성
cat > .env << EOF
OPENAI_API_KEY=your_openai_key_here
COHERE_API_KEY=your_cohere_key_here
PINECONE_API_KEY=your_pinecone_key_here
EOF
```

**참고**: API 키가 없어도 로컬 모델(HuggingFace, 한국어)과 벡터 DB는 사용 가능합니다.

## 3. 벤치마크 실행 (10-30분)

### 3.1 기본 실험 실행

```bash
# 모든 실험 실행
python main.py
```

실행되는 실험:
1. ✅ HuggingFace + ChromaDB (무료, 로컬)
2. ✅ HuggingFace Multilingual + ChromaDB
3. ✅ Korean SRoBERTa + FAISS
4. ✅ Korean SimCSE + Qdrant
5. ⭐ OpenAI + ChromaDB (API 키 필요)
6. ⭐ Cohere + FAISS (API 키 필요)
7. 🐳 Korean SRoBERTa + Milvus (Docker)
8. 🐳 HuggingFace MiniLM + Weaviate (Docker)
9. 🐳 Multilingual + Elasticsearch (Docker)
10. 🐳 Korean SimCSE + pgvector (Docker)
11. 🐳 HuggingFace MiniLM + Redis (Docker)

### 3.2 진행 상황 확인

터미널에서 실시간으로 진행 상황을 확인할 수 있습니다:

```
================================================================================
Running experiment: korean_sroberta_milvus
================================================================================
Loading documents...
Embedding documents...
Indexing documents: 100%|████████████████████| 100/100
Running search benchmark...
Search queries: 100%|████████████████████| 10/10
Experiment completed: korean_sroberta_milvus
```

## 4. 결과 분석 (5분)

### 4.1 결과 파일 확인

```bash
# results/ 디렉토리에 JSON 파일 생성됨
ls results/

# 출력 예시:
# hf_miniLM_chromadb.json
# korean_sroberta_milvus.json
# multilingual_elasticsearch.json
# ...
```

### 4.2 분석 및 시각화

```bash
# 테이블 형식의 비교 결과 생성
python analysis/analyze_results.py

# 시각화 차트 생성
python analysis/visualize.py
```

생성되는 차트:
- `results/plots/search_time_comparison.png` - 검색 속도 비교
- `results/plots/indexing_time_comparison.png` - 인덱싱 시간 비교
- `results/plots/quality_metrics.png` - 검색 품질 비교
- `results/plots/dimension_vs_performance.png` - 차원별 성능
- `results/plots/search_time_heatmap.png` - 히트맵

## 5. 개별 벡터 DB 테스트

특정 벡터 데이터베이스만 테스트하고 싶다면:

### 5.1 Milvus만 테스트

```bash
# Milvus 시작
docker-compose up -d milvus etcd minio

# main.py 수정하여 Milvus 실험만 실행
# 또는 직접 Python에서:
python -c "
from embeddings.korean_embeddings import KoSRoBERTaEmbedding
from vector_stores.milvus_store import MilvusStore
from experiments.benchmark import RAGBenchmark

embedding = KoSRoBERTaEmbedding()
store = MilvusStore(dimension=embedding.dimension)
benchmark = RAGBenchmark(embedding, store)
# ... 벤치마크 실행
"
```

### 5.2 Weaviate만 테스트

```bash
docker-compose up -d weaviate
# ... 유사하게 테스트
```

### 5.3 Elasticsearch만 테스트

```bash
docker-compose up -d elasticsearch
# ... 유사하게 테스트
```

### 5.4 pgvector만 테스트

```bash
docker-compose up -d postgres
# ... 유사하게 테스트
```

### 5.5 Redis만 테스트

```bash
docker-compose up -d redis
# ... 유사하게 테스트
```

## 6. 관리 도구 접속

일부 벡터 데이터베이스는 웹 UI를 제공합니다:

### Redis Insight
```
http://localhost:8001
```
- Redis 데이터 확인
- 벡터 인덱스 모니터링

### Qdrant Dashboard
```
http://localhost:6333/dashboard
```
- 컬렉션 관리
- 벡터 검색 테스트

### Elasticsearch (Kibana 없음)
```bash
# REST API로 확인
curl http://localhost:9200/_cluster/health?pretty
```

## 7. 정리

### 7.1 벡터 DB 중지

```bash
# 모든 컨테이너 중지
docker-compose down

# 볼륨까지 삭제 (데이터 완전 삭제)
docker-compose down -v
```

### 7.2 디스크 공간 정리

```bash
# Docker 시스템 정리
docker system prune -a

# Python 가상환경 삭제
deactivate
rm -rf venv/
```

## 8. 문제 해결

### 8.1 Docker 메모리 부족

```bash
# Docker Desktop 설정에서 메모리 할당 증가 (최소 8GB 권장)
# Mac: Docker Desktop > Settings > Resources > Memory
# Windows: Docker Desktop > Settings > Resources > Advanced
```

### 8.2 포트 충돌

기존에 실행 중인 서비스와 포트가 충돌하는 경우:

```bash
# docker-compose.yml에서 포트 변경
# 예: 6379 -> 16379로 변경
```

### 8.3 Milvus 시작 실패

```bash
# 로그 확인
docker-compose logs milvus

# etcd, minio가 먼저 시작되었는지 확인
docker-compose ps

# 재시작
docker-compose restart milvus
```

### 8.4 Python 패키지 설치 오류

```bash
# 최신 pip로 업그레이드
pip install --upgrade pip

# 개별 패키지 설치 시도
pip install pymilvus
pip install weaviate-client
# ...
```

## 9. 다음 단계

### 9.1 커스텀 데이터 사용

`data/documents.json` 파일을 수정하여 자신의 데이터로 테스트:

```json
{
  "documents": [
    "여기에 자신의 문서를 추가하세요",
    "한국어 문서도 지원합니다",
    "문서가 많을수록 정확한 비교 가능"
  ]
}
```

### 9.2 테스트 쿼리 수정

`experiments/config.py`에서 테스트 쿼리 수정:

```python
TEST_QUERIES_KO = [
    "자신의 질문 1",
    "자신의 질문 2",
    # ...
]
```

### 9.3 보고서 작성

`COMPARISON_GUIDE.md` 파일을 참고하여 보고서 작성:
- 실험 방법론
- 결과 분석
- 비교 및 평가
- 결론 및 추천

## 10. 유용한 명령어

```bash
# 모든 컨테이너 상태 확인
docker-compose ps

# 특정 서비스 로그 확인
docker-compose logs -f milvus

# 특정 서비스 재시작
docker-compose restart weaviate

# 리소스 사용량 확인
docker stats

# 벤치마크 결과 요약
cat results/*.json | jq '.metrics.avg_search_time'

# 차트 이미지 확인
open results/plots/search_time_comparison.png
```

## 참고 자료

- **전체 문서**: `README.md`
- **비교 가이드**: `COMPARISON_GUIDE.md`
- **벡터 DB 문서**:
  - [Milvus](https://milvus.io/docs)
  - [Weaviate](https://weaviate.io/developers/weaviate)
  - [Elasticsearch](https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html)
  - [pgvector](https://github.com/pgvector/pgvector)
  - [Redis](https://redis.io/docs/interact/search-and-query/search/vectors/)

## 지원

문제가 발생하면:
1. `COMPARISON_GUIDE.md`의 문제 해결 섹션 확인
2. Docker 로그 확인: `docker-compose logs`
3. GitHub Issues 검색
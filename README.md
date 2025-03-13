# BGE-M3-ko 임베딩 API 서버 (Google Colab 배포용)

Google Colab에서 BGE-M3-ko 임베딩 모델을 사용하여 FastAPI 기반의 API 서버를 구축하고 공개 접근 가능한 엔드포인트를 생성하는 프로젝트입니다.

## 주요 특징

- 🌐 **BGE-M3-ko 모델 사용**: 한국어에 최적화된 고성능 임베딩 모델 활용
- 🔒 **API 키 인증**: 개인만 사용할 수 있도록 API 키 기반 인증 제공
- 🚀 **Gradio로 공개 URL 생성**: ngrok 없이 Colab에서 공개 접근 가능한 URL 제공
- 📊 **웹 인터페이스**: 간단한 테스트를 위한 웹 인터페이스 제공

## Colab에서 바로 실행하기 (빠른 시작)

1. Google Colab 노트북을 열고 다음 코드를 실행하세요:

```python
# 1. 저장소 클론
!git clone https://github.com/your-username/embed_api.git
%cd embed_api

# 2. 필요한 패키지 설치
!pip install -r requirements.txt

# 3. API 서버 실행 (colab_deploy.ipynb를 그대로 복사하여 실행하세요)
# 다음 코드를 복사하여 새 셀에 붙여넣고 실행하세요
```

2. 아래 코드를 새 셀에 복사하여 실행하세요:

```python
import gradio as gr
import uvicorn
import threading
import time
import random
import string
import os
from typing import List, Optional, Dict
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from FlagEmbedding import FlagModel
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(title="BGE-M3-ko 임베딩 API", 
              description="한국어 텍스트 임베딩을 위한 BGE-M3-ko 모델 API",
              version="1.0.0")

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 키 생성 함수
def generate_api_key(length=32):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

# API 키 생성
API_KEY = generate_api_key()
print(f"생성된 API 키: {API_KEY}")
print(f"이 키를 API 요청 시 'api_key' 헤더에 포함해야 합니다.")

# 모델 로드
@app.on_event("startup")
async def startup_event():
    global embedding_model
    try:
        logger.info("모델 로드 중...")
        embedding_model = FlagModel('BAAI/bge-m3-ko', 
                                   use_fp16=True)  # 경량화를 위해 fp16 사용
        logger.info("모델 로드 완료")
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        raise

# 인증 함수
def verify_api_key(api_key: str = Header(...)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API 키가 유효하지 않습니다")
    return api_key

# API 요청 모델
class TextEmbeddingRequest(BaseModel):
    texts: List[str]
    truncate: Optional[bool] = True

# API 응답 모델
class TextEmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    tokens: Optional[List[int]] = None

# 임베딩 API 엔드포인트
@app.post("/api/embeddings", response_model=TextEmbeddingResponse)
async def create_embeddings(
    request: TextEmbeddingRequest, 
    api_key: str = Depends(verify_api_key)
):
    try:
        # 텍스트 임베딩 생성
        embeddings = embedding_model.encode(request.texts)
        
        # numpy 배열을 파이썬 목록으로 변환
        embeddings_list = embeddings.tolist()
        
        # 토큰 수 계산 (간단한 근사치)
        tokens = [len(text.split()) for text in request.texts]
        
        return TextEmbeddingResponse(
            embeddings=embeddings_list,
            tokens=tokens
        )
    except Exception as e:
        logger.error(f"임베딩 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=f"임베딩 생성 중 오류 발생: {str(e)}")

# 상태 확인 엔드포인트
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "BGE-M3-ko"}

# FastAPI 서버 스레드 생성
def run_fastapi_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# 백그라운드에서 FastAPI 서버 시작
thread = threading.Thread(target=run_fastapi_server, daemon=True)
thread.start()

# FastAPI 서버가 시작될 때까지 잠시 대기
print("FastAPI 서버 시작 중...")
time.sleep(5)
print("FastAPI 서버가 8000 포트에서 실행 중입니다")

# Gradio 인터페이스 생성
with gr.Blocks() as demo:
    gr.Markdown("# BGE-M3-ko 임베딩 API 서버")
    gr.Markdown(f"## API 키: `{API_KEY}`")
    gr.Markdown("## 사용 방법:")
    gr.Markdown("""
    1. API 엔드포인트: `/api/embeddings`
    2. 요청 메소드: `POST`
    3. 헤더에 `api_key` 추가: 위에 표시된 API 키 값 사용
    4. 요청 본문 예시:
    ```json
    {
        "texts": ["한국어 문장 임베딩 예시입니다."]
    }
    ```
    5. Swagger 문서: `/docs`에서 확인 가능
    """)
    
    with gr.Row():
        input_text = gr.Textbox(label="테스트할 텍스트 입력", lines=5, placeholder="여기에 임베딩할 텍스트를 입력하세요.")
        output_text = gr.JSON(label="임베딩 결과")
    
    def get_embedding(text):
        import requests
        import json
        
        url = "http://localhost:8000/api/embeddings"
        headers = {"api_key": API_KEY, "Content-Type": "application/json"}
        data = {"texts": [text]}
        
        response = requests.post(url, headers=headers, json=data)
        return response.json()
    
    test_btn = gr.Button("임베딩 생성 테스트")
    test_btn.click(fn=get_embedding, inputs=input_text, outputs=output_text)

# Gradio 인터페이스 시작 (공개 URL 생성)
demo.launch(share=True, server_name="0.0.0.0")
```

3. API 서버가 시작되면 Gradio 인터페이스에서 생성된 공개 URL을 통해 API에 접근할 수 있습니다.

## API 사용 방법

API가 실행되면 다음과 같이 사용할 수 있습니다:

### 임베딩 생성 (POST /api/embeddings)

**요청 예시:**
```bash
curl -X POST "https://your-gradio-public-url.gradio.app/api/embeddings" \
  -H "Content-Type: application/json" \
  -H "api_key: your_api_key_here" \
  -d '{"texts": ["한국어 문장 임베딩 예시입니다."]}'
```

**요청 본문:**
```json
{
  "texts": ["한국어 문장 임베딩 예시입니다."],
  "truncate": true
}
```

**응답 예시:**
```json
{
  "embeddings": [[0.123, 0.456, ...]],
  "tokens": [6]
}
```

### 상태 확인 (GET /health)

**요청 예시:**
```bash
curl "https://your-gradio-public-url.gradio.app/health"
```

**응답 예시:**
```json
{
  "status": "healthy",
  "model": "BGE-M3-ko"
}
```

## 파일 구조

- `requirements.txt`: 필요한 라이브러리와 의존성
- `app.py`: FastAPI 기반 임베딩 API 서버 코드 (로컬 개발용)
- `.env`: API 키 등의 환경 변수 관리 (로컬 개발용)
- `colab_setup.py`: Colab에서 실행하기 위한 코드

## Colab 무료 버전 제한사항

- **연결 시간 제한**: 약 12시간 후 자동 종료
- **리소스 제한**: CPU, 메모리 제한으로 대형 모델 로드 시 문제 발생 가능
- **간헐적 연결 해제**: 비활성 상태 시 연결 끊김
- **GPU 제한**: 무료 버전은 GPU 할당이 보장되지 않음

## 주의사항 및 개선 방안

- **보안**: API 키를 안전하게 관리하고 정기적으로 변경하세요.
- **모델 경량화**: `use_fp16=True` 옵션으로 메모리 사용량 최소화
- **세션 유지**: 정기적인 활동으로 세션 유지
- **프로덕션 환경**: 실제 서비스에는 클라우드 서비스 고려

## 라이선스

이 프로젝트는 MIT 라이선스 하에 제공됩니다.

## 참고 자료

- [FastAPI 공식 문서](https://fastapi.tiangolo.com/)
- [BGE-M3 모델 정보](https://huggingface.co/BAAI/bge-m3-ko)
- [Google Colab 사용 가이드](https://colab.research.google.com/)

## colab_deploy.ipynb 파일 직접 생성하기

Jupyter Notebook 파일 크기가 크거나 편집에 문제가 있는 경우, 다음과 같이 수동으로 파일을 생성할 수 있습니다:

1. Google Colab에서 **새 노트북**을 생성합니다.
2. 아래 코드 셀과 마크다운 셀을 차례로 추가합니다:

### 셀 1 (마크다운):
```
# BGE-M3-ko 임베딩 API 서버 - Colab 배포 가이드

이 노트북은 Google Colab에서 BGE-M3-ko 한국어 임베딩 모델을 사용하여 API 서버를 배포하는 방법을 안내합니다.
```

### 셀 2 (마크다운):
```
## 1. 저장소 복제 및 설정

먼저 GitHub 저장소를 복제하고 필요한 라이브러리를 설치합니다.
```

### 셀 3 (코드):
```python
# GitHub 저장소 복제
!git clone https://github.com/your-username/embed_api.git
%cd embed_api

# 필요한 라이브러리 설치
!pip install -r requirements.txt
```

### 셀 4 (마크다운):
```
## 2. API 서버 실행

아래 코드를 실행하여 FastAPI 서버와 Gradio 인터페이스를 시작합니다.
```

### 셀 5 (코드):
```python
import gradio as gr
import uvicorn
import threading
import time
import random
import string
import os
from typing import List, Optional, Dict
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from FlagEmbedding import FlagModel
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(title="BGE-M3-ko 임베딩 API", 
              description="한국어 텍스트 임베딩을 위한 BGE-M3-ko 모델 API",
              version="1.0.0")

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 키 생성 함수
def generate_api_key(length=32):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

# API 키 생성
API_KEY = generate_api_key()
print(f"생성된 API 키: {API_KEY}")
print(f"이 키를 API 요청 시 'api_key' 헤더에 포함해야 합니다.")

# 모델 로드
@app.on_event("startup")
async def startup_event():
    global embedding_model
    try:
        logger.info("모델 로드 중...")
        embedding_model = FlagModel('BAAI/bge-m3-ko', 
                                   use_fp16=True)  # 경량화를 위해 fp16 사용
        logger.info("모델 로드 완료")
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        raise

# 인증 함수
def verify_api_key(api_key: str = Header(...)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API 키가 유효하지 않습니다")
    return api_key

# API 요청 모델
class TextEmbeddingRequest(BaseModel):
    texts: List[str]
    truncate: Optional[bool] = True

# API 응답 모델
class TextEmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    tokens: Optional[List[int]] = None

# 임베딩 API 엔드포인트
@app.post("/api/embeddings", response_model=TextEmbeddingResponse)
async def create_embeddings(
    request: TextEmbeddingRequest, 
    api_key: str = Depends(verify_api_key)
):
    try:
        # 텍스트 임베딩 생성
        embeddings = embedding_model.encode(request.texts)
        
        # numpy 배열을 파이썬 목록으로 변환
        embeddings_list = embeddings.tolist()
        
        # 토큰 수 계산 (간단한 근사치)
        tokens = [len(text.split()) for text in request.texts]
        
        return TextEmbeddingResponse(
            embeddings=embeddings_list,
            tokens=tokens
        )
    except Exception as e:
        logger.error(f"임베딩 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=f"임베딩 생성 중 오류 발생: {str(e)}")

# 상태 확인 엔드포인트
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "BGE-M3-ko"}

# FastAPI 서버 스레드 생성
def run_fastapi_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# 백그라운드에서 FastAPI 서버 시작
thread = threading.Thread(target=run_fastapi_server, daemon=True)
thread.start()

# FastAPI 서버가 시작될 때까지 잠시 대기
print("FastAPI 서버 시작 중...")
time.sleep(5)
print("FastAPI 서버가 8000 포트에서 실행 중입니다")

# Gradio 인터페이스 생성
with gr.Blocks() as demo:
    gr.Markdown("# BGE-M3-ko 임베딩 API 서버")
    gr.Markdown(f"## API 키: `{API_KEY}`")
    gr.Markdown("## 사용 방법:")
    gr.Markdown("""
    1. API 엔드포인트: `/api/embeddings`
    2. 요청 메소드: `POST`
    3. 헤더에 `api_key` 추가: 위에 표시된 API 키 값 사용
    4. 요청 본문 예시:
    ```json
    {
        "texts": ["한국어 문장 임베딩 예시입니다."]
    }
    ```
    5. Swagger 문서: `/docs`에서 확인 가능
    """)
    
    with gr.Row():
        input_text = gr.Textbox(label="테스트할 텍스트 입력", lines=5, placeholder="여기에 임베딩할 텍스트를 입력하세요.")
        output_text = gr.JSON(label="임베딩 결과")
    
    def get_embedding(text):
        import requests
        import json
        
        url = "http://localhost:8000/api/embeddings"
        headers = {"api_key": API_KEY, "Content-Type": "application/json"}
        data = {"texts": [text]}
        
        response = requests.post(url, headers=headers, json=data)
        return response.json()
    
    test_btn = gr.Button("임베딩 생성 테스트")
    test_btn.click(fn=get_embedding, inputs=input_text, outputs=output_text)

# Gradio 인터페이스 시작 (공개 URL 생성)
demo.launch(share=True, server_name="0.0.0.0")
```

### 셀 6 (마크다운):
```
## 3. 세션 유지하기

아래 코드를 실행하여 Colab 세션이 자동으로 종료되지 않도록 유지합니다.
```

### 셀 7 (코드):
```python
# 세션 유지 코드
from IPython.display import display, Javascript
import time
import threading

def keep_session_alive():
    while True:
        print("세션 유지 중...", time.strftime("%Y-%m-%d %H:%M:%S"))
        time.sleep(1800)  # 30분마다 실행

# 백그라운드 스레드에서 세션 유지 함수 실행
session_thread = threading.Thread(target=keep_session_alive, daemon=True)
session_thread.start()

print("세션 유지 스크립트가 활성화되었습니다.")
```

### 셀 8 (마크다운):
```
## 주의사항

- Google Colab의 무료 버전은 최대 12시간 동안만 실행됩니다.
- 모델 로드에 시간이 걸릴 수 있으니 인내심을 가지고 기다려주세요.
- API 키를 안전하게 관리하고, 필요한 경우 새로 생성하세요.
- 트래픽이 많을 경우 성능이 저하될 수 있습니다.
- 프로덕션 환경에서는 클라우드 서비스 사용을 고려하세요.
```

3. 모든 셀을 추가한 후 **파일 > 다운로드 > .ipynb 다운로드**를 선택하여 파일을 저장합니다.
4. 저장한 파일을 프로젝트 저장소에 업로드합니다.

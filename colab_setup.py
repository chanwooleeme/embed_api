# Colab에서 코드를 복사하여 셀에 실행하세요

# 1. 필요한 라이브러리 설치
# !pip install fastapi uvicorn FlagEmbedding gradio pyjwt python-dotenv

# 2. app.py 파일 생성
# %%writefile app.py
import os
from typing import List, Optional, Dict
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from FlagEmbedding import FlagModel
import jwt
from datetime import datetime, timedelta
import logging

# API 키 설정 (실제 배포 시에는 환경 변수 또는 시크릿으로 관리)
API_KEY = "your_secret_api_key_here"

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
    allow_origins=["*"],  # 개발용으로는 모든 오리진 허용, 프로덕션에서는 특정 도메인만 허용하세요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        
        # 토큰 수 계산 (실제 구현에서는 모델의 토크나이저로 계산해야 합니다)
        tokens = [len(text.split()) for text in request.texts]  # 간단한 근사치
        
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

# 3. Gradio와 FastAPI 서버 실행
# 다음 코드를 새 셀에 복사하여 실행하세요
"""
import gradio as gr
import uvicorn
import threading
import time
import random
import string
from app import app as fastapi_app

# API 키 재생성 - 보안을 위해 랜덤한 API 키 생성
def generate_api_key(length=32):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

API_KEY = generate_api_key()
print(f"생성된 API 키: {API_KEY}")
print(f"이 키를 API 요청 시 'api_key' 헤더에 포함해야 합니다.")

# FastAPI에서 API 키 업데이트
import app
app.API_KEY = API_KEY

# Gradio 인터페이스 생성
with gr.Blocks() as demo:
    gr.Markdown("# BGE-M3-ko 임베딩 API 서버")
    gr.Markdown(f"## API 키: `{API_KEY}`")
    gr.Markdown("## 사용 방법:")
    gr.Markdown(\"\"\"
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
    \"\"\")
    
    with gr.Row():
        input_text = gr.Textbox(label="테스트할 텍스트 입력", lines=5)
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

# FastAPI 서버 스레드 생성
def run_fastapi_server():
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)

# 백그라운드에서 FastAPI 서버 시작
thread = threading.Thread(target=run_fastapi_server, daemon=True)
thread.start()

# FastAPI 서버가 시작될 때까지 잠시 대기
print("FastAPI 서버 시작 중...")
time.sleep(5)
print("FastAPI 서버가 8000 포트에서 실행 중입니다")

# Gradio 인터페이스 시작 (공개 URL 생성)
demo.launch(share=True, server_name="0.0.0.0")
"""

# 4. API 테스트
# 다음 코드를 새 셀에 복사하여 실행하세요
"""
import requests
import json

# 위에서 생성된 API 키를 입력하세요
API_KEY = "여기에 API 키를 입력하세요"

# 공개 URL을 입력하세요 (gradio에서 생성된 URL)
PUBLIC_URL = "여기에 gradio에서 생성된 공개 URL 입력"

# API 엔드포인트
url = f"{PUBLIC_URL}/api/embeddings"

# 헤더 설정
headers = {
    "api_key": API_KEY,
    "Content-Type": "application/json"
}

# 요청 데이터
data = {
    "texts": ["한국어 문장 임베딩 테스트입니다.", "BGE-M3-ko 모델은 한국어에 최적화되어 있습니다."]
}

# API 요청 보내기
response = requests.post(url, headers=headers, json=data)

# 응답 확인
if response.status_code == 200:
    result = response.json()
    print("임베딩 차원:", len(result["embeddings"][0]))
    print("첫 번째 문장의 처음 5개 임베딩 값:", result["embeddings"][0][:5])
else:
    print(f"오류 발생: {response.status_code}")
    print(response.text)
"""

# 5. 세션 유지 (필요한 경우)
# 다음 코드를 새 셀에 복사하여 실행하세요
"""
# 세션 유지를 위한 코드
import time
import IPython.display as display
from datetime import datetime

def keep_session_alive(minutes=30):
    \"\"\"Colab 세션을 유지하기 위해 주기적으로 출력을 생성합니다.\"\"\"
    end_time = time.time() + minutes * 60
    
    while time.time() < end_time:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[세션 유지 중...] {current_time}", end="\\r")
        display.display(display.Javascript('IPython.notebook.kernel.execute("print(\\'\\')")'))
        time.sleep(60)  # 1분마다 출력
        
# 30분 동안 세션 유지하기
keep_session_alive(minutes=30)
""" 
import os
from typing import List, Optional, Dict
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from FlagEmbedding import FlagModel
import jwt
from datetime import datetime, timedelta
import dotenv
import logging

# 환경 변수 로드
dotenv.load_dotenv()
API_KEY = os.getenv("API_KEY")

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
        embedding_model = FlagModel('dragonkue/BGE-m3-ko', 
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

# 메인 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 
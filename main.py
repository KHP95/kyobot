# std lib
import os
import time
from typing import *

# external lib
import joblib
from dotenv import load_dotenv

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# local lib import이전 환경변수 불러오기
load_dotenv()

# local lib
from llmpkg.chatbots import ChatBot
from schema import *


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['https://localhost:8080',
                   'http://localhost:8080'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)
chatbot = ChatBot() # 챗봇 불러오기 (6초 가량 소요)


# 기본 QA api
@app.post('/api/v1/query')
async def query_respond(query: ChatInput):
    output_text:str = await chatbot.invoke(query.query)
    return {'ans':output_text}

# 정적 파일 마운트 (나중에 해야 엔드포인트 충돌 x)
app.mount("/", StaticFiles(directory="dist", html=True), name="dist")


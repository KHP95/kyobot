# std lib
import os
import time
import regex as re
from typing import *
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# external lib
from dotenv import load_dotenv
import yaml
import tiktoken
import sqlite3

from langchain_community.chat_models import ChatOpenAI

from langchain.callbacks.base import AsyncCallbackHandler
from langchain_core.messages.base import BaseMessage
from langchain.schema import LLMResult
from langchain_core.output_parsers import StrOutputParser

# local lib
try:
    from .llm_prompts import basic_chatbot_prompt
    from .recommendation import InsuranceEstimator, EstimatorPreProcesseor
except ImportError:
    from llmpkg.llm_prompts import basic_chatbot_prompt
    from llmpkg.recommendation import InsuranceEstimator, EstimatorPreProcesseor


# API key 불러오기
load_dotenv()

# 설정 불러오기
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(FILE_PATH, '..', 'insurance_info.db')
CONFIG_PATH = os.path.join(FILE_PATH, '..', 'config.yaml')
with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
    cfg = yaml.safe_load(file)
CHATBOT_LLM_NAME = cfg['chatbot']['llm_model'] # 사용할 llm 모델
CHATBOT_LLM_TEMP = cfg['chatbot']['llm_temp'] # llm temperature
CHATBOT_LLM_MAX_OUT= cfg['chatbot']['llm_max_out'] # llm 최대 아웃풋 토큰


class TokenTracker:
    """openai의 api 토큰 사용량을 추적하는 클래스"""
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    successful_requests: int = 0

    def sum_prompt_tokens(self, tokens: int ):
      self.prompt_tokens += tokens
      self.total_tokens += tokens

    def sum_completion_tokens(self, tokens: int ):
      self.completion_tokens += tokens
      self.total_tokens += tokens

    def sum_successful_requests(self, requests: int ):
      self.successful_requests += requests
    
    def __repr__(self) -> str:
        return (f'token tracker\ntotal_tokens : {self.total_tokens}\nprompt_tokens : {self.prompt_tokens}\n'
                f'completion tokens : {self.completion_tokens}\nsuccessful_requests : {self.successful_requests}')


class TokenTrackerHandler(AsyncCallbackHandler):
    """llm에 콜백으로 bind하여 토큰트래커의 동작을 제어하는 핸들러"""
    socketprint = None
    websocketaction: str = "appendtext"
    tracker: TokenTracker

    def __init__(self, tracker:TokenTracker):
        self.tracker = tracker
        self.encoder = tiktoken.encoding_for_model(CHATBOT_LLM_NAME)

    # ChatOpenAI는 on_chat_model_start에 콜백이 걸린다. (on_llm_start는 안걸림)
    # serialized엔 각종 메타데이터가, messages에는 입력 메시지들이 리스트로 들어온다.
    async def on_chat_model_start(self, serialized:Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any) -> None:
        for message_row in messages:
            for message in message_row:
                self.tracker.sum_prompt_tokens(len(self.encoder.encode(message.content)))
    
    # stream상황에서 토큰수 합산을 가능케 하기 위해 비동기로 토큰 올때마다 1개씩 추가(함수 호출횟수 많아지는 단점..)
    async def on_llm_new_token(self, token: str, **kwargs:Any) -> None:
        self.tracker.sum_completion_tokens(1)
    
    def on_llm_end(self, response:LLMResult, **kwargs:Any) -> None:
        self.tracker.sum_successful_requests(1)


class ChatBot:
    """기본 챗봇"""
    def __init__(self):
        # init chatbot infra
        self.insurance_estimator = InsuranceEstimator()
        self.llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'),
                              model_name=CHATBOT_LLM_NAME,
                              temperature=CHATBOT_LLM_TEMP,
                              max_tokens=CHATBOT_LLM_MAX_OUT)
        self.chain = (
            {
                'ins_name': lambda x: x['ins_name'],
                'ins_benefits': lambda x: x['ins_benefits'],
                'ins_necessity': lambda x: x['ins_necessity'],
                'ad_text': lambda x: x['ad_text'],
                'keyword_string': lambda x: x['keyword_string'],
                'question': lambda x: x['question']
            }
            | basic_chatbot_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def find_insurance_info(self, ins_name:str) -> dict:
        """
        DB에 보험이름으로 검색해서 정보(마케팅문구, 이점, 필요성)를 가져오는 함수\n
        ['ins_name', 'ad_text', 'benefits', 'necessity'] 영역을 dict로 반환한다.\n

        inputs:
            ins_name: 보험 이름
        returns:
            ans: 보험정보(dict)
        """
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(f"SELECT ins_name, ad_text, benefits, necessity FROM insurance_info WHERE ins_name='{ins_name}'")
        ans:tuple = cursor.fetchall()[0]
        conn.close()
        ans:dict = {key:val for key, val in zip(['ins_name', 'ad_text', 'ins_benefits', 'ins_necessity'], ans)}
        return ans

    async def invoke(self, question:str) -> str:
        """
        챗봇쿼리함수\n
        입력 질문을 받고 키워드추출 => 보험추천 => 추천보험정보 => llm 입력 => 마케팅문구 생성 순서로 동작\n
        서버처리량 확보를 위해 비동기로 동작\n

        inputs:
            question: 사용자 질문
        returns:
            output_text: 생성된 마케팅 문구
        """
        # 키워드 추출 및 보험추천
        ins_name, keyword_dict = await self.insurance_estimator.invoke(question)

        # ins_name, ad_text, ins_benefits, ins_necessity 가져오기
        input_dict:dict = self.find_insurance_info(ins_name)

        # 키워드별로 -1 혹은 '모름' 이면 keyword_string에 불포함
        keyword_string = ""
        for key, value in keyword_dict.items():
            if value not in {'모름', -1}:
                keyword_string += f"{key} : {value}\n"
        
        # 최종 입력 dict
        input_dict.update({'keyword_string':keyword_string, 'question':question})

        # 마케팅 문구생성 체인
        output_text = await self.chain.ainvoke(input_dict)
        output_text = f"키워드:\n{keyword_string}\n추천보험 : {ins_name}\n생성된 마케팅문구:\n" + output_text
        return output_text
        
        
        

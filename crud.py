import os
import json
import re
import datetime

import yaml
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from pymongo.cursor import Cursor

from schema import *

with open('config.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
MONGO_DBNAME = cfg['mongo']['db_name']
MONGO_CASECOL = cfg['mongo']['case_col']
MONGO_QACOL = cfg['mongo']['qa_col']
MONGO_URI = f"mongodb+srv://{os.getenv('ATLAS_ID')}:{os.getenv('ATLAS_PASSWD')}@cluster0.aqiqj9a.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# 아틀라스 연결
mongo_client = MongoClient(MONGO_URI)
try:
    mongo_client.admin.command('ping')
except ConnectionFailure as e:
    raise ConnectionFailure('MongoDB 연결 실패')
case_collection = mongo_client[MONGO_DBNAME][MONGO_CASECOL]
qa_collection = mongo_client[MONGO_DBNAME][MONGO_QACOL]

# Read
def mongo_read_cases(case_nums:List[str], max_docs=5) -> MongoContainer:
    """
    mongoDB의 판례정보를 가져오는 함수. 필터를 걸어서 가져올 수 있다.

    inputs: 
        case_nums: List[str] = 판례(사건)번호의 리스트 (Ex. ['12다3456', 78누9102', ...])
        max_docs: int = 가져올 최대 문서 숫자
    returns: MongoContainer = doc들의 리스트
    """
    # 파싱이후 검색
    parsed_nums: List[str|None] = []
    for i in case_nums:
        match = re.search(r'\d{2,}[가-힣]+\d{2,}', re.sub(r'\s+', '', i))
        if match:
            parsed_nums.append(match.group())
    
    cursor: Cursor = case_collection.find(filter={'사건번호':{'$in':parsed_nums}}).limit(max_docs)
    docs: List[CaseDoc] = [CaseDoc(**doc) for doc in cursor]
    container = MongoContainer(docs=docs)

    return container

# Create
def mongo_insert_qa(question:str, answer:str) -> str:
    """
    MongoDB에 한개의 문답 기록을 저장하는 함수. qa와 현재 시각을 db에 저장한다.
    insert 시도 이후 결과를 반환한다.

    inputs:
        question: str = 유저 질문 문자열
        answer: str = 챗봇 응답 문자열
    returns:
        result: str = ('ok' or error msg)
    """
    now: datetime.datetime = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
    try:
        res = qa_collection.insert_one({'question': question,
                                        'answer': answer,
                                        'timestamp': now})
        return 'ok'
    except Exception as e:
        return f'mongo_insert_qa중 오류 발생 : {str(e)}'



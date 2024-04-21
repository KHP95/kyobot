# std lib
import os
import ast
from typing import *
import warnings
warnings.filterwarnings('ignore')

# external lib
from dotenv import load_dotenv
import yaml
import numpy as np
import pandas as pd
import joblib

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from langchain_community.chat_models import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

# local lib
try:
    from .llm_prompts import basic_keyword_extractor_prompt
    from . import CATEGORIES, CATEGORY_NAMES, DEFAULT_KEYWORDS
except ImportError:
    pass



"""
보험추천을 위한 파이프라인과 키워드추출 모델을 정의하는 모듈
모듈이 아닌 스크립트로 실행시 Mock_data.csv에서 데이터 받아와서 학습 후 파이프라인 저장
"""

load_dotenv()
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
PIPELINE_PATH = os.path.join(FILE_PATH, 'recommend_pipeline.joblib')
CONFIG_PATH = os.path.join(FILE_PATH, '..', 'config.yaml')
with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
    cfg = yaml.safe_load(file)
EXTRACTOR_LLM_NAME = cfg['extractor']['llm_model']
EXTRACTOR_LLM_TEMP = cfg['extractor']['llm_temp']
EXTRACTOR_LLM_MAX_OUT= cfg['extractor']['llm_max_out']
"""
CATEGORIES ={
    0 : {1, 2, 3, 4, 5, 6, 7, 8, 9, 0}, # 연령대
    1 : {'남성', '여성', '모름'}, # 성별
    2 : {'간암', '갑상선암', '고혈압', '관절염', '뇌질환', '당뇨', '모름', '없음', '위암', '피부병'}, # 질병이력
    3 : {'없음', '적음', '중간', '많음', '모름'}, # 음주유무/량
    4 : {'없음', '적음', '중간', '많음', '모름'}, # 흡연유무/량
    5 : {'미혼' ,'기혼', '모름'} # 결혼유무
}
CATEGORY_NAMES = ['연령대', '성별', '질병이력', '음주유무/량', '흡연유무/량', '결혼유무']
DEFAULT_KEYWORDS = [-1, '모름', '모름', '모름', '모름', '모름']
"""


class EstimatorPreProcesseor(TransformerMixin, BaseEstimator):
    """
    sklearn의 pipeline에 커스텀 전처리기를 bind하는 클래스\n
    feature의 각 범주를 미리 지정해놓고 범주외의 데이터가 들어오거나, 데이터타입이 다르면 기본값으로 변환\n
    [[연령대, 성별, 질병이력, 음주유무, 흡연유무, 결혼여부]] 순서로 데이터 입력\n
    반드시 2중첩 리스트 혹은 판다스 데이터프레임, 시리즈로 입력\n
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X:List[list], y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values.tolist()
        elif isinstance(X, pd.Series):
            X = [X.values.tolist()]
        elif not isinstance(X[0], list):
            X = [X]

        for i in range(len(X)):
            for j in range(len(CATEGORIES)):
                # 카테고리 라벨에 없으면 정수형은 -1로, string은 '모름'으로 변경
                if X[i][j] not in CATEGORIES[j]:
                    X[i][j] = -1 if isinstance(X[i][j], int) else '모름'
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class KeywordExtractor():
    """
    키워드 추출기능이 캡슐화된 클래스\n
    llm모델 기반의 키워드 추출을 진행\n
    InsuranceEstimator에 포함되어있음\n
    """
    def __init__(self):
        self.llm = ChatOpenAI(model_name=EXTRACTOR_LLM_NAME,
                              temperature=EXTRACTOR_LLM_TEMP,
                              max_tokens=EXTRACTOR_LLM_MAX_OUT)
        self.prompt = basic_keyword_extractor_prompt
        self.chain = {'query':RunnablePassthrough()} | self.prompt | self.llm
    
    async def invoke(self, query:str) -> list:
        """
        문장에서 키워드 추출후 리스트로 반환\n
        제대로된 키워드가 없거나 인젝션 시도시 기본 키워드 반환\n
        서버처리량 확보를 위해 비동기로 동작\n

        inputs:
            query: 입력 질문
        returns:
            ans: 추출된 키워드 리스트
        """
        ans = await self.chain.ainvoke(query)
        ans:str = ans.content
        try: # 인젝션 방지를 위해 ast의 literal eval 사용
            ans:list = ast.literal_eval(ans)
        except: # 문제가생겨 eval이 안되는 경우 기본 키워드 반환
            ans:list = DEFAULT_KEYWORDS
        return ans
        

class InsuranceEstimator():
    """
    보험 추천(분류)에 관련된 기능이 캡슐화된 클래스
    """
    def __init__(self):
        # recommend_pipeline.joblib load
        self.pipeline:Pipeline = joblib.load(PIPELINE_PATH)
        self.extractor:KeywordExtractor = KeywordExtractor()

    async def predict(self, query:str) -> Tuple[str, dict]:
        """
        입력 문장에서 키워드를 추출한뒤 추천보험과 키워드 딕셔너리를 반환하는 함수\n
        단일 문장과 단일 보험을 반환\n
        서버처리량 확보를 위해 비동기로 동작\n

        inputs:
            query: 입력 문장
        returns:
            ans: 추천보험
            keyword_dict: 추출된 키워드(딕셔너리)
        """
        keywords:list = await self.extractor.invoke(query)
        ans:str = self.pipeline.predict([keywords]).tolist()[0]
        keyword_dict = {CATEGORY_NAMES[i]:value for i, value in enumerate(keywords)}
        return ans, keyword_dict

    async def invoke(self, query:str) -> str:
        """
        predict와 같은기능을 하는 함수
        """
        return await self.predict(query)


if __name__ == '__main__':
    print('loading & processing df..', end='\t')
    df = pd.read_csv(os.path.join(FILE_PATH, '..', 'datapkg','Mock_data.csv'), encoding='utf-8', sep=',')

    # 연령대 컬럼 추가
    df['연령대'] = df['연령'].apply(lambda x: int(x//10))
    # 6개 데이터와 target 활용
    df = df[['연령대', '성별', '질병이력', '음주유무/량', '흡연유무/량', '결혼여부', '추천보험']]

    # '성별', '질병이력', '음주유무/량', '흡연유무/량', '결혼여부', '추천보험' 컬럼에 일정비율로 '모름' 추가
    # '연령대' 일정비율로 -1 (모른다는 의미) 추가
    def add_missing(x:pd.Series, missing_rate=0.15):
        n = len(x)
        missing_rows = np.random.choice(n, int(n*missing_rate), replace=False)
        if x.name == '추천보험':
            pass
        elif x.name == '연령대':
            x[missing_rows] = int(-1)
        else:
            x[missing_rows] = '모름'
        return x

    df2 = df.apply(add_missing)

    # feature, target 분리
    y = df2['추천보험']
    X = df2.drop(columns=['추천보험'])

    # 훈련 및 평가 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=.1,
                                                        shuffle=True,
                                                        random_state=42)
    print('done')

    #### ColumnTransformer ####
    # 연령대, 성별, 질병이력, 음주유무, 흡연유무, 결혼여부 순서로 입력
    ct = ColumnTransformer(
        transformers=[
            ('pass', 'passthrough', [0]),
            ('ordinal', OrdinalEncoder(), [1, 2, 3, 4, 5]),
        ]
    )
    # 인코더 범주에 연령대 -1 과 나머지 카테고리항목 '모름' 추가 (데이터 미기입시 모름으로 변경을 위함)
    df.loc[len(df)] = {'연령대':int(-1), '성별':'모름', '질병이력':'모름', '음주유무/량':'모름', '흡연유무/량':'모름', '결혼여부':'모름', '추천보험':'교보실속종신보험'}
    # ct 학습은 모든 범주가 포함된 df로
    print('column_transformer fit..', end='\t')
    ct.fit(df.drop(columns=['추천보험']).values.tolist())
    print('done')


    #### 추천모델 학습 ####
    # 랜덤포레스트로 학습
    estimator = RandomForestClassifier(n_estimators=100,
                                    criterion='entropy',
                                    max_depth=4,
                                    n_jobs=-1, # 학습시 멀티프로세싱
                                    random_state=42)
    print('classifier fit..', end='\t')
    estimator.fit(ct.transform(X_train), y_train)
    estimator.n_jobs = 1 # 추론시 다시 단일 cpu로 변경 (소규모 데이터 추론시 오버헤드 방지)
    print('done')

    # 파이프라인 구성
    pipeline = Pipeline([
        ('preprocessor', EstimatorPreProcesseor()),
        ('columntransforemr', ct),
        ('estimator', estimator)
    ])

    # 저장(recommend_pipeline.joblib)
    print('saving..', end='\t')
    joblib.dump(pipeline, PIPELINE_PATH)
    print('done')
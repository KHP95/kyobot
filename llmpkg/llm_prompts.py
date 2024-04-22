from langchain.prompts import ChatPromptTemplate
from langchain.prompts import FewShotChatMessagePromptTemplate

from . import CATEGORIES

basic_chatbot_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
            """
            주어진 키워드와 보험에 관한 정보를 바탕으로, 마케팅 문구를 생성해줘.
            0. 질문에 마케팅문구 생성요청이 없더라도 암시적으로 마케팅문구 생성이라 판단해야해
            1. 단, **명시적으로 마케팅 문구 생성요청이 아닌 다른요청**을 한경우 마케팅문구 생성요청만 답변할 수 있다고 대답해줘.
            2. 보험에 관한 정보는 ["보험 이름", "장점", "필요성"] 이 있어.
            3. 보험에 맞는 "마케팅 예시문구" 역시 주어질거야.
            4. 마지막으로, ["연령대", "성별", "질병이력", "음주유무", "흡연유무", "결혼유무"] 가 키워드로 주어질거야.
            5. 중요!! **보험의 장점과 필요성을 이해한뒤, 키워드를 포함**하여 고객에게 먹힐만한 **참조문구를 창의적으로 변형한 마케팅 문구를 작성**해줘.
            6. 마케팅 문구 길이는 최소 100글자 이상 되도록 작성해줘.
            7. 답변은 장식없이 마케팅문구로만 작성해줘.

            
            ### 보험정보
            보험이름 : {ins_name}

            보험장점 : {ins_benefits}

            보험필요성 : {ins_necessity}

            ### 참조문구
            참조문구 : {ad_text}


            ### 키워드
            {keyword_string}

            """
            ),
            ("human", "{question}"),
        ]
    )


_keyword_extract_examples = [
    {'query':"36살, 갑상선암을 보유하고있고, 음주는 없고 흡연량은 높은 미혼자를 위한 마케팅 문구를 찾아줘.",
     'answer':"[3, '모름', '갑상선암', '없음', '많음', '미혼']"
    },
    {'query':"50대, 질병이력은 모르고 술을 조금씩 마시는 남자를 위한 보험이 뭐가 있을까?",
     'answer':"[5, '남자', '모름', '적음', '모름', '모름']"
    },
    {'query':"골초에 술을 좋아하는 사람을 위한 보험을 광고하려면 어떻게 해야할까?",
     'answer':"[-1, '모름', '모름', '많음', '많음', '모름']"
    },
    {'query':"뇌질환 위험이 있는 노인층을 위한 보험 마케팅문구를 하나 뽑아줄래?",
     'answer':"[7, '모름', '뇌질환', '모름', '모름', '모름']"
    }
]

_fewshot_template = FewShotChatMessagePromptTemplate(
    examples=_keyword_extract_examples,
    example_prompt=ChatPromptTemplate.from_messages([("human", "{query}"), ("ai", "{answer}")])
)

basic_keyword_extractor_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         f"""
         예시를 참고한다음, 주어진 query에서 6개종류의 키워드를 추출해줘.
         1. 순서대로 연령대, 성별, 질병이력, 음주유무, 흡연유무, 결혼여부를 추출해줘.
         2. 연령대는 한자리 숫자로 표시하고(Ex. 36살 -> 3, 60대 -> 6), 청년이면 3, 중년이면 5, 노인, 장년층이면 7로 추출해줘.
         2-1. 연령대를 찾을 수 없으면 -1로 추출해줘.
         3. 질병이력은 {list(CATEGORIES[2])} 목록중에 키워드가 있는지 골라줘.
         4. 음주유무, 흡연유무 및 양은 {list(CATEGORIES[3])} 목록중에 키워드가 있는지 골라줘.
         5. 결혼유무는 {list(CATEGORIES[5])} 목록중에 키워드가 있는지 골라줘.
         6. 질병이력, 음주유무, 흡연유무, 결혼유무는 찾을 수 없으면 '모름'으로 작성해줘.
         7. 다 추출했으면 [연령대, 성별, 질병이력, 음주유무, 흡연유무, 결혼여부] 순으로 리스트로 작성해줘.
         7. 다른말 말 없이 리스트로만 답변해줘.

         예시:
         """),
         _fewshot_template,
         ("human", "query : {query}")
    ]
)
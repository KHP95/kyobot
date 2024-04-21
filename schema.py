from typing import *
import datetime

from pydantic import BaseModel, Field


class ChatInput(BaseModel):
    # 입력 텍스트 
    query: str = Field(description='사용자 입력 텍스트')

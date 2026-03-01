#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

def analyze_apple_with_langchain():
    # 환경변수 로드
    load_dotenv()

    # LangChain ChatOpenAI 초기화
    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-3.5-turbo",
    )

    # 메시지 구성
    messages = [
        SystemMessage(content="주요 사업 부문, 긍정적 요인, 위험 요인 등을 포함한 종합적인 분석을 제공해주세요."),
        HumanMessage(content="Apple 주식의 현재 상황을 분석해주세요.")
    ]

    # LLM 호출
    response = llm.invoke(messages)

    return response.content

if __name__ == "__main__":
    analysis = analyze_apple_with_langchain()
    print(analysis)

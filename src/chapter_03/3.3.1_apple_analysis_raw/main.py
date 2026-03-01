#!/usr/bin/env python3
import os
from dotenv import load_dotenv
import openai

def analyze_apple_stock():
    # 환경변수 로드
    load_dotenv()

    # OpenAI 클라이언트 생성
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Apple 분석 요청
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "주요 사업 부문, 긍정적 요인, 위험 요인 등을 포함한 종합적인 분석을 제공해주세요."
            },
            {
                "role": "user",
                "content": "Apple 주식의 현재 상황을 분석해주세요."
            }
        ],
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    analysis = analyze_apple_stock()
    print(analysis)

#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

def demonstrate_temperature_consistency():
    """Temperature 설정에 따른 일관성 차이를 보여주는 예제"""
    load_dotenv()

    # Apple 분석 요청
    prompt = "Apple 주식의 현재 상황을 분석해주세요."

    # Temperature별 모델 설정
    models = {
        "일관성 우선 (0.0)": ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0),
        "균형잡힌 (0.7)": ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7),
        "창의적 (1.0)": ChatOpenAI(model="gpt-3.5-turbo", temperature=1.0)
    }

    results = {}

    for name, model in models.items():
        print(f"\n=== {name} ===")
        response = model.invoke([HumanMessage(content=prompt)])
        results[name] = response.content
        print(response.content[:200] + "..." if len(response.content) > 200 else response.content)

    return results

if __name__ == "__main__":
    demonstrate_temperature_consistency()

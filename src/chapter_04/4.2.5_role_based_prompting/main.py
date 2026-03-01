#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

def demonstrate_role_based_prompting():
    """Role-based Prompting의 효과를 보여주는 예제"""
    load_dotenv()

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

    # 일반적인 방식
    general_prompt = "기업의 재무상태를 어떻게 평가하나요?"

    # Role-based 방식들
    role_prompts = {
        "투자 분석가": """당신은 15년 경력의 투자 분석가입니다.
주요 증권사에서 기업 분석을 담당하며,
특히 재무제표 분석에 전문성을 가지고 있습니다.

기업의 재무상태를 어떻게 평가하는지 설명해주세요.""",

        "신경과 전문의": """당신은 30년 경력의 신경과 전문의입니다.
두통 환자를 진료할 때 어떤 원인들을 고려하는지 설명해주세요.""",

        "기업법 변호사": """당신은 기업법 전문 변호사입니다.
20년간 M&A 계약을 담당해왔습니다.
계약서 검토 시 가장 중요하게 보는 조항들을 설명해주세요."""
    }

    print("=== 일반적인 방식 ===")
    general_response = llm.invoke([HumanMessage(content=general_prompt)])
    print(general_response.content[:300] + "..." if len(general_response.content) > 300 else general_response.content)

    for role, prompt in role_prompts.items():
        print(f"\n=== {role} 역할 ===")
        response = llm.invoke([HumanMessage(content=prompt)])
        print(response.content[:300] + "..." if len(response.content) > 300 else response.content)

if __name__ == "__main__":
    demonstrate_role_based_prompting()

#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

def demonstrate_prompt_template():
    """PromptTemplate을 활용한 입력 표준화 예제"""
    load_dotenv()

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

    # 다중 변수 템플릿 생성
    investment_template = PromptTemplate(
        input_variables=["company_name", "analysis_focus", "time_horizon"],
        template="""당신은 {time_horizon} 투자를 전문으로 하는 분석가입니다.

{company_name}에 대해 다음 관점에서 분석해주세요:
분석 초점: {analysis_focus}
투자 기간: {time_horizon}

다음 형식으로 답변해주세요:
1. 현재 상황
2. 주요 강점
3. 리스크 요인
4. 투자 의견"""
    )

    # LLM과 연결
    chain = investment_template | llm

    # 다양한 분석 시나리오
    scenarios = [
        {"company_name": "Apple", "analysis_focus": "신제품 출시 영향", "time_horizon": "단기"},
        {"company_name": "Apple", "analysis_focus": "시장 점유율 변화", "time_horizon": "장기"},
        {"company_name": "Microsoft", "analysis_focus": "클라우드 사업 성장", "time_horizon": "중기"},
        {"company_name": "Tesla", "analysis_focus": "전기차 시장 경쟁", "time_horizon": "중기"}
    ]

    for scenario in scenarios:
        print(f"\n=== {scenario['company_name']} - {scenario['analysis_focus']} ({scenario['time_horizon']}) ===")
        result = chain.invoke(scenario).content
        print(result[:400] + "..." if len(result) > 400 else result)

if __name__ == "__main__":
    demonstrate_prompt_template()

#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from pydantic import BaseModel, Field
from enum import Enum
from typing import List

class InvestmentGrade(str, Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

class RiskLevel(str, Enum):
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"

class StockAnalysis(BaseModel):
    company_name: str = Field(description="회사명")
    investment_grade: InvestmentGrade = Field(description="투자 등급")
    target_price: float = Field(description="목표 주가 (달러)")
    confidence_score: float = Field(description="분석 신뢰도 (0-1)")
    key_strengths: List[str] = Field(description="주요 강점들")
    risk_factors: List[str] = Field(description="주요 리스크 요인들")
    analysis_summary: str = Field(description="분석 요약")

def demonstrate_structured_output():
    """구조화된 출력 파서 예제"""
    load_dotenv()

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

    # 1. 기본 파서 생성
    parser = PydanticOutputParser(pydantic_object=StockAnalysis)

    # 2. 프롬프트 템플릿 생성
    prompt_template = PromptTemplate(
        template="""당신은 투자 분석가입니다.

다음 회사에 대한 분석을 수행해주세요:
회사명: {company_name}
현재 주가: ${current_price}
시가총액: ${market_cap}B

{format_instructions}

분석 결과:""",
        input_variables=["company_name", "current_price", "market_cap"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    # 3. 체인 구성
    chain = prompt_template | llm | parser

    # 4. 분석 실행
    companies = [
        {"company_name": "Apple", "current_price": 175.43, "market_cap": 2.75},
        {"company_name": "Tesla", "current_price": 220.50, "market_cap": 0.70}
    ]

    for company_data in companies:
        print(f"\n=== {company_data['company_name']} 구조화된 분석 ===")
        try:
            result = chain.invoke(company_data)
            print(f"회사명: {result.company_name}")
            print(f"투자 등급: {result.investment_grade}")
            print(f"목표 주가: ${result.target_price}")
            print(f"신뢰도: {result.confidence_score:.2f}")
            print(f"주요 강점: {', '.join(result.key_strengths[:2])}")
            print(f"리스크 요인: {', '.join(result.risk_factors[:2])}")
            print(f"분석 요약: {result.analysis_summary[:100]}...")
        except Exception as e:
            print(f"파싱 오류: {e}")

            # 5. OutputFixingParser로 자동 수정 시도
            print("자동 수정 시도 중...")
            fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
            fixing_chain = prompt_template | llm | fixing_parser

            try:
                fixed_result = fixing_chain.invoke(company_data)
                print(f"수정된 결과 - 투자 등급: {fixed_result.investment_grade}")
            except Exception as fix_error:
                print(f"수정 실패: {fix_error}")

if __name__ == "__main__":
    demonstrate_structured_output()

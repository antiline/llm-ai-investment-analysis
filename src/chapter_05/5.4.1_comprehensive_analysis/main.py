import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# 각 분석 모듈 import
from yahoo_finance_analyzer import YahooFinanceAnalyzer
from sec_edgar_analyzer import SECAnalyzer
from google_news_analyzer import NewsEnhancedAnalyzer


class ComprehensiveAnalysis(BaseModel):
    """종합 투자 분석 결과 구조화"""
    yahoo_analysis: Dict[str, Any] = Field(description="Yahoo Finance 분석 결과")
    sec_analysis: Dict[str, Any] = Field(description="SEC EDGAR 분석 결과")
    news_analysis: str = Field(description="Google News 분석 결과")
    final_investment_grade: str = Field(description="최종 투자 등급")
    consensus_analysis: str = Field(description="통합 분석 결과")
    key_insights: List[str] = Field(description="핵심 인사이트")
    risk_assessment: Dict[str, str] = Field(description="통합 리스크 평가")
    investment_recommendation: str = Field(description="최종 투자 권고")


class ComprehensiveInvestmentAnalyzer:
    """종합 투자 분석 시스템"""

    def __init__(self, company_name: str, ticker: str):
        self.company_name = company_name
        self.ticker = ticker

        # LLM 초기화
        load_dotenv()
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-3.5-turbo",
            temperature=0
        )

        # 각 분석기 초기화
        self.yahoo_analyzer = YahooFinanceAnalyzer(company_name, ticker)
        self.sec_analyzer = SECAnalyzer(company_name, ticker)
        self.news_analyzer = NewsEnhancedAnalyzer(company_name, ticker)

        # 출력 파서 초기화
        self.output_parser = PydanticOutputParser(pydantic_object=ComprehensiveAnalysis)

        # 종합 분석 프롬프트
        self.comprehensive_prompt = PromptTemplate(
            input_variables=["company_name", "ticker", "yahoo_result", "sec_result", "news_result"],
            template="""
당신은 {company_name} ({ticker}) 전문 투자 애널리스트입니다.

다음 세 가지 데이터 소스의 분석 결과를 종합하여 최종 투자 권고를 제시하세요:

## 1. Yahoo Finance 실시간 데이터 분석
<yahoo_result>
{yahoo_result}
</yahoo_result>

## 2. SEC EDGAR 공식 데이터 분석
<sec_result>
{sec_result}
</sec_result>

## 3. Google News 뉴스 분석
<news_result>
{news_result}
</news_result>

## 종합 분석 요구사항

### 1. 데이터 소스 신뢰도 평가
- Yahoo Finance: [실시간 시장 데이터의 장단점]
- SEC EDGAR: [공식 재무 데이터의 장단점]
- Google News: [뉴스 데이터의 장단점]

### 2. 투자 등급 통합
- **최종 투자 등급**: [Strong Buy/Buy/Hold/Sell/Strong Sell]
- **통합 근거**: [세 데이터 소스의 일치/불일치 분석]

### 3. 핵심 인사이트 도출
- **주요 발견사항**: [3-5가지 핵심 인사이트]
- **데이터 간 상관관계**: [시장 데이터, 재무 데이터, 뉴스 데이터 간 연관성]

### 4. 리스크 통합 평가
- **시장 리스크**: [Yahoo Finance 기반]
- **재무 리스크**: [SEC EDGAR 기반]
- **뉴스 리스크**: [Google News 기반]
- **통합 리스크 등급**: [높음/보통/낮음]

### 5. 최종 투자 권고
- **투자 전략**: [구체적 투자 방향]
- **진입 시점**: [적절한 진입 타이밍]
- **포트폴리오 비중**: [권장 비중]
- **모니터링 포인트**: [중요 지표들]

모든 분석은 세 데이터 소스의 신뢰성을 고려하여 객관적으로 수행하세요.

{format_instructions}
"""
        )

    def run_comprehensive_analysis(self) -> ComprehensiveAnalysis:
        """종합 투자 분석 실행"""
        print(f"🚀 {self.company_name} ({self.ticker}) 종합 투자 분석 시작...")
        print("=" * 60)

        try:
            # 1. Yahoo Finance 분석
            print("📊 1단계: Yahoo Finance 실시간 데이터 분석")
            yahoo_result = self.yahoo_analyzer.run_analysis()
            yahoo_summary = self._summarize_yahoo_result(yahoo_result) if yahoo_result else "분석 실패"

            # 2. SEC EDGAR 분석
            print("\n📄 2단계: SEC EDGAR 공식 데이터 분석")
            sec_result = self.sec_analyzer.run_sec_analysis(self.ticker)
            sec_summary = self._summarize_sec_result(sec_result) if sec_result else "분석 실패"

            # 3. Google News 분석
            print("\n📰 3단계: Google News 뉴스 분석")
            news_result = self.news_analyzer.run_news_analysis()
            news_summary = news_result if news_result and "분석 실패" not in news_result else "분석 실패"

            # 4. 종합 분석
            print("\n🤖 4단계: 종합 분석 수행")
            comprehensive_result = self._perform_comprehensive_analysis(
                yahoo_summary, sec_summary, news_summary
            )

            # 5. 결과 출력
            self._print_comprehensive_results(comprehensive_result)

            return comprehensive_result

        except Exception as e:
            print(f"❌ 종합 분석 중 오류 발생: {e}")
            return self._create_default_comprehensive_analysis()

    def _summarize_yahoo_result(self, yahoo_result) -> str:
        """Yahoo Finance 결과 요약"""
        if not yahoo_result:
            return "Yahoo Finance 분석 실패"

        return f"""
투자 등급: {yahoo_result.investment_grade}
목표가: {yahoo_result.target_price}
투자 기간: {yahoo_result.investment_period}
핵심 근거: {', '.join(yahoo_result.key_reasons)}
주요 리스크: {', '.join(yahoo_result.risk_factors)}
"""

    def _summarize_sec_result(self, sec_result) -> str:
        """SEC EDGAR 결과 요약"""
        if not sec_result:
            return "SEC EDGAR 분석 실패"

        return f"""
투자 등급: {sec_result.investment_grade_revision}
수정 근거: {sec_result.revision_reason}
목표가 조정: {sec_result.target_price_adjustment}
신뢰도: {sec_result.confidence_level}
추가 리스크: {', '.join(sec_result.additional_risks)}
"""

    def _perform_comprehensive_analysis(self, yahoo_summary: str, sec_summary: str, news_summary: str) -> ComprehensiveAnalysis:
        """종합 분석 수행"""
        try:
            chain = self.comprehensive_prompt | self.llm | self.output_parser
            result = chain.invoke({
                "company_name": self.company_name,
                "ticker": self.ticker,
                "yahoo_result": yahoo_summary,
                "sec_result": sec_summary,
                "news_result": news_summary,
                "format_instructions": self.output_parser.get_format_instructions()
            })
            return result
        except Exception as e:
            print(f"❌ 종합 분석 오류: {e}")
            return self._create_default_comprehensive_analysis()

    def _print_comprehensive_results(self, result: ComprehensiveAnalysis):
        """종합 분석 결과 출력"""
        print("\n" + "=" * 60)
        print("🎯 종합 투자 분석 결과")
        print("=" * 60)
        print(f"📊 최종 투자 등급: {result.final_investment_grade}")
        print(f"💡 핵심 인사이트:")
        for i, insight in enumerate(result.key_insights[:3], 1):
            print(f"   {i}. {insight}")
        print(f"⚠️  통합 리스크 등급: {result.risk_assessment.get('통합_리스크_등급', '평가 불가')}")
        print(f"📈 최종 투자 권고: {result.investment_recommendation}")

    def _create_default_comprehensive_analysis(self) -> ComprehensiveAnalysis:
        """기본 종합 분석 결과 생성"""
        return ComprehensiveAnalysis(
            yahoo_analysis={"status": "분석 실패"},
            sec_analysis={"status": "분석 실패"},
            news_analysis="분석 실패",
            final_investment_grade="Hold",
            consensus_analysis="종합 분석 실패",
            key_insights=["분석 실패"],
            risk_assessment={"통합_리스크_등급": "평가 불가"},
            investment_recommendation="분석 실패로 인한 투자 권고 불가"
        )


def main():
    """메인 실행 함수"""
    print("🚀 종합 투자 분석 시스템")
    print("Yahoo Finance + SEC EDGAR + Google News 통합 분석")
    print("=" * 60)

    # Apple Inc. 종합 분석 예시
    analyzer = ComprehensiveInvestmentAnalyzer("Apple Inc.", "AAPL")
    result = analyzer.run_comprehensive_analysis()

    if result and result.final_investment_grade != "Hold":
        print("\n✅ 종합 분석 완료!")
    else:
        print("\n❌ 종합 분석 실패!")


if __name__ == "__main__":
    main()

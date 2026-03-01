#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Dict
from collections import Counter

class InvestmentGrade(str, Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"

class MarketSentiment(str, Enum):
    VERY_BULLISH = "VERY_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    VERY_BEARISH = "VERY_BEARISH"

class RiskLevel(str, Enum):
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"

class FinancialMetrics(BaseModel):
    revenue: float = Field(description="연간 매출 (백만 달러)")
    revenue_growth: float = Field(description="매출 성장률 (%)")
    operating_margin: float = Field(description="영업이익률 (%)")
    pe_ratio: float = Field(description="P/E 비율")
    debt_to_equity: float = Field(description="부채비율")
    roe: float = Field(description="ROE (%)")

class BusinessReport(BaseModel):
    company_name: str = Field(description="회사명")
    investment_grade: InvestmentGrade = Field(description="투자 등급")
    target_price: float = Field(description="목표 주가 (달러)")
    market_sentiment: MarketSentiment = Field(description="시장 심리")
    risk_level: RiskLevel = Field(description="리스크 수준")
    confidence_score: float = Field(description="분석 신뢰도 (0-1)")
    key_assumptions: List[str] = Field(description="핵심 가정들")
    financial_metrics: FinancialMetrics = Field(description="재무 지표")
    analysis_summary: str = Field(description="분석 요약")
    investment_thesis: str = Field(description="투자 논리")
    risk_factors: List[str] = Field(description="주요 리스크 요인들")

class BusinessReportGenerator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.parser = PydanticOutputParser(pydantic_object=BusinessReport)

        # Temperature별 모델 설정
        self.models = {
            "creative": ChatOpenAI(model="gpt-3.5-turbo", temperature=0.8, api_key=api_key),
            "balanced": ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, api_key=api_key),
            "conservative": ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, api_key=api_key)
        }

        # Role-based 프롬프트 정의
        self.roles = {
            "creative": """당신은 혁신적 기술과 산업 트렌드를 깊이 이해하는 산업 전문가입니다.
            기존의 틀에 얽매이지 않고 미래 지향적 관점에서 기업의 장기적 성장 가능성과
            파괴적 혁신의 잠재력을 평가하는 데 특화되어 있습니다.""",

            "balanced": """당신은 15년 이상의 주식 분석 경험을 가진 시니어 분석가입니다.
            다양한 시장 사이클을 경험하며 거시경제적 변화와 기업의 펀더멘털을 종합적으로
            고려한 실용적인 투자 의견을 제공합니다.""",

            "conservative": """당신은 데이터 기반의 객관적 분석을 통해 투자 리스크를
            최소화하는 것을 최우선으로 하는 정량 분석가입니다. 검증 가능한 재무 데이터와
            객관적인 지표들을 중심으로 보수적 접근을 선호합니다."""
        }

    def generate_report(self, company_data: dict, analysis_type: str = "balanced") -> BusinessReport:
        """비즈니스 리포트 생성"""
        model = self.models[analysis_type]
        role = self.roles[analysis_type]

        # 프롬프트 템플릿 구성
        prompt_template = PromptTemplate(
            template="""
            {role}

            다음 회사에 대한 종합적인 비즈니스 분석을 수행해주세요:

            회사명: {company_name}
            재무 데이터: {financial_data}
            시장 데이터: {market_data}
            뉴스 데이터: {news_data}

            {format_instructions}
            """,
            input_variables=["role", "company_name", "financial_data", "market_data", "news_data"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

        # 데이터 전처리
        processed_data = self._preprocess_data(company_data)

        # 프롬프트 실행
        chain = prompt_template | model | self.parser

        try:
            result = chain.invoke({
                "role": role,
                "company_name": processed_data["company_name"],
                "financial_data": processed_data["financial_data"],
                "market_data": processed_data["market_data"],
                "news_data": processed_data["news_data"]
            })
            return result
        except Exception as e:
            print(f"분석 오류: {e}")
            return None

    def _preprocess_data(self, data: dict) -> dict:
        """데이터 전처리"""
        processed = {}

        # 재무 데이터 포맷팅
        if "financial" in data:
            fin = data["financial"]
            processed["financial_data"] = f"""
            매출: ${fin.get('revenue', 0):,.0f}M (성장률 {fin.get('revenue_growth', 0):.1f}%)
            영업이익률: {fin.get('operating_margin', 0):.1f}%
            P/E 비율: {fin.get('pe_ratio', 0):.1f}
            부채비율: {fin.get('debt_to_equity', 0):.1f}
            ROE: {fin.get('roe', 0):.1f}%
            """

        # 시장 데이터 포맷팅
        if "market" in data:
            mkt = data["market"]
            processed["market_data"] = f"""
            현재가: ${mkt.get('current_price', 0):.2f}
            시가총액: ${mkt.get('market_cap', 0):,.0f}M
            52주 최고가: ${mkt.get('high_52w', 0):.2f}
            52주 최저가: ${mkt.get('low_52w', 0):.2f}
            """

        # 뉴스 데이터 포맷팅
        if "news" in data:
            news_items = []
            for i, news in enumerate(data["news"][:5], 1):
                impact = news.get("impact", "중립적")
                news_items.append(f"{i}. {news['title']} ({impact})")
            processed["news_data"] = "\n".join(news_items)

        processed["company_name"] = data.get("company_name", "Unknown Company")

        return processed

class PortfolioAnalyzer:
    def __init__(self, report_generator: BusinessReportGenerator):
        self.generator = report_generator

    def analyze_portfolio(self, portfolio_data: List[Dict], analysis_type: str = "balanced") -> Dict:
        """포트폴리오 전체 분석"""
        reports = []
        for company_data in portfolio_data:
            try:
                report = self.generator.generate_report(company_data, analysis_type)
                if report:
                    reports.append(report)
            except Exception as e:
                print(f"Error analyzing {company_data.get('company_name', 'Unknown')}: {e}")
                continue

        # 포트폴리오 요약 생성
        summary = self._generate_portfolio_summary(reports)

        return {
            "individual_reports": reports,
            "portfolio_summary": summary
        }

    def _generate_portfolio_summary(self, reports: List[BusinessReport]) -> Dict:
        """포트폴리오 요약 생성"""
        if not reports:
            return {"error": "분석 가능한 리포트가 없습니다."}

        # 투자 등급 분포
        grade_counts = Counter([r.investment_grade for r in reports])

        # 리스크 프로파일
        risk_counts = Counter([r.risk_level for r in reports])

        # 평균 신뢰도
        avg_confidence = sum(r.confidence_score for r in reports) / len(reports)

        # 상위 추천 종목 (BUY 등급, 높은 신뢰도)
        top_recommendations = [
            r for r in reports
            if r.investment_grade == InvestmentGrade.BUY and r.confidence_score > 0.8
        ]
        top_recommendations.sort(key=lambda x: x.confidence_score, reverse=True)

        return {
            "total_companies": len(reports),
            "grade_distribution": dict(grade_counts),
            "risk_profile": dict(risk_counts),
            "average_confidence": round(avg_confidence, 3),
            "top_recommendations": [
                {
                    "company": r.company_name,
                    "target_price": r.target_price,
                    "confidence": r.confidence_score
                } for r in top_recommendations[:3]
            ]
        }

def demonstrate_business_report_system():
    """비즈니스 리포트 생성 시스템 통합 예제"""
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        return

    # 시스템 초기화
    generator = BusinessReportGenerator(api_key)
    analyzer = PortfolioAnalyzer(generator)

    # Apple 샘플 데이터
    apple_data = {
        "company_name": "Apple Inc.",
        "financial": {
            "revenue": 394328,
            "revenue_growth": 8.1,
            "operating_margin": 29.0,
            "pe_ratio": 28.5,
            "debt_to_equity": 1.2,
            "roe": 15.8
        },
        "market": {
            "current_price": 175.43,
            "market_cap": 2750000,
            "high_52w": 198.23,
            "low_52w": 124.17
        },
        "news": [
            {"title": "Apple Vision Pro 출시 성공", "impact": "긍정적"},
            {"title": "중국 iPhone 판매량 감소", "impact": "부정적"},
            {"title": "서비스 매출 20% 성장", "impact": "긍정적"}
        ]
    }

    # Temperature별 분석 결과 비교
    print("=== Apple Inc. 분석 결과 비교 ===")

    for analysis_type in ["conservative", "balanced", "creative"]:
        print(f"\n[{analysis_type.upper()} 분석]")
        report = generator.generate_report(apple_data, analysis_type)
        if report:
            print(f"투자 등급: {report.investment_grade}")
            print(f"목표 주가: ${report.target_price}")
            print(f"시장 심리: {report.market_sentiment}")
            print(f"리스크 수준: {report.risk_level}")
            print(f"신뢰도: {report.confidence_score:.2f}")
            if report.key_assumptions:
                print(f"핵심 가정: {report.key_assumptions[0]}")

    # 포트폴리오 분석 예제
    portfolio_data = [
        apple_data,
        {
            "company_name": "Tesla Inc.",
            "financial": {
                "revenue": 96773,
                "revenue_growth": 18.8,
                "operating_margin": 9.2,
                "pe_ratio": 45.2,
                "debt_to_equity": 0.8,
                "roe": 12.5
            },
            "market": {
                "current_price": 220.50,
                "market_cap": 700000,
                "high_52w": 299.29,
                "low_52w": 138.80
            },
            "news": [
                {"title": "EV 시장 경쟁 심화", "impact": "부정적"},
                {"title": "자율주행 기술 발전", "impact": "긍정적"}
            ]
        }
    ]

    # 포트폴리오 분석 실행
    print("\n=== 포트폴리오 분석 결과 ===")
    portfolio_result = analyzer.analyze_portfolio(portfolio_data, "balanced")
    summary = portfolio_result["portfolio_summary"]

    if "error" not in summary:
        print(f"총 종목 수: {summary['total_companies']}")
        print(f"투자 등급 분포: {summary['grade_distribution']}")
        print(f"리스크 프로파일: {summary['risk_profile']}")
        print(f"평균 신뢰도: {summary['average_confidence']}")

        if summary['top_recommendations']:
            print("\n상위 추천 종목:")
            for rec in summary['top_recommendations']:
                print(f"- {rec['company']}: ${rec['target_price']} (신뢰도: {rec['confidence']:.2f})")
    else:
        print(summary["error"])

if __name__ == "__main__":
    demonstrate_business_report_system()

"""
5.3.1 Yahoo Finance 데이터 수집 및 분석

이 섹션에서는 Yahoo Finance API를 사용하여 실시간 주식 데이터를 수집하고,
LLM을 활용하여 구조화된 투자 분석을 수행합니다.

주요 기능:
- 실시간 시장 데이터 수집 (주가, 거래량, 시가총액 등)
- 재무 비율 데이터 수집 (P/E, ROE, 부채비율 등)
- LLM을 통한 구조화된 투자 분석
- Pydantic을 활용한 구조화된 출력
"""

import yfinance as yf
import json
from datetime import datetime
from typing import Dict, Any, List
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Pydantic 모델 정의
class InvestmentAnalysis(BaseModel):
    """투자 분석 결과 구조화"""
    current_situation: Dict[str, str] = Field(description="현재 상황 진단")
    financial_health: Dict[str, str] = Field(description="재무 건전성 평가")
    investment_grade: str = Field(description="투자 등급 (Strong Buy/Buy/Hold/Sell/Strong Sell)")
    target_price: str = Field(description="목표가 및 현재가 대비 비율")
    investment_period: str = Field(description="투자 기간 (단기/중기/장기)")
    key_reasons: list = Field(description="핵심 근거 3가지")
    risk_factors: list = Field(description="주요 리스크 2-3가지")
    monitoring_indicators: list = Field(description="모니터링 지표 2-3가지")

class YahooFinanceAnalyzer:
    """Yahoo Finance 데이터 수집 및 LLM 분석 클래스"""

    def __init__(self, company_name: str, ticker: str):
        self.company_name = company_name
        self.ticker = ticker
        self.yf_ticker = yf.Ticker(ticker)

        # LLM 초기화 (재무 분석을 위해 temperature=0 사용)
        load_dotenv()
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-3.5-turbo",
            temperature=0
        )

        # 출력 파서 초기화
        self.output_parser = PydanticOutputParser(pydantic_object=InvestmentAnalysis)

        # 프롬프트 템플릿 설정 (Chapter 4 구조화 기법 적용)
        self.analysis_prompt = PromptTemplate(
            input_variables=["company_name", "market_data", "financial_data"],
            template="""
당신은 {company_name} 전문 투자 애널리스트입니다.

다음 실시간 데이터를 바탕으로 종합적인 투자 분석을 수행하세요:

## 📊 실시간 시장 데이터
{market_data}

## 📈 재무 비율 데이터
{financial_data}

## 분석 요구사항

### 1. 현재 상황 진단
- 주가 적정성: [과대평가/적정/저평가] - 근거: [구체적 수치 기반 분석]
- 시장 포지션: [선도/중위/후발] - 근거: [섹터 내 위치 분석]

### 2. 재무 건전성 평가
- 수익성: [상/중/하] - 근거: [ROE, 영업이익률 기준]
- 성장성: [상/중/하] - 근거: [매출 성장률, P/E 비율 기준]
- 안전성: [상/중/하] - 근거: [부채비율, 유동비율 기준]

### 3. 투자 등급 및 추천
- **투자 등급**: [Strong Buy/Buy/Hold/Sell/Strong Sell]
- **목표가**: $XXX (현재가 대비 ±X%)
- **투자 기간**: [단기/중기/장기]
- **핵심 근거**: [3가지 주요 이유]

### 4. 리스크 요인
- 주요 리스크: [2-3가지]
- 모니터링 지표: [2-3가지]

모든 분석은 제공된 실시간 데이터를 기반으로 하며, 추측이 아닌 구체적 수치와 근거를 제시하세요.

{format_instructions}
"""
        )

        # LCEL 체인 생성
        self.analysis_chain = self.analysis_prompt | self.llm | self.output_parser

    def collect_market_data(self) -> Dict[str, Any]:
        """실시간 시장 데이터 수집"""
        try:
            info = self.yf_ticker.info
            hist = self.yf_ticker.history(period="5d")
            latest = hist.iloc[-1] if not hist.empty else None

            if latest is None:
                raise ValueError("주가 데이터를 가져올 수 없습니다")

            # 전일 대비 변화율 계산
            prev_close = hist.iloc[-2]['Close'] if len(hist) > 1 else latest['Close']
            change = latest['Close'] - prev_close
            change_percent = (change / prev_close) * 100

            return {
                "company_name": self.company_name,
                "ticker": self.ticker,
                "current_price": round(latest['Close'], 2),
                "change": round(change, 2),
                "change_percent": round(change_percent, 2),
                "volume": int(latest['Volume']),
                "market_cap": info.get('marketCap', 'N/A'),
                "pe_ratio": info.get('trailingPE', 'N/A'),
                "dividend_yield": info.get('dividendYield', 'N/A'),
                "52_week_high": info.get('fiftyTwoWeekHigh', 'N/A'),
                "52_week_low": info.get('fiftyTwoWeekLow', 'N/A'),
                "sector": info.get('sector', 'N/A'),
                "industry": info.get('industry', 'N/A'),
                "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

        except Exception as e:
            return {"error": str(e)}

    def collect_financial_ratios(self) -> Dict[str, Any]:
        """재무 비율 데이터 수집"""
        try:
            info = self.yf_ticker.info

            return {
                "valuation_ratios": {
                    "pe_ratio": info.get('trailingPE'),
                    "forward_pe": info.get('forwardPE'),
                    "peg_ratio": info.get('pegRatio'),
                    "price_to_book": info.get('priceToBook'),
                    "price_to_sales": info.get('priceToSalesTrailing12Months')
                },
                "profitability_ratios": {
                    "profit_margin": info.get('profitMargins'),
                    "operating_margin": info.get('operatingMargins'),
                    "return_on_equity": info.get('returnOnEquity'),
                    "return_on_assets": info.get('returnOnAssets')
                },
                "financial_health": {
                    "debt_to_equity": info.get('debtToEquity'),
                    "current_ratio": info.get('currentRatio'),
                    "quick_ratio": info.get('quickRatio'),
                    "cash_per_share": info.get('totalCashPerShare')
                }
            }

        except Exception as e:
            return {"error": f"재무 비율 수집 실패: {e}"}

    def analyze_with_llm(self, market_data: Dict, financial_data: Dict) -> InvestmentAnalysis:
        """LLM을 사용한 투자 분석 수행"""
        result = self.analysis_chain.invoke({
            "company_name": self.company_name,
            "market_data": json.dumps(market_data, indent=2, ensure_ascii=False),
            "financial_data": json.dumps(financial_data, indent=2, ensure_ascii=False),
            "format_instructions": self.output_parser.get_format_instructions()
        })
        return result

    def run_analysis(self) -> Dict[str, Any]:
        """종합 투자 분석 실행"""
        print(f"🔍 {self.company_name} ({self.ticker}) 분석 시작...")

        # 1. 시장 데이터 수집
        market_data = self.collect_market_data()
        if "error" in market_data:
            print(f"❌ 시장 데이터 수집 실패: {market_data['error']}")
            return market_data

        # 2. 재무 데이터 수집
        financial_data = self.collect_financial_ratios()
        if "error" in financial_data:
            print(f"❌ 재무 데이터 수집 실패: {financial_data['error']}")
            return financial_data

        # 3. LLM 분석
        try:
            analysis_result = self.analyze_with_llm(market_data, financial_data)
        except Exception as e:
            print(f"❌ LLM 분석 실패: {e}")
            return {"error": f"분석 실패: {e}"}

        return {
            "company_name": self.company_name,
            "ticker": self.ticker,
            "market_data": market_data,
            "financial_data": financial_data,
            "analysis_result": analysis_result,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

def demonstrate_analysis():
    """투자 분석 시연"""
    analyzer = YahooFinanceAnalyzer("Apple Inc.", "AAPL")
    result = analyzer.run_analysis()

    if "error" in result:
        print(f"❌ 분석 실패: {result['error']}")
        return

    print("\n" + "="*60)
    print("📊 Apple Inc. (AAPL) 투자 분석 결과")
    print("="*60)

    # 시장 데이터 출력
    market = result["market_data"]
    print(f"\n📈 현재 시장 상황")
    print(f"  현재가: ${market['current_price']} ({market['change']:+.2f}, {market['change_percent']:+.2f}%)")
    print(f"  거래량: {market['volume']:,}")
    print(f"  시가총액: ${market['market_cap']:,}" if market['market_cap'] != 'N/A' else f"  시가총액: {market['market_cap']}")
    print(f"  섹터: {market['sector']} | 업종: {market['industry']}")

    # 재무 데이터 출력
    financial = result["financial_data"]
    print(f"\n💰 주요 재무 지표")
    if financial.get("valuation_ratios", {}).get("pe_ratio"):
        print(f"  P/E 비율: {financial['valuation_ratios']['pe_ratio']:.2f}")
    if financial.get("profitability_ratios", {}).get("profit_margin"):
        print(f"  순이익률: {financial['profitability_ratios']['profit_margin']:.2%}")

    # LLM 분석 결과 출력
    analysis = result["analysis_result"]
    print(f"\n🎯 투자 분석 결과")
    print(f"  투자 등급: {analysis.investment_grade}")
    print(f"  목표가: {analysis.target_price}")
    print(f"  투자 기간: {analysis.investment_period}")

    print(f"\n✅ 핵심 근거:")
    for i, reason in enumerate(analysis.key_reasons, 1):
        print(f"  {i}. {reason}")

    print(f"\n⚠️  주요 리스크:")
    for i, risk in enumerate(analysis.risk_factors, 1):
        print(f"  {i}. {risk}")

    print(f"\n📊 모니터링 지표:")
    for i, indicator in enumerate(analysis.monitoring_indicators, 1):
        print(f"  {i}. {indicator}")

    return result

if __name__ == "__main__":
    demonstrate_analysis()

# 실행 예시 출력:
"""
🔍 Apple Inc. (AAPL) 분석 시작...

============================================================
📊 Apple Inc. (AAPL) 투자 분석 결과
============================================================

📈 현재 시장 상황
  현재가: $173.50 (+2.15, +1.26%)
  거래량: 52,847,123
  시가총액: $2,734,567,890,000
  섹터: Technology | 업종: Consumer Electronics

💰 주요 재무 지표
  P/E 비율: 28.45
  순이익률: 25.34%

🎯 투자 분석 결과
  투자 등급: Buy
  목표가: $185.00 (현재가 대비 +6.6%)
  투자 기간: 중기

✅ 핵심 근거:
  1. 강력한 브랜드 가치와 고객 충성도로 인한 안정적 수익 창출
  2. 서비스 부문 매출 성장으로 인한 수익성 개선 및 다각화
  3. 현금 흐름이 우수하고 재무 건전성이 높음

⚠️  주요 리스크:
  1. 중국 시장 의존도 및 지속적인 공급망 불확실성
  2. 규제 환경 변화 및 앱스토어 수수료 정책 관련 리스크
  3. 경쟁사와의 기술 경쟁 심화

📊 모니터링 지표:
  1. iPhone 판매량 및 서비스 부문 매출 성장률
  2. 중국 시장 상황 및 공급망 안정성
  3. AI 기술 개발 및 경쟁사 대비 혁신 속도
"""

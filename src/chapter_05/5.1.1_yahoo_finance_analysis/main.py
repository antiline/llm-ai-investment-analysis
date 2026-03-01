import os
import yfinance as yf
from datetime import datetime
from typing import Dict, Any, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


class InvestmentAnalysis(BaseModel):
    """투자 분석 결과 구조화"""
    current_situation: Dict[str, str] = Field(description="현재 상황 진단")
    financial_health: Dict[str, str] = Field(description="재무 건전성 평가")
    investment_grade: str = Field(description="투자 등급 (Strong Buy/Buy/Hold/Sell/Strong Sell)")
    target_price: str = Field(description="목표가 및 현재가 대비 비율")
    investment_period: str = Field(description="투자 기간 (단기/중기/장기)")
    key_reasons: List[str] = Field(description="핵심 근거 3가지")
    risk_factors: List[str] = Field(description="주요 리스크 2-3가지")
    monitoring_indicators: List[str] = Field(description="모니터링 지표 2-3가지")


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

        # 핵심: 프롬프트 템플릿 설정
        self.analysis_prompt = PromptTemplate(
            input_variables=["company_name", "market_data", "financial_data"],
            template="""
당신은 {company_name} 전문 투자 애널리스트입니다.

다음 실시간 데이터를 바탕으로 종합적인 투자 분석을 수행하세요:

## 📊 실시간 시장 데이터
<market_data>
{market_data}
</market_data>

## 📈 재무 비율 데이터
<financial_data>
{financial_data}
</financial_data>

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
        """시장 데이터 수집"""
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

    def collect_financial_ratios(self) -> Dict[str, Any]:
        """재무 비율 데이터 수집"""
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

    def analyze_with_llm(self, market_data: Dict[str, Any], financial_data: Dict[str, Any]) -> InvestmentAnalysis:
        """LLM을 통한 투자 분석"""
        try:
            result = self.analysis_chain.invoke({
                "company_name": self.company_name,
                "market_data": str(market_data),
                "financial_data": str(financial_data),
                "format_instructions": self.output_parser.get_format_instructions()
            })
            return result
        except Exception as e:
            print(f"LLM 분석 오류: {e}")
            # 기본 분석 결과 반환
            return InvestmentAnalysis(
                current_situation={"주가 적정성": "분석 실패", "시장 포지션": "분석 실패"},
                financial_health={"수익성": "분석 실패", "성장성": "분석 실패", "안전성": "분석 실패"},
                investment_grade="Hold",
                target_price="분석 실패",
                investment_period="단기",
                key_reasons=["분석 실패"],
                risk_factors=["분석 실패"],
                monitoring_indicators=["분석 실패"]
            )

    def run_analysis(self):
        """종합 투자 분석 실행"""
        print(f"🔍 {self.company_name} ({self.ticker}) 분석 시작...")

        try:
            # 1. 시장 데이터 수집
            print("📈 시장 데이터 수집 중...")
            market_data = self.collect_market_data()
            print(f"   현재가: ${market_data['current_price']} ({market_data['change']:+.2f}, {market_data['change_percent']:+.2f}%)")
            print(f"   거래량: {market_data['volume']:,}")
            print(f"   시가총액: ${market_data['market_cap']:,}" if market_data['market_cap'] != 'N/A' else "   시가총액: N/A")
            print(f"   섹터: {market_data['sector']} | 업종: {market_data['industry']}")

            # 2. 재무 데이터 수집
            print("💰 재무 데이터 수집 중...")
            financial_data = self.collect_financial_ratios()
            pe_ratio = financial_data['valuation_ratios']['pe_ratio']
            print(f"   P/E 비율: {pe_ratio:.2f}" if pe_ratio else "   P/E 비율: N/A")

            # 3. LLM 분석
            print("🤖 LLM 분석 중...")
            analysis_result = self.analyze_with_llm(market_data, financial_data)

            # 4. 결과 출력
            print(f"\n🎯 투자 분석 결과")
            print(f"   투자 등급: {analysis_result.investment_grade}")
            print(f"   목표가: {analysis_result.target_price}")
            print(f"   투자 기간: {analysis_result.investment_period}")
            print(f"   ✅ 핵심 근거: {', '.join(analysis_result.key_reasons[:2])}")
            print(f"   ⚠️  주요 리스크: {', '.join(analysis_result.risk_factors[:2])}")
            print(f"   📊 모니터링 지표: {', '.join(analysis_result.monitoring_indicators[:2])}")

            return analysis_result

        except Exception as e:
            print(f"❌ 분석 중 오류 발생: {e}")
            return None


def main():
    """메인 실행 함수"""
    print("🚀 Yahoo Finance 투자 분석 시스템")
    print("=" * 50)

    # Apple Inc. 분석 예시
    analyzer = YahooFinanceAnalyzer("Apple Inc.", "AAPL")
    result = analyzer.run_analysis()

    if result:
        print("\n✅ 분석 완료!")
    else:
        print("\n❌ 분석 실패!")


if __name__ == "__main__":
    main()

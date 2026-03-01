#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from typing import List, Dict, Any
import time

class AppleStockTools:
    """Apple 주식 분석을 위한 도구들"""

    def __init__(self):
        # 샘플 데이터 (실제로는 API 호출)
        self.sample_data = {
            "stock_price": {
                "current": 191.45,
                "change": 2.1,
                "volume": 45678900,
                "market_cap": 3000000000000
            },
            "financial_data": {
                "revenue": 94900000000,
                "eps": 1.46,
                "pe_ratio": 31.2,
                "peg_ratio": 1.8,
                "pb_ratio": 8.4
            },
            "news": [
                "Apple Q4 실적 발표, 매출 예상치 상회",
                "Vision Pro 출시로 새로운 성장 동력 확보",
                "서비스 사업 매출 22% 증가로 수익성 개선",
                "중국 시장에서 경쟁 심화 우려"
            ]
        }

    @tool
    def get_stock_price(self, symbol: str) -> str:
        """주식 현재가 정보를 가져옵니다."""
        if symbol.upper() == "AAPL":
            data = self.sample_data["stock_price"]
            return f"Apple (AAPL) 현재가: ${data['current']}, 전일대비 {data['change']}% 변화, 거래량: {data['volume']:,}"
        return f"{symbol} 주가 정보를 찾을 수 없습니다."

    @tool
    def get_financial_data(self, symbol: str, period: str = "quarterly") -> str:
        """재무 데이터를 가져옵니다."""
        if symbol.upper() == "AAPL":
            data = self.sample_data["financial_data"]
            return f"Apple (AAPL) {period} 재무 데이터:\n매출: ${data['revenue']:,}\nEPS: ${data['eps']}\nP/E: {data['pe_ratio']}\nPEG: {data['peg_ratio']}\nP/B: {data['pb_ratio']}"
        return f"{symbol} 재무 데이터를 찾을 수 없습니다."

    @tool
    def get_latest_news(self, company: str) -> str:
        """최신 뉴스를 가져옵니다."""
        if company.lower() in ["apple", "aapl"]:
            news_list = self.sample_data["news"]
            return f"Apple 관련 최신 뉴스:\n" + "\n".join([f"- {news}" for news in news_list])
        return f"{company} 관련 뉴스를 찾을 수 없습니다."

    @tool
    def calculate_valuation_metrics(self, symbol: str) -> str:
        """밸류에이션 지표를 계산합니다."""
        if symbol.upper() == "AAPL":
            data = self.sample_data["financial_data"]
            return f"Apple (AAPL) 밸류에이션 지표:\nP/E 비율: {data['pe_ratio']}\nPEG 비율: {data['peg_ratio']}\nP/B 비율: {data['pb_ratio']}\n\n해석: P/E 31.2는 기술주 평균 대비 적정 수준, PEG 1.8은 성장 대비 적정 밸류에이션"
        return f"{symbol} 밸류에이션 지표를 계산할 수 없습니다."

    @tool
    def generate_analysis_report(self, stock_data: str, news: str, financial_data: str, valuation: str) -> str:
        """종합 분석 보고서를 생성합니다."""
        return f"""
## Apple Inc. 종합 투자 분석 보고서

### 1. 주가 현황
{stock_data}

### 2. 재무 성과
{financial_data}

### 3. 최신 뉴스 및 이슈
{news}

### 4. 밸류에이션 분석
{valuation}

### 5. 투자 의견
현재 Apple은 안정적인 재무 성과와 혁신적인 제품 라인업을 보유하고 있습니다.
Vision Pro 출시로 새로운 성장 동력이 확보되었으며, 서비스 사업의 성장도 긍정적입니다.
다만 중국 시장의 경쟁 심화는 주의가 필요한 요소입니다.

**투자 등급: BUY**
**목표가: $220**
**투자 기간: 12개월**
        """

class ReActAppleAgent:
    """ReAct 패턴을 활용한 Apple 분석 에이전트"""

    def __init__(self):
        load_dotenv()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

        # 도구 설정
        self.tools = AppleStockTools()
        self.setup_agent()

    def setup_agent(self):
        """ReAct 에이전트 설정"""
        # 도구 리스트 생성
        tools = [
            Tool(
                name="get_stock_price",
                func=self.tools.get_stock_price,
                description="주식 현재가 정보를 가져옵니다. 입력: 주식 심볼 (예: AAPL)"
            ),
            Tool(
                name="get_financial_data",
                func=self.tools.get_financial_data,
                description="재무 데이터를 가져옵니다. 입력: 주식 심볼, 기간 (예: AAPL, quarterly)"
            ),
            Tool(
                name="get_latest_news",
                func=self.tools.get_latest_news,
                description="최신 뉴스를 가져옵니다. 입력: 회사명 (예: Apple)"
            ),
            Tool(
                name="calculate_valuation_metrics",
                func=self.tools.calculate_valuation_metrics,
                description="밸류에이션 지표를 계산합니다. 입력: 주식 심볼 (예: AAPL)"
            ),
            Tool(
                name="generate_analysis_report",
                func=self.tools.generate_analysis_report,
                description="종합 분석 보고서를 생성합니다. 입력: 주가, 뉴스, 재무, 밸류에이션 데이터"
            )
        ]

        # ReAct 에이전트 초기화
        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )

    def analyze_apple_stock(self, request: str) -> str:
        """Apple 주식 분석 수행"""
        print(f"=== ReAct 에이전트 분석 시작: {request} ===")

        try:
            start_time = time.time()
            result = self.agent.invoke({"input": request})["output"]
            end_time = time.time()

            print(f"분석 완료 (소요시간: {end_time - start_time:.2f}초)")
            return result

        except Exception as e:
            return f"분석 중 오류 발생: {e}"

    def demonstrate_react_pattern(self, request: str):
        """ReAct 패턴 시연"""
        print(f"\n=== ReAct 패턴 시연 ===")
        print(f"사용자 요청: {request}")
        print("\nReAct 에이전트의 작업 과정:")
        print("1. Thought: 필요한 정보 파악")
        print("2. Action: 적절한 도구 선택 및 실행")
        print("3. Observation: 결과 관찰")
        print("4. Thought: 다음 단계 계획")
        print("5. Action: 추가 정보 수집")
        print("6. ... (목표 달성까지 반복)")
        print("7. Final Answer: 종합 분석 결과")

        result = self.analyze_apple_stock(request)
        return result

class TraditionalSystem:
    """전통적인 시스템 (비교용)"""

    def __init__(self):
        load_dotenv()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

    def analyze_apple_stock(self, request: str) -> str:
        """전통적인 방식으로 Apple 주식 분석"""
        print(f"=== 전통적 시스템 분석: {request} ===")

        prompt = PromptTemplate(
            template="Apple Inc.에 대한 투자 분석을 해주세요: {request}",
            input_variables=["request"]
        )

        try:
            start_time = time.time()
            result = self.llm.predict(prompt.format(request=request))
            end_time = time.time()

            print(f"분석 완료 (소요시간: {end_time - start_time:.2f}초)")
            return result

        except Exception as e:
            return f"분석 중 오류 발생: {e}"

def demonstrate_react_agent():
    """ReAct 에이전트 예제 실행"""
    react_agent = ReActAppleAgent()
    traditional_system = TraditionalSystem()

    # 분석 요청들
    analysis_requests = [
        "Apple 주식을 분석해주세요",
        "Apple의 현재 투자 가치를 평가해주세요",
        "Apple의 최신 실적과 전망을 분석해주세요"
    ]

    print("🤖 ReAct 에이전트 Apple 분석 시스템")
    print("="*60)

    for i, request in enumerate(analysis_requests, 1):
        print(f"\n{'='*80}")
        print(f"요청 {i}: {request}")
        print('='*80)

        # ReAct 에이전트 분석
        print("\n[ReAct 에이전트 분석]")
        react_result = react_agent.demonstrate_react_pattern(request)

        print(f"\n[전통적 시스템 분석]")
        traditional_result = traditional_system.analyze_apple_stock(request)

        # 결과 비교
        print(f"\n{'='*80}")
        print("분석 결과 비교")
        print('='*80)

        print(f"\nReAct 에이전트 결과:")
        print(react_result[:500] + "..." if len(react_result) > 500 else react_result)

        print(f"\n전통적 시스템 결과:")
        print(traditional_result[:500] + "..." if len(traditional_result) > 500 else traditional_result)

    # ReAct의 장점 설명
    print(f"\n{'='*80}")
    print("ReAct 에이전트의 장점")
    print('='*80)

    benefits = [
        "1. 자율적 정보 수집: 필요한 정보를 스스로 판단하고 수집",
        "2. 동적 문제 해결: 상황에 따라 다른 접근 방식 선택",
        "3. 투명한 의사결정: 매 단계의 사고 과정이 투명하게 공개",
        "4. 체계적 분석: 단계별로 체계적인 분석 수행",
        "5. 확장 가능성: 새로운 도구와 기능을 쉽게 추가"
    ]

    for benefit in benefits:
        print(f"✓ {benefit}")

    # ReAct vs 전통적 방식 비교
    print(f"\n{'='*80}")
    print("ReAct vs 전통적 방식 비교")
    print('='*80)

    comparison = {
        "정보 수집": {
            "ReAct": "자율적, 필요한 정보를 스스로 판단",
            "전통적": "사용자가 모든 정보를 미리 제공"
        },
        "문제 해결": {
            "ReAct": "동적, 상황에 따라 접근 방식 변경",
            "전통적": "정적, 정해진 절차만 수행"
        },
        "투명성": {
            "ReAct": "높음, 사고 과정 공개",
            "전통적": "낮음, 결론만 제시"
        },
        "확장성": {
            "ReAct": "높음, 도구 추가 용이",
            "전통적": "낮음, 구조 변경 필요"
        }
    }

    for aspect, methods in comparison.items():
        print(f"\n{aspect}:")
        print(f"  ReAct: {methods['ReAct']}")
        print(f"  전통적: {methods['전통적']}")

if __name__ == "__main__":
    demonstrate_react_agent()

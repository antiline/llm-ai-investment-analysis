#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from typing import List, Dict, Any
import time

class FinancialAnalystAgent:
    """재무 분석 전문 에이전트"""

    def __init__(self):
        load_dotenv()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
        self.setup_tools()
        self.setup_agent()

    def setup_tools(self):
        """재무 분석 도구 설정"""
        self.tools = [
            Tool(
                name="analyze_financial_ratios",
                func=self.analyze_financial_ratios,
                description="재무 비율을 분석하여 회사의 재무 건전성을 평가합니다."
            ),
            Tool(
                name="evaluate_cash_flow",
                func=self.evaluate_cash_flow,
                description="현금 흐름을 분석하여 회사의 유동성을 평가합니다."
            ),
            Tool(
                name="assess_debt_levels",
                func=self.assess_debt_levels,
                description="부채 수준을 평가하여 재무 리스크를 분석합니다."
            )
        ]

    def setup_agent(self):
        """재무 분석 에이전트 설정"""
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )

    @tool
    def analyze_financial_ratios(self, company: str) -> str:
        """재무 비율 분석"""
        if company.lower() == "apple":
            return """
            Apple 재무 비율 분석:
            - P/E 비율: 31.2 (기술주 평균 대비 적정)
            - PEG 비율: 1.8 (성장 대비 적정 밸류에이션)
            - P/B 비율: 8.4 (자산 대비 적정)
            - ROE: 147% (매우 우수한 수익성)
            - 영업이익률: 29.0% (높은 수익성)

            결론: 재무 비율이 전반적으로 우수하며, 높은 수익성을 보여줍니다.
            """
        return f"{company} 재무 비율 데이터를 찾을 수 없습니다."

    @tool
    def evaluate_cash_flow(self, company: str) -> str:
        """현금 흐름 평가"""
        if company.lower() == "apple":
            return """
            Apple 현금 흐름 분석:
            - 영업현금흐름: $1,100억 (매우 강력)
            - 자유현금흐름: $900억 (높은 현금 창출 능력)
            - 현금 보유량: $1,600억 (풍부한 유동성)
            - 배당 지급: $150억 (안정적 배당)

            결론: 매우 강력한 현금 창출 능력과 풍부한 유동성을 보유하고 있습니다.
            """
        return f"{company} 현금 흐름 데이터를 찾을 수 없습니다."

    @tool
    def assess_debt_levels(self, company: str) -> str:
        """부채 수준 평가"""
        if company.lower() == "apple":
            return """
            Apple 부채 수준 분석:
            - 총 부채: $1,200억
            - 부채비율: 0.15 (매우 낮음)
            - 이자보상배율: 50배 (매우 우수)
            - 순부채: -$400억 (순현금 보유)

            결론: 매우 낮은 부채 수준과 우수한 재무 건전성을 보여줍니다.
            """
        return f"{company} 부채 데이터를 찾을 수 없습니다."

    def analyze(self, request: str) -> str:
        """재무 분석 수행"""
        return self.agent.invoke({"input": f"재무 분석 관점에서 {request}"})["output"]

class MarketAnalystAgent:
    """시장 분석 전문 에이전트"""

    def __init__(self):
        load_dotenv()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
        self.setup_tools()
        self.setup_agent()

    def setup_tools(self):
        """시장 분석 도구 설정"""
        self.tools = [
            Tool(
                name="analyze_market_trends",
                func=self.analyze_market_trends,
                description="시장 트렌드를 분석하여 산업 동향을 파악합니다."
            ),
            Tool(
                name="evaluate_competition",
                func=self.evaluate_competition,
                description="경쟁 환경을 분석하여 시장 포지션을 평가합니다."
            ),
            Tool(
                name="assess_market_risks",
                func=self.assess_market_risks,
                description="시장 리스크를 평가하여 투자 환경을 분석합니다."
            )
        ]

    def setup_agent(self):
        """시장 분석 에이전트 설정"""
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )

    @tool
    def analyze_market_trends(self, industry: str) -> str:
        """시장 트렌드 분석"""
        if industry.lower() in ["technology", "tech", "smartphone", "mobile"]:
            return """
            기술 산업 시장 트렌드:
            - 스마트폰 시장: 성숙 단계, 고급화 추세
            - AI/ML 시장: 급속 성장, 연 25% 성장 예상
            - AR/VR 시장: 초기 단계, Vision Pro로 새로운 기회
            - 서비스 시장: 구독 모델 확산, 높은 수익성

            결론: 기술 산업은 AI와 서비스 중심으로 전환되고 있습니다.
            """
        return f"{industry} 시장 트렌드 데이터를 찾을 수 없습니다."

    @tool
    def evaluate_competition(self, company: str) -> str:
        """경쟁 환경 평가"""
        if company.lower() == "apple":
            return """
            Apple 경쟁 환경 분석:
            - 스마트폰: Samsung, Huawei와 경쟁, 프리미엄 포지션
            - 서비스: Google, Microsoft와 경쟁, 생태계 우위
            - 하드웨어: 자체 칩 개발로 차별화
            - 브랜드: 강력한 브랜드 파워로 경쟁 우위

            결론: 프리미엄 포지션과 생태계 우위로 경쟁력이 강합니다.
            """
        return f"{company} 경쟁 환경 데이터를 찾을 수 없습니다."

    @tool
    def assess_market_risks(self, company: str) -> str:
        """시장 리스크 평가"""
        if company.lower() == "apple":
            return """
            Apple 시장 리스크 분석:
            - 규제 리스크: 앱스토어 규제 강화 우려
            - 기술 리스크: AI 경쟁 심화
            - 지역 리스크: 중국 시장 의존도
            - 환율 리스크: 글로벌 사업으로 환율 영향

            결론: 규제와 기술 변화가 주요 리스크 요소입니다.
            """
        return f"{company} 시장 리스크 데이터를 찾을 수 없습니다."

    def analyze(self, request: str) -> str:
        """시장 분석 수행"""
        return self.agent.invoke({"input": f"시장 분석 관점에서 {request}"})["output"]

class TechnologyAnalystAgent:
    """기술 분석 전문 에이전트"""

    def __init__(self):
        load_dotenv()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
        self.setup_tools()
        self.setup_agent()

    def setup_tools(self):
        """기술 분석 도구 설정"""
        self.tools = [
            Tool(
                name="analyze_technology_roadmap",
                func=self.analyze_technology_roadmap,
                description="기술 로드맵을 분석하여 미래 기술 방향을 파악합니다."
            ),
            Tool(
                name="evaluate_rd_investment",
                func=self.evaluate_rd_investment,
                description="R&D 투자를 분석하여 혁신 능력을 평가합니다."
            ),
            Tool(
                name="assess_technology_risks",
                func=self.assess_technology_risks,
                description="기술 리스크를 평가하여 기술적 위험을 분석합니다."
            )
        ]

    def setup_agent(self):
        """기술 분석 에이전트 설정"""
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )

    @tool
    def analyze_technology_roadmap(self, company: str) -> str:
        """기술 로드맵 분석"""
        if company.lower() == "apple":
            return """
            Apple 기술 로드맵:
            - AI/ML: 자체 AI 칩 개발, Siri 고도화
            - AR/VR: Vision Pro 플랫폼 확장
            - 자율주행: Project Titan 진행 중
            - 웨어러블: Apple Watch 기능 확장
            - 서비스: 클라우드 및 엔터테인먼트 확대

            결론: AI, AR/VR, 자율주행 등 미래 기술에 집중 투자하고 있습니다.
            """
        return f"{company} 기술 로드맵 데이터를 찾을 수 없습니다."

    @tool
    def evaluate_rd_investment(self, company: str) -> str:
        """R&D 투자 평가"""
        if company.lower() == "apple":
            return """
            Apple R&D 투자 분석:
            - R&D 지출: $300억 (매출의 3.2%)
            - 주요 투자 분야: AI, AR/VR, 자율주행
            - 특허 출원: 연 2,000건 이상
            - 인수합병: AI 스타트업 다수 인수

            결론: 지속적인 R&D 투자로 기술 혁신을 주도하고 있습니다.
            """
        return f"{company} R&D 투자 데이터를 찾을 수 없습니다."

    @tool
    def assess_technology_risks(self, company: str) -> str:
        """기술 리스크 평가"""
        if company.lower() == "apple":
            return """
            Apple 기술 리스크 분석:
            - AI 경쟁: Google, Microsoft와의 AI 경쟁
            - 기술 변화: 빠른 기술 변화에 대응 필요
            - 인재 확보: AI 인재 확보 경쟁
            - 특허 분쟁: 기술 특허 관련 소송 위험

            결론: AI 경쟁과 기술 변화가 주요 기술 리스크입니다.
            """
        return f"{company} 기술 리스크 데이터를 찾을 수 없습니다."

    def analyze(self, request: str) -> str:
        """기술 분석 수행"""
        return self.agent.invoke({"input": f"기술 분석 관점에서 {request}"})["output"]

class MetaAgent:
    """메타 에이전트 - 여러 전문 에이전트를 조율"""

    def __init__(self):
        load_dotenv()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

        # 전문 에이전트들 초기화
        self.financial_agent = FinancialAnalystAgent()
        self.market_agent = MarketAnalystAgent()
        self.technology_agent = TechnologyAnalystAgent()

    def coordinate_analysis(self, request: str) -> Dict[str, Any]:
        """여러 에이전트를 조율하여 종합 분석 수행"""
        print(f"=== 메타 에이전트 종합 분석: {request} ===")

        results = {}

        # 1. 재무 분석
        print("\n--- 재무 분석 에이전트 실행 ---")
        try:
            financial_result = self.financial_agent.analyze(request)
            results["financial_analysis"] = financial_result
            print("✓ 재무 분석 완료")
        except Exception as e:
            results["financial_analysis"] = f"재무 분석 오류: {e}"
            print(f"✗ 재무 분석 오류: {e}")

        # 2. 시장 분석
        print("\n--- 시장 분석 에이전트 실행 ---")
        try:
            market_result = self.market_agent.analyze(request)
            results["market_analysis"] = market_result
            print("✓ 시장 분석 완료")
        except Exception as e:
            results["market_analysis"] = f"시장 분석 오류: {e}"
            print(f"✗ 시장 분석 오류: {e}")

        # 3. 기술 분석
        print("\n--- 기술 분석 에이전트 실행 ---")
        try:
            technology_result = self.technology_agent.analyze(request)
            results["technology_analysis"] = technology_result
            print("✓ 기술 분석 완료")
        except Exception as e:
            results["technology_analysis"] = f"기술 분석 오류: {e}"
            print(f"✗ 기술 분석 오류: {e}")

        # 4. 종합 분석
        print("\n--- 종합 분석 수행 ---")
        synthesis = self.synthesize_results(results)
        results["synthesis"] = synthesis

        return results

    def synthesize_results(self, agent_results: Dict[str, str]) -> str:
        """여러 에이전트의 결과를 종합"""
        synthesis_prompt = PromptTemplate(
            template="""
            다음 세 전문 에이전트의 분석 결과를 종합하여 최종 투자 의견을 제시해주세요:

            재무 분석:
            {financial_analysis}

            시장 분석:
            {market_analysis}

            기술 분석:
            {technology_analysis}

            다음 형식으로 종합 분석을 작성해주세요:

            ## 종합 투자 분석 보고서

            ### 1. 핵심 결론
            [세 관점을 종합한 핵심 결론]

            ### 2. 재무 관점
            [재무 분석의 주요 인사이트]

            ### 3. 시장 관점
            [시장 분석의 주요 인사이트]

            ### 4. 기술 관점
            [기술 분석의 주요 인사이트]

            ### 5. 투자 권고
            [최종 투자 의견과 근거]

            ### 6. 리스크 요인
            [주요 리스크 요소들]

            종합 분석:""",
            input_variables=["financial_analysis", "market_analysis", "technology_analysis"]
        )

        chain = synthesis_prompt.format_prompt(
            financial_analysis=agent_results.get("financial_analysis", ""),
            market_analysis=agent_results.get("market_analysis", ""),
            technology_analysis=agent_results.get("technology_analysis", "")
        )

        return self.llm.predict(chain.to_string())

def demonstrate_multi_agent_system():
    """멀티 에이전트 시스템 예제 실행"""
    meta_agent = MetaAgent()

    # 분석 요청들
    analysis_requests = [
        "Apple의 투자 가치를 종합적으로 평가해주세요",
        "Apple의 미래 성장 가능성을 분석해주세요",
        "Apple의 주요 리스크와 기회를 평가해주세요"
    ]

    print("🤖 멀티 에이전트 Apple 분석 시스템")
    print("="*60)

    for i, request in enumerate(analysis_requests, 1):
        print(f"\n{'='*80}")
        print(f"요청 {i}: {request}")
        print('='*80)

        # 멀티 에이전트 분석
        start_time = time.time()
        results = meta_agent.coordinate_analysis(request)
        end_time = time.time()

        print(f"\n{'='*80}")
        print("멀티 에이전트 분석 결과")
        print('='*80)

        # 각 에이전트별 결과
        for agent_type, result in results.items():
            if agent_type != "synthesis":
                print(f"\n[{agent_type.upper()} 분석]")
                print(result[:300] + "..." if len(result) > 300 else result)

        # 종합 분석 결과
        print(f"\n{'='*80}")
        print("종합 분석 결과")
        print('='*80)
        print(results["synthesis"])

        print(f"\n분석 완료 (소요시간: {end_time - start_time:.2f}초)")

    # 멀티 에이전트의 장점 설명
    print(f"\n{'='*80}")
    print("멀티 에이전트 시스템의 장점")
    print('='*80)

    benefits = [
        "1. 전문성 분화: 각 에이전트가 특정 분야에 전문성 집중",
        "2. 포괄적 분석: 여러 관점에서 종합적인 분석 제공",
        "3. 확장성: 새로운 전문 에이전트를 쉽게 추가",
        "4. 병렬 처리: 여러 에이전트가 동시에 작업 수행",
        "5. 신뢰성: 각 전문 분야의 정확한 분석 보장"
    ]

    for benefit in benefits:
        print(f"✓ {benefit}")

    # 에이전트별 역할 설명
    print(f"\n{'='*80}")
    print("에이전트별 전문 분야")
    print('='*80)

    roles = {
        "재무 분석 에이전트": "재무 비율, 현금 흐름, 부채 수준 분석",
        "시장 분석 에이전트": "시장 트렌드, 경쟁 환경, 시장 리스크 분석",
        "기술 분석 에이전트": "기술 로드맵, R&D 투자, 기술 리스크 분석",
        "메타 에이전트": "여러 전문 에이전트 조율 및 종합 분석"
    }

    for agent, role in roles.items():
        print(f"• {agent}: {role}")

if __name__ == "__main__":
    demonstrate_multi_agent_system()

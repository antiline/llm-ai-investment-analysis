#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from typing import List, Dict, Any
import time

class TutorialAgentSystem:
    """단계별 학습을 위한 튜토리얼 에이전트 시스템"""

    def __init__(self):
        load_dotenv()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

        # 단계별 진행 상태
        self.stages = {
            "basic_agent": False,
            "react_agent": False,
            "multi_agent": False,
            "advanced_workflow": False
        }

        # 튜토리얼 도구 설정
        self.setup_tutorial_tools()

    def setup_tutorial_tools(self):
        """튜토리얼 도구 설정"""
        self.tools = [
            Tool(
                name="get_apple_info",
                func=self.get_apple_info,
                description="Apple의 기본 정보를 가져옵니다."
            ),
            Tool(
                name="analyze_stock_price",
                func=self.analyze_stock_price,
                description="Apple 주가를 분석합니다."
            ),
            Tool(
                name="get_financial_data",
                func=self.get_financial_data,
                description="Apple의 재무 데이터를 가져옵니다."
            ),
            Tool(
                name="generate_report",
                func=self.generate_report,
                description="분석 결과를 바탕으로 보고서를 생성합니다."
            )
        ]

    @tool
    def get_apple_info(self) -> str:
        """Apple 기본 정보"""
        return """
        Apple Inc. 기본 정보:
        - 설립: 1976년
        - CEO: Tim Cook
        - 본사: Cupertino, California
        - 주요 제품: iPhone, iPad, Mac, Apple Watch, AirPods
        - 시가총액: $3조 (세계 최대)
        - 직원 수: 164,000명
        """

    @tool
    def analyze_stock_price(self) -> str:
        """Apple 주가 분석"""
        return """
        Apple (AAPL) 주가 분석:
        - 현재가: $191.45
        - 전일대비: +2.1%
        - 52주 최고가: $198.23
        - 52주 최저가: $124.17
        - 거래량: 45,678,900주
        - 시가총액: $3조
        """

    @tool
    def get_financial_data(self) -> str:
        """Apple 재무 데이터"""
        return """
        Apple 재무 데이터 (2023년):
        - 매출: $394.3B
        - 순이익: $97.0B
        - 영업이익률: 29.0%
        - P/E 비율: 31.2
        - 현금 보유량: $1,600B
        - 부채비율: 0.15
        """

    @tool
    def generate_report(self, info: str, stock: str, financial: str) -> str:
        """분석 보고서 생성"""
        return f"""
        ## Apple Inc. 투자 분석 보고서

        ### 1. 회사 개요
        {info}

        ### 2. 주가 현황
        {stock}

        ### 3. 재무 성과
        {financial}

        ### 4. 투자 의견
        Apple은 강력한 브랜드 파워와 혁신적인 제품 라인업을 보유하고 있습니다.
        높은 수익성과 풍부한 현금 보유량으로 안정적인 재무 상태를 유지하고 있습니다.

        **투자 등급: BUY**
        **목표가: $220**
        **투자 기간: 12개월**
        """

    def stage_1_basic_agent(self):
        """1단계: 기본 에이전트"""
        print("=== 1단계: 기본 에이전트 ===")

        # 기본 에이전트 설정
        basic_agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

        # 기본 질문으로 테스트
        question = "Apple에 대한 기본 정보를 알려주세요"
        print(f"질문: {question}")

        try:
            result = basic_agent.invoke({"input": question})["output"]
            print(f"답변: {result}")
            self.stages["basic_agent"] = True
            print("✓ 기본 에이전트 완료")
        except Exception as e:
            print(f"오류: {e}")

        return result

    def stage_2_react_agent(self):
        """2단계: ReAct 에이전트"""
        print("\n=== 2단계: ReAct 에이전트 ===")

        # ReAct 에이전트 설정
        react_agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )

        # 복잡한 질문으로 테스트
        question = "Apple의 투자 가치를 분석해주세요"
        print(f"질문: {question}")

        try:
            result = react_agent.invoke({"input": question})["output"]
            print(f"답변: {result}")
            self.stages["react_agent"] = True
            print("✓ ReAct 에이전트 완료")
        except Exception as e:
            print(f"오류: {e}")

        return result

    def stage_3_multi_agent(self):
        """3단계: 멀티 에이전트 개념"""
        print("\n=== 3단계: 멀티 에이전트 개념 ===")

        # 여러 에이전트 시뮬레이션
        agents = {
            "재무 분석가": "재무 데이터 분석",
            "시장 분석가": "주가 및 시장 동향 분석",
            "기술 분석가": "제품 및 기술 전망 분석"
        }

        question = "Apple의 종합 투자 분석을 수행해주세요"
        print(f"질문: {question}")

        results = {}
        for agent_name, specialty in agents.items():
            print(f"\n--- {agent_name} ({specialty}) ---")

            # 각 전문 에이전트가 특정 도구만 사용
            if agent_name == "재무 분석가":
                agent_tools = [tool for tool in self.tools if "financial" in tool.name or "get_apple_info" in tool.name]
            elif agent_name == "시장 분석가":
                agent_tools = [tool for tool in self.tools if "stock" in tool.name or "get_apple_info" in tool.name]
            else:  # 기술 분석가
                agent_tools = [tool for tool in self.tools if "get_apple_info" in tool.name]

            agent = initialize_agent(
                tools=agent_tools,
                llm=self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=False
            )

            try:
                result = agent.invoke({"input": f"{specialty} 관점에서 {question}"})["output"]
                results[agent_name] = result
                print(f"✓ {agent_name} 분석 완료")
            except Exception as e:
                results[agent_name] = f"오류: {e}"
                print(f"✗ {agent_name} 분석 오류: {e}")

        self.stages["multi_agent"] = True
        print("✓ 멀티 에이전트 개념 완료")

        return results

    def stage_4_advanced_workflow(self):
        """4단계: 고급 워크플로우"""
        print("\n=== 4단계: 고급 워크플로우 ===")

        # 워크플로우 시뮬레이션
        workflow_steps = [
            "1. 기본 정보 수집",
            "2. 주가 데이터 분석",
            "3. 재무 데이터 분석",
            "4. 종합 보고서 생성"
        ]

        print("워크플로우 단계:")
        for step in workflow_steps:
            print(f"  {step}")

        # 순차적 실행
        try:
            # 1단계: 기본 정보 수집
            print("\n--- 1단계: 기본 정보 수집 ---")
            basic_info = self.get_apple_info()
            print("✓ 기본 정보 수집 완료")

            # 2단계: 주가 데이터 분석
            print("\n--- 2단계: 주가 데이터 분석 ---")
            stock_analysis = self.analyze_stock_price()
            print("✓ 주가 분석 완료")

            # 3단계: 재무 데이터 분석
            print("\n--- 3단계: 재무 데이터 분석 ---")
            financial_data = self.get_financial_data()
            print("✓ 재무 데이터 분석 완료")

            # 4단계: 종합 보고서 생성
            print("\n--- 4단계: 종합 보고서 생성 ---")
            final_report = self.generate_report(basic_info, stock_analysis, financial_data)
            print("✓ 종합 보고서 생성 완료")

            self.stages["advanced_workflow"] = True
            print("✓ 고급 워크플로우 완료")

            return final_report

        except Exception as e:
            print(f"워크플로우 오류: {e}")
            return f"오류: {e}"

    def run_tutorial(self):
        """전체 튜토리얼 실행"""
        print("🚀 에이전트 시스템 튜토리얼 시작")
        print("="*60)

        try:
            # 1단계: 기본 에이전트
            basic_result = self.stage_1_basic_agent()

            # 2단계: ReAct 에이전트
            react_result = self.stage_2_react_agent()

            # 3단계: 멀티 에이전트
            multi_results = self.stage_3_multi_agent()

            # 4단계: 고급 워크플로우
            workflow_result = self.stage_4_advanced_workflow()

            # 튜토리얼 완료
            print("\n" + "="*60)
            print("🎉 튜토리얼 완료!")
            print("="*60)

            completed_stages = sum(self.stages.values())
            total_stages = len(self.stages)
            print(f"완료된 단계: {completed_stages}/{total_stages}")

            # 다음 단계 안내
            print("\n다음 단계:")
            print("1. 더 복잡한 도구 추가하기")
            print("2. LangGraph 워크플로우 구현하기")
            print("3. 실제 API 연동하기")
            print("4. 성능 최적화하기")

            return {
                "basic_agent": basic_result,
                "react_agent": react_result,
                "multi_agent": multi_results,
                "workflow": workflow_result
            }

        except Exception as e:
            print(f"튜토리얼 실행 중 오류 발생: {e}")
            return None

    def interactive_demo(self):
        """대화형 데모"""
        print("\n=== 대화형 에이전트 데모 ===")
        print("질문을 입력하세요 (종료하려면 'quit' 입력):")

        # 기본 에이전트 설정
        agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

        while True:
            try:
                question = input("\n질문: ").strip()

                if question.lower() in ['quit', 'exit', '종료']:
                    print("데모를 종료합니다.")
                    break

                if not question:
                    continue

                print("에이전트가 분석 중...")
                start_time = time.time()
                answer = agent.invoke({"input": question})["output"]
                end_time = time.time()

                print(f"\n답변: {answer}")
                print(f"응답 시간: {end_time - start_time:.2f}초")

            except KeyboardInterrupt:
                print("\n데모를 종료합니다.")
                break
            except Exception as e:
                print(f"오류 발생: {e}")

def demonstrate_tutorial_agent():
    """튜토리얼 에이전트 시스템 예제 실행"""
    tutorial = TutorialAgentSystem()

    # 튜토리얼 실행
    results = tutorial.run_tutorial()

    if results:
        # 대화형 데모 실행 (선택사항)
        print("\n대화형 데모를 시작하시겠습니까? (y/n): ", end="")
        try:
            choice = input().strip().lower()
            if choice in ['y', 'yes', '예']:
                tutorial.interactive_demo()
        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")

if __name__ == "__main__":
    demonstrate_tutorial_agent()

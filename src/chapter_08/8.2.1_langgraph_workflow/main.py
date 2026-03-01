#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import PromptTemplate
from typing import List, Dict, Any, TypedDict
import time

class WorkflowState(TypedDict):
    """워크플로우 상태 정의"""
    request: str
    stock_data: str
    financial_data: str
    news_data: str
    analysis_result: str
    error: str

class AppleWorkflowTools:
    """Apple 분석 워크플로우 도구들"""

    def __init__(self):
        # 샘플 데이터
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

    def gather_stock_data(self, state: WorkflowState) -> WorkflowState:
        """주식 데이터 수집"""
        print("=== 1단계: 주식 데이터 수집 ===")

        try:
            data = self.sample_data["stock_price"]
            stock_summary = f"""
            Apple (AAPL) 주식 정보:
            - 현재가: ${data['current']}
            - 전일대비: {data['change']}%
            - 거래량: {data['volume']:,}
            - 시가총액: ${data['market_cap']:,}
            """

            state["stock_data"] = stock_summary
            print("✓ 주식 데이터 수집 완료")

        except Exception as e:
            state["error"] = f"주식 데이터 수집 실패: {e}"
            print(f"✗ 주식 데이터 수집 실패: {e}")

        return state

    def gather_financial_data(self, state: WorkflowState) -> WorkflowState:
        """재무 데이터 수집"""
        print("=== 2단계: 재무 데이터 수집 ===")

        try:
            data = self.sample_data["financial_data"]
            financial_summary = f"""
            Apple (AAPL) 재무 데이터:
            - 매출: ${data['revenue']:,}
            - EPS: ${data['eps']}
            - P/E 비율: {data['pe_ratio']}
            - PEG 비율: {data['peg_ratio']}
            - P/B 비율: {data['pb_ratio']}
            """

            state["financial_data"] = financial_summary
            print("✓ 재무 데이터 수집 완료")

        except Exception as e:
            state["error"] = f"재무 데이터 수집 실패: {e}"
            print(f"✗ 재무 데이터 수집 실패: {e}")

        return state

    def gather_news_data(self, state: WorkflowState) -> WorkflowState:
        """뉴스 데이터 수집"""
        print("=== 3단계: 뉴스 데이터 수집 ===")

        try:
            news_list = self.sample_data["news"]
            news_summary = f"""
            Apple 관련 최신 뉴스:
            {chr(10).join([f"- {news}" for news in news_list])}
            """

            state["news_data"] = news_summary
            print("✓ 뉴스 데이터 수집 완료")

        except Exception as e:
            state["error"] = f"뉴스 데이터 수집 실패: {e}"
            print(f"✗ 뉴스 데이터 수집 실패: {e}")

        return state

    def perform_analysis(self, state: WorkflowState) -> WorkflowState:
        """종합 분석 수행"""
        print("=== 4단계: 종합 분석 수행 ===")

        try:
            # LLM을 사용한 분석
            load_dotenv()
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

            analysis_prompt = PromptTemplate(
                template="""
                다음 Apple 관련 데이터를 바탕으로 종합 투자 분석을 수행해주세요:

                주식 데이터:
                {stock_data}

                재무 데이터:
                {financial_data}

                뉴스 데이터:
                {news_data}

                다음 형식으로 분석해주세요:

                ## Apple Inc. 종합 투자 분석

                ### 1. 현재 상황 요약
                [주가, 재무, 뉴스 종합 요약]

                ### 2. 투자 가치 평가
                [밸류에이션 관점에서의 평가]

                ### 3. 성장 동력 분석
                [미래 성장 가능성 분석]

                ### 4. 리스크 요인
                [주요 리스크 요소들]

                ### 5. 투자 권고
                [최종 투자 의견과 근거]

                분석:""",
                input_variables=["stock_data", "financial_data", "news_data"]
            )

            analysis_chain = analysis_prompt | llm
            result = analysis_chain.invoke({
                "stock_data": state["stock_data"],
                "financial_data": state["financial_data"],
                "news_data": state["news_data"]
            }).content

            state["analysis_result"] = result
            print("✓ 종합 분석 완료")

        except Exception as e:
            state["error"] = f"분석 수행 실패: {e}"
            print(f"✗ 분석 수행 실패: {e}")

        return state

class LangGraphWorkflowManager:
    """LangGraph를 활용한 워크플로우 관리자"""

    def __init__(self):
        self.tools = AppleWorkflowTools()
        self.setup_workflow()

    def setup_workflow(self):
        """워크플로우 설정"""
        print("=== LangGraph 워크플로우 설정 ===")

        # 상태 그래프 생성
        self.workflow = StateGraph(WorkflowState)

        # 노드 추가
        self.workflow.add_node("gather_stock_data", self.tools.gather_stock_data)
        self.workflow.add_node("gather_financial_data", self.tools.gather_financial_data)
        self.workflow.add_node("gather_news_data", self.tools.gather_news_data)
        self.workflow.add_node("perform_analysis", self.tools.perform_analysis)

        # 에러 처리
        def should_continue(state: WorkflowState) -> str:
            if state.get("error"):
                return "error_handler"
            return "continue"

        def error_handler(state: WorkflowState) -> WorkflowState:
            print(f"=== 에러 처리: {state['error']} ===")
            return state

        self.workflow.add_node("error_handler", error_handler)
        self.workflow.add_edge("error_handler", END)

        # 엣지 추가
        self.workflow.add_edge(START, "gather_stock_data")

        self.workflow.add_conditional_edges(
            "gather_stock_data",
            should_continue,
            {"continue": "gather_financial_data", "error_handler": "error_handler"}
        )

        self.workflow.add_conditional_edges(
            "gather_financial_data",
            should_continue,
            {"continue": "gather_news_data", "error_handler": "error_handler"}
        )

        self.workflow.add_conditional_edges(
            "gather_news_data",
            should_continue,
            {"continue": "perform_analysis", "error_handler": "error_handler"}
        )

        self.workflow.add_edge("perform_analysis", END)

        # 워크플로우 컴파일
        self.app = self.workflow.compile()

        print("✓ 워크플로우 설정 완료")

    def execute_workflow(self, request: str) -> Dict[str, Any]:
        """워크플로우 실행"""
        print(f"=== 워크플로우 실행: {request} ===")

        # 초기 상태 설정
        initial_state = {
            "request": request,
            "stock_data": "",
            "financial_data": "",
            "news_data": "",
            "analysis_result": "",
            "error": ""
        }

        try:
            start_time = time.time()
            result = self.app.invoke(initial_state)
            end_time = time.time()

            print(f"워크플로우 실행 완료 (소요시간: {end_time - start_time:.2f}초)")
            return result

        except Exception as e:
            print(f"워크플로우 실행 실패: {e}")
            return {"error": str(e)}

class SequentialWorkflow:
    """순차적 워크플로우 (비교용)"""

    def __init__(self):
        self.tools = AppleWorkflowTools()

    def execute_workflow(self, request: str) -> Dict[str, Any]:
        """순차적 워크플로우 실행"""
        print(f"=== 순차적 워크플로우 실행: {request} ===")

        state = {
            "request": request,
            "stock_data": "",
            "financial_data": "",
            "news_data": "",
            "analysis_result": "",
            "error": ""
        }

        try:
            start_time = time.time()

            # 순차적 실행
            state = self.tools.gather_stock_data(state)
            if state.get("error"):
                return state

            state = self.tools.gather_financial_data(state)
            if state.get("error"):
                return state

            state = self.tools.gather_news_data(state)
            if state.get("error"):
                return state

            state = self.tools.perform_analysis(state)

            end_time = time.time()
            print(f"순차적 워크플로우 실행 완료 (소요시간: {end_time - start_time:.2f}초)")

            return state

        except Exception as e:
            print(f"순차적 워크플로우 실행 실패: {e}")
            return {"error": str(e)}

def demonstrate_langgraph_workflow():
    """LangGraph 워크플로우 예제 실행"""
    langgraph_manager = LangGraphWorkflowManager()
    sequential_workflow = SequentialWorkflow()

    # 분석 요청들
    analysis_requests = [
        "Apple 주식을 종합적으로 분석해주세요",
        "Apple의 투자 가치를 평가해주세요",
        "Apple의 현재 상황과 전망을 분석해주세요"
    ]

    print("🔄 LangGraph 워크플로우 Apple 분석 시스템")
    print("="*60)

    for i, request in enumerate(analysis_requests, 1):
        print(f"\n{'='*80}")
        print(f"요청 {i}: {request}")
        print('='*80)

        # LangGraph 워크플로우 실행
        print("\n[LangGraph 워크플로우]")
        langgraph_result = langgraph_manager.execute_workflow(request)

        # 순차적 워크플로우 실행
        print("\n[순차적 워크플로우]")
        sequential_result = sequential_workflow.execute_workflow(request)

        # 결과 비교
        print(f"\n{'='*80}")
        print("워크플로우 결과 비교")
        print('='*80)

        if "error" not in langgraph_result:
            print(f"\nLangGraph 결과:")
            print(langgraph_result["analysis_result"][:500] + "..." if len(langgraph_result["analysis_result"]) > 500 else langgraph_result["analysis_result"])
        else:
            print(f"LangGraph 오류: {langgraph_result['error']}")

        if "error" not in sequential_result:
            print(f"\n순차적 워크플로우 결과:")
            print(sequential_result["analysis_result"][:500] + "..." if len(sequential_result["analysis_result"]) > 500 else sequential_result["analysis_result"])
        else:
            print(f"순차적 워크플로우 오류: {sequential_result['error']}")

    # LangGraph의 장점 설명
    print(f"\n{'='*80}")
    print("LangGraph 워크플로우의 장점")
    print('='*80)

    benefits = [
        "1. 복잡한 워크플로우 관리: 조건부 분기와 루프 지원",
        "2. 에러 처리: 자동 에러 감지 및 복구 메커니즘",
        "3. 시각화: 워크플로우 구조를 그래프로 시각화",
        "4. 확장성: 새로운 노드와 엣지를 쉽게 추가",
        "5. 모니터링: 각 단계별 실행 상태 추적"
    ]

    for benefit in benefits:
        print(f"✓ {benefit}")

    # 워크플로우 비교
    print(f"\n{'='*80}")
    print("LangGraph vs 순차적 워크플로우 비교")
    print('='*80)

    comparison = {
        "복잡성": {
            "LangGraph": "높음, 복잡한 분기와 조건 처리 가능",
            "순차적": "낮음, 단순한 순차 실행만 가능"
        },
        "에러 처리": {
            "LangGraph": "강력함, 자동 에러 감지 및 복구",
            "순차적": "제한적, 수동 에러 처리 필요"
        },
        "확장성": {
            "LangGraph": "높음, 새로운 노드 추가 용이",
            "순차적": "낮음, 구조 변경 시 전체 수정 필요"
        },
        "시각화": {
            "LangGraph": "지원, 워크플로우 그래프 제공",
            "순차적": "없음, 코드로만 표현"
        }
    }

    for aspect, methods in comparison.items():
        print(f"\n{aspect}:")
        print(f"  LangGraph: {methods['LangGraph']}")
        print(f"  순차적: {methods['순차적']}")

if __name__ == "__main__":
    demonstrate_langgraph_workflow()

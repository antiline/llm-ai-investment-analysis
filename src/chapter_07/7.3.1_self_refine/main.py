#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from typing import List, Dict, Any
import time

class SelfRefineAnalyzer:
    """Self-Refine 기법을 활용한 Apple 분석 시스템"""

    def __init__(self):
        load_dotenv()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

        # Self-Refine 프롬프트 설정
        self.setup_refine_prompts()

    def setup_refine_prompts(self):
        """Self-Refine 프롬프트 설정"""
        # 초기 분석 프롬프트
        self.initial_analysis_prompt = PromptTemplate(
            template="""
            당신은 Apple Inc. 전문 투자 분석가입니다.

            다음 질문에 대해 상세한 분석을 제공해주세요:

            질문: {question}

            다음 형식으로 분석해주세요:

            ## 초기 분석

            ### 1. 핵심 분석
            [질문에 대한 직접적인 분석]

            ### 2. 주요 근거
            [분석의 근거가 되는 정보들]

            ### 3. 결론
            [최종 결론과 권고사항]

            분석:""",
            input_variables=["question"]
        )

        # 자기 검토 프롬프트
        self.self_review_prompt = PromptTemplate(
            template="""
            당신은 엄격한 품질 관리자입니다.

            다음 Apple 분석 결과를 검토하고 개선점을 찾아주세요:

            ## 원본 분석
            {original_analysis}

            다음 관점에서 검토해주세요:

            1. **논리적 일관성**: 분석 과정이 논리적으로 일관된가?
            2. **근거의 충분성**: 제시된 근거가 결론을 뒷받침하는가?
            3. **반대 관점 고려**: 반대 의견이나 대안적 관점은 고려되었는가?
            4. **정보의 정확성**: 제시된 정보가 정확하고 최신인가?
            5. **분석의 깊이**: 충분히 깊이 있는 분석인가?

            ## 검토 결과

            ### 발견된 문제점
            [구체적인 문제점들을 나열]

            ### 개선 제안
            [구체적인 개선 방안들을 제시]

            ### 개선된 분석
            [개선된 분석 내용]

            검토 결과:""",
            input_variables=["original_analysis"]
        )

        # 최종 개선 프롬프트
        self.final_improvement_prompt = PromptTemplate(
            template="""
            당신은 Apple Inc. 전문 투자 분석가입니다.

            다음 검토 결과를 바탕으로 최종 개선된 분석을 작성해주세요:

            ## 검토 결과
            {review_result}

            ## 개선된 최종 분석

            다음 형식으로 개선된 분석을 작성해주세요:

            ### 1. 개선된 핵심 분석
            [검토 결과를 반영한 개선된 분석]

            ### 2. 강화된 근거
            [추가된 근거와 정보]

            ### 3. 반대 관점 고려
            [고려된 반대 의견과 대안적 관점]

            ### 4. 최종 결론
            [개선된 최종 결론과 권고사항]

            ### 5. 분석 신뢰도
            [개선 후 분석의 신뢰도 평가]

            개선된 분석:""",
            input_variables=["review_result"]
        )

    def initial_analysis(self, question: str) -> str:
        """초기 분석 수행"""
        print(f"=== 초기 분석 시작 ===")

        chain = self.initial_analysis_prompt | self.llm

        try:
            result = chain.invoke({"question": question}).content
            print("✓ 초기 분석 완료")
            return result
        except Exception as e:
            print(f"✗ 초기 분석 오류: {e}")
            return ""

    def self_review(self, original_analysis: str) -> str:
        """자기 검토 수행"""
        print(f"=== 자기 검토 시작 ===")

        chain = self.self_review_prompt | self.llm

        try:
            result = chain.invoke({"original_analysis": original_analysis}).content
            print("✓ 자기 검토 완료")
            return result
        except Exception as e:
            print(f"✗ 자기 검토 오류: {e}")
            return ""

    def final_improvement(self, review_result: str) -> str:
        """최종 개선 수행"""
        print(f"=== 최종 개선 시작 ===")

        chain = self.final_improvement_prompt | self.llm

        try:
            result = chain.invoke({"review_result": review_result}).content
            print("✓ 최종 개선 완료")
            return result
        except Exception as e:
            print(f"✗ 최종 개선 오류: {e}")
            return ""

    def iterative_refinement(self, question: str, max_iterations: int = 3) -> Dict[str, Any]:
        """반복적 개선 과정"""
        print(f"=== Self-Refine 반복 개선: {question} ===")

        refinement_history = []

        # 초기 분석
        current_analysis = self.initial_analysis(question)
        refinement_history.append({
            "iteration": 1,
            "type": "initial",
            "content": current_analysis
        })

        # 반복적 개선
        for iteration in range(2, max_iterations + 1):
            print(f"\n--- 반복 {iteration} ---")

            # 자기 검토
            review_result = self.self_review(current_analysis)
            refinement_history.append({
                "iteration": iteration,
                "type": "review",
                "content": review_result
            })

            # 개선된 분석
            improved_analysis = self.final_improvement(review_result)
            refinement_history.append({
                "iteration": iteration,
                "type": "improved",
                "content": improved_analysis
            })

            current_analysis = improved_analysis

            # 개선 효과 측정 (간단한 버전)
            improvement_score = self.measure_improvement(refinement_history)
            print(f"개선 점수: {improvement_score:.2f}")

            # 개선 효과가 미미하면 중단
            if iteration > 2 and improvement_score < 0.1:
                print("개선 효과가 미미하여 반복을 중단합니다.")
                break

        return {
            "final_analysis": current_analysis,
            "refinement_history": refinement_history,
            "total_iterations": len(refinement_history) // 2
        }

    def measure_improvement(self, history: List[Dict[str, Any]]) -> float:
        """개선 효과 측정 (간단한 버전)"""
        if len(history) < 4:
            return 0.0

        # 분석 길이와 키워드 다양성으로 개선 효과 측정
        keywords = ["근거", "분석", "결론", "위험", "기회", "전망", "전략"]

        recent_analysis = history[-1]["content"]
        previous_analysis = history[-3]["content"] if len(history) >= 3 else ""

        # 키워드 다양성 증가 측정
        recent_keywords = sum(1 for keyword in keywords if keyword in recent_analysis)
        previous_keywords = sum(1 for keyword in keywords if keyword in previous_analysis)

        if previous_keywords > 0:
            improvement = (recent_keywords - previous_keywords) / previous_keywords
            return max(0, min(1, improvement))

        return 0.0

    def compare_with_single_shot(self, question: str) -> Dict[str, Any]:
        """Single-shot과 Self-Refine 비교"""
        print(f"\n=== Single-shot vs Self-Refine 비교 ===")

        # Single-shot 분석
        single_shot_prompt = PromptTemplate(
            template="Apple Inc.에 대한 투자 분석을 간단히 해주세요: {question}",
            input_variables=["question"]
        )
        single_shot_chain = single_shot_prompt | self.llm

        try:
            start_time = time.time()
            single_shot_result = single_shot_chain.invoke({"question": question}).content
            single_shot_time = time.time() - start_time

            # Self-Refine 분석
            start_time = time.time()
            refine_result = self.iterative_refinement(question, max_iterations=2)
            refine_time = time.time() - start_time

            return {
                "single_shot": {
                    "result": single_shot_result,
                    "time": single_shot_time
                },
                "self_refine": {
                    "result": refine_result["final_analysis"],
                    "time": refine_time,
                    "iterations": refine_result["total_iterations"]
                }
            }
        except Exception as e:
            return {"error": str(e)}

def demonstrate_self_refine():
    """Self-Refine 예제 실행"""
    analyzer = SelfRefineAnalyzer()

    # Apple 관련 복잡한 분석 질문
    complex_questions = [
        "Apple의 Vision Pro가 새로운 성장 동력이 될 수 있을까요?",
        "Apple의 서비스 사업 확장이 수익성에 미치는 영향은?",
        "Apple의 중국 시장 의존도 리스크는 어느 정도인가요?"
    ]

    print("✨ Self-Refine Apple 분석 시스템")
    print("="*60)

    for i, question in enumerate(complex_questions, 1):
        print(f"\n{'='*80}")
        print(f"질문 {i}: {question}")
        print('='*80)

        # Self-Refine vs Single-shot 비교
        comparison = analyzer.compare_with_single_shot(question)

        if "error" not in comparison:
            print(f"\n[Single-shot 분석]")
            print(f"소요 시간: {comparison['single_shot']['time']:.2f}초")
            print(comparison["single_shot"]["result"][:300] + "...")

            print(f"\n[Self-Refine 분석]")
            print(f"소요 시간: {comparison['self_refine']['time']:.2f}초")
            print(f"반복 횟수: {comparison['self_refine']['iterations']}")
            print(comparison["self_refine"]["result"][:300] + "...")
        else:
            print(f"오류: {comparison['error']}")

    # Self-Refine의 장점 설명
    print(f"\n{'='*80}")
    print("Self-Refine의 장점")
    print('='*80)

    benefits = [
        "1. 품질 향상: 반복적 검토로 분석 품질 개선",
        "2. 오류 감소: 자기 검토를 통한 오류 발견 및 수정",
        "3. 깊이 증대: 단계적 개선으로 분석 깊이 향상",
        "4. 신뢰성 증대: 검증된 분석 결과 제공",
        "5. 완성도 향상: 반복적 개선으로 완성도 증대"
    ]

    for benefit in benefits:
        print(f"✓ {benefit}")

if __name__ == "__main__":
    demonstrate_self_refine()

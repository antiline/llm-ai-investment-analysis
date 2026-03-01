#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from typing import List, Dict, Any
import time
import statistics

class IntegratedAdvancedAnalyzer:
    """고급 프롬프트 기법들을 통합한 Apple 분석 시스템"""

    def __init__(self):
        load_dotenv()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

        # 통합 시스템 설정
        self.setup_integrated_system()

    def setup_integrated_system(self):
        """통합 시스템 설정"""
        # Chain-of-Thought 예시들
        self.cot_examples = [
            {
                "question": "Tesla의 전기차 시장 지배력은?",
                "reasoning": """
1단계: 시장 점유율 분석
- 글로벌 전기차 시장 점유율: 18%
- 미국 시장 점유율: 65%
- 유럽 시장 점유율: 12%

2단계: 기술적 우위 평가
- 배터리 기술: 자체 개발로 경쟁 우위
- 자율주행: FSD 기술 선도
- 생산 효율성: 기가팩토리 혁신

3단계: 경쟁 환경 분석
- 전통 자동차 업체들의 전기차 진입
- 중국 업체들의 가격 경쟁
- 신규 스타트업들의 도전

결론: 현재는 강력한 지배력, 향후 경쟁 심화 예상
                """
            }
        ]

        # Self-Consistency 관점들
        self.perspectives = {
            "성장주_관점": "성장성과 혁신 중심 분석",
            "가치주_관점": "안정성과 가치 중심 분석",
            "기술주_관점": "기술력과 경쟁 우위 중심 분석",
            "글로벌_관점": "글로벌 전략과 국제 시장 중심 분석"
        }

    def chain_of_thought_analysis(self, question: str) -> str:
        """Chain-of-Thought 분석"""
        print("=== Chain-of-Thought 분석 ===")

        cot_prompt = PromptTemplate(
            template="""
            당신은 전문 투자 분석가입니다. 다음 예시를 참고하여 단계별로 분석해주세요:

            예시:
            질문: Tesla의 전기차 시장 지배력은?
            답변:
            1단계: 시장 점유율 분석
            2단계: 기술적 우위 평가
            3단계: 경쟁 환경 분석
            결론: 현재는 강력한 지배력, 향후 경쟁 심화 예상

            질문: {question}

            단계별 분석:""",
            input_variables=["question"]
        )

        chain = cot_prompt | self.llm
        return chain.invoke({"question": question}).content

    def self_consistency_analysis(self, question: str) -> Dict[str, Any]:
        """Self-Consistency 분석"""
        print("=== Self-Consistency 분석 ===")

        results = {}

        for perspective, description in self.perspectives.items():
            prompt = PromptTemplate(
                template=f"""
                당신은 {description} 전문가입니다.

                질문: {{question}}

                {description} 관점에서 분석해주세요:""",
                input_variables=["question"]
            )

            chain = prompt | self.llm
            results[perspective] = chain.invoke({"question": question}).content

        return results

    def self_refine_analysis(self, initial_analysis: str) -> str:
        """Self-Refine 분석"""
        print("=== Self-Refine 분석 ===")

        # 자기 검토
        review_prompt = PromptTemplate(
            template="""
            다음 분석을 검토하고 개선점을 찾아주세요:

            원본 분석: {analysis}

            다음 관점에서 검토:
            1. 논리적 일관성
            2. 근거의 충분성
            3. 반대 관점 고려
            4. 분석의 깊이

            개선된 분석:""",
            input_variables=["analysis"]
        )

        chain = review_prompt | self.llm
        return chain.invoke({"analysis": initial_analysis}).content

    def integrated_analysis(self, question: str) -> Dict[str, Any]:
        """통합 분석 수행"""
        print(f"=== 통합 고급 분석: {question} ===")

        start_time = time.time()

        # 1단계: Chain-of-Thought 분석
        cot_result = self.chain_of_thought_analysis(question)

        # 2단계: Self-Consistency 분석
        consistency_results = self.self_consistency_analysis(question)

        # 3단계: Self-Refine 분석
        refined_result = self.self_refine_analysis(cot_result)

        # 4단계: 결과 종합
        synthesis = self.synthesize_results(cot_result, consistency_results, refined_result)

        end_time = time.time()

        return {
            "chain_of_thought": cot_result,
            "self_consistency": consistency_results,
            "self_refine": refined_result,
            "synthesis": synthesis,
            "execution_time": end_time - start_time
        }

    def synthesize_results(self, cot_result: str, consistency_results: Dict[str, str], refined_result: str) -> Dict[str, Any]:
        """결과 종합"""
        print("=== 결과 종합 ===")

        # 각 관점별 주요 키워드 추출
        keywords = ["성장", "위험", "기회", "매수", "매도", "보유", "긍정", "부정", "안정", "혁신"]

        perspective_keywords = {}
        for perspective, result in consistency_results.items():
            keyword_count = sum(1 for keyword in keywords if keyword in result)
            perspective_keywords[perspective] = keyword_count

        # 일관성 점수 계산
        consistency_scores = list(perspective_keywords.values())
        overall_consistency = statistics.mean(consistency_scores) if consistency_scores else 0

        # 최종 종합 분석
        synthesis_prompt = PromptTemplate(
            template="""
            다음 세 가지 분석 결과를 종합하여 최종 투자 의견을 제시해주세요:

            Chain-of-Thought 분석:
            {cot_analysis}

            Self-Consistency 분석 (다양한 관점):
            {consistency_analysis}

            Self-Refine 분석 (개선된 분석):
            {refined_analysis}

            다음 형식으로 종합 분석을 작성해주세요:

            ## 종합 투자 분석 보고서

            ### 1. 핵심 결론
            [세 기법을 종합한 핵심 결론]

            ### 2. 주요 근거
            [각 기법에서 도출된 주요 근거들]

            ### 3. 관점별 인사이트
            [각 관점에서 얻은 주요 인사이트]

            ### 4. 투자 권고
            [최종 투자 권고사항]

            ### 5. 분석 신뢰도
            [통합 분석의 신뢰도 평가]

            종합 분석:""",
            input_variables=["cot_analysis", "consistency_analysis", "refined_analysis"]
        )

        chain = synthesis_prompt | self.llm
        final_synthesis = chain.invoke({
            "cot_analysis": cot_result,
            "consistency_analysis": "\n\n".join([f"{k}: {v}" for k, v in consistency_results.items()]),
            "refined_analysis": refined_result
        }).content

        return {
            "final_analysis": final_synthesis,
            "perspective_keywords": perspective_keywords,
            "overall_consistency": overall_consistency,
            "analysis_quality": "high" if overall_consistency > 0.6 else "medium"
        }

    def compare_methods(self, question: str) -> Dict[str, Any]:
        """다양한 방법 비교"""
        print(f"\n=== 방법별 비교 분석 ===")

        methods = {}

        # 1. 기본 분석
        basic_prompt = PromptTemplate(
            template="Apple Inc.에 대한 투자 분석을 해주세요: {question}",
            input_variables=["question"]
        )
        basic_chain = basic_prompt | self.llm

        start_time = time.time()
        methods["basic"] = {
            "result": basic_chain.invoke({"question": question}).content,
            "time": time.time() - start_time
        }

        # 2. Chain-of-Thought만
        start_time = time.time()
        methods["cot_only"] = {
            "result": self.chain_of_thought_analysis(question),
            "time": time.time() - start_time
        }

        # 3. 통합 분석
        start_time = time.time()
        integrated_result = self.integrated_analysis(question)
        methods["integrated"] = {
            "result": integrated_result["synthesis"]["final_analysis"],
            "time": integrated_result["execution_time"],
            "consistency": integrated_result["synthesis"]["overall_consistency"],
            "quality": integrated_result["synthesis"]["analysis_quality"]
        }

        return methods

def demonstrate_integrated_system():
    """통합 고급 시스템 예제 실행"""
    analyzer = IntegratedAdvancedAnalyzer()

    # Apple 관련 복잡한 분석 질문
    complex_questions = [
        "Apple의 Vision Pro가 새로운 성장 동력이 될 수 있을까요?",
        "Apple의 서비스 사업 확장이 수익성에 미치는 영향은?",
        "Apple의 중국 시장 의존도 리스크는 어느 정도인가요?"
    ]

    print("🚀 통합 고급 프롬프트 기법 Apple 분석 시스템")
    print("="*70)

    for i, question in enumerate(complex_questions, 1):
        print(f"\n{'='*80}")
        print(f"질문 {i}: {question}")
        print('='*80)

        # 방법별 비교
        comparison = analyzer.compare_methods(question)

        print(f"\n[기본 분석]")
        print(f"소요 시간: {comparison['basic']['time']:.2f}초")
        print(comparison["basic"]["result"][:200] + "...")

        print(f"\n[Chain-of-Thought 분석]")
        print(f"소요 시간: {comparison['cot_only']['time']:.2f}초")
        print(comparison["cot_only"]["result"][:200] + "...")

        print(f"\n[통합 고급 분석]")
        print(f"소요 시간: {comparison['integrated']['time']:.2f}초")
        print(f"일관성 점수: {comparison['integrated']['consistency']:.2f}")
        print(f"분석 품질: {comparison['integrated']['quality']}")
        print(comparison["integrated"]["result"][:200] + "...")

    # 통합 시스템의 장점 설명
    print(f"\n{'='*80}")
    print("통합 고급 프롬프트 기법의 장점")
    print('='*80)

    benefits = [
        "1. 체계적 분석: Chain-of-Thought로 논리적 추론",
        "2. 다각도 검증: Self-Consistency로 편향성 감소",
        "3. 품질 향상: Self-Refine으로 지속적 개선",
        "4. 신뢰성 증대: 세 기법의 시너지 효과",
        "5. 전문가 수준: 실제 투자 전문가와 유사한 분석 과정"
    ]

    for benefit in benefits:
        print(f"✓ {benefit}")

    # 성능 비교 요약
    print(f"\n{'='*80}")
    print("성능 비교 요약")
    print('='*80)

    print("기본 분석: 빠르지만 단순함")
    print("Chain-of-Thought: 논리적이지만 일관성 부족")
    print("통합 분석: 복잡하지만 최고 품질과 신뢰성")

if __name__ == "__main__":
    demonstrate_integrated_system()

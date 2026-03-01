#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List, Dict, Any

class ChainOfThoughtAnalyzer:
    """Chain-of-Thought 기법을 활용한 Apple 분석 시스템"""

    def __init__(self):
        load_dotenv()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

        # CoT 예시들 설정
        self.setup_cot_examples()

        # CoT 프롬프트 템플릿 설정
        self.setup_cot_prompt()

    def setup_cot_examples(self):
        """Chain-of-Thought 예시들 설정"""
        self.cot_examples = [
            {
                "question": "Tesla 주식을 매수해야 할까요?",
                "reasoning": """
1단계: 재무 현황 분석
- 매출 성장률: 전년 대비 19% 증가
- 순이익: 150억 달러로 흑자 전환
- 현금 보유량: 290억 달러로 안정적

2단계: 시장 환경 분석
- 전기차 시장 성장률: 연 25% 성장 예상
- 정부 정책: 친환경 지원 정책 확대
- 소비자 선호도: 전기차 수요 증가

3단계: 경쟁 우위 분석
- 기술력: 배터리 기술 선도
- 브랜드: 강력한 브랜드 파워
- 생산 능력: 글로벌 공장 확장

결론: 장기 성장 잠재력이 높아 매수 추천
근거: 안정적 재무구조 + 성장하는 시장 + 기술적 우위
                """
            },
            {
                "question": "Microsoft의 클라우드 사업 전망은?",
                "reasoning": """
1단계: 사업 현황 분석
- Azure 매출: 연 40% 성장
- 시장 점유율: AWS 다음 2위 (22%)
- 고객 기반: 대기업 중심 확대

2단계: 시장 동향 분석
- 클라우드 시장: 연 20% 성장 예상
- AI 서비스 수요: 급증 중
- 하이브리드 클라우드: 대세

3단계: 경쟁력 평가
- 기술력: AI/ML 서비스 우수
- 생태계: Office 365와 연계
- 보안: 엔터프라이즈급 보안

결론: 클라우드 사업은 Microsoft의 핵심 성장 동력
근거: 높은 성장률 + 강력한 기술력 + 시장 확장성
                """
            }
        ]

    def setup_cot_prompt(self):
        """Chain-of-Thought 프롬프트 설정"""
        # 예시 템플릿
        example_prompt = PromptTemplate(
            input_variables=["question", "reasoning"],
            template="질문: {question}\n\n답변: {reasoning}"
        )

        # Few-shot 프롬프트 템플릿
        self.cot_prompt = FewShotPromptTemplate(
            examples=self.cot_examples,
            example_prompt=example_prompt,
            prefix="당신은 전문 투자 분석가입니다. 다음 예시들을 참고하여 단계별로 분석해주세요:",
            suffix="질문: {input}\n\n답변:",
            input_variables=["input"],
            example_separator="\n\n"
        )

        # CoT 체인 생성
        self.cot_chain = self.cot_prompt | self.llm

    def analyze_with_cot(self, question: str) -> str:
        """Chain-of-Thought로 분석"""
        print(f"=== Chain-of-Thought 분석: {question} ===")

        try:
            result = self.cot_chain.invoke({"input": question}).content
            return result
        except Exception as e:
            return f"분석 중 오류 발생: {e}"

    def compare_with_zero_shot(self, question: str) -> Dict[str, str]:
        """Zero-shot과 CoT 비교"""
        print(f"\n=== Zero-shot vs Chain-of-Thought 비교 ===")

        # Zero-shot 분석
        zero_shot_prompt = PromptTemplate(
            template="Apple Inc.에 대한 투자 의견을 간단히 제시해주세요: {question}",
            input_variables=["question"]
        )
        zero_shot_chain = zero_shot_prompt | self.llm

        try:
            zero_shot_result = zero_shot_chain.invoke({"question": question}).content
            cot_result = self.analyze_with_cot(question)

            return {
                "zero_shot": zero_shot_result,
                "chain_of_thought": cot_result
            }
        except Exception as e:
            return {"error": str(e)}

    def demonstrate_cot_benefits(self):
        """CoT의 장점 시연"""
        print("=== Chain-of-Thought의 장점 시연 ===")

        benefits = {
            "투명성": "추론 과정을 명시하여 신뢰도 향상",
            "정확성": "단계별 분석으로 정확도와 깊이 향상",
            "이해도": "사용자가 추론 과정을 따라가며 납득 가능",
            "검증성": "각 단계별로 논리적 오류 검증 가능"
        }

        for benefit, description in benefits.items():
            print(f"✓ {benefit}: {description}")

def demonstrate_chain_of_thought():
    """Chain-of-Thought 예제 실행"""
    analyzer = ChainOfThoughtAnalyzer()

    # Apple 관련 분석 질문들
    apple_questions = [
        "Apple의 iPhone 매출 감소가 미래 성장에 미치는 영향은?",
        "Apple Vision Pro가 새로운 성장 동력이 될 수 있을까요?",
        "Apple의 서비스 사업 확장이 수익성에 미치는 영향은?",
        "Apple의 중국 시장 의존도 리스크는 어느 정도인가요?"
    ]

    print("🚀 Chain-of-Thought Apple 분석 시스템")
    print("="*60)

    # CoT의 장점 설명
    analyzer.demonstrate_cot_benefits()

    # 각 질문에 대한 CoT 분석
    for i, question in enumerate(apple_questions, 1):
        print(f"\n{'='*80}")
        print(f"질문 {i}: {question}")
        print('='*80)

        # Zero-shot vs CoT 비교
        comparison = analyzer.compare_with_zero_shot(question)

        if "error" not in comparison:
            print(f"\n[Zero-shot 분석]")
            print(comparison["zero_shot"])

            print(f"\n[Chain-of-Thought 분석]")
            print(comparison["chain_of_thought"])
        else:
            print(f"오류: {comparison['error']}")

    # CoT 분석의 특징 요약
    print(f"\n{'='*80}")
    print("Chain-of-Thought 분석의 특징")
    print('='*80)

    features = [
        "1. 단계별 체계적 접근",
        "2. 명시적 추론 과정",
        "3. 논리적 근거 제시",
        "4. 검증 가능한 분석",
        "5. 투명한 의사결정"
    ]

    for feature in features:
        print(f"✓ {feature}")

if __name__ == "__main__":
    demonstrate_chain_of_thought()

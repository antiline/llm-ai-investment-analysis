#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from typing import List, Dict, Any
import statistics

class SelfConsistencyAnalyzer:
    """Self-Consistency 기법을 활용한 Apple 분석 시스템"""

    def __init__(self):
        load_dotenv()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)  # 다양성을 위해 높은 temperature

        # 다양한 관점별 프롬프트 설정
        self.setup_perspective_prompts()

    def setup_perspective_prompts(self):
        """다양한 관점별 프롬프트 설정"""
        self.perspectives = {
            "성장주_관점": {
                "prompt": PromptTemplate(
                    template="""
                    당신은 성장주 전문 분석가입니다. 성장성과 혁신에 중점을 두고 분석해주세요.

                    질문: {question}

                    다음 관점에서 분석해주세요:
                    1. 매출 성장률과 성장 동력
                    2. 신기술 개발 및 혁신성
                    3. 시장 확장 가능성
                    4. 미래 성장 잠재력

                    분석:""",
                    input_variables=["question"]
                ),
                "description": "성장성과 혁신 중심 분석"
            },
            "가치주_관점": {
                "prompt": PromptTemplate(
                    template="""
                    당신은 가치주 전문 분석가입니다. 안정성과 가치에 중점을 두고 분석해주세요.

                    질문: {question}

                    다음 관점에서 분석해주세요:
                    1. 재무 건전성과 안정성
                    2. 배당 및 현금 흐름
                    3. 자산 가치와 부채 상황
                    4. 위험 대비 수익률

                    분석:""",
                    input_variables=["question"]
                ),
                "description": "안정성과 가치 중심 분석"
            },
            "기술주_관점": {
                "prompt": PromptTemplate(
                    template="""
                    당신은 기술주 전문 분석가입니다. 기술력과 경쟁 우위에 중점을 두고 분석해주세요.

                    질문: {question}

                    다음 관점에서 분석해주세요:
                    1. 기술적 경쟁 우위
                    2. R&D 투자 및 혁신 능력
                    3. 생태계와 플랫폼 파워
                    4. 기술 트렌드 대응력

                    분석:""",
                    input_variables=["question"]
                ),
                "description": "기술력과 경쟁 우위 중심 분석"
            },
            "글로벌_관점": {
                "prompt": PromptTemplate(
                    template="""
                    당신은 글로벌 투자 전문가입니다. 국제 시장과 글로벌 전략에 중점을 두고 분석해주세요.

                    질문: {question}

                    다음 관점에서 분석해주세요:
                    1. 글로벌 시장 점유율과 확장
                    2. 지역별 성과와 전략
                    3. 환율 및 지정학적 리스크
                    4. 글로벌 공급망 관리

                    분석:""",
                    input_variables=["question"]
                ),
                "description": "글로벌 전략과 국제 시장 중심 분석"
            }
        }

    def analyze_from_multiple_perspectives(self, question: str, num_runs: int = 3) -> Dict[str, Any]:
        """여러 관점에서 반복 분석"""
        print(f"=== Self-Consistency 분석: {question} ===")

        all_results = {}

        for perspective_name, perspective_config in self.perspectives.items():
            print(f"\n--- {perspective_name} 분석 중... ---")

            # 각 관점에서 여러 번 분석
            perspective_results = []
            chain = perspective_config["prompt"] | self.llm

            for run in range(num_runs):
                try:
                    result = chain.invoke({"question": question}).content
                    perspective_results.append(result)
                    print(f"  실행 {run + 1}: 완료")
                except Exception as e:
                    print(f"  실행 {run + 1}: 오류 - {e}")

            all_results[perspective_name] = {
                "results": perspective_results,
                "description": perspective_config["description"],
                "consistency_score": self.calculate_consistency(perspective_results)
            }

        return all_results

    def calculate_consistency(self, results: List[str]) -> float:
        """결과 간 일관성 점수 계산 (간단한 버전)"""
        if len(results) < 2:
            return 1.0

        # 키워드 기반 일관성 측정 (실제로는 더 정교한 방법 사용)
        keywords = ["매수", "매도", "보유", "성장", "위험", "기회", "긍정", "부정"]
        keyword_counts = []

        for result in results:
            count = sum(1 for keyword in keywords if keyword in result)
            keyword_counts.append(count)

        # 표준편차가 작을수록 일관성 높음
        if len(keyword_counts) > 1:
            std_dev = statistics.stdev(keyword_counts)
            max_possible_std = max(keyword_counts) - min(keyword_counts)
            if max_possible_std > 0:
                consistency = 1 - (std_dev / max_possible_std)
                return max(0, min(1, consistency))

        return 1.0

    def synthesize_results(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """여러 관점의 결과를 종합"""
        print("\n=== 결과 종합 분석 ===")

        synthesis = {
            "perspective_analyses": all_results,
            "overall_consistency": 0.0,
            "consensus_opinion": "",
            "key_insights": []
        }

        # 전체 일관성 점수 계산
        consistency_scores = [
            result["consistency_score"]
            for result in all_results.values()
        ]
        synthesis["overall_consistency"] = statistics.mean(consistency_scores)

        # 각 관점별 주요 인사이트 추출
        for perspective, data in all_results.items():
            print(f"\n{perspective} ({data['description']}):")
            print(f"  일관성 점수: {data['consistency_score']:.2f}")

            # 가장 자주 나타나는 키워드나 의견 추출
            common_keywords = self.extract_common_keywords(data["results"])
            synthesis["key_insights"].append({
                "perspective": perspective,
                "consistency": data["consistency_score"],
                "common_themes": common_keywords
            })

        return synthesis

    def extract_common_keywords(self, results: List[str]) -> List[str]:
        """결과에서 공통 키워드 추출"""
        keywords = ["성장", "위험", "기회", "매수", "매도", "보유", "긍정", "부정", "안정", "혁신"]
        keyword_frequency = {}

        for keyword in keywords:
            count = sum(1 for result in results if keyword in result)
            if count > 0:
                keyword_frequency[keyword] = count

        # 빈도순으로 정렬하여 상위 키워드 반환
        sorted_keywords = sorted(keyword_frequency.items(), key=lambda x: x[1], reverse=True)
        return [keyword for keyword, count in sorted_keywords[:5]]

    def generate_final_recommendation(self, synthesis: Dict[str, Any]) -> str:
        """최종 투자 권고 생성"""
        print(f"\n=== 최종 투자 권고 ===")
        print(f"전체 일관성 점수: {synthesis['overall_consistency']:.2f}")

        # 일관성 점수에 따른 신뢰도 평가
        if synthesis["overall_consistency"] >= 0.8:
            confidence = "높음"
        elif synthesis["overall_consistency"] >= 0.6:
            confidence = "보통"
        else:
            confidence = "낮음"

        print(f"분석 신뢰도: {confidence}")

        # 각 관점별 주요 인사이트 요약
        print("\n관점별 주요 인사이트:")
        for insight in synthesis["key_insights"]:
            print(f"  • {insight['perspective']}: {', '.join(insight['common_themes'])}")

        return f"신뢰도 {confidence}의 종합 분석 완료"

def demonstrate_self_consistency():
    """Self-Consistency 예제 실행"""
    analyzer = SelfConsistencyAnalyzer()

    # Apple 관련 복잡한 분석 질문
    complex_questions = [
        "Apple의 Vision Pro가 새로운 성장 동력이 될 수 있을까요?",
        "Apple의 서비스 사업 확장이 주가에 미치는 영향은?",
        "Apple의 중국 시장 의존도 리스크는 어느 정도인가요?"
    ]

    print("🧠 Self-Consistency Apple 분석 시스템")
    print("="*60)

    for i, question in enumerate(complex_questions, 1):
        print(f"\n{'='*80}")
        print(f"질문 {i}: {question}")
        print('='*80)

        # 여러 관점에서 반복 분석
        all_results = analyzer.analyze_from_multiple_perspectives(question, num_runs=3)

        # 결과 종합
        synthesis = analyzer.synthesize_results(all_results)

        # 최종 권고 생성
        final_recommendation = analyzer.generate_final_recommendation(synthesis)

        print(f"\n{final_recommendation}")

    # Self-Consistency의 장점 설명
    print(f"\n{'='*80}")
    print("Self-Consistency의 장점")
    print('='*80)

    benefits = [
        "1. 편향성 감소: 다양한 관점으로 편향 방지",
        "2. 신뢰도 향상: 일관성 검증으로 신뢰성 증대",
        "3. 포괄적 분석: 여러 측면을 종합적으로 고려",
        "4. 불확실성 정량화: 일관성 점수로 불확실성 측정",
        "5. 견고한 결론: 다양한 관점에서 검증된 분석"
    ]

    for benefit in benefits:
        print(f"✓ {benefit}")

if __name__ == "__main__":
    demonstrate_self_consistency()

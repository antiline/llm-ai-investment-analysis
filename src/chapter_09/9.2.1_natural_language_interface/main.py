#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from typing import Dict, List, Any
import re

class NaturalLanguageInterface:
    """자연어 대화 인터페이스"""

    def __init__(self):
        load_dotenv()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        self.setup_conversation_system()

    def setup_conversation_system(self):
        """대화 시스템 설정"""
        self.conversation_prompt = PromptTemplate(
            template="""
            당신은 친근하고 전문적인 투자 상담사입니다. 사용자와 자연스럽게 대화하면서 투자 상담을 제공합니다.

            대화 스타일:
            - 친근하고 자연스러운 톤 사용
            - 이모지와 시각적 요소 활용
            - 핵심 정보를 간결하게 제시
            - 추가 질문을 유도하는 개방형 구조
            - 전문적이지만 이해하기 쉬운 설명

            사용자 질문: {user_input}

            답변:""",
            input_variables=["user_input"]
        )

        self.analysis_prompt = PromptTemplate(
            template="""
            Apple Inc.에 대한 투자 분석을 친근하고 이해하기 쉽게 설명해주세요.

            다음 형식으로 답변해주세요:

            🍎 Apple 주식 분석

            💡 핵심 포인트:
            • [주요 정보 1]
            • [주요 정보 2]
            • [주요 정보 3]

            📈 투자 의견:
            [간단한 투자 의견]

            💬 추가 질문이 있으시면 언제든 말씀해주세요!

            분석:""",
            input_variables=[]
        )

    def process_natural_language_input(self, user_input: str) -> str:
        """자연어 입력 처리"""
        print(f"사용자: {user_input}")

        # 입력 의도 파악
        intent = self.analyze_intent(user_input)

        # 의도에 따른 응답 생성
        if intent == "stock_analysis":
            response = self.generate_stock_analysis(user_input)
        elif intent == "general_question":
            response = self.generate_general_response(user_input)
        else:
            response = self.generate_fallback_response(user_input)

        return response

    def analyze_intent(self, user_input: str) -> str:
        """사용자 의도 분석"""
        input_lower = user_input.lower()

        # 주식 분석 관련 키워드
        stock_keywords = [
            "주식", "투자", "분석", "어떻게", "생각", "전망", "가격",
            "stock", "investment", "analysis", "think", "outlook", "price"
        ]

        # Apple 관련 키워드
        apple_keywords = [
            "apple", "애플", "aapl", "아이폰", "iphone", "티머", "tim"
        ]

        # 의도 판단
        if any(keyword in input_lower for keyword in stock_keywords):
            if any(keyword in input_lower for keyword in apple_keywords):
                return "stock_analysis"
            else:
                return "stock_analysis"  # Apple이 언급되지 않아도 주식 분석으로 간주

        return "general_question"

    def generate_stock_analysis(self, user_input: str) -> str:
        """주식 분석 응답 생성"""
        chain = self.analysis_prompt | self.llm

        try:
            response = chain.invoke({}).content
            return response
        except Exception as e:
            return f"분석 중 오류가 발생했습니다: {e}"

    def generate_general_response(self, user_input: str) -> str:
        """일반 질문 응답 생성"""
        general_prompt = PromptTemplate(
            template="""
            사용자의 일반적인 질문에 친근하게 답변해주세요.

            질문: {user_input}

            답변:""",
            input_variables=["user_input"]
        )

        chain = general_prompt | self.llm

        try:
            response = chain.invoke({"user_input": user_input}).content
            return response
        except Exception as e:
            return f"답변 중 오류가 발생했습니다: {e}"

    def generate_fallback_response(self, user_input: str) -> str:
        """기본 응답 생성"""
        return """
        안녕하세요! 👋 투자 상담사입니다.

        Apple 주식에 대해 궁금한 점이 있으시면 언제든 말씀해주세요!
        예를 들어:
        • "Apple 주식 어떻게 생각해?"
        • "Apple 투자 가치가 어때?"
        • "Apple 전망이 어떨까?"

        어떤 것이든 편하게 물어보세요! 😊
        """

class ConversationalInterface:
    """대화형 인터페이스"""

    def __init__(self):
        self.nli = NaturalLanguageInterface()
        self.conversation_history = []

    def start_conversation(self):
        """대화 시작"""
        print("🤖 안녕하세요! Apple 투자 상담사입니다.")
        print("궁금한 점이 있으시면 언제든 말씀해주세요!")
        print("(종료하려면 'quit' 또는 '종료'를 입력하세요)")
        print("-" * 50)

        while True:
            try:
                user_input = input("\n사용자: ").strip()

                if user_input.lower() in ['quit', 'exit', '종료', '끝']:
                    print("\n🤖 감사합니다! 또 궁금한 점이 있으시면 언제든 찾아주세요! 👋")
                    break

                if not user_input:
                    continue

                # 대화 기록 저장
                self.conversation_history.append({"user": user_input, "timestamp": "now"})

                # 응답 생성
                print("\nAI 상담사: ", end="", flush=True)
                response = self.nli.process_natural_language_input(user_input)
                print(response)

                # 응답 기록 저장
                self.conversation_history.append({"ai": response, "timestamp": "now"})

            except KeyboardInterrupt:
                print("\n\n🤖 대화를 종료합니다. 감사합니다! 👋")
                break
            except Exception as e:
                print(f"\n오류가 발생했습니다: {e}")

    def demonstrate_natural_language_examples(self):
        """자연어 인터페이스 예시 시연"""
        print("=== 자연어 대화 인터페이스 예시 ===")

        example_conversations = [
            "Apple 주식 어떻게 생각해?",
            "Apple 투자 가치가 어때?",
            "Apple 전망이 어떨까?",
            "Apple 주식 사도 될까?",
            "Apple 실적은 어때?",
            "안녕하세요",
            "도움말"
        ]

        for example in example_conversations:
            print(f"\n사용자: {example}")
            response = self.nli.process_natural_language_input(example)
            print(f"AI: {response}")
            print("-" * 50)

def demonstrate_natural_language_interface():
    """자연어 인터페이스 예제 실행"""
    interface = ConversationalInterface()

    print("🎯 자연어 대화 인터페이스 시스템")
    print("="*60)

    # 예시 시연
    interface.demonstrate_natural_language_examples()

    # 대화형 인터페이스 시작 (선택사항)
    print("\n대화형 인터페이스를 시작하시겠습니까? (y/n): ", end="")
    try:
        choice = input().strip().lower()
        if choice in ['y', 'yes', '예']:
            interface.start_conversation()
    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")

    # 자연어 인터페이스의 장점 설명
    print(f"\n{'='*60}")
    print("자연어 대화 인터페이스의 장점")
    print('='*60)

    benefits = [
        "1. 사용자 진입 장벽 제거: 복잡한 명령어 불필요",
        "2. 직관적 상호작용: 자연스러운 대화 방식",
        "3. 친근한 사용자 경험: 이모지와 시각적 요소 활용",
        "4. 유연한 질문 처리: 다양한 표현 방식 지원",
        "5. 개방형 구조: 추가 질문을 통한 심화 학습"
    ]

    for benefit in benefits:
        print(f"✓ {benefit}")

    # 기존 시스템과의 비교
    print(f"\n{'='*60}")
    print("기존 시스템 vs 자연어 인터페이스 비교")
    print('='*60)

    comparison = {
        "입력 방식": {
            "기존": "복잡한 JSON 형식",
            "자연어": "자연스러운 대화"
        },
        "학습 곡선": {
            "기존": "높음 (기술적 지식 필요)",
            "자연어": "낮음 (직관적)"
        },
        "사용자 경험": {
            "기존": "기술적이고 딱딱함",
            "자연어": "친근하고 자연스러움"
        },
        "접근성": {
            "기존": "제한적 (전문가 중심)",
            "자연어": "보편적 (모든 사용자)"
        }
    }

    for aspect, methods in comparison.items():
        print(f"\n{aspect}:")
        print(f"  기존: {methods['기존']}")
        print(f"  자연어: {methods['자연어']}")

if __name__ == "__main__":
    demonstrate_natural_language_interface()

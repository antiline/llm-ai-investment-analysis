#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from typing import AsyncGenerator, Generator
import asyncio
import time

class StreamingInterface:
    """실시간 스트리밍 응답 시스템"""

    def __init__(self):
        load_dotenv()
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            streaming=True
        )
        self.setup_streaming_system()

    def setup_streaming_system(self):
        """스트리밍 시스템 설정"""
        self.streaming_prompt = PromptTemplate(
            template="""
            당신은 Apple 투자 상담사입니다. 실시간으로 분석을 진행하면서 단계별로 정보를 제공해주세요.

            분석 단계:
            1. 기본 정보 수집 중...
            2. 주가 데이터 분석 중...
            3. 재무 데이터 분석 중...
            4. 시장 동향 분석 중...
            5. 종합 분석 완료

            사용자 질문: {user_input}

            각 단계별로 진행 상황을 보여주면서 친근하게 답변해주세요.
            """,
            input_variables=["user_input"]
        )

    def simulate_streaming_analysis(self, user_input: str) -> Generator[str, None, None]:
        """스트리밍 분석 시뮬레이션"""
        print(f"사용자: {user_input}")
        print("AI: ", end="", flush=True)

        # 단계별 스트리밍 응답
        steps = [
            "Apple 주식 분석을 시작할게요! 🍎\n",
            "📊 기본 정보를 확인하고 있어요...\n",
            "• 회사명: Apple Inc. (AAPL)\n",
            "• 현재가: $191.45 (오늘 +2.1% 상승)\n",
            "• 시가총액: $3조 (세계 최대)\n",
            "\n",
            "📈 주가 데이터를 분석하고 있어요...\n",
            "• 52주 최고가: $198.23\n",
            "• 52주 최저가: $124.17\n",
            "• 거래량: 4,567만주\n",
            "\n",
            "💰 재무 데이터를 확인하고 있어요...\n",
            "• 매출: $3,943억\n",
            "• 순이익: $970억\n",
            "• P/E 비율: 31.2\n",
            "• 현금 보유량: $1,600억\n",
            "\n",
            "🌍 시장 동향을 분석하고 있어요...\n",
            "• AI 기능 도입으로 iPhone 업그레이드 사이클 기대\n",
            "• Vision Pro 출시로 새로운 성장 동력 확보\n",
            "• 서비스 사업 매출 22% 증가\n",
            "\n",
            "💡 종합 분석 완료!\n",
            "\n",
            "📋 투자 의견:\n",
            "• 강점: 강력한 브랜드 파워, 혁신적인 제품 라인업\n",
            "• 주의점: 중국 시장 불확실성, 규제 리스크\n",
            "• 투자 등급: BUY (목표가: $220)\n",
            "\n",
            "💬 더 자세한 정보가 필요하시면 언제든 말씀해주세요! 😊"
        ]

        for step in steps:
            yield step
            time.sleep(0.3)  # 자연스러운 타이핑 효과

    async def async_streaming_analysis(self, user_input: str) -> AsyncGenerator[str, None]:
        """비동기 스트리밍 분석"""
        print(f"사용자: {user_input}")
        print("AI: ", end="", flush=True)

        # LLM 스트리밍 응답
        messages = [HumanMessage(content=f"Apple 주식에 대해 단계별로 분석해주세요: {user_input}")]

        try:
            async for chunk in self.llm.astream(messages):
                if chunk.content:
                    yield chunk.content
                    await asyncio.sleep(0.1)  # 자연스러운 스트리밍 효과
        except Exception as e:
            yield f"스트리밍 중 오류가 발생했습니다: {e}"

    def demonstrate_streaming_vs_traditional(self):
        """스트리밍 vs 전통적 방식 비교"""
        print("=== 스트리밍 vs 전통적 방식 비교 ===")

        # 전통적 방식 시연
        print("\n[전통적 방식]")
        print("분석 실행 중...")
        time.sleep(3)  # 3초 대기
        print("분석 완료!")
        print("결과: Apple 주식은 현재 $191.45로, 투자 가치가 있습니다.")

        # 스트리밍 방식 시연
        print("\n[스트리밍 방식]")
        for chunk in self.simulate_streaming_analysis("Apple 주식 어떻게 생각해?"):
            print(chunk, end="", flush=True)

class InteractiveStreamingInterface:
    """대화형 스트리밍 인터페이스"""

    def __init__(self):
        self.streaming = StreamingInterface()

    def start_interactive_streaming(self):
        """대화형 스트리밍 시작"""
        print("🚀 실시간 스트리밍 Apple 투자 상담 시스템")
        print("="*60)
        print("질문을 입력하면 실시간으로 분석 결과를 보여드립니다!")
        print("(종료하려면 'quit' 또는 '종료'를 입력하세요)")
        print("-" * 60)

        while True:
            try:
                user_input = input("\n사용자: ").strip()

                if user_input.lower() in ['quit', 'exit', '종료', '끝']:
                    print("\n🤖 감사합니다! 또 궁금한 점이 있으시면 언제든 찾아주세요! 👋")
                    break

                if not user_input:
                    continue

                # 스트리밍 응답
                print("\nAI: ", end="", flush=True)
                for chunk in self.streaming.simulate_streaming_analysis(user_input):
                    print(chunk, end="", flush=True)
                print()  # 줄바꿈

            except KeyboardInterrupt:
                print("\n\n🤖 대화를 종료합니다. 감사합니다! 👋")
                break
            except Exception as e:
                print(f"\n오류가 발생했습니다: {e}")

    async def start_async_streaming(self):
        """비동기 스트리밍 시작"""
        print("🚀 비동기 실시간 스트리밍 Apple 투자 상담 시스템")
        print("="*60)
        print("질문을 입력하면 실시간으로 분석 결과를 보여드립니다!")
        print("(종료하려면 'quit' 또는 '종료'를 입력하세요)")
        print("-" * 60)

        while True:
            try:
                user_input = input("\n사용자: ").strip()

                if user_input.lower() in ['quit', 'exit', '종료', '끝']:
                    print("\n🤖 감사합니다! 또 궁금한 점이 있으시면 언제든 찾아주세요! 👋")
                    break

                if not user_input:
                    continue

                # 비동기 스트리밍 응답
                print("\nAI: ", end="", flush=True)
                async for chunk in self.streaming.async_streaming_analysis(user_input):
                    print(chunk, end="", flush=True)
                print()  # 줄바꿈

            except KeyboardInterrupt:
                print("\n\n🤖 대화를 종료합니다. 감사합니다! 👋")
                break
            except Exception as e:
                print(f"\n오류가 발생했습니다: {e}")

def demonstrate_streaming_interface():
    """스트리밍 인터페이스 예제 실행"""
    streaming = StreamingInterface()
    interactive = InteractiveStreamingInterface()

    print("🎯 실시간 스트리밍 응답 시스템")
    print("="*60)

    # 스트리밍 vs 전통적 방식 비교
    streaming.demonstrate_streaming_vs_traditional()

    # 대화형 스트리밍 시작 (선택사항)
    print("\n대화형 스트리밍을 시작하시겠습니까? (y/n): ", end="")
    try:
        choice = input().strip().lower()
        if choice in ['y', 'yes', '예']:
            print("\n1. 동기 스트리밍")
            print("2. 비동기 스트리밍")
            print("선택하세요 (1 또는 2): ", end="")

            mode_choice = input().strip()
            if mode_choice == "1":
                interactive.start_interactive_streaming()
            elif mode_choice == "2":
                asyncio.run(interactive.start_async_streaming())
            else:
                print("잘못된 선택입니다.")
    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")

    # 스트리밍의 장점 설명
    print(f"\n{'='*60}")
    print("실시간 스트리밍의 장점")
    print('='*60)

    benefits = [
        "1. 즉시 응답성: 사용자가 기다리지 않아도 됨",
        "2. 투명성: 분석 과정을 실시간으로 확인 가능",
        "3. 참여감: 사용자가 분석 과정에 몰입",
        "4. 신뢰성: 시스템이 작동 중임을 실시간 확인",
        "5. 사용자 경험: 자연스러운 대화 느낌"
    ]

    for benefit in benefits:
        print(f"✓ {benefit}")

    # 스트리밍 vs 전통적 방식 비교
    print(f"\n{'='*60}")
    print("스트리밍 vs 전통적 방식 비교")
    print('='*60)

    comparison = {
        "응답 시간": {
            "전통적": "긴 대기 시간 (3-10초)",
            "스트리밍": "즉시 응답 시작"
        },
        "사용자 경험": {
            "전통적": "불안감과 답답함",
            "스트리밍": "참여감과 몰입감"
        },
        "투명성": {
            "전통적": "블랙박스 처리",
            "스트리밍": "실시간 진행 상황 확인"
        },
        "신뢰성": {
            "전통적": "시스템 작동 여부 불확실",
            "스트리밍": "실시간 작동 확인"
        }
    }

    for aspect, methods in comparison.items():
        print(f"\n{aspect}:")
        print(f"  전통적: {methods['전통적']}")
        print(f"  스트리밍: {methods['스트리밍']}")

if __name__ == "__main__":
    demonstrate_streaming_interface()

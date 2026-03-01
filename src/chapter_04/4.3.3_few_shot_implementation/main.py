#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

def demonstrate_few_shot_prompting():
    """Few-shot 프롬프팅의 효과를 보여주는 예제"""
    load_dotenv()

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

    # 예시 데이터 준비
    examples = [
        ("지구의 대기 중 가장 많은 비율을 차지하는 기체는 무엇인가요?", "지구 대기의 약 78%를 차지하는 질소입니다."),
        ("광합성에 필요한 주요 요소들은 무엇인가요?", "광합성에 필요한 주요 요소는 빛, 이산화탄소, 물입니다."),
        ("피타고라스 정리를 설명해주세요.", "피타고라스 정리는 직각삼각형에서 빗변의 제곱이 다른 두 변의 제곱의 합과 같다는 것입니다.")
    ]

    # 새로운 질문
    new_question = "화성의 표면이 붉은 이유는 무엇인가요?"

    # Few-shot 프롬프트 직접 구성
    prompt = "다음 예시들을 참고하여 비슷한 형식으로 답변해주세요.\n\n"

    # 예시들 추가
    for i, (question, answer) in enumerate(examples, 1):
        prompt += f"예시 {i}:\n"
        prompt += f"질문: {question}\n"
        prompt += f"답변: {answer}\n\n"

    # 새로운 질문 추가
    prompt += "새로운 문제:\n"
    prompt += f"질문: {new_question}\n"
    prompt += "답변:"

    print("=== Few-shot 프롬프트 ===")
    print(prompt)

    print("\n=== Few-shot 응답 ===")
    response = llm.invoke([HumanMessage(content=prompt)])
    print(response.content)

    # Zero-shot과 비교
    print("\n=== Zero-shot 응답 (비교용) ===")
    zero_shot_prompt = f"질문: {new_question}\n답변:"
    zero_shot_response = llm.invoke([HumanMessage(content=zero_shot_prompt)])
    print(zero_shot_response.content)

if __name__ == "__main__":
    demonstrate_few_shot_prompting()

"""
Chapter 6.4.1: 텍스트 분할 예제
다양한 텍스트 분할 전략과 파라미터 튜닝 예제
"""

from langchain_text_splitters import (
    CharacterTextSplitter,
    TokenTextSplitter,
    RecursiveCharacterTextSplitter
)
from langchain_core.documents import Document

def demonstrate_character_text_splitter():
    """CharacterTextSplitter 예제"""
    print("=== CharacterTextSplitter 예제 ===")

    # 샘플 텍스트
    sample_text = """
    Apple Inc.는 2024년 1분기에 강력한 실적을 기록했습니다. iPhone 판매량이 전년 대비 15% 증가했으며, 특히 iPhone 15 Pro 시리즈의 성공이 두드러집니다.
    Apple의 서비스 부문도 12% 성장하여 Apple Music, iCloud, App Store의 강세를 보였습니다. 구독자 수가 전년 대비 20% 증가했습니다.
    Apple의 재무 건전성은 여전히 우수합니다. 현금 보유량이 1,500억 달러를 유지하고 있으며, 부채 비율도 업계 평균 대비 낮은 수준입니다.
    """

    # CharacterTextSplitter 설정
    text_splitter = CharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separator="\n"
    )

    # 텍스트 분할
    chunks = text_splitter.split_text(sample_text)

    print(f"원본 텍스트 길이: {len(sample_text)}자")
    print(f"생성된 청크 수: {len(chunks)}")
    print("\n=== 분할된 청크들 ===")

    for i, chunk in enumerate(chunks, 1):
        print(f"\n청크 {i} ({len(chunk)}자):")
        print(f"'{chunk}'")
        print("-" * 50)
    print()

def demonstrate_token_text_splitter():
    """TokenTextSplitter 예제"""
    print("=== TokenTextSplitter 예제 ===")

    # 샘플 텍스트
    sample_text = """
    Apple의 2024년 1분기 실적이 발표되었습니다. 매출은 119.6조원으로 전년 대비 4% 증가했습니다.
    iPhone 판매량이 전년 대비 15% 증가했으며, 특히 iPhone 15 Pro 시리즈의 성공이 두드러집니다.
    Apple의 서비스 부문도 12% 성장하여 Apple Music, iCloud, App Store의 강세를 보였습니다.
    """

    # TokenTextSplitter 설정
    text_splitter = TokenTextSplitter(
        chunk_size=50,  # 50 토큰
        chunk_overlap=10,  # 10 토큰 겹침
        encoding_name="cl100k_base"  # GPT-4 토크나이저
    )

    # 텍스트 분할
    chunks = text_splitter.split_text(sample_text)

    print(f"원본 텍스트 길이: {len(sample_text)}자")
    print(f"생성된 청크 수: {len(chunks)}")
    print("\n=== 분할된 청크들 ===")

    for i, chunk in enumerate(chunks, 1):
        print(f"\n청크 {i} ({len(chunk)}자):")
        print(f"'{chunk}'")
        print("-" * 50)
    print()

def demonstrate_recursive_character_text_splitter():
    """RecursiveCharacterTextSplitter 예제"""
    print("=== RecursiveCharacterTextSplitter 예제 ===")

    # Apple 재무 보고서 샘플 텍스트
    apple_report = """
    Apple Inc.는 2024년 1분기에 강력한 실적을 기록했습니다. iPhone 판매량이 전년 대비 15% 증가했으며, 특히 iPhone 15 Pro 시리즈의 성공이 두드러집니다.

    Apple의 서비스 부문도 12% 성장하여 Apple Music, iCloud, App Store의 강세를 보였습니다. 구독자 수가 전년 대비 20% 증가했습니다.

    Apple의 재무 건전성은 여전히 우수합니다. 현금 보유량이 1,500억 달러를 유지하고 있으며, 부채 비율도 업계 평균 대비 낮은 수준입니다.

    Apple의 중국 시장에서의 성과가 저조하며, 규제 강화로 인해 8% 감소했습니다. 이는 Apple의 주요 시장 중 하나인 중국에서의 도전을 보여줍니다.

    Apple의 연구개발 투자는 전년 대비 25% 증가하여 300억 달러에 달했습니다. 이는 AI 기술과 자율주행 기술 개발에 집중한 결과입니다.
    """

    # RecursiveCharacterTextSplitter 설정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )

    # 텍스트 분할
    chunks = text_splitter.split_text(apple_report)

    print(f"원본 텍스트 길이: {len(apple_report)}자")
    print(f"생성된 청크 수: {len(chunks)}")
    print("\n=== 분할된 청크들 ===")

    for i, chunk in enumerate(chunks, 1):
        print(f"\n청크 {i} ({len(chunk)}자):")
        print(f"'{chunk}'")
        print("-" * 50)

    # Document 객체로 변환하여 메타데이터 추가
    documents = []
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                "source": "Apple_Financial_Report_2024",
                "chunk_id": i,
                "chunk_size": len(chunk),
                "type": "financial_analysis"
            }
        )
        documents.append(doc)

    # 청크 겹침 확인
    print("\n=== 청크 겹침 확인 ===")
    for i in range(len(chunks) - 1):
        chunk1_end = chunks[i][-50:]  # 첫 번째 청크의 마지막 50자
        chunk2_start = chunks[i+1][:50]  # 두 번째 청크의 처음 50자

        # 겹치는 부분 찾기
        overlap = ""
        for j in range(min(len(chunk1_end), len(chunk2_start))):
            if chunk1_end[-j-1:] == chunk2_start[:j+1]:
                overlap = chunk1_end[-j-1:]

        if overlap:
            print(f"청크 {i+1}과 {i+2} 간 겹침: '{overlap}'")
        else:
            print(f"청크 {i+1}과 {i+2} 간 겹침 없음")
    print()

def demonstrate_chunk_parameters():
    """청킹 파라미터 튜닝 예제"""
    print("=== 청킹 파라미터 튜닝 예제 ===")

    # 샘플 텍스트
    sample_text = """
    Apple의 iPhone은 2023년에 2,050억 달러의 매출을 기록했습니다. 이는 전체 Apple 매출의 52%를 차지하는 핵심 제품입니다.
    iPhone 15 시리즈 출시로 인해 4분기 매출이 697억 달러를 기록했습니다. 주요 시장에서 성장세를 보였으며, 특히 미국에서 40% 성장했습니다.
    Apple의 서비스 사업은 2023년 867억 달러의 매출을 기록했습니다. 전체 매출의 22%를 차지하며, 가장 빠른 성장세를 보이는 사업입니다.
    App Store, Apple Music, iCloud, Apple TV+ 등 다양한 서비스를 제공합니다. 하드웨어와 소프트웨어의 통합된 생태계를 통해 높은 고객 충성도를 확보하고 있습니다.
    """

    # 다양한 파라미터로 테스트
    parameters = [
        {"chunk_size": 200, "chunk_overlap": 20, "name": "작은 청크 (200자, 20자 겹침)"},
        {"chunk_size": 400, "chunk_overlap": 40, "name": "중간 청크 (400자, 40자 겹침)"},
        {"chunk_size": 600, "chunk_overlap": 60, "name": "큰 청크 (600자, 60자 겹침)"}
    ]

    for params in parameters:
        print(f"\n--- {params['name']} ---")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=params["chunk_size"],
            chunk_overlap=params["chunk_overlap"],
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = text_splitter.split_text(sample_text)

        print(f"청크 수: {len(chunks)}")
        print(f"평균 청크 크기: {sum(len(chunk) for chunk in chunks) / len(chunks):.1f}자")

        for i, chunk in enumerate(chunks, 1):
            print(f"  청크 {i}: {len(chunk)}자")
    print()

def main():
    """메인 실행 함수"""
    print("Chapter 6.4.1: 텍스트 분할 예제")
    print("=" * 50)

    # 1. CharacterTextSplitter 예제
    demonstrate_character_text_splitter()

    # 2. TokenTextSplitter 예제
    demonstrate_token_text_splitter()

    # 3. RecursiveCharacterTextSplitter 예제
    demonstrate_recursive_character_text_splitter()

    # 4. 청킹 파라미터 튜닝 예제
    demonstrate_chunk_parameters()

if __name__ == "__main__":
    main()

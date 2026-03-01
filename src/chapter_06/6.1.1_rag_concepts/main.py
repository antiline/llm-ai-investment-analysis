"""
Chapter 6.1.1: RAG 개념과 FAISS 설정 예제
"""

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

def basic_faiss_example():
    """기본 FAISS 설정 예제"""
    print("=== 기본 FAISS 설정 예제 ===")

    # 임베딩 모델 초기화
    embeddings = OpenAIEmbeddings()

    # 샘플 텍스트
    texts = [
        "Apple의 iPhone 판매량이 전년 대비 15% 증가했습니다.",
        "Apple의 서비스 부문 매출이 12% 성장했습니다.",
        "Apple의 현금 보유량이 1,500억 달러를 유지하고 있습니다.",
        "Apple의 중국 시장에서의 성과가 저조합니다."
    ]

    # FAISS 벡터 스토어 생성
    vector_store = FAISS.from_texts(texts, embeddings)

    # 검색 수행
    query = "Apple의 iPhone 성과는?"
    results = vector_store.similarity_search_with_score(query, k=2)

    print(f"검색: {query}")
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. {doc.page_content} (점수: {score:.3f})")
    print()

def enhanced_faiss_example():
    """RecursiveCharacterTextSplitter를 사용한 향상된 FAISS 예제"""
    print("=== 향상된 FAISS 예제 (RecursiveCharacterTextSplitter) ===")

    # 임베딩 모델 초기화
    embeddings = OpenAIEmbeddings()

    # 텍스트 분할기 설정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    # 긴 텍스트 샘플
    long_text = """
    Apple Inc.는 2024년 1분기에 강력한 실적을 기록했습니다. iPhone 판매량이 전년 대비 15% 증가했으며, 특히 iPhone 15 Pro 시리즈의 성공이 두드러집니다.
    Apple의 서비스 부문도 12% 성장하여 Apple Music, iCloud, App Store의 강세를 보였습니다. 구독자 수가 전년 대비 20% 증가했습니다.
    Apple의 재무 건전성은 여전히 우수합니다. 현금 보유량이 1,500억 달러를 유지하고 있으며, 부채 비율도 업계 평균 대비 낮은 수준입니다.
    Apple의 중국 시장에서의 성과가 저조하며, 규제 강화로 인해 8% 감소했습니다. 이는 Apple의 주요 시장 중 하나인 중국에서의 도전을 보여줍니다.
    """

    # 텍스트 분할
    chunks = text_splitter.split_text(long_text)
    documents = [Document(page_content=chunk, metadata={"source": "Apple_Analysis"}) for chunk in chunks]

    # FAISS 벡터 스토어 생성
    vector_store = FAISS.from_documents(documents, embeddings)

    # 검색 수행
    query = "Apple의 iPhone 성과는?"
    results = vector_store.similarity_search_with_score(query, k=3)

    print(f"검색: {query}")
    print(f"총 청크 수: {len(chunks)}")
    print("\n=== 검색 결과 ===")
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. 점수: {score:.3f}")
        print(f"   내용: {doc.page_content}")
        print()

def main():
    """메인 실행 함수"""
    print("Chapter 6.1.1: RAG 개념과 FAISS 설정 예제")
    print("=" * 50)

    # 기본 FAISS 예제
    basic_faiss_example()

    # 향상된 FAISS 예제
    enhanced_faiss_example()

if __name__ == "__main__":
    main()

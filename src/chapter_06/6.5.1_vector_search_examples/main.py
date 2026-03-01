"""
Chapter 6.5.1: 벡터 검색 예제
벡터 검색의 기본 원리와 다양한 검색 파라미터 튜닝 예제
"""

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

def demonstrate_basic_vector_search():
    """기본 벡터 검색 예제"""
    print("=== 기본 벡터 검색 예제 ===")

    # 임베딩 모델 초기화
    embeddings = OpenAIEmbeddings()

    # 샘플 텍스트
    texts = [
        "Apple의 iPhone 판매량이 전년 대비 15% 증가했습니다.",
        "Apple의 서비스 부문 매출이 12% 성장했습니다.",
        "Apple의 현금 보유량이 1,500억 달러를 유지하고 있습니다.",
        "Apple의 중국 시장에서의 성과가 저조합니다.",
        "Apple의 P/E 비율은 28.5배로 업계 평균 대비 높습니다.",
        "Apple의 2024년 1분기 매출은 119.6조원으로 증가했습니다."
    ]

    # FAISS 벡터 스토어 생성
    vector_store = FAISS.from_texts(texts, embeddings)

    # 검색 수행
    query = "Apple의 iPhone 성과는?"
    results = vector_store.similarity_search_with_score(query, k=3)

    print(f"검색: {query}")
    print("=" * 50)
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. {doc.page_content} (점수: {score:.3f})")
    print()

def demonstrate_search_parameters():
    """검색 파라미터 튜닝 예제"""
    print("=== 검색 파라미터 튜닝 예제 ===")

    # 임베딩 모델 초기화
    embeddings = OpenAIEmbeddings()

    # 샘플 텍스트
    texts = [
        "Apple의 iPhone 판매량이 전년 대비 15% 증가했습니다.",
        "Apple의 서비스 부문 매출이 12% 성장했습니다.",
        "Apple의 P/E 비율은 28.5배로 업계 평균 대비 높습니다.",
        "Apple의 2024년 1분기 매출은 119.6조원으로 증가했습니다.",
        "Apple의 현금 보유량이 1,500억 달러를 유지하고 있습니다.",
        "Apple의 중국 시장에서의 성과가 저조합니다.",
        "Apple의 연구개발 투자는 전년 대비 25% 증가했습니다.",
        "Apple의 Vision Pro는 2024년 2월 출시되었습니다."
    ]

    # FAISS 벡터 스토어 생성
    vector_store = FAISS.from_texts(texts, embeddings)

    # 다양한 k 값으로 검색
    query = "Apple의 재무 성과"
    k_values = [2, 4, 6]

    for k in k_values:
        print(f"\n--- k={k} 검색 결과 ---")
        results = vector_store.similarity_search_with_score(query, k=k)

        for i, (doc, score) in enumerate(results, 1):
            print(f"{i}. {doc.page_content} (점수: {score:.3f})")
    print()

def demonstrate_distance_metrics():
    """거리 측정 방식 비교 예제"""
    print("=== 거리 측정 방식 비교 예제 ===")

    # 임베딩 모델 초기화
    embeddings = OpenAIEmbeddings()

    # 샘플 텍스트
    texts = [
        "Apple의 iPhone 판매량이 전년 대비 15% 증가했습니다.",
        "Apple의 서비스 부문 매출이 12% 성장했습니다.",
        "Apple의 P/E 비율은 28.5배로 업계 평균 대비 높습니다."
    ]

    # FAISS 벡터 스토어 생성
    vector_store = FAISS.from_texts(texts, embeddings)

    # 다양한 거리 측정 방식으로 검색
    query = "Apple의 매출 성과"

    print(f"검색: {query}")
    print("=" * 50)

    # 코사인 유사도 (기본값)
    results_cosine = vector_store.similarity_search_with_score(query, k=2)
    print("코사인 유사도 결과:")
    for i, (doc, score) in enumerate(results_cosine, 1):
        print(f"{i}. {doc.page_content} (점수: {score:.3f})")

    print()
    print("참고: FAISS는 기본적으로 코사인 유사도를 사용합니다.")
    print("다른 거리 측정 방식을 사용하려면 별도의 설정이 필요합니다.")
    print()

def demonstrate_mmr_search():
    """MMR (Maximal Marginal Relevance) 검색 예제"""
    print("=== MMR 검색 예제 ===")

    # 임베딩 모델 초기화
    embeddings = OpenAIEmbeddings()

    # 샘플 텍스트
    texts = [
        "Apple의 iPhone 판매량이 전년 대비 15% 증가했습니다.",
        "Apple의 서비스 부문 매출이 12% 성장했습니다.",
        "Apple의 P/E 비율은 28.5배로 업계 평균 대비 높습니다.",
        "Apple의 2024년 1분기 매출은 119.6조원으로 증가했습니다.",
        "Apple의 현금 보유량이 1,500억 달러를 유지하고 있습니다.",
        "Apple의 중국 시장에서의 성과가 저조합니다.",
        "Apple의 연구개발 투자는 전년 대비 25% 증가했습니다.",
        "Apple의 Vision Pro는 2024년 2월 출시되었습니다."
    ]

    # FAISS 벡터 스토어 생성
    vector_store = FAISS.from_texts(texts, embeddings)

    # 일반적인 유사도 검색
    query = "Apple의 성과"
    print(f"검색: {query}")
    print("=" * 50)

    print("일반 유사도 검색 결과:")
    results_similarity = vector_store.similarity_search_with_score(query, k=3)
    for i, (doc, score) in enumerate(results_similarity, 1):
        print(f"{i}. {doc.page_content} (점수: {score:.3f})")

    print("\nMMR 검색 결과 (다양성 고려):")
    # MMR 검색 (FAISS에서는 max_marginal_relevance_search 사용)
    results_mmr = vector_store.max_marginal_relevance_search(query, k=3, fetch_k=6)
    for i, doc in enumerate(results_mmr, 1):
        print(f"{i}. {doc.page_content}")

    print("\n참고: MMR은 관련성과 다양성을 동시에 고려하여 검색 결과의 다양성을 높입니다.")
    print()

def demonstrate_search_with_metadata():
    """메타데이터를 포함한 검색 예제"""
    print("=== 메타데이터를 포함한 검색 예제 ===")

    # 임베딩 모델 초기화
    embeddings = OpenAIEmbeddings()

    # 텍스트 분할기 설정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    # 샘플 문서
    documents = [
        {
            "content": "Apple의 iPhone 판매량이 전년 대비 15% 증가했습니다. 특히 iPhone 15 Pro 시리즈의 성공이 두드러집니다.",
            "source": "earnings_report",
            "type": "financial",
            "date": "2024-01-01"
        },
        {
            "content": "Apple의 서비스 부문 매출이 12% 성장했습니다. Apple Music, iCloud, App Store의 강세를 보였습니다.",
            "source": "financial_analysis",
            "type": "financial",
            "date": "2024-01-15"
        },
        {
            "content": "Apple의 P/E 비율은 28.5배로 업계 평균 대비 높습니다. 투자자들의 높은 기대를 반영합니다.",
            "source": "market_analysis",
            "type": "valuation",
            "date": "2024-01-20"
        }
    ]

    # Document 객체로 변환
    all_documents = []
    for doc in documents:
        chunks = text_splitter.split_text(doc["content"])
        for i, chunk in enumerate(chunks):
            document = Document(
                page_content=chunk,
                metadata={
                    "source": doc["source"],
                    "type": doc["type"],
                    "date": doc["date"],
                    "chunk_id": i
                }
            )
            all_documents.append(document)

    # FAISS 벡터 스토어 생성
    vector_store = FAISS.from_documents(all_documents, embeddings)

    # 검색 수행
    query = "Apple의 재무 성과"
    results = vector_store.similarity_search_with_score(query, k=3)

    print(f"검색: {query}")
    print("=" * 50)
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. 점수: {score:.3f}")
        print(f"   내용: {doc.page_content}")
        print(f"   출처: {doc.metadata['source']} ({doc.metadata['type']})")
        print(f"   날짜: {doc.metadata['date']}")
        print()
    print()

def main():
    """메인 실행 함수"""
    print("Chapter 6.5.1: 벡터 검색 예제")
    print("=" * 50)

    # 1. 기본 벡터 검색 예제
    demonstrate_basic_vector_search()

    # 2. 검색 파라미터 튜닝 예제
    demonstrate_search_parameters()

    # 3. 거리 측정 방식 비교 예제
    demonstrate_distance_metrics()

    # 4. MMR 검색 예제
    demonstrate_mmr_search()

    # 5. 메타데이터를 포함한 검색 예제
    demonstrate_search_with_metadata()

if __name__ == "__main__":
    main()

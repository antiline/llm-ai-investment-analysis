"""
Chapter 6.2.1: BM25와 하이브리드 검색 예제
"""

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi
import numpy as np
from dotenv import load_dotenv

load_dotenv()

def simple_bm25_example():
    """간단한 BM25 예제"""
    print("=== 간단한 BM25 예제 ===")

    # 샘플 텍스트
    texts = [
        "Apple의 iPhone 판매량이 전년 대비 15% 증가했습니다.",
        "Apple의 서비스 부문 매출이 12% 성장했습니다.",
        "Apple의 P/E 비율은 28.5배로 업계 평균 대비 높습니다.",
        "Apple의 2024년 1분기 매출은 119.6조원으로 증가했습니다."
    ]

    # 토큰화 함수
    def tokenize(text):
        return text.replace('.', ' ').replace(',', ' ').split()

    # BM25 인덱스 생성
    tokenized_texts = [tokenize(text) for text in texts]
    bm25 = BM25Okapi(tokenized_texts)

    # 검색 수행
    query = "Apple의 P/E 비율"
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    print(f"검색: {query}")
    for i, (text, score) in enumerate(zip(texts, scores), 1):
        print(f"{i}. {text} (점수: {score:.3f})")
    print()

def hybrid_search_example():
    """FAISS와 BM25를 결합한 하이브리드 검색 예제"""
    print("=== 하이브리드 검색 예제 (FAISS + BM25) ===")

    # 임베딩 모델 초기화
    embeddings = OpenAIEmbeddings()

    # 샘플 텍스트
    texts = [
        "Apple의 iPhone 판매량이 전년 대비 15% 증가했습니다.",
        "Apple의 서비스 부문 매출이 12% 성장했습니다.",
        "Apple의 P/E 비율은 28.5배로 업계 평균 대비 높습니다.",
        "Apple의 2024년 1분기 매출은 119.6조원으로 증가했습니다.",
        "Apple의 현금 보유량이 1,500억 달러를 유지하고 있습니다.",
        "Apple의 중국 시장에서의 성과가 저조합니다."
    ]

    # FAISS 벡터 스토어 생성
    vector_store = FAISS.from_texts(texts, embeddings)

    # BM25 인덱스 생성
    def tokenize(text):
        return text.replace('.', ' ').replace(',', ' ').split()

    tokenized_texts = [tokenize(text) for text in texts]
    bm25 = BM25Okapi(tokenized_texts)

    # 하이브리드 검색 함수
    def hybrid_search(query, k=3, alpha=0.5):
        # 벡터 검색
        vector_results = vector_store.similarity_search_with_score(query, k=k)

        # BM25 검색
        query_tokens = tokenize(query)
        bm25_scores = bm25.get_scores(query_tokens)
        bm25_indices = np.argsort(bm25_scores)[::-1][:k]
        bm25_results = [(texts[i], bm25_scores[i]) for i in bm25_indices]

        # 결과 결합
        combined_results = {}

        # 벡터 검색 결과 처리
        for i, (doc, score) in enumerate(vector_results):
            doc_content = doc.page_content
            combined_results[doc_content] = {
                'vector_score': 1 - score,
                'bm25_score': 0,
                'combined_score': 0
            }

        # BM25 검색 결과 처리
        for doc_content, score in bm25_results:
            if doc_content not in combined_results:
                combined_results[doc_content] = {
                    'vector_score': 0,
                    'bm25_score': score,
                    'combined_score': 0
                }
            else:
                combined_results[doc_content]['bm25_score'] = score

        # 결합 점수 계산
        for doc_content in combined_results:
            vector_score = combined_results[doc_content]['vector_score']
            bm25_score = combined_results[doc_content]['bm25_score']
            combined_results[doc_content]['combined_score'] = alpha * vector_score + (1 - alpha) * bm25_score

        # 결합 점수로 정렬
        sorted_results = sorted(combined_results.items(), key=lambda x: x[1]['combined_score'], reverse=True)
        return sorted_results[:k]

    # 검색 수행
    query = "Apple의 P/E 비율"
    results = hybrid_search(query, k=3, alpha=0.5)

    print(f"하이브리드 검색: {query}")
    print("=" * 50)
    for i, (text, scores) in enumerate(results, 1):
        print(f"{i}. {text}")
        print(f"   벡터 점수: {scores['vector_score']:.3f}")
        print(f"   BM25 점수: {scores['bm25_score']:.3f}")
        print(f"   결합 점수: {scores['combined_score']:.3f}")
        print()

def main():
    """메인 실행 함수"""
    print("Chapter 6.2.1: BM25와 하이브리드 검색 예제")
    print("=" * 50)

    # BM25 예제
    simple_bm25_example()

    # 하이브리드 검색 예제
    hybrid_search_example()

if __name__ == "__main__":
    main()

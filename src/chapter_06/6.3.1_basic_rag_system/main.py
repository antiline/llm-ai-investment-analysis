"""
Chapter 6.3.1: 통합 RAG 시스템 구축
챕터 4,5를 바탕으로 한 하이브리드 검색과 Structured Output을 활용한 통합 분석
"""

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List
from rank_bm25 import BM25Okapi
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# 챕터 5의 데이터 수집기 (Mocked for example)
class YahooFinanceCollector:
    def collect_apple_data(self):
        return [
            {
                "content": "Apple의 2024년 1분기 매출은 119.6조원으로 전년 대비 4% 증가했습니다. iPhone 판매량이 전년 대비 15% 증가했으며, 특히 iPhone 15 Pro 시리즈의 성공이 두드러집니다. Apple의 서비스 부문도 12% 성장하여 Apple Music, iCloud, App Store의 강세를 보였습니다.",
                "source": "yahoo_finance",
                "type": "financials",
                "date": "2024-01-01"
            }
        ]

class SECDataCollector:
    def collect_apple_filings(self):
        return [
            {
                "content": "Apple의 10-K 보고서에 따르면 2023년 회계연도에 3943억 달러의 매출을 기록했습니다. 제품별 매출 비중은 iPhone 52%, Mac 7%, iPad 8%, Wearables 8%, 서비스 22%입니다. 영업이익률은 29.0%로 전년 대비 1.2%p 하락했습니다.",
                "source": "sec_edgar",
                "type": "10k",
                "date": "2023-12-31"
            }
        ]

class NewsCollector:
    def collect_apple_news(self):
        return [
            {
                "content": "Apple Vision Pro는 2024년 2월 출시된 혁신적인 공간 컴퓨팅 기기입니다. 초기 시장 반응은 예상보다 긍정적이며, 프리미엄 가격대에도 불구하고 높은 관심을 받고 있습니다. 2024년 40만대 판매 예상으로 매출 14억 달러를 기대하고 있습니다.",
                "source": "news",
                "type": "article",
                "date": "2024-02-01"
            }
        ]

# Structured Output 모델 정의
class AppleAnalysis(BaseModel):
    summary: str = Field(description="핵심 요약 - 주요 성과 지표와 핵심 변화 사항")
    financial_analysis: str = Field(description="재무 분석 - 매출, 수익성, 성장률, 주요 재무 지표")
    investment_perspective: str = Field(description="투자 관점 - 투자 매력도, 리스크 요인, 투자 권고사항")
    sources: List[str] = Field(description="출처 정보 - 데이터 출처 및 신뢰도")
    confidence_score: float = Field(description="분석 신뢰도 (0-1)", ge=0, le=1)

def build_vector_database():
    """데이터 수집 및 벡터 데이터베이스 구축"""
    print("=== 데이터 수집 및 벡터 데이터베이스 구축 ===")

    # 임베딩 모델 초기화
    embeddings = OpenAIEmbeddings()

    # 텍스트 분할기 설정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    # 챕터 5의 데이터 수집기 사용
    yahoo_collector = YahooFinanceCollector()
    sec_collector = SECDataCollector()
    news_collector = NewsCollector()

    # 데이터 수집
    yahoo_data = yahoo_collector.collect_apple_data()
    sec_data = sec_collector.collect_apple_filings()
    news_data = news_collector.collect_apple_news()

    # 문서 변환
    all_documents = []
    for data in [yahoo_data, sec_data, news_data]:
        for item in data:
            chunks = text_splitter.split_text(item['content'])
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": item['source'],
                        "type": item['type'],
                        "date": item.get('date', ''),
                        "chunk_id": i
                    }
                )
                all_documents.append(doc)

    # FAISS 벡터 스토어 생성
    vector_store = FAISS.from_documents(all_documents, embeddings)

    print(f"✓ {len(all_documents)}개 문서로 벡터 스토어 구축 완료")
    return vector_store, all_documents

def create_bm25_index(documents):
    """BM25 인덱스 생성"""
    def tokenize(text):
        return text.replace('.', ' ').replace(',', ' ').split()

    texts = [doc.page_content for doc in documents]
    tokenized_texts = [tokenize(text) for text in texts]
    return BM25Okapi(tokenized_texts)

def hybrid_search(vector_store, bm25, documents, query, k=5, alpha=0.6):
    """하이브리드 검색 함수"""
    # 벡터 검색
    vector_results = vector_store.similarity_search_with_score(query, k=k)

    # BM25 검색
    def tokenize(text):
        return text.replace('.', ' ').replace(',', ' ').split()

    query_tokens = tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_indices = np.argsort(bm25_scores)[::-1][:k]
    bm25_results = [(documents[i], bm25_scores[i]) for i in bm25_indices]

    # 결과 결합 및 점수 계산
    combined_results = {}

    # 벡터 검색 결과 처리
    for doc, score in vector_results:
        doc_content = doc.page_content
        combined_results[doc_content] = {
            'doc': doc,
            'vector_score': 1 - score,
            'bm25_score': 0,
            'combined_score': 0
        }

    # BM25 검색 결과 처리
    for doc, score in bm25_results:
        doc_content = doc.page_content
        if doc_content in combined_results:
            combined_results[doc_content]['bm25_score'] = score
        else:
            combined_results[doc_content] = {
                'doc': doc,
                'vector_score': 0,
                'bm25_score': score,
                'combined_score': 0
            }

    # 결합 점수 계산
    for doc_content in combined_results:
        vector_score = combined_results[doc_content]['vector_score']
        bm25_score = combined_results[doc_content]['bm25_score']
        combined_results[doc_content]['combined_score'] = alpha * vector_score + (1 - alpha) * bm25_score

    # 결합 점수로 정렬
    sorted_results = sorted(combined_results.items(), key=lambda x: x[1]['combined_score'], reverse=True)
    return [(item[1]['doc'], item[1]['combined_score']) for item in sorted_results[:k]]

def generate_hybrid_structured_analysis(vector_store, bm25, documents, question, k=5):
    """하이브리드 검색 + Structured Output 통합 분석 함수"""
    # LLM 초기화
    llm = ChatOpenAI(model="gpt-4", temperature=0.1)

    # Apple 분석 프롬프트 (챕터 5 스타일)
    apple_analysis_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
당신은 전문 투자 분석가입니다. 주어진 컨텍스트를 바탕으로 Apple Inc.에 대한 종합적인 투자 분석을 제공해주세요.

컨텍스트:
<context>
{context}
</context>

질문: {question}

다음 구조로 상세한 분석을 제공해주세요:

## 📊 핵심 요약
- 주요 성과 지표
- 핵심 변화 사항

## 📈 재무 분석
- 매출 및 수익성 분석
- 성장률 및 전망
- 주요 재무 지표

## 🎯 투자 관점
- 투자 매력도 평가
- 리스크 요인
- 투자 권고사항

## 📋 출처 정보
- 데이터 출처 및 신뢰도
- 분석 근거

답변:
"""
    )

    # 1. 하이브리드 검색 수행
    search_results = hybrid_search(vector_store, bm25, documents, question, k=k)

    # 2. 컨텍스트 구성
    context_parts = []
    sources = []

    for doc, score in search_results:
        context_parts.append(f"문서 (하이브리드 점수: {score:.3f}): {doc.page_content}")
        if doc.metadata.get('source'):
            sources.append(f"{doc.metadata['source']} - {doc.metadata.get('type', 'unknown')}")

    context = "\n\n".join(context_parts)

    # 3. Structured Output을 사용한 응답 생성
    structured_chain = llm.with_structured_output(AppleAnalysis)

    prompt = apple_analysis_prompt.format(context=context, question=question)
    result = structured_chain.invoke([HumanMessage(content=prompt)])

    return {
        "summary": result.summary,
        "financial_analysis": result.financial_analysis,
        "investment_perspective": result.investment_perspective,
        "sources": result.sources,
        "confidence_score": result.confidence_score,
        "hybrid_scores": [score for _, score in search_results]
    }

def main():
    """메인 실행 함수"""
    print("Chapter 6.3.1: 통합 RAG 시스템 구축")
    print("=" * 60)

    # 1. 벡터 데이터베이스 구축
    vector_store, documents = build_vector_database()

    # 2. BM25 인덱스 생성
    bm25 = create_bm25_index(documents)

    # 3. 실제 사용 예시
    questions = [
        "Apple의 최근 재무 성과는 어떤가요?",
        "iPhone 판매량 추이는 어떻나요?",
        "Apple의 서비스 부문 성장 전망은?"
    ]

    print("\n=== 통합 RAG 시스템 분석 결과 ===")

    for question in questions:
        print(f"\n질문: {question}")
        print("-" * 50)

        result = generate_hybrid_structured_analysis(vector_store, bm25, documents, question)

        print(f"📊 핵심 요약: {result['summary']}")
        print(f"📈 재무 분석: {result['financial_analysis']}")
        print(f"🎯 투자 관점: {result['investment_perspective']}")
        print(f"📋 출처: {', '.join(result['sources'])}")
        print(f"신뢰도: {result['confidence_score']:.2f}")
        print(f"하이브리드 검색 점수: {result['hybrid_scores']}")
        print()

if __name__ == "__main__":
    main()

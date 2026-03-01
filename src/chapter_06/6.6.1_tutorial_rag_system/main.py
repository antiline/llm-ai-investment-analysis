"""
Chapter 6.6.1: 튜토리얼 RAG 시스템
실제 사용 가능한 완전한 RAG 시스템 구현
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

class TutorialRAGSystem:
    """튜토리얼용 완전한 RAG 시스템"""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.1)
        self.vector_store = None
        self.bm25 = None
        self.documents = []

    def create_sample_data(self):
        """샘플 Apple 데이터 생성"""
        sample_data = [
            {
                "content": """
                Apple Inc. 2024년 1분기 실적 발표

                Apple은 2024년 1분기에 119.6조원의 매출을 기록했습니다. 이는 전년 대비 4% 증가한 수치입니다.
                iPhone 판매량이 전년 대비 15% 증가했으며, 특히 iPhone 15 Pro 시리즈의 성공이 두드러집니다.
                Apple의 서비스 부문도 12% 성장하여 Apple Music, iCloud, App Store의 강세를 보였습니다.
                구독자 수가 전년 대비 20% 증가했습니다.
                """,
                "source": "earnings_report",
                "type": "financial",
                "date": "2024-01-01"
            },
            {
                "content": """
                Apple Vision Pro 시장 전망

                Apple Vision Pro는 2024년 2월 출시된 혁신적인 공간 컴퓨팅 기기입니다.
                초기 시장 반응은 예상보다 긍정적이며, 프리미엄 가격대에도 불구하고 높은 관심을 받고 있습니다.
                2024년 40만대 판매 예상으로 매출 14억 달러를 기대하고 있습니다.
                2025년 100만대 판매 목표를 설정했습니다.
                """,
                "source": "product_analysis",
                "type": "product",
                "date": "2024-02-01"
            },
            {
                "content": """
                Apple의 재무 건전성 분석

                Apple의 재무 건전성은 여전히 우수합니다. 현금 보유량이 1,500억 달러를 유지하고 있으며,
                부채 비율도 업계 평균 대비 낮은 수준입니다. 영업이익률은 29.0%로 전년 대비 1.2%p 하락했지만
                여전히 업계 최고 수준을 유지하고 있습니다.
                """,
                "source": "financial_analysis",
                "type": "financial",
                "date": "2024-01-15"
            },
            {
                "content": """
                Apple의 중국 시장 도전

                Apple의 중국 시장에서의 성과가 저조하며, 규제 강화로 인해 8% 감소했습니다.
                이는 Apple의 주요 시장 중 하나인 중국에서의 도전을 보여줍니다.
                중국 정부의 데이터 보안 규제 강화와 현지 경쟁사들의 성장이 주요 요인입니다.
                """,
                "source": "market_analysis",
                "type": "market",
                "date": "2024-01-20"
            }
        ]
        return sample_data

    def build_knowledge_base(self):
        """지식 베이스 구축"""
        print("=== 지식 베이스 구축 ===")

        # 1. 샘플 데이터 생성
        sample_data = self.create_sample_data()

        # 2. 텍스트 분할기 설정
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # 3. 문서 변환 및 분할
        for item in sample_data:
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
                self.documents.append(doc)

        # 4. FAISS 벡터 스토어 생성
        self.vector_store = FAISS.from_documents(self.documents, self.embeddings)

        # 5. BM25 인덱스 생성
        def tokenize(text):
            return text.replace('.', ' ').replace(',', ' ').split()

        texts = [doc.page_content for doc in self.documents]
        tokenized_texts = [tokenize(text) for text in texts]
        self.bm25 = BM25Okapi(tokenized_texts)

        print(f"✓ {len(self.documents)}개 문서로 지식 베이스 구축 완료")
        print(f"✓ FAISS 벡터 스토어 생성 완료")
        print(f"✓ BM25 인덱스 생성 완료")

    def hybrid_search(self, query, k=5, alpha=0.6):
        """하이브리드 검색 수행"""
        # 벡터 검색
        vector_results = self.vector_store.similarity_search_with_score(query, k=k)

        # BM25 검색
        def tokenize(text):
            return text.replace('.', ' ').replace(',', ' ').split()

        query_tokens = tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_indices = np.argsort(bm25_scores)[::-1][:k]
        bm25_results = [(self.documents[i], bm25_scores[i]) for i in bm25_indices]

        # 결과 결합
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

    def create_analysis_prompt(self):
        """분석용 프롬프트 생성"""
        return PromptTemplate(
            input_variables=["context", "question"],
            template="""
당신은 Apple Inc. 전문 투자 분석가입니다. 주어진 컨텍스트를 바탕으로 Apple에 대한 종합적인 투자 분석을 제공해주세요.

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

    def analyze_question(self, question, k=5):
        """질문 분석 및 답변 생성"""
        print(f"\n=== 질문 분석: {question} ===")

        # 1. 하이브리드 검색 수행
        search_results = self.hybrid_search(question, k=k)

        # 2. 컨텍스트 구성
        context_parts = []
        sources = []

        for doc, score in search_results:
            context_parts.append(f"문서 (하이브리드 점수: {score:.3f}): {doc.page_content}")
            if doc.metadata.get('source'):
                sources.append(f"{doc.metadata['source']} - {doc.metadata.get('type', 'unknown')}")

        context = "\n\n".join(context_parts)

        # 3. 프롬프트 생성 및 답변 생성
        prompt = self.create_analysis_prompt()
        formatted_prompt = prompt.format(context=context, question=question)

        # 4. LLM으로 답변 생성
        response = self.llm.invoke([HumanMessage(content=formatted_prompt)])

        return {
            "question": question,
            "answer": response.content,
            "sources": sources,
            "search_scores": [score for _, score in search_results],
            "num_sources": len(search_results)
        }

    def run_tutorial(self):
        """튜토리얼 실행"""
        print("Chapter 6.6.1: 튜토리얼 RAG 시스템")
        print("=" * 60)

        # 1. 지식 베이스 구축
        self.build_knowledge_base()

        # 2. 다양한 질문으로 테스트
        test_questions = [
            "Apple의 최근 재무 성과는 어떤가요?",
            "iPhone 판매량 추이는 어떻나요?",
            "Apple의 서비스 부문 성장 전망은?",
            "Vision Pro의 시장 전망은 어떻나요?",
            "Apple의 중국 시장에서의 도전은?"
        ]

        print("\n=== RAG 시스템 테스트 ===")

        for i, question in enumerate(test_questions, 1):
            print(f"\n질문 {i}: {question}")
            print("-" * 50)

            result = self.analyze_question(question)

            print(f"답변:")
            print(result["answer"])
            print(f"\n참고 문서: {result['num_sources']}개")
            print(f"출처: {', '.join(result['sources'])}")
            print(f"검색 점수: {result['search_scores']}")
            print()

def main():
    """메인 실행 함수"""
    # 튜토리얼 RAG 시스템 생성 및 실행
    rag_system = TutorialRAGSystem()
    rag_system.run_tutorial()

if __name__ == "__main__":
    main()

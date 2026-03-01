import os
import requests
import feedparser
from typing import Dict, Any, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


class NewsAnalysis(BaseModel):
    """뉴스 데이터 분석 결과 구조화"""
    market_sentiment: str = Field(description="뉴스 기반 시장 분위기")
    key_issues: List[str] = Field(description="뉴스에서 발견된 핵심 이슈 3가지")
    market_reaction_prediction: str = Field(description="뉴스가 주가에 미칠 영향")
    final_investment_grade: str = Field(description="최종 투자 등급")
    news_impact: str = Field(description="뉴스가 투자 등급에 미친 영향")
    target_price_final_adjustment: str = Field(description="목표가 최종 조정")
    short_term_outlook: str = Field(description="단기 전망 (1-3개월)")
    long_term_outlook: str = Field(description="장기 전망 (6-12개월)")
    monitoring_points: List[str] = Field(description="핵심 모니터링 포인트 3-5가지")
    additional_risks: List[str] = Field(description="뉴스 기반 추가 리스크")
    risk_level: str = Field(description="리스크 등급")
    response_strategy: str = Field(description="리스크 대응 방안")
    investment_strategy: str = Field(description="투자 전략")
    entry_timing: str = Field(description="진입 시점")
    portfolio_weight: str = Field(description="포트폴리오 비중")
    stop_loss_criteria: str = Field(description="손절매 기준")


class GoogleNewsCollector:
    """Google News RSS를 통한 뉴스 데이터 수집 클래스"""

    def __init__(self, company_name: str, ticker: str):
        self.company_name = company_name
        self.ticker = ticker
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def search_google_news(self, keyword: str, days: int = 1, country: str = 'en', limit: int = 10) -> List[Dict[str, Any]]:
        """Google News RSS를 통한 뉴스 검색"""
        url = f'https://news.google.com/rss/search?q={keyword}+when:{days}d'

        if country == 'en':
            url += '&hl=en-NG&gl=NG&ceid=NG:en'
        elif country == 'ko':
            url += '&hl=ko&gl=KR&ceid=KR:ko'
        else:
            url += '&hl=en&gl=US&ceid=US:en'

        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                feed = feedparser.parse(response.text)
                news_items = []

                for entry in feed.entries[:limit]:
                    news_item = {
                        "title": entry.get('title', ''),
                        "summary": entry.get('summary', ''),
                        "link": entry.get('link', ''),
                        "published": entry.get('published', ''),
                        "source": entry.get('source', {}).get('title', 'Unknown'),
                        "relevance_score": self._calculate_relevance(entry.get('title', ''), entry.get('summary', ''))
                    }
                    news_items.append(news_item)

                return news_items
            else:
                print(f"❌ Google News 검색 실패: {response.status_code}")
                return []
        except Exception as e:
            print(f"❌ Google News 검색 오류: {e}")
            return []

    def _calculate_relevance(self, title: str, summary: str) -> float:
        """뉴스 관련성 점수 계산"""
        text = f"{title} {summary}".lower()
        score = 0.0

        # 회사명 매칭 (높은 가중치)
        if self.company_name.lower() in text:
            score += 3.0
        if self.ticker.lower() in text:
            score += 2.0

        return score

    def collect_company_news(self, limit: int = 10, country: str = 'en') -> List[Dict[str, Any]]:
        """회사명 기반 뉴스 수집"""
        print(f"📰 {self.company_name} Google News 검색 중...")

        # 회사명으로 뉴스 검색
        news_items = self.search_google_news(self.company_name, days=1, country=country, limit=limit)

        # 관련성 점수로 정렬
        news_items.sort(key=lambda x: x['relevance_score'], reverse=True)

        print(f"   수집된 뉴스: {len(news_items)}개")
        return news_items[:limit]

    def collect_news_by_keywords(self, keywords: List[str], limit: int = 10, country: str = 'en') -> List[Dict[str, Any]]:
        """키워드 기반 뉴스 수집"""
        all_news = []
        for keyword in keywords:
            news_items = self.search_google_news(keyword, days=1, country=country, limit=limit//len(keywords))
            all_news.extend(news_items)

        # 관련성 점수로 정렬
        all_news.sort(key=lambda x: x['relevance_score'], reverse=True)

        return all_news[:limit]


class NewsDataProcessor:
    """뉴스 데이터 처리 및 마크다운 변환 클래스"""

    def __init__(self, llm):
        self.llm = llm

    def refine_news_with_llm(self, news_items: List[Dict[str, Any]]) -> str:
        """LLM을 사용하여 뉴스 데이터를 정제된 마크다운으로 변환"""
        if not news_items:
            return "수집된 뉴스가 없습니다."

        # 뉴스 데이터를 텍스트로 구성
        raw_news_text = ""
        for i, news in enumerate(news_items, 1):
            raw_news_text += f"뉴스 {i}:\n"
            raw_news_text += f"제목: {news['title']}\n"
            raw_news_text += f"출처: {news['source']}\n"
            raw_news_text += f"발행일: {news['published']}\n"
            raw_news_text += f"요약: {news['summary']}\n"
            raw_news_text += f"링크: {news['link']}\n\n"

        # LLM을 통한 마크다운 변환 프롬프트
        prompt = PromptTemplate(
            input_variables=["company_name", "raw_news"],
            template="""
당신은 뉴스 데이터를 깔끔하고 구조화된 마크다운 형식으로 변환하는 전문가입니다.

다음 뉴스 데이터를 읽고, 투자 분석에 유용한 형태로 정리해주세요:

## 원본 뉴스 데이터
<raw_news>
{raw_news}
</raw_news>

## 변환 요구사항
1. 마크다운 형식으로 깔끔하게 정리
2. 중요도에 따라 뉴스를 분류 (핵심/일반)
3. 각 뉴스의 핵심 내용을 간결하게 요약
4. 투자자 관점에서 중요한 정보 강조
5. 시장 영향도 평가 포함

{company_name} 관련 뉴스를 분석하여 투자 결정에 도움이 되는 형태로 정리해주세요.
"""
        )

        try:
            chain = prompt | self.llm
            result = chain.invoke({
                "company_name": news_items[0].get('company_name', 'Company'),
                "raw_news": raw_news_text
            })
            return result.content
        except Exception as e:
            print(f"❌ LLM 뉴스 정제 실패: {e}")
            return "뉴스 정제 실패"


class NewsEnhancedAnalyzer:
    """Google News 데이터 기반 투자 분석기"""

    def __init__(self, company_name: str, ticker: str):
        self.company_name = company_name
        self.ticker = ticker

        # LLM 초기화
        load_dotenv()
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-3.5-turbo",
            temperature=0
        )

        # 뉴스 수집기 및 처리기 초기화
        self.news_collector = GoogleNewsCollector(company_name, ticker)
        self.news_processor = NewsDataProcessor(self.llm)

    def run_news_analysis(self, country: str = 'en') -> str:
        """Google News 데이터 기반 투자 분석 실행"""
        print(f"📰 {self.company_name} Google News 기반 분석 시작...")

        try:
            # 1단계: Google News에서 뉴스 수집
            print("1단계: Google News 뉴스 수집 중...")
            news_items = self.news_collector.collect_company_news(limit=10, country=country)

            if not news_items:
                print("❌ 수집된 뉴스가 없습니다.")
                return "뉴스 데이터 부족"

            # 2단계: LLM을 통한 뉴스 데이터 정제
            print("2단계: LLM을 통한 뉴스 데이터 정제 중...")
            refined_news_data = self.news_processor.refine_news_with_llm(news_items)

            # 3단계: 정제된 데이터로 투자 분석
            print("3단계: 정제된 뉴스 데이터로 투자 분석 중...")

            # 분석 프롬프트
            analysis_prompt = PromptTemplate(
                input_variables=["company_name", "ticker", "refined_news_data"],
                template="""
당신은 {company_name} ({ticker}) 전문 투자 애널리스트입니다.

다음 Google News에서 수집하고 정제된 뉴스 데이터를 바탕으로 종합적인 투자 분석을 수행하세요:

## 정제된 뉴스 데이터
<refined_news_data>
{refined_news_data}
</refined_news_data>

## 분석 요구사항

### 1. 시장 분위기 분석
- 뉴스 기반 시장 분위기: [긍정적/부정적/중립적]
- 시장 반응 예측: [구체적 예측 내용]

### 2. 핵심 이슈 식별
- 뉴스에서 발견된 핵심 이슈 3가지: [구체적 이슈들]
- 각 이슈의 투자 영향도: [높음/보통/낮음]

### 3. 투자 전략 수립
- 단기 투자 전략: [1-3개월 관점]
- 중기 투자 전략: [3-12개월 관점]
- 포트폴리오 비중: [증가/유지/감소]

### 4. 리스크 관리
- 손절매 기준: [구체적 기준]
- 모니터링 포인트: [중요 지표들]

모든 분석은 Google News의 최신 뉴스 데이터를 기반으로 제공하세요.
"""
            )

            chain = analysis_prompt | self.llm
            result = chain.invoke({
                "company_name": self.company_name,
                "ticker": self.ticker,
                "refined_news_data": refined_news_data
            })

            print("📈 Google News 기반 투자 분석 완료")
            return result.content

        except Exception as e:
            print(f"❌ 분석 실패: {e}")
            return "분석 실패"


def main():
    """메인 실행 함수"""
    print("🚀 Google News 뉴스 분석 시스템")
    print("=" * 50)

    # Apple Inc. 뉴스 분석 예시
    analyzer = NewsEnhancedAnalyzer("Apple Inc.", "AAPL")
    result = analyzer.run_news_analysis()

    if result and "분석 실패" not in result:
        print("\n✅ 뉴스 분석 완료!")
        print("\n📰 분석 결과 요약:")
        print(result[:500] + "..." if len(result) > 500 else result)
    else:
        print("\n❌ 뉴스 분석 실패!")


if __name__ == "__main__":
    main()

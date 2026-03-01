"""
5.4.1 AI 기반 키워드 추천 시스템

이 섹션에서는 1단계와 2단계의 분석 결과를 바탕으로 사용자가 놓칠 수 있는
중요한 뉴스 검색 키워드를 AI가 추천하여 3단계 뉴스 검색의 정확도를 높입니다.

주요 기능:
- 분석 결과 기반 키워드 자동 추천
- 재무, 비즈니스, 리스크 관련 키워드 분류
- 우선순위 기반 키워드 정렬
"""

import json
from typing import List, Dict, Any
from datetime import datetime
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

class KeywordRecommender:
    """AI 기반 키워드 추천 시스템"""

    def __init__(self, llm):
        self.llm = llm

    def recommend_keywords_from_analysis(self, step1_result: Dict, step2_result: Dict, user_keywords: List[str]) -> List[str]:
        """분석 결과를 바탕으로 키워드 추천"""

        prompt = f"""
당신은 투자 분석 전문가입니다. 다음 분석 결과를 바탕으로 뉴스 검색에 유용한 키워드를 추천해주세요.

## 📊 1단계 분석 결과
{step1_result.get('analysis_result', 'N/A')}

## 📈 2단계 SEC 데이터 분석 결과
{step2_result.get('enhanced_analysis', 'N/A')}

## 🔍 사용자가 이미 입력한 키워드
{', '.join(user_keywords)}

## 키워드 추천 요구사항

### 1. 재무 관련 키워드
- SEC 데이터에서 발견된 중요한 재무 지표나 트렌드
- 예: "revenue growth", "profit margin", "debt ratio"

### 2. 비즈니스 전략 키워드
- 회사의 주요 사업 영역이나 전략적 이니셔티브
- 예: "new product launch", "market expansion", "acquisition"

### 3. 리스크 관련 키워드
- 분석에서 발견된 주요 리스크 요인
- 예: "regulatory risk", "competition", "supply chain"

### 4. 시장 동향 키워드
- 업계나 시장 전반의 트렌드
- 예: "industry trend", "market share", "consumer demand"

### 5. 특정 이벤트 키워드
- 예상되는 중요한 이벤트나 발표
- 예: "earnings call", "product announcement", "legal case"

## 출력 형식
다음 JSON 형식으로 추천 키워드를 제공하세요:

```json
{{
    "recommended_keywords": [
        "keyword1",
        "keyword2",
        "keyword3",
        "keyword4",
        "keyword5"
    ],
    "reasoning": {{
        "financial_keywords": ["keyword1", "keyword2"],
        "business_strategy_keywords": ["keyword3"],
        "risk_keywords": ["keyword4"],
        "market_trend_keywords": ["keyword5"]
    }},
    "priority": {{
        "high": ["keyword1", "keyword2"],
        "medium": ["keyword3", "keyword4"],
        "low": ["keyword5"]
    }}
}}
```

추천 키워드는 구체적이고 검색 가능해야 하며, 사용자가 놓칠 수 있는 중요한 정보를 포함해야 합니다.
"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])

            # JSON 파싱 시도
            try:
                result = json.loads(response.content)
                return result.get("recommended_keywords", [])
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 텍스트에서 키워드 추출
                return self._extract_keywords_from_text(response.content)

        except Exception as e:
            print(f"키워드 추천 실패: {e}")
            return []

    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """텍스트에서 키워드 추출"""
        keywords = []

        # 따옴표로 둘러싸인 키워드 찾기
        import re
        quoted_keywords = re.findall(r'"([^"]+)"', text)
        keywords.extend(quoted_keywords)

        # 예시 키워드들 찾기
        example_pattern = r'예:\s*"([^"]+)"'
        example_keywords = re.findall(example_pattern, text)
        keywords.extend(example_keywords)

        return list(set(keywords))  # 중복 제거

class KeywordRecommendationAnalyzer:
    """키워드 추천 분석기"""

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

        self.keyword_recommender = KeywordRecommender(self.llm)

    def analyze_keyword_recommendation(self, step1_result: Dict, step2_result: Dict, initial_keywords: List[str]) -> Dict[str, Any]:
        """키워드 추천 분석 실행"""
        print(f"🔍 {self.company_name} ({self.ticker}) 키워드 추천 분석 시작...")

        # AI 키워드 추천
        recommended_keywords = self.keyword_recommender.recommend_keywords_from_analysis(
            step1_result, step2_result, initial_keywords
        )

        # 사용자 키워드와 추천 키워드 결합
        combined_keywords = list(set(initial_keywords + recommended_keywords))

        # 키워드 분석 결과
        analysis_result = {
            "company_name": self.company_name,
            "ticker": self.ticker,
            "initial_keywords": initial_keywords,
            "recommended_keywords": recommended_keywords,
            "combined_keywords": combined_keywords,
            "keyword_improvement": {
                "original_count": len(initial_keywords),
                "recommended_count": len(recommended_keywords),
                "combined_count": len(combined_keywords),
                "improvement_percentage": round((len(recommended_keywords) / len(initial_keywords)) * 100, 1) if initial_keywords else 0
            },
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        return analysis_result

    def generate_keyword_analysis_report(self, analysis_result: Dict) -> str:
        """키워드 분석 보고서 생성"""

        prompt = f"""
다음 키워드 추천 분석 결과를 바탕으로 상세한 보고서를 작성해주세요:

## 📊 키워드 추천 분석 결과
{json.dumps(analysis_result, indent=2, ensure_ascii=False)}

## 📝 보고서 요구사항

### 1. 키워드 개선 효과
- 원본 키워드 수: {analysis_result['keyword_improvement']['original_count']}개
- 추천 키워드 수: {analysis_result['keyword_improvement']['recommended_count']}개
- 개선율: {analysis_result['keyword_improvement']['improvement_percentage']}%

### 2. 추천 키워드 분석
- **재무 관련**: [재무 관련 키워드 분석]
- **비즈니스 전략**: [비즈니스 전략 관련 키워드 분석]
- **리스크 관리**: [리스크 관련 키워드 분석]
- **시장 동향**: [시장 동향 관련 키워드 분석]

### 3. 키워드 활용 전략
- **우선순위**: [높은 우선순위 키워드와 그 이유]
- **검색 전략**: [효과적인 뉴스 검색 방법]
- **모니터링 계획**: [키워드별 모니터링 계획]

### 4. 예상 효과
- **뉴스 검색 정확도**: [예상 개선 효과]
- **분석 품질**: [분석 품질 향상 예상]
- **투자 의사결정**: [투자 의사결정에 미칠 영향]

### 5. 추가 권고사항
- **키워드 조정**: [필요한 키워드 조정 사항]
- **지속적 개선**: [지속적 키워드 개선 방안]
- **시스템 활용**: [AI 키워드 추천 시스템 활용 방안]

보고서는 투자 분석가가 실제로 활용할 수 있는 실용적인 내용으로 작성해주세요.
"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            return f"보고서 생성 실패: {str(e)}"

def main():
    """메인 실행 함수"""
    print("🤖 AI 기반 키워드 추천 시스템")
    print("="*50)

    # 샘플 데이터 (실제로는 step1, step2에서 받아옴)
    step1_result = {
        "analysis_result": "1단계 Yahoo Finance 분석 결과 (예시)"
    }

    step2_result = {
        "enhanced_analysis": "2단계 SEC 데이터 분석 결과 (예시)"
    }

    # 초기 사용자 키워드
    initial_keywords = ["iPhone", "AI", "earnings"]

    # 키워드 추천 분석 실행
    analyzer = KeywordRecommendationAnalyzer("Apple Inc.", "AAPL")
    result = analyzer.analyze_keyword_recommendation(step1_result, step2_result, initial_keywords)

    print(f"\n📝 키워드 추천 결과:")
    print(f"   초기 키워드: {', '.join(result['initial_keywords'])}")
    print(f"   추천 키워드: {', '.join(result['recommended_keywords'])}")
    print(f"   결합 키워드: {', '.join(result['combined_keywords'])}")
    print(f"   개선율: {result['keyword_improvement']['improvement_percentage']}%")

    # 상세 보고서 생성
    report = analyzer.generate_keyword_analysis_report(result)

    print(f"\n📊 키워드 분석 보고서:")
    print("="*50)
    print(report)

if __name__ == "__main__":
    main()

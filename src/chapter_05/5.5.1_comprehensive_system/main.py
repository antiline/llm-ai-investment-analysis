"""
5.5.1 4단계 종합 분석 시스템

이 섹션에서는 앞서 구축한 모든 단계를 통합하여 완전한 투자 분석 시스템을 제공합니다.

주요 기능:
- 1단계: Yahoo Finance 실시간 데이터 수집 및 분석
- 2단계: SEC EDGAR 공식 재무제표 데이터 수집 및 분석
- 3단계: AI 기반 키워드 추천 시스템
- 4단계: 향상된 뉴스 데이터 수집 및 최종 분석
"""

import sys
import os
from datetime import datetime
from typing import Dict, Any, List

# 상위 디렉토리 import를 위한 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 각 단계별 모듈 import (실제 실행 시에는 해당 모듈들이 필요)
# from chapter_05.5.1.1_llm_limitations.main import YahooFinanceAnalyzer
# from chapter_05.5.2.1_sequential_chain.main import SECEnhancedAnalyzer
# from chapter_05.5.4.1_ai_keyword_recommendation.main import KeywordRecommendationAnalyzer
# from chapter_05.5.3.1_yahoo_finance_integration.main import NewsEnhancedAnalyzer

# 임시 클래스 정의 (실제로는 위의 import를 사용)
class YahooFinanceAnalyzer:
    def __init__(self, company_name, ticker):
        self.company_name = company_name
        self.ticker = ticker

    def run_analysis(self):
        return {"analysis_result": "1단계 Yahoo Finance 분석 결과 (예시)"}

class SECEnhancedAnalyzer:
    def __init__(self, company_name, ticker, sec_info):
        self.company_name = company_name
        self.ticker = ticker
        self.sec_info = sec_info

    def run_analysis(self, step1_result):
        return {"enhanced_analysis": "2단계 SEC 데이터 분석 결과 (예시)"}

class KeywordRecommendationAnalyzer:
    def __init__(self, company_name, ticker):
        self.company_name = company_name
        self.ticker = ticker

    def analyze_keyword_recommendation(self, step1_result, step2_result, initial_keywords):
        return {
            "initial_keywords": initial_keywords,
            "recommended_keywords": ["revenue", "profit", "growth"],
            "combined_keywords": initial_keywords + ["revenue", "profit", "growth"],
            "keyword_improvement": {"improvement_percentage": 42}
        }

class NewsEnhancedAnalyzer:
    def __init__(self, company_name, ticker, sec_info, keywords):
        self.company_name = company_name
        self.ticker = ticker
        self.sec_info = sec_info
        self.keywords = keywords

    def run_analysis(self, step2_result):
        return {"final_analysis": "4단계 뉴스 데이터 분석 결과 (예시)"}

class ComprehensiveAnalyzer:
    """4단계 종합 분석기"""

    def __init__(self, company_name: str, ticker: str, sec_info: str, initial_keywords: List[str]):
        self.company_name = company_name
        self.ticker = ticker
        self.sec_info = sec_info
        self.initial_keywords = initial_keywords

        # 각 단계별 분석기 초기화
        self.step1_analyzer = YahooFinanceAnalyzer(company_name, ticker)
        self.step2_analyzer = SECEnhancedAnalyzer(company_name, ticker, sec_info)
        self.step3_analyzer = KeywordRecommendationAnalyzer(company_name, ticker)

    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """4단계 종합 분석 실행"""

        print(f"🚀 {self.company_name} ({self.ticker}) 4단계 종합 분석 시작...")
        print("="*60)

        # 1단계: Yahoo Finance 데이터 분석
        print("📊 1단계: Yahoo Finance 데이터 수집 및 분석")
        step1_result = self.step1_analyzer.run_analysis()

        if "error" in step1_result:
            return {"error": f"1단계 실패: {step1_result['error']}"}

        print("✅ 1단계 완료")

        # 2단계: SEC 데이터 분석
        print("\n📈 2단계: SEC EDGAR 데이터 수집 및 분석")
        step2_result = self.step2_analyzer.run_analysis(step1_result)

        if "error" in step2_result:
            return {"error": f"2단계 실패: {step2_result['error']}"}

        print("✅ 2단계 완료")

        # 3단계: AI 키워드 추천
        print("\n🔍 3단계: AI 키워드 추천")
        keyword_result = self.step3_analyzer.analyze_keyword_recommendation(
            step1_result, step2_result, self.initial_keywords
        )

        # 사용자 키워드와 추천 키워드 결합
        combined_keywords = keyword_result["combined_keywords"]
        print(f"📝 결합된 키워드: {', '.join(combined_keywords)}")

        # 4단계: 향상된 뉴스 분석
        print("\n📰 4단계: 향상된 뉴스 데이터 수집 및 분석")
        step4_analyzer = NewsEnhancedAnalyzer(self.company_name, self.ticker, self.sec_info, combined_keywords)
        step4_result = step4_analyzer.run_analysis(step2_result)

        if "error" in step4_result:
            return {"error": f"4단계 실패: {step4_result['error']}"}

        print("✅ 4단계 완료")

        # 최종 결과 통합
        final_result = {
            "company_name": self.company_name,
            "ticker": self.ticker,
            "analysis_steps": {
                "step1_yahoo_finance": step1_result,
                "step2_sec_data": step2_result,
                "step3_keyword_recommendation": keyword_result,
                "step4_enhanced_news": step4_result
            },
            "final_analysis": step4_result["final_analysis"],
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "analysis_quality": {
                "data_sources": 3,  # Yahoo Finance, SEC, News
                "keyword_optimization": True,
                "sentiment_analysis": True,
                "confidence_level": "High"
            }
        }

        return final_result

    def generate_executive_summary(self, final_result: Dict[str, Any]) -> str:
        """실행 요약 보고서 생성"""

        step3 = final_result['analysis_steps']['step3_keyword_recommendation']
        improvement = step3.get('keyword_improvement', {}).get('improvement_percentage', 'N/A')

        summary = f"""
# 🍎 {self.company_name} ({self.ticker}) 종합 투자 분석 보고서

## 📊 분석 개요
- **분석 일시**: {final_result['timestamp']}
- **데이터 소스**: Yahoo Finance, SEC EDGAR, 뉴스 피드
- **키워드 최적화**: {improvement}% 개선
- **분석 신뢰도**: {final_result['analysis_quality']['confidence_level']}

## 🎯 핵심 분석 결과

### 3단계: 키워드 최적화
- 원본 키워드: {len(step3['initial_keywords'])}개
- 추천 키워드: {len(step3['recommended_keywords'])}개
- 최종 키워드: {len(step3['combined_keywords'])}개

## 📈 최종 투자 분석
{final_result['final_analysis']}

## 🔍 분석 품질 지표
- **데이터 소스 다양성**: {final_result['analysis_quality']['data_sources']}/3
- **키워드 최적화**: {'활성화' if final_result['analysis_quality']['keyword_optimization'] else '비활성화'}
- **감정 분석**: {'활성화' if final_result['analysis_quality']['sentiment_analysis'] else '비활성화'}
- **전체 신뢰도**: {final_result['analysis_quality']['confidence_level']}

---
*이 보고서는 AI 기반 종합 분석 시스템을 통해 생성되었습니다.*
"""

        return summary

def main():
    """메인 실행 함수"""
    print("🍎 Apple Inc. 종합 투자 분석 시스템")
    print("="*60)

    # 사용자 입력 (실제로는 CLI 인자나 GUI에서 받음)
    company_name = "Apple Inc."
    ticker = "AAPL"
    sec_info = "Apple Inc. SEC filings"
    initial_keywords = ["iPhone", "AI", "earnings", "China"]

    print(f"회사: {company_name}")
    print(f"티커: {ticker}")
    print(f"초기 키워드: {', '.join(initial_keywords)}")
    print("\n분석을 시작합니다...\n")

    # 종합 분석 실행
    analyzer = ComprehensiveAnalyzer(company_name, ticker, sec_info, initial_keywords)
    result = analyzer.run_comprehensive_analysis()

    if "error" in result:
        print(f"❌ 분석 실패: {result['error']}")
        return

    print("\n" + "="*60)
    print("🎯 4단계 종합 분석 완료")
    print("="*60)

    # 키워드 추천 결과 출력
    keyword_info = result["analysis_steps"]["step3_keyword_recommendation"]
    print(f"\n📝 키워드 추천 결과:")
    print(f"   초기 키워드: {', '.join(keyword_info['initial_keywords'])}")
    print(f"   추천 키워드: {', '.join(keyword_info['recommended_keywords'])}")
    print(f"   결합 키워드: {', '.join(keyword_info['combined_keywords'])}")

    # 최종 분석 결과 출력
    print(f"\n📊 최종 분석 결과:")
    print(result["final_analysis"])

    # 실행 요약 보고서 생성
    summary = analyzer.generate_executive_summary(result)

    print(f"\n📋 실행 요약 보고서:")
    print("="*60)
    print(summary)

    return result

if __name__ == "__main__":
    main()

"""
5.2.1 SEC EDGAR 데이터 수집 및 분석

이 섹션에서는 SEC EDGAR에서 최신 재무제표 데이터를 크롤링하여
1단계 Yahoo Finance 결과와 결합하여 더 정확한 분석을 제공합니다.

주요 기능:
- SEC EDGAR 크롤링을 통한 공식 재무제표 데이터 수집
- Yahoo Finance 데이터와의 비교 분석
- LangChain SequentialChain을 활용한 순차 분석
"""

import requests
import re
from bs4 import BeautifulSoup
import time
from typing import List, Dict, Any
import json
from datetime import datetime
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.chains import LLMChain, SequentialChain
from pydantic import BaseModel, Field

class SECAnalysis(BaseModel):
    """SEC 데이터 분석 결과 구조화"""
    data_reliability: Dict[str, str] = Field(description="데이터 신뢰성 평가")
    financial_reassessment: Dict[str, Any] = Field(description="재무 건전성 재평가")
    investment_grade_revision: str = Field(description="수정된 투자 등급")
    revision_reason: str = Field(description="수정 근거")
    target_price_adjustment: str = Field(description="목표가 조정")
    additional_risks: list = Field(description="추가 리스크 요인")
    regulatory_issues: list = Field(description="규제 관련 이슈")
    confidence_level: str = Field(description="전체 분석 신뢰도")

class SECDataCollector:
    """SEC EDGAR 데이터 수집 클래스"""

    def __init__(self, company_name: str, ticker: str, sec_info: str):
        self.company_name = company_name
        self.ticker = ticker
        self.sec_info = sec_info
        self.base_url = "https://www.sec.gov/Archives/edgar/data"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def extract_cik_number(self) -> str:
        """CIK 번호 추출"""
        search_url = f"https://www.sec.gov/cgi-bin/browse-edgar?company={self.ticker}&owner=exclude&action=getcompany"

        try:
            response = requests.get(search_url, headers=self.headers)
            soup = BeautifulSoup(response.content, 'html.parser')

            # CIK 번호 패턴 찾기
            cik_pattern = r'CIK=(\d{10})'
            match = re.search(cik_pattern, response.text)

            if match:
                return match.group(1)
            else:
                # Apple 기본 CIK
                return "0000320193"

        except Exception as e:
            print(f"CIK 번호 추출 실패: {e}")
            return "0000320193"

    def get_latest_filing_urls(self, form_type: str = "10-K", limit: int = 3) -> List[str]:
        """최신 SEC 파일링 URL 수집"""
        cik = self.extract_cik_number()
        search_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type={form_type}&dateb=&owner=exclude&count={limit}"

        try:
            response = requests.get(search_url, headers=self.headers)
            soup = BeautifulSoup(response.content, 'html.parser')

            filing_links = []
            for link in soup.find_all('a', href=True):
                if '/Archives/edgar/data/' in link['href'] and form_type in link['href']:
                    filing_links.append("https://www.sec.gov" + link['href'])

            return filing_links[:limit]

        except Exception as e:
            print(f"파일링 URL 수집 실패: {e}")
            return []

    def extract_financial_data(self, filing_url: str) -> Dict[str, Any]:
        """SEC 파일링에서 재무 데이터 추출"""
        try:
            response = requests.get(filing_url, headers=self.headers)
            soup = BeautifulSoup(response.content, 'html.parser')

            financial_data = {
                "filing_date": "",
                "revenue": "",
                "net_income": "",
                "total_assets": "",
                "total_liabilities": "",
                "cash_flow": "",
                "key_metrics": {}
            }

            text_content = soup.get_text()

            # 매출 패턴 찾기
            revenue_pattern = r'Total net sales[:\s]*\$?([\d,]+\.?\d*)'
            revenue_match = re.search(revenue_pattern, text_content, re.IGNORECASE)
            if revenue_match:
                financial_data["revenue"] = revenue_match.group(1)

            # 순이익 패턴 찾기
            net_income_pattern = r'Net income[:\s]*\$?([\d,]+\.?\d*)'
            net_income_match = re.search(net_income_pattern, text_content, re.IGNORECASE)
            if net_income_match:
                financial_data["net_income"] = net_income_match.group(1)

            return financial_data

        except Exception as e:
            print(f"재무 데이터 추출 실패: {e}")
            return {"error": str(e)}

    def collect_sec_data(self) -> Dict[str, Any]:
        """SEC 데이터 종합 수집"""
        print(f"📄 {self.company_name} SEC 데이터 수집 중...")

        filing_urls = self.get_latest_filing_urls("10-K", 2)

        sec_data = {
            "company_name": self.company_name,
            "ticker": self.ticker,
            "filing_data": [],
            "summary": {}
        }

        for url in filing_urls:
            filing_data = self.extract_financial_data(url)
            if "error" not in filing_data:
                sec_data["filing_data"].append(filing_data)
            time.sleep(0.1)

        if sec_data["filing_data"]:
            latest_filing = sec_data["filing_data"][0]
            sec_data["summary"] = {
                "latest_revenue": latest_filing.get("revenue", "N/A"),
                "latest_net_income": latest_filing.get("net_income", "N/A"),
                "filing_count": len(sec_data["filing_data"]),
                "data_quality": "Good" if len(sec_data["filing_data"]) > 0 else "Poor"
            }

        return sec_data

class SECEnhancedAnalyzer:
    """SEC 데이터가 추가된 향상된 분석기"""

    def __init__(self, company_name: str, ticker: str, sec_info: str):
        self.company_name = company_name
        self.ticker = ticker
        self.sec_collector = SECDataCollector(company_name, ticker, sec_info)

        # LLM 초기화
        load_dotenv()
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-3.5-turbo",
            temperature=0
        )

        # 출력 파서 초기화
        self.output_parser = PydanticOutputParser(pydantic_object=SECAnalysis)

        # SequentialChain 구성
        self.setup_sequential_chains()

    def setup_sequential_chains(self):
        """SequentialChain 설정"""

        # 1단계: 데이터 신뢰성 평가 체인
        self.reliability_prompt = PromptTemplate(
            input_variables=["step1_result", "sec_data"],
            template="""
다음 데이터의 신뢰성을 평가하세요:

## 1단계 분석 결과
{step1_result}

## SEC EDGAR 공식 데이터
{sec_data}

Yahoo Finance와 SEC 데이터의 일치도를 분석하고, 데이터 불일치가 있다면 구체적인 차이점과 원인을 분석하세요.

출력 형식:
- 일치도: [높음/보통/낮음]
- 차이점: [구체적 차이점과 원인]
- 신뢰성 평가: [전체적인 데이터 신뢰성 평가]
"""
        )

        self.reliability_chain = LLMChain(
            llm=self.llm,
            prompt=self.reliability_prompt,
            output_key="reliability_analysis"
        )

        # 2단계: 재무 건전성 재평가 체인
        self.financial_prompt = PromptTemplate(
            input_variables=["step1_result", "sec_data", "reliability_analysis"],
            template="""
SEC 공식 데이터를 바탕으로 재무 건전성을 재평가하세요:

## 1단계 분석 결과
{step1_result}

## SEC EDGAR 공식 데이터
{sec_data}

## 신뢰성 분석
{reliability_analysis}

SEC 공식 데이터를 우선으로 하여 수익성, 성장성, 안전성을 재평가하세요.

출력 형식:
- 수익성: [상/중/하] - 근거: [SEC 데이터 기반]
- 성장성: [상/중/하] - 근거: [SEC 데이터 기반]
- 안전성: [상/중/하] - 근거: [SEC 데이터 기반]
"""
        )

        self.financial_chain = LLMChain(
            llm=self.llm,
            prompt=self.financial_prompt,
            output_key="financial_reassessment"
        )

        # 3단계: 최종 투자 등급 조정 체인
        self.final_prompt = PromptTemplate(
            input_variables=["step1_result", "sec_data", "reliability_analysis", "financial_reassessment"],
            template="""
모든 분석을 종합하여 최종 투자 등급을 조정하세요:

## 1단계 분석 결과
{step1_result}

## SEC EDGAR 공식 데이터
{sec_data}

## 신뢰성 분석
{reliability_analysis}

## 재무 재평가
{financial_reassessment}

SEC 데이터를 반영하여 투자 등급을 재검토하고, 추가 리스크 요인과 규제 이슈를 분석하세요.

{format_instructions}
"""
        )

        self.final_chain = LLMChain(
            llm=self.llm,
            prompt=self.final_prompt,
            output_parser=self.output_parser,
            output_key="final_analysis"
        )

        # SequentialChain 구성
        self.sequential_chain = SequentialChain(
            chains=[self.reliability_chain, self.financial_chain, self.final_chain],
            input_variables=["step1_result", "sec_data", "format_instructions"],
            output_variables=["reliability_analysis", "financial_reassessment", "final_analysis"],
            verbose=True
        )

    def analyze_with_sec_data(self, step1_result: Dict, sec_data: Dict) -> SECAnalysis:
        """SEC 데이터를 포함한 향상된 LLM 분석"""

        try:
            # SequentialChain 실행
            result = self.sequential_chain.invoke({
                "step1_result": step1_result.get('analysis_result', 'N/A'),
                "sec_data": json.dumps(sec_data, indent=2, ensure_ascii=False),
                "format_instructions": self.output_parser.get_format_instructions()
            })

            return result["final_analysis"]
        except Exception as e:
            print(f"SEC 데이터 분석 실패: {str(e)}")
            # 기본값 반환
            return SECAnalysis(
                data_reliability={"일치도": "분석 실패", "차이점": "분석 실패"},
                financial_reassessment={"수익성": "분석 실패", "성장성": "분석 실패", "안전성": "분석 실패"},
                investment_grade_revision="Hold",
                revision_reason="분석 실패",
                target_price_adjustment="분석 실패",
                additional_risks=["분석 실패"],
                regulatory_issues=["분석 실패"],
                confidence_level="낮음"
            )

    def run_analysis(self, step1_result: Dict) -> Dict[str, Any]:
        """SEC 데이터 기반 향상된 분석 실행"""
        print(f"🔍 {self.company_name} ({self.ticker}) SEC 데이터 분석 시작...")

        # SEC 데이터 수집
        sec_data = self.sec_collector.collect_sec_data()

        # 향상된 분석 (SequentialChain 사용)
        enhanced_analysis = self.analyze_with_sec_data(step1_result, sec_data)

        return {
            "step": 2,
            "company_name": self.company_name,
            "ticker": self.ticker,
            "step1_result": step1_result,
            "sec_data": sec_data,
            "enhanced_analysis": enhanced_analysis,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

def main():
    """메인 실행 함수"""
    print("📄 SEC EDGAR 데이터 분석")
    print("="*50)

    # 1단계 결과 (실제로는 step1에서 받아옴)
    step1_result = {
        "analysis_result": "1단계 Yahoo Finance 분석 결과 (예시)"
    }

    # SEC 데이터 분석 실행
    analyzer = SECEnhancedAnalyzer("Apple Inc.", "AAPL", "Apple Inc. SEC filings")
    result = analyzer.run_analysis(step1_result)

    # 구조화된 결과 출력
    analysis = result["enhanced_analysis"]
    print("\n📊 SEC 데이터 분석 결과:")
    print("="*50)
    print(f"수정된 투자 등급: {analysis.investment_grade_revision}")
    print(f"수정 근거: {analysis.revision_reason}")
    print(f"목표가 조정: {analysis.target_price_adjustment}")
    print(f"신뢰도: {analysis.confidence_level}")
    print(f"\n추가 리스크: {', '.join(analysis.additional_risks)}")
    print(f"규제 이슈: {', '.join(analysis.regulatory_issues)}")

if __name__ == "__main__":
    main()

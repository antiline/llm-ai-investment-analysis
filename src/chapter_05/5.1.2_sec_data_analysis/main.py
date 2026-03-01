"""
5.1.2 SEC EDGAR 데이터 수집 및 분석

이 섹션에서는 SEC EDGAR에서 공식 재무 데이터를 수집하고,
SequentialChain을 활용하여 단계별 분석을 수행합니다.

주요 기능:
- SEC EDGAR에서 실시간 재무 데이터 수집
- Yahoo Finance 데이터와의 비교 분석
- SequentialChain을 활용한 단계별 분석
- 구조화된 출력을 통한 신뢰성 있는 분석 결과
"""

import requests
import json
import time
import re
from typing import Dict, Any, Optional, List
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import os
from dotenv import load_dotenv
from datetime import datetime

# 1. SEC 분석 결과 모델 정의
class SECAnalysis(BaseModel):
    financial_health: str = Field(description="재무 건전성 평가")
    growth_potential: str = Field(description="성장 잠재력 평가")
    risk_assessment: str = Field(description="리스크 평가")
    investment_grade: str = Field(description="투자 등급")
    target_price: str = Field(description="목표가")
    key_findings: List[str] = Field(description="주요 발견사항")
    confidence_level: str = Field(description="분석 신뢰도")

# 2. SEC 데이터 수집기
class SECDataCollector:
    def __init__(self):
        self.base_url = "https://www.sec.gov/Archives/edgar/data"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def ticker_to_cik(self, ticker: str) -> str:
        """티커를 CIK로 변환"""
        url = "https://www.sec.gov/files/company_tickers.json"

        try:
            response = requests.get(url, headers=self.headers)
            companies_data = response.json()

            ticker_upper = ticker.upper()
            for cik, company_info in companies_data.items():
                if company_info.get('ticker', '').upper() == ticker_upper:
                    cik_padded = cik.zfill(10)
                    company_name = company_info.get('title', 'Unknown')
                    print(f"✅ {ticker} -> CIK: {cik_padded} ({company_name})")
                    return cik_padded

            raise ValueError(f"티커 {ticker}에 해당하는 CIK를 찾을 수 없습니다.")

        except Exception as e:
            print(f"CIK 변환 실패: {e}")
            return None

    def get_company_info(self, ticker: str) -> Dict[str, Any]:
        """티커에 대한 기업 기본 정보 조회"""
        url = "https://www.sec.gov/files/company_tickers.json"

        try:
            response = requests.get(url, headers=self.headers)
            companies_data = response.json()

            ticker_upper = ticker.upper()
            for cik, company_info in companies_data.items():
                if company_info.get('ticker', '').upper() == ticker_upper:
                    return {
                        'cik': cik.zfill(10),
                        'name': company_info.get('title', 'Unknown'),
                        'ticker': company_info.get('ticker', ticker),
                        'exchange': company_info.get('exchange', 'Unknown')
                    }

            return None

        except Exception as e:
            print(f"기업 정보 조회 실패: {e}")
            return None

    def get_10k_links(self, cik: str, limit: int = 1) -> List[Dict[str, str]]:
        """CIK로 10-K 문서 링크 찾기"""
        submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"

        try:
            print(f"📄 10-K 파일링 정보 조회 중...")
            response = requests.get(submissions_url, headers=self.headers)
            data = response.json()
            filings = data["filings"]["recent"]

            filing_info = []
            for i, form in enumerate(filings["form"]):
                if form == "10-K" and len(filing_info) < limit:
                    accession = filings["accessionNumber"][i].replace("-", "")
                    primary_doc = filings["primaryDocument"][i]
                    filing_date = filings["filingDate"][i]

                    filing_info.append({
                        "accession": accession,
                        "primary_doc": primary_doc,
                        "filing_date": filing_date,
                        "file_url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{primary_doc}"
                    })

            print(f"✅ {len(filing_info)}개의 10-K 파일링을 찾았습니다.")
            return filing_info

        except Exception as e:
            print(f"10-K 링크 조회 실패: {e}")
            return []

    def download_10k_document(self, file_url: str) -> str:
        """10-K 문서 다운로드 및 텍스트 추출"""
        try:
            print(f"📖 10-K 문서 다운로드 중...")
            response = requests.get(file_url, headers=self.headers)
            soup = BeautifulSoup(response.content, "html.parser")

            # 불필요한 태그 제거
            for tag in soup(["script", "style", "table"]):
                tag.extract()

            text = soup.get_text(separator="\n")

            # 공백 정리
            lines = [line.strip() for line in text.splitlines()]
            text = "\n".join([line for line in lines if line])

            # 섹션 헤더를 마크다운 형식으로 변환
            text = re.sub(r"(Item\s+\d+[A]?\.\s+.+)", r"\n# \1\n", text, flags=re.IGNORECASE)
            text = re.sub(r"(PART\s+[IVX]+)", r"\n## \1\n", text, flags=re.IGNORECASE)

            print(f"✅ 10-K 문서 변환 완료 (길이: {len(text)} 문자)")
            return text

        except Exception as e:
            print(f"10-K 문서 다운로드 실패: {e}")
            return ""

# 3. SEC 데이터 분석기
class SECAnalyzer:
    def __init__(self):
        load_dotenv()
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-3.5-turbo",
            temperature=0
        )
        self.output_parser = PydanticOutputParser(pydantic_object=SECAnalysis)
        self.setup_prompts()

    def setup_prompts(self):
        """프롬프트 설정"""
        self.analysis_prompt = PromptTemplate(
            input_variables=["company_name", "ticker", "sec_data"],
            template="""
다음 SEC 10-K 문서를 바탕으로 {company_name} ({ticker})의 투자 분석을 수행하세요.

## SEC 10-K 문서 내용
{sec_data}

다음 항목들을 분석하여 구조화된 결과를 제공하세요:

1. 재무 건전성: 수익성, 유동성, 부채 비율 등을 종합 평가
2. 성장 잠재력: 매출 성장, 시장 확장, 혁신 능력 등을 평가
3. 리스크 평가: 경영 리스크, 시장 리스크, 규제 리스크 등을 분석
4. 투자 등급: Buy, Hold, Sell 중 하나로 평가
5. 목표가: 현재 주가 대비 합리적인 목표가 제시
6. 주요 발견사항: 3-5개의 핵심 발견사항 나열
7. 분석 신뢰도: 높음/보통/낮음으로 평가

{format_instructions}
"""
        )

    def analyze_sec_data(self, company_name: str, ticker: str, sec_data: str) -> SECAnalysis:
        """SEC 데이터 기반 분석 수행"""
        try:
            chain = self.analysis_prompt | self.llm | self.output_parser

            result = chain.invoke({
                "company_name": company_name,
                "ticker": ticker,
                "sec_data": sec_data[:8000],  # 토큰 제한 고려
                "format_instructions": self.output_parser.get_format_instructions()
            })

            return result

        except Exception as e:
            print(f"SEC 데이터 분석 실패: {e}")
            return SECAnalysis(
                financial_health="분석 실패",
                growth_potential="분석 실패",
                risk_assessment="분석 실패",
                investment_grade="Hold",
                target_price="분석 실패",
                key_findings=["분석 실패"],
                confidence_level="낮음"
            )

# 4. 메인 실행 함수
def run_sec_analysis(ticker: str):
    """SEC 데이터 기반 투자 분석 실행"""
    print(f"🔍 {ticker} SEC 데이터 기반 분석 시작...")
    print("="*50)

    # SEC 데이터 수집기 초기화
    sec_collector = SECDataCollector()

    # 1단계: 티커를 CIK로 변환 및 기업 정보 조회
    cik = sec_collector.ticker_to_cik(ticker)
    if not cik:
        print("❌ CIK 변환 실패")
        return

    # 기업 정보 조회
    company_info = sec_collector.get_company_info(ticker)
    if not company_info:
        print("❌ 기업 정보 조회 실패")
        return

    print(f"📊 기업 정보: {company_info['name']} ({company_info['ticker']}) - {company_info['exchange']}")

    # 2단계: 10-K 문서 링크 찾기
    filing_info = sec_collector.get_10k_links(cik, 1)
    if not filing_info:
        print("❌ 10-K 문서 링크를 찾을 수 없습니다.")
        return

    # 3단계: 10-K 문서 다운로드
    sec_data = sec_collector.download_10k_document(filing_info[0]["file_url"])
    if not sec_data:
        print("❌ 10-K 문서 다운로드 실패")
        return

    # 4단계: SEC 데이터 분석
    analyzer = SECAnalyzer()

    # 찾은 CIK를 바탕으로 실제 회사명 사용
    company_name = company_info['name']

    analysis_result = analyzer.analyze_sec_data(company_name, ticker, sec_data)

    # 결과 출력
    print("\n" + "="*50)
    print("SEC 데이터 기반 분석 결과")
    print("="*50)
    print(f"재무 건전성: {analysis_result.financial_health}")
    print(f"성장 잠재력: {analysis_result.growth_potential}")
    print(f"리스크 평가: {analysis_result.risk_assessment}")
    print(f"투자 등급: {analysis_result.investment_grade}")
    print(f"목표가: {analysis_result.target_price}")
    print(f"분석 신뢰도: {analysis_result.confidence_level}")
    print(f"\n주요 발견사항:")
    for i, finding in enumerate(analysis_result.key_findings, 1):
        print(f"  {i}. {finding}")

    return {
        "ticker": ticker,
        "cik": cik,
        "filing_date": filing_info[0]["filing_date"],
        "analysis_result": analysis_result,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

# 실행 예시
if __name__ == "__main__":
    # Apple Inc. 분석 예시
    result = run_sec_analysis("AAPL")

    if result:
        print(f"\n✅ 분석 완료: {result['ticker']} ({result['filing_date']})")

import requests
import re
import json
from bs4 import BeautifulSoup
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

class EnhancedSECDataCollector:
    """개선된 SEC EDGAR 데이터 수집 클래스"""

    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.headers = {
            'User-Agent': 'LLM Book Project (contact@example.com)'
        }
        self.cik = None
        self.company_info = None

    def get_cik_from_ticker(self) -> str:
        """SEC 공식 JSON API를 사용하여 티커로부터 CIK 번호 추출"""
        try:
            # SEC 공식 company_tickers.json 사용
            cik_url = "https://www.sec.gov/files/company_tickers.json"
            resp = requests.get(cik_url, headers=self.headers)
            companies = resp.json()

            for _, info in companies.items():
                if info["ticker"].lower() == self.ticker.lower():
                    self.cik = str(info["cik_str"]).zfill(10)
                    self.company_info = info
                    print(f"✅ {self.ticker} → CIK: {self.cik} ({info['title']})")
                    return self.cik

            raise ValueError(f"티커 {self.ticker}를 찾을 수 없습니다.")

        except Exception as e:
            print(f"❌ CIK 번호 추출 실패: {e}")
            raise

    def get_latest_filings(self, form_type: str = "10-K", limit: int = 3) -> List[Dict]:
        """SEC JSON API를 사용하여 최신 파일링 정보 수집"""
        if not self.cik:
            self.get_cik_from_ticker()

        try:
            # SEC JSON API 사용
            sub_url = f"https://data.sec.gov/submissions/CIK{self.cik}.json"
            resp = requests.get(sub_url, headers=self.headers)
            data = resp.json()

            filings = data["filings"]["recent"]
            latest_filings = []

            for i, form in enumerate(filings["form"]):
                if form == form_type and len(latest_filings) < limit:
                    filing_info = {
                        "accession_number": filings["accessionNumber"][i].replace("-", ""),
                        "primary_document": filings["primaryDocument"][i],
                        "filing_date": filings["filingDate"][i],
                        "form": form,
                        "description": filings["description"][i] if "description" in filings else ""
                    }
                    latest_filings.append(filing_info)

            print(f"📄 {len(latest_filings)}개의 {form_type} 파일링 발견")
            return latest_filings

        except Exception as e:
            print(f"❌ 파일링 정보 수집 실패: {e}")
            return []

    def fetch_10k_markdown(self, filing_info: Dict) -> str:
        """10-K 보고서를 마크다운 형식으로 변환하여 가져오기"""
        try:
            # 문서 URL 구성
            file_url = f"https://www.sec.gov/Archives/edgar/data/{int(self.cik)}/{filing_info['accession_number']}/{filing_info['primary_document']}"

            print(f"📥 문서 다운로드 중: {filing_info['filing_date']}")
            resp = requests.get(file_url, headers=self.headers)
            html_content = resp.content

            # HTML → 텍스트 변환
            soup = BeautifulSoup(html_content, "html.parser")

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

            return text

        except Exception as e:
            print(f"❌ 문서 변환 실패: {e}")
            return ""

    def extract_financial_metrics(self, markdown_text: str) -> Dict[str, Any]:
        """마크다운 텍스트에서 재무 지표 추출"""
        metrics = {
            "revenue": None,
            "net_income": None,
            "total_assets": None,
            "total_liabilities": None,
            "cash_and_equivalents": None,
            "debt": None,
            "key_metrics": {}
        }

        # 다양한 패턴으로 재무 데이터 검색
        patterns = {
            "revenue": [
                r'Total net sales[:\s]*\$?([\d,]+\.?\d*)',
                r'Revenue[:\s]*\$?([\d,]+\.?\d*)',
                r'Net sales[:\s]*\$?([\d,]+\.?\d*)'
            ],
            "net_income": [
                r'Net income[:\s]*\$?([\d,]+\.?\d*)',
                r'Net earnings[:\s]*\$?([\d,]+\.?\d*)'
            ],
            "total_assets": [
                r'Total assets[:\s]*\$?([\d,]+\.?\d*)'
            ],
            "total_liabilities": [
                r'Total liabilities[:\s]*\$?([\d,]+\.?\d*)'
            ],
            "cash_and_equivalents": [
                r'Cash and cash equivalents[:\s]*\$?([\d,]+\.?\d*)',
                r'Cash and equivalents[:\s]*\$?([\d,]+\.?\d*)'
            ],
            "debt": [
                r'Total debt[:\s]*\$?([\d,]+\.?\d*)',
                r'Long-term debt[:\s]*\$?([\d,]+\.?\d*)'
            ]
        }

        for metric, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, markdown_text, re.IGNORECASE)
                if match:
                    metrics[metric] = match.group(1)
                    break

        return metrics

    def collect_comprehensive_sec_data(self) -> Dict[str, Any]:
        """종합적인 SEC 데이터 수집"""
        print(f"🔍 {self.ticker} SEC 데이터 수집 시작...")

        try:
            # CIK 번호 가져오기
            self.get_cik_from_ticker()

            # 최신 10-K 파일링 정보 수집
            filings = self.get_latest_filings("10-K", 2)

            sec_data = {
                "company_info": self.company_info,
                "ticker": self.ticker,
                "cik": self.cik,
                "filings": [],
                "summary": {},
                "collection_timestamp": datetime.now().isoformat()
            }

            for filing in filings:
                print(f"📊 {filing['filing_date']} 파일링 분석 중...")

                # 마크다운 형식으로 문서 가져오기
                markdown_content = self.fetch_10k_markdown(filing)

                if markdown_content:
                    # 재무 지표 추출
                    financial_metrics = self.extract_financial_metrics(markdown_content)

                    filing_data = {
                        "filing_date": filing["filing_date"],
                        "accession_number": filing["accession_number"],
                        "financial_metrics": financial_metrics,
                        "content_preview": markdown_content[:1000] + "..." if len(markdown_content) > 1000 else markdown_content,
                        "content_length": len(markdown_content)
                    }

                    sec_data["filings"].append(filing_data)

                # SEC 요청 제한 준수
                time.sleep(0.1)

            # 데이터 요약 생성
            if sec_data["filings"]:
                latest_filing = sec_data["filings"][0]
                metrics = latest_filing["financial_metrics"]

                sec_data["summary"] = {
                    "latest_filing_date": latest_filing["filing_date"],
                    "filing_count": len(sec_data["filings"]),
                    "latest_revenue": metrics.get("revenue", "N/A"),
                    "latest_net_income": metrics.get("net_income", "N/A"),
                    "latest_total_assets": metrics.get("total_assets", "N/A"),
                    "data_quality": "Good" if len(sec_data["filings"]) > 0 else "Poor",
                    "content_available": any(f["content_length"] > 1000 for f in sec_data["filings"])
                }

            print(f"✅ {self.ticker} SEC 데이터 수집 완료")
            return sec_data

        except Exception as e:
            print(f"❌ SEC 데이터 수집 실패: {e}")
            return {"error": str(e)}

def demonstrate_enhanced_sec_collector():
    """개선된 SEC 데이터 수집기 시연"""
    print("🚀 개선된 SEC 데이터 수집기 시연")
    print("=" * 50)

    # Apple Inc. 데이터 수집
    collector = EnhancedSECDataCollector("AAPL")
    sec_data = collector.collect_comprehensive_sec_data()

    if "error" not in sec_data:
        print("\n📊 수집된 데이터 요약:")
        print(f"회사명: {sec_data['company_info']['title']}")
        print(f"티커: {sec_data['ticker']}")
        print(f"CIK: {sec_data['cik']}")
        print(f"파일링 수: {sec_data['summary']['filing_count']}")
        print(f"최신 파일링: {sec_data['summary']['latest_filing_date']}")
        print(f"최신 매출: {sec_data['summary']['latest_revenue']}")
        print(f"최신 순이익: {sec_data['summary']['latest_net_income']}")

        # 첫 번째 파일링의 재무 지표 출력
        if sec_data["filings"]:
            print("\n📈 재무 지표:")
            metrics = sec_data["filings"][0]["financial_metrics"]
            for key, value in metrics.items():
                if value:
                    print(f"  {key}: {value}")
    else:
        print(f"❌ 오류: {sec_data['error']}")

    return sec_data

if __name__ == "__main__":
    demonstrate_enhanced_sec_collector()

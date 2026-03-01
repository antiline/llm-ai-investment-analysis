import os
import requests
import html2text
from typing import Dict, Any, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from bs4 import BeautifulSoup


class SECAnalysis(BaseModel):
    """SEC 데이터 분석 결과 구조화"""
    data_reliability: Dict[str, str] = Field(description="데이터 신뢰성 평가")
    financial_reassessment: Dict[str, str] = Field(description="재무 건전성 재평가")
    investment_grade_revision: str = Field(description="수정된 투자 등급")
    revision_reason: str = Field(description="수정 근거")
    target_price_adjustment: str = Field(description="목표가 조정")
    additional_risks: List[str] = Field(description="추가 리스크 요인")
    regulatory_issues: List[str] = Field(description="규제 관련 이슈")
    confidence_level: str = Field(description="전체 분석 신뢰도")


class SECDataCollector:
    """SEC EDGAR 데이터 수집 클래스"""

    def __init__(self, company_name: str, ticker: str):
        self.company_name = company_name
        self.ticker = ticker
        self.base_url = "https://www.sec.gov/Archives/edgar/data"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def get_cik_from_ticker(self, ticker: str) -> str:
        """SEC API를 활용하여 티커 심볼을 CIK 번호로 변환"""
        try:
            # SEC의 기업 정보 JSON API 엔드포인트
            url = "https://www.sec.gov/files/company_tickers.json"
            response = requests.get(url, headers=self.headers)
            companies_data = response.json()

            # 티커로 CIK 찾기 (대소문자 무시)
            ticker_upper = ticker.upper()
            for cik, company_info in companies_data.items():
                if company_info.get('ticker', '').upper() == ticker_upper:
                    return cik.zfill(10)  # CIK는 10자리로 패딩

            print(f"⚠️  {ticker}에 대한 CIK를 찾을 수 없습니다.")
            return None
        except Exception as e:
            print(f"❌ CIK 조회 오류: {e}")
            return None

    def get_latest_filing_info(self, form_type: str = "10-K", limit: int = 3) -> List[Dict[str, Any]]:
        """SEC JSON API를 활용한 최신 파일링 정보 수집"""
        try:
            cik = self.get_cik_from_ticker(self.ticker)
            if not cik:
                return []

            # SEC JSON API 엔드포인트
            submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            response = requests.get(submissions_url, headers=self.headers)
            data = response.json()
            filings = data["filings"]["recent"]

            # 지정된 형태의 파일링 찾기
            filing_info = []
            for i, form in enumerate(filings["form"]):
                if form == form_type and len(filing_info) < limit:
                    accession = filings["accessionNumber"][i].replace("-", "")
                    primary_doc = filings["primaryDocument"][i]
                    filing_date = filings["filingDate"][i]

                    filing_info.append({
                        "accession": accession,
                        "primary_doc": primary_doc,
                        "filing_date": filing_date,
                        "file_url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{primary_doc}"
                    })

            return filing_info
        except Exception as e:
            print(f"❌ 파일링 정보 조회 오류: {e}")
            return []

    def fetch_10k_markdown(self, ticker: str) -> str:
        """10-K 보고서를 마크다운 형식으로 변환하여 수집"""
        try:
            # 1. CIK 번호 가져오기
            cik = self.get_cik_from_ticker(ticker)
            if not cik:
                return "CIK 번호를 찾을 수 없습니다."

            # 2. 최신 10-K 파일링 정보 조회
            filing_info = self.get_latest_filing_info("10-K", 1)
            if not filing_info:
                return "10-K 파일링 정보를 찾을 수 없습니다."

            latest_filing = filing_info[0]
            file_url = latest_filing["file_url"]

            # 3. 문서 다운로드
            print(f"📄 SEC 10-K 문서 다운로드 중: {file_url}")
            response = requests.get(file_url, headers=self.headers)
            html_content = response.content

            # 4. HTML 정리
            soup = BeautifulSoup(html_content, "html.parser")

            # 불필요한 태그 제거 (script, style 태그만 제거하고 table은 유지)
            for tag in soup(["script", "style"]):
                tag.extract()

            text = str(soup)
            # 마크다운 텍스트 정리 (빈 줄 제거 및 공백 정리)
            lines = [line.strip() for line in text.splitlines()]
            formated_text = "\n".join([line for line in lines if line])

            # 5. html-to-markdown 라이브러리를 사용하여 HTML을 마크다운으로 변환
            h = html2text.HTML2Text()
            h.ignore_links = False      # 링크 정보 유지
            h.ignore_images = False     # 이미지 정보 유지
            h.body_width = 0            # 줄바꿈 비활성화로 원본 형식 유지

            markdown_text = h.handle(formated_text)
            print(f"✅ 마크다운 변환 완료 (길이: {len(markdown_text)} 문자)")
            return markdown_text

        except Exception as e:
            print(f"❌ 10-K 마크다운 변환 오류: {e}")
            return f"문서 변환 실패: {e}"


class SECAnalyzer:
    """SEC 데이터 분석기"""

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

        # 출력 파서 초기화
        self.output_parser = PydanticOutputParser(pydantic_object=SECAnalysis)

        # 프롬프트 템플릿 설정
        self.analysis_prompt = PromptTemplate(
            input_variables=["company_name", "ticker", "sec_data"],
            template="""
당신은 {company_name} ({ticker}) 전문 투자 애널리스트입니다.

다음 SEC 10-K 마크다운 문서를 바탕으로 종합적인 투자 분석을 수행하세요:

## SEC 10-K 문서 내용
<sec-data>
{sec_data}
</sec-data>

## 분석 요구사항

### 1. 데이터 신뢰성 평가
- SEC 공식 데이터의 신뢰도: [높음/보통/낮음]
- 데이터 품질 평가: [구체적 평가 내용]
- 분석 가능한 정보 범위: [재무/경영/리스크/규제 등]

### 2. 재무 건전성 재평가
- **수익성**: 매출, 순이익, 영업이익률 분석
- **유동성**: 유동비율, 당좌비율, 현금흐름 분석
- **안정성**: 부채비율, 이자보상배율, 자본구조 분석
- **성장성**: 매출성장률, 자산성장률, 수익성장률 분석

### 3. 투자 등급 수정
- **수정된 투자 등급**: [Strong Buy/Buy/Hold/Sell/Strong Sell]
- **수정 근거**: SEC 데이터 기반 구체적 근거
- **목표가 조정**: $XXX (SEC 데이터 반영 후)

### 4. 리스크 요인 분석
- **추가 리스크 요인**: [SEC에서 발견된 새로운 리스크 3-5가지]
- **규제 관련 이슈**: [규제 환경 변화, 법적 리스크 등]
- **경영 리스크**: [경영진 변경, 전략적 리스크 등]

### 5. 신뢰도 평가
- **전체 분석 신뢰도**: [높음/보통/낮음]
- **신뢰도 근거**: [SEC 공식 데이터 활용의 장점]

모든 분석은 SEC 공식 데이터의 신뢰성을 최대한 활용하여 제공하세요.

{format_instructions}
"""
        )

    def run_sec_analysis(self, ticker: str) -> SECAnalysis:
        """SEC 데이터 분석 실행"""
        print(f"📈 {self.company_name} SEC 데이터 분석 시작...")

        try:
            # SEC 데이터 수집
            sec_collector = SECDataCollector(self.company_name, ticker)
            markdown_data = sec_collector.fetch_10k_markdown(ticker)

            if "실패" in markdown_data or "찾을 수 없습니다" in markdown_data:
                print(f"❌ SEC 데이터 수집 실패: {markdown_data}")
                return self._create_default_analysis()

            # LLM 분석
            print("🤖 LLM SEC 데이터 분석 중...")
            chain = self.analysis_prompt | self.llm | self.output_parser

            # 분석 실행 (토큰 제한 고려)
            result = chain.invoke({
                "company_name": self.company_name,
                "ticker": ticker,
                "sec_data": markdown_data[:8000],  # 토큰 제한 고려
                "format_instructions": self.output_parser.get_format_instructions()
            })

            # 결과 출력
            print("📈 SEC 데이터 분석 결과")
            print(f"   투자 등급: {result.investment_grade_revision}")
            print(f"   수정 근거: {result.revision_reason}")
            print(f"   목표가 조정: {result.target_price_adjustment}")
            print(f"   신뢰도: {result.confidence_level}")
            print(f"   추가 리스크: {', '.join(result.additional_risks[:3])}")
            print(f"   규제 이슈: {', '.join(result.regulatory_issues[:3])}")

            return result

        except Exception as e:
            print(f"❌ SEC 분석 오류: {e}")
            return self._create_default_analysis()

    def _create_default_analysis(self) -> SECAnalysis:
        """기본 분석 결과 생성"""
        return SECAnalysis(
            data_reliability={"신뢰도": "분석 실패", "품질": "분석 실패", "범위": "분석 실패"},
            financial_reassessment={"수익성": "분석 실패", "유동성": "분석 실패", "안정성": "분석 실패", "성장성": "분석 실패"},
            investment_grade_revision="Hold",
            revision_reason="SEC 데이터 수집 실패",
            target_price_adjustment="분석 실패",
            additional_risks=["분석 실패"],
            regulatory_issues=["분석 실패"],
            confidence_level="낮음"
        )

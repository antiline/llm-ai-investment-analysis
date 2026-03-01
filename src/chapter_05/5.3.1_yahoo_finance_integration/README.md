# 개선된 SEC 데이터 수집기

이 모듈은 SEC EDGAR에서 공식 재무 데이터를 수집하고 분석하는 개선된 시스템입니다.

## 🚀 주요 개선사항

### 1. **안정적인 CIK 번호 추출**
- **기존**: HTML 파싱으로 CIK 추출 (불안정)
- **개선**: SEC 공식 JSON API (`company_tickers.json`) 사용
- **장점**: 100% 정확한 티커 → CIK 매핑

### 2. **구조화된 파일링 검색**
- **기존**: HTML 파싱으로 파일링 URL 수집
- **개선**: SEC JSON API (`CIK{cik}.json`) 사용
- **장점**: 구조화된 데이터로 정확한 파일링 정보

### 3. **마크다운 형식 문서 변환**
- **기존**: 기본 텍스트 추출
- **개선**: HTML → 마크다운 변환, 섹션 구조화
- **장점**: 읽기 쉽고 구조화된 문서

### 4. **향상된 재무 지표 추출**
- **기존**: 단순한 정규식 패턴
- **개선**: 다양한 패턴과 대안 검색
- **장점**: 더 정확한 재무 데이터 수집

## 📁 파일 구조

```
5.3.1_yahoo_finance_integration/
├── main.py                    # 메인 실행 파일
├── enhanced_sec_collector.py  # 개선된 SEC 데이터 수집기
├── test_enhanced_sec.py       # 테스트 스크립트
├── requirements.txt           # 의존성 패키지
└── README.md                 # 이 파일
```

## 🔧 설치 및 설정

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정
```bash
# .env 파일 생성
OPENAI_API_KEY=your_openai_api_key_here
```

## 📖 사용법

### 1. 기본 SEC 데이터 수집
```python
from enhanced_sec_collector import EnhancedSECDataCollector

# Apple Inc. 데이터 수집
collector = EnhancedSECDataCollector("AAPL")
sec_data = collector.collect_comprehensive_sec_data()

print(f"회사명: {sec_data['company_info']['title']}")
print(f"CIK: {sec_data['cik']}")
print(f"파일링 수: {sec_data['summary']['filing_count']}")
```

### 2. 종합 투자 분석
```python
from main import EnhancedInvestmentAnalyzer

# Apple Inc. 종합 분석
analyzer = EnhancedInvestmentAnalyzer("Apple Inc.", "AAPL")
result = analyzer.run_comprehensive_analysis()

print(result["integrated_analysis"])
```

### 3. 테스트 실행
```bash
python test_enhanced_sec.py
```

## 🔍 주요 기능

### EnhancedSECDataCollector 클래스

#### `get_cik_from_ticker()`
- SEC 공식 JSON API를 사용하여 티커로부터 CIK 번호 추출
- 100% 정확한 매핑 보장

#### `get_latest_filings(form_type, limit)`
- SEC JSON API를 사용하여 최신 파일링 정보 수집
- 구조화된 데이터 반환

#### `fetch_10k_markdown(filing_info)`
- 10-K 보고서를 마크다운 형식으로 변환
- 섹션 헤더 자동 구조화

#### `extract_financial_metrics(markdown_text)`
- 마크다운 텍스트에서 재무 지표 추출
- 다양한 패턴으로 정확도 향상

#### `collect_comprehensive_sec_data()`
- 종합적인 SEC 데이터 수집
- 에러 처리 및 요약 정보 포함

### EnhancedInvestmentAnalyzer 클래스

#### `compare_data_sources(yahoo_data, sec_data)`
- Yahoo Finance와 SEC 데이터 비교 분석
- 데이터 신뢰성 평가

#### `generate_integrated_analysis(yahoo_result, sec_data, comparison)`
- 통합 투자 분석 결과 생성
- SEC 데이터 우선 분석

## 📊 데이터 구조

### SEC 데이터 응답 구조
```json
{
  "company_info": {
    "cik_str": "0000320193",
    "ticker": "AAPL",
    "title": "Apple Inc."
  },
  "ticker": "AAPL",
  "cik": "0000320193",
  "filings": [
    {
      "filing_date": "2023-10-27",
      "accession_number": "000032019323000106",
      "financial_metrics": {
        "revenue": "394,328",
        "net_income": "96,995",
        "total_assets": "352,755"
      },
      "content_preview": "...",
      "content_length": 150000
    }
  ],
  "summary": {
    "latest_filing_date": "2023-10-27",
    "filing_count": 2,
    "latest_revenue": "394,328",
    "data_quality": "Good"
  }
}
```

## ⚠️ 주의사항

### 1. SEC 요청 제한
- SEC는 요청 빈도에 제한을 둡니다
- `time.sleep(0.1)`로 요청 간격 조절
- 대량 요청 시 주의 필요

### 2. User-Agent 설정
- SEC는 적절한 User-Agent를 요구합니다
- 프로덕션 환경에서는 실제 연락처 정보로 수정

### 3. 에러 처리
- 네트워크 오류, API 변경 등에 대비
- 적절한 예외 처리 구현

## 🎯 활용 사례

### 1. 투자 분석 시스템
- SEC 공식 데이터 기반 투자 등급 산정
- Yahoo Finance 데이터와의 비교 분석

### 2. 재무 모니터링
- 정기적인 재무 지표 추적
- 자동화된 재무 보고서 분석

### 3. 리스크 평가
- SEC 파일링 기반 리스크 요인 분석
- 규제 관련 이슈 모니터링

## 🔄 향후 개선 계획

1. **더 많은 재무 지표 추출**
   - 현금흐름표, 자본변동표 등
   - 분기별 데이터 수집

2. **자동화된 데이터 검증**
   - 데이터 일관성 검사
   - 이상치 탐지

3. **실시간 모니터링**
   - 새로운 파일링 자동 감지
   - 알림 시스템 구축

## 📞 지원

문제가 발생하거나 개선 제안이 있으시면 이슈를 등록해 주세요.

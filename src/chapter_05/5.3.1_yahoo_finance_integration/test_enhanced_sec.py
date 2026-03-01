#!/usr/bin/env python3
"""
개선된 SEC 데이터 수집기 테스트 스크립트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_sec_collector import EnhancedSECDataCollector

def test_sec_collector():
    """SEC 데이터 수집기 테스트"""
    print("🧪 SEC 데이터 수집기 테스트")
    print("=" * 50)

    # 테스트할 티커들
    test_tickers = ["AAPL", "GOOGL", "MSFT"]

    for ticker in test_tickers:
        print(f"\n📊 {ticker} 테스트 중...")
        try:
            collector = EnhancedSECDataCollector(ticker)

            # CIK 번호 테스트
            cik = collector.get_cik_from_ticker()
            print(f"✅ CIK 번호: {cik}")

            # 파일링 정보 테스트
            filings = collector.get_latest_filings("10-K", 1)
            if filings:
                print(f"✅ 파일링 발견: {len(filings)}개")
                print(f"   최신 파일링: {filings[0]['filing_date']}")
            else:
                print("❌ 파일링을 찾을 수 없습니다")

        except Exception as e:
            print(f"❌ {ticker} 테스트 실패: {e}")

    print("\n🎉 테스트 완료!")

def test_single_company():
    """단일 회사 상세 테스트"""
    print("\n🔍 Apple Inc. 상세 테스트")
    print("=" * 50)

    try:
        collector = EnhancedSECDataCollector("AAPL")
        sec_data = collector.collect_comprehensive_sec_data()

        if "error" not in sec_data:
            print("✅ 데이터 수집 성공!")
            print(f"회사명: {sec_data['company_info']['title']}")
            print(f"CIK: {sec_data['cik']}")
            print(f"파일링 수: {sec_data['summary']['filing_count']}")
            print(f"최신 파일링: {sec_data['summary']['latest_filing_date']}")

            if sec_data["filings"]:
                metrics = sec_data["filings"][0]["financial_metrics"]
                print("\n📈 재무 지표:")
                for key, value in metrics.items():
                    if value:
                        print(f"  {key}: {value}")
        else:
            print(f"❌ 오류: {sec_data['error']}")

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")

if __name__ == "__main__":
    test_sec_collector()
    test_single_company()

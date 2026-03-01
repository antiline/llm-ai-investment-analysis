# LLM으로 만드는 AI 투자 분석 시스템 - 소스코드

Apple 투자 분석 시스템을 단계적으로 구축하며 LLM 애플리케이션 개발을 학습하는 **37개 예제** (챕터 3~9) 모음입니다.

---

## 빠른 시작

아래 명령어를 순서대로 터미널에 붙여넣으세요.

```bash
# 1. uv 설치 (이미 설치되어 있으면 생략)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 프로젝트 클론
git clone https://github.com/antiline/llm-ai-investment-analysis.git
cd llm-ai-investment-analysis

# 3. 의존성 설치
uv sync

# 4. API 키 설정
cp .env.example .env
# .env 파일을 열어서 OPENAI_API_KEY 값을 입력하세요

# 5. 예제 실행
uv run scripts/runner.py --list          # 전체 예제 목록
uv run scripts/runner.py 3.2.7           # 특정 예제 실행
```

> **요구사항**: Python 3.11+, [OpenAI API 키](https://platform.openai.com/api-keys)

---

## 실행 방법

### 러너로 실행

```bash
uv run scripts/runner.py --list          # 예제 목록
uv run scripts/runner.py 3.2.7           # 특정 예제
```

### 직접 실행

중간 출력 확인이나 대화형 예제(9.2~9.4)는 직접 실행하세요:

```bash
uv run python src/chapter_05/5.1.1_yahoo_finance_analysis/main.py
```

---

## 예제 목록

### Chapter 3. LLM 실습 환경 (3개)

| 섹션 | 제목 | 핵심 내용 |
|---|---|---|
| 3.2.7 | 환경변수 설정 | `.env`, `load_dotenv()`, API 키 보안 관리 |
| 3.3.1 | Raw API Apple 분석 | OpenAI SDK 직접 호출 |
| 3.4.2 | LangChain 구현 | LangChain LCEL 파이프라인 |

### Chapter 4. 프롬프트 엔지니어링 (6개)

| 섹션 | 제목 | 핵심 내용 |
|---|---|---|
| 4.1.1 | Temperature 일관성 | temperature별 응답 변동성 비교 |
| 4.2.5 | 역할 기반 프롬프팅 | system message 전문가 역할 부여 |
| 4.3.3 | Few-shot 구현 | 예시 기반 출력 형식/품질 제어 |
| 4.4.2 | PromptTemplate 활용 | PromptTemplate, 변수 바인딩 |
| 4.5.1 | 구조화된 출력 파서 | PydanticOutputParser JSON 출력 |
| 4.6.1 | 비즈니스 리포트 시스템 | 종합 프롬프트 기법 리포트 생성 |

### Chapter 5. 외부 데이터 연동 (10개)

| 섹션 | 제목 | 핵심 내용 |
|---|---|---|
| 5.1.1 | LLM 한계 분석 | 학습 데이터 컷오프, 할루시네이션 |
| 5.1.1 | Yahoo Finance 분석 | yfinance 실시간 주가/재무 데이터 + LLM |
| 5.1.2 | SEC 데이터 분석 | SEC EDGAR 크롤링, 공식 재무제표 |
| 5.2.1 | SEC EDGAR 분석 | SEC 데이터 기반 심화 분석 |
| 5.2.1 | SequentialChain | 다단계 분석 파이프라인 |
| 5.3.1 | Google 뉴스 분석 | RSS 뉴스 수집 + 감성 분석 |
| 5.3.1 | Yahoo Finance 통합 | 주가 + 뉴스 통합 분석 |
| 5.4.1 | AI 키워드 추천 | LLM 검색 키워드 자동 추천 |
| 5.4.1 | 종합 분석 | 다중 데이터 소스 결합 |
| 5.5.1 | 종합 시스템 | 4단계 파이프라인 통합 |

### Chapter 6. RAG와 벡터 검색 (6개)

| 섹션 | 제목 | 핵심 내용 |
|---|---|---|
| 6.1.1 | RAG 개념 | Retrieval-Augmented Generation 구현 |
| 6.2.1 | 하이브리드 검색 | 벡터 검색 + BM25 결합 |
| 6.3.1 | 기본 RAG 시스템 | FAISS 벡터 스토어 기반 RAG |
| 6.4.1 | 텍스트 분할 | TextSplitter 전략 비교 |
| 6.5.1 | 벡터 검색 예제 | OpenAI Embeddings + 유사도 검색 |
| 6.6.1 | 튜토리얼 RAG 시스템 | 실전 RAG 전체 구현 |

### Chapter 7. 고급 추론 기법 (4개)

| 섹션 | 제목 | 핵심 내용 |
|---|---|---|
| 7.1.1 | Chain-of-Thought | 단계별 추론 분석 |
| 7.2.1 | Self-Consistency | 다중 추론 + 다수결 투표 |
| 7.3.1 | Self-Refine | LLM 자기 평가 + 반복 개선 |
| 7.4.1 | 통합 고급 시스템 | CoT + SC + Self-Refine 결합 |

### Chapter 8. 에이전트 (4개)

| 섹션 | 제목 | 핵심 내용 |
|---|---|---|
| 8.1.1 | ReAct 에이전트 | Reasoning + Acting, 도구 사용 |
| 8.2.1 | LangGraph 워크플로우 | StateGraph 조건부 분기 |
| 8.3.1 | 멀티 에이전트 시스템 | 역할별 에이전트 협업 |
| 8.4.1 | 튜토리얼 에이전트 | 실전 에이전트 전체 구현 |

### Chapter 9. 인터페이스 (4개)

| 섹션 | 제목 | 핵심 내용 |
|---|---|---|
| 9.1.1 | 인터페이스 문제점 | CLI 인터페이스 한계 분석 |
| 9.2.1 | 자연어 인터페이스 | 대화형 자연어 입력 (interactive) |
| 9.3.1 | 스트리밍 인터페이스 | 실시간 토큰 스트리밍 (interactive) |
| 9.4.1 | 통합 시스템 | 최종 통합 인터페이스 (interactive) |

---

## 프로젝트 구조

```
llm-ai-investment-analysis/
├── pyproject.toml          # 의존성
├── uv.lock                 # 버전 잠금
├── .env.example            # API 키 템플릿
├── .env                    # API 키 (gitignored)
├── scripts/
│   └── runner.py           # 통합 러너
└── src/
    ├── chapter_03/         # 3개 예제
    ├── chapter_04/         # 6개 예제
    ├── chapter_05/         # 10개 예제
    ├── chapter_06/         # 6개 예제
    ├── chapter_07/         # 4개 예제
    ├── chapter_08/         # 4개 예제
    └── chapter_09/         # 4개 예제
```

---

## 책 원본 대비 변경사항

이 리포지토리의 소스코드는 **최신 LangChain API에 맞게 업데이트**되었습니다.
책에 수록된 코드는 일부 deprecated된 함수를 사용하고 있어, 현재 버전의 라이브러리에서 경고가 발생하거나 동작하지 않을 수 있습니다.
비즈니스 로직과 프롬프트 내용은 책과 동일합니다.

| 항목 | 책 (원본) | 리포지토리 (최신) |
|---|---|---|
| 의존성 관리 | 예제별 `requirements.txt` | 단일 `pyproject.toml` + `uv sync` |
| LangChain import | `from langchain.chat_models import ChatOpenAI` | `from langchain_openai import ChatOpenAI` |
| 체인 생성 | `LLMChain(llm=llm, prompt=prompt)` | `prompt \| llm` (LCEL) |
| 체인 실행 | `chain.run(key=val)` | `chain.invoke({"key": val}).content` |
| API 키 관리 | `os.environ["OPENAI_API_KEY"] = "your-key"` | `load_dotenv()` |

---

## 트러블슈팅

### API 키 오류

```bash
cat .env
# OPENAI_API_KEY=sk-... 형태로 설정되어 있어야 합니다
```

### 의존성 문제

```bash
rm -rf .venv && uv sync
```

### 대화형 예제 타임아웃

9.2.1, 9.3.1, 9.4.1은 `input()` 입력이 필요하므로 직접 실행하세요:

```bash
uv run python src/chapter_09/9.2.1_natural_language_interface/main.py
```

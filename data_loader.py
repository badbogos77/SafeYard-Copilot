"""
=====================================================================
 data_loader.py  —  SafeYard Copilot 데이터 계층
---------------------------------------------------------------------
 역할
  1) 작업 기록(위험성평가·작업허가·사고/아차사고) CSV 로딩 & 검증
  2) 개발용 합성 데이터 생성 (실데이터 확보 전 파이프라인 검증용)
  3) ML 피처 엔지니어링 (체감온도, 폭염 단계, 고소/화기/밀폐 플래그 …)
  4) 데이터 리터러시 KPI 계산
  5) RAG용 문서 로딩 & '조항(섹션) 단위' 텍스트 분할

 ※ 데이터 한 행(row) = "작업 1건"
    - 입력 피처 : 작업종류, 작업높이, 밀폐여부, 기온·습도, 시각, 경력, 언어장벽, 하청여부
    - 라벨      : risk_level  (위험성평가 등급 0=낮음,1=보통,2=높음)
                  hazard_type (해당 작업의 지배적 재해유형)
                  incident    (사고/아차사고 발생 여부 0/1) → KPI 계산용
=====================================================================
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------
# 0. 경로 & 도메인 상수
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"
SAMPLE_CSV = DATA_DIR / "work_records_sample.csv"

# 조선소 주요 작업 유형 (사내 작업허가서 분류를 단순화)
WORK_TYPES = [
    "용접",
    "가스·플라즈마 절단",
    "도장",
    "족장(비계) 설치·해체",
    "중량물 인양(크레인)",
    "탱크·이중저 내부 작업",
    "의장품 설치",
    "사상(그라인딩)",
]
# 불꽃·고열이 발생하는 '화기작업'
HOT_WORK_TYPES = {"용접", "가스·플라즈마 절단", "사상(그라인딩)"}

# 조선업 6대 재해유형
HAZARD_TYPES = ["추락", "끼임", "화재·폭발", "질식", "온열질환", "낙하·맞음"]

RISK_LABELS = {0: "낮음", 1: "보통", 2: "높음"}
RISK_TEXT_TO_INT = {"낮음": 0, "하": 0, "low": 0, "보통": 1, "중": 1, "medium": 1, "높음": 2, "상": 2, "high": 2}

# UI 표시명 → 언어코드 (TBM 카드·챗봇 출력 언어)
LANGUAGES = {
    "한국어": "ko",
    "Tiếng Việt (베트남어)": "vi",
    "नेपाली (네팔어)": "ne",
    "O‘zbekcha (우즈베크어)": "uz",
    "ภาษาไทย (태국어)": "th",
    "Bahasa Indonesia (인도네시아어)": "id",
    "English (영어)": "en",
}

# 기상청 폭염 영향예보 체감온도 단계(31/33/35/38℃)를 참고한 구간
HEAT_STAGES = ["정상", "관심", "주의", "경고", "위험"]
HEAT_BINS = [-np.inf, 31, 33, 35, 38, np.inf]

EXP_BANDS = ["3개월 미만", "3~12개월", "1~3년", "3년 이상"]
HOUR_BANDS = ["~09시", "09-12시", "12-15시", "15-18시", "18시~"]

# 원본 CSV가 가져야 할 컬럼
REQUIRED_COLUMNS = ["work_type", "height_m", "confined_space", "temp_c",
                    "humidity", "hour", "experience_months"]
OPTIONAL_DEFAULTS = {"language_barrier": 0, "subcontractor": 0, "language": "ko"}
LABEL_COLUMNS = ["risk_level", "hazard_type", "incident"]

# ML 모델에 투입되는 피처 목록 (ai_engine.py 에서 import)
CATEGORICAL_FEATURES = ["work_type", "heat_stage", "exp_band"]
NUMERIC_FEATURES = [
    "height_m", "confined_space", "feels_like", "temp_c", "humidity", "hour",
    "experience_months", "language_barrier", "subcontractor",
    "is_high_work", "is_hot_work", "hot_x_confined", "is_novice", "is_night",
]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


# ---------------------------------------------------------------------
# 1. 피처 엔지니어링
# ---------------------------------------------------------------------
def feels_like_temperature(temp_c, humidity) -> np.ndarray:
    """기상청 여름철 체감온도 산출식 (Stull 습구온도 기반).

    - 기온 20℃ 미만에서는 여름철 공식이 의미가 없으므로 기온을 그대로 사용
    - 조선소 철판 위 작업은 복사열로 실제 체감이 더 높다는 점을 발표 때 함께 언급하면 좋음
    """
    t = np.asarray(temp_c, dtype=float)
    rh = np.clip(np.asarray(humidity, dtype=float), 0, 100)

    # 1) 습구온도(Tw) 근사식 (Stull, 2011)
    tw = (t * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
          + np.arctan(t + rh) - np.arctan(rh - 1.67633)
          + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh) - 4.686035)
    # 2) 체감온도 회귀식
    fl = (-0.2442 + 0.55399 * tw + 0.45535 * t
          - 0.0022 * tw ** 2 + 0.00278 * tw * t + 3.0)
    return np.round(np.where(t >= 20, fl, t), 1)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """원본 컬럼으로부터 ML용 파생 피처를 만든다. (원본은 변경하지 않음)

    도메인 지식 → 피처로 번역하는 단계가 이 프로젝트의 '데이터 리터러시' 핵심:
      * 높이 2m 이상      → 추락 방지 조치 의무 구간 (is_high_work)
      * 화기 × 밀폐공간   → 화재·폭발 + 질식이 겹치는 최악 조합 (hot_x_confined)
      * 경력 3개월 미만   → 신규 투입자 (is_novice)
      * 체감온도 단계     → 폭염 영향예보 구간 (heat_stage)
    """
    out = df.copy()
    out["feels_like"] = feels_like_temperature(out["temp_c"], out["humidity"])
    out["heat_stage"] = pd.cut(out["feels_like"], HEAT_BINS, right=False,
                               labels=HEAT_STAGES).astype(str)
    out["exp_band"] = pd.cut(out["experience_months"], [-np.inf, 3, 12, 36, np.inf],
                             right=False, labels=EXP_BANDS).astype(str)
    out["hour_band"] = pd.cut(out["hour"], [0, 9, 12, 15, 18, 25], right=False,
                              labels=HOUR_BANDS).astype(str)

    out["is_high_work"] = (out["height_m"] >= 2).astype(int)
    out["is_hot_work"] = out["work_type"].isin(HOT_WORK_TYPES).astype(int)
    out["hot_x_confined"] = out["is_hot_work"] * out["confined_space"].astype(int)
    out["is_novice"] = (out["experience_months"] < 3).astype(int)
    out["is_night"] = ((out["hour"] >= 19) | (out["hour"] < 6)).astype(int)
    return out


# ---------------------------------------------------------------------
# 2. 합성 데이터 생성기 (실데이터 대체용)
# ---------------------------------------------------------------------
def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def generate_synthetic_records(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    """조선소 1년치 작업 기록을 '그럴듯하게' 흉내 낸 합성 데이터.

    ⚠️ 실제 통계가 아닙니다. 파이프라인 개발·데모용이며,
       실데이터(KOSHA 재해사례, 사내 위험성평가 기록)로 교체해야 합니다.
    """
    rng = np.random.default_rng(seed)

    # --- 작업 종류 & 날짜/시각 ---
    work_type = rng.choice(WORK_TYPES, size=n,
                           p=[0.22, 0.10, 0.12, 0.10, 0.10, 0.10, 0.14, 0.12])
    dates = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    doy = dates.dayofyear.to_numpy()
    hour_w = np.array([3, 6, 6, 6, 6, 3, 6, 6, 6, 6, 5, 2, 2, 2, 1, 1], dtype=float)  # 07~22시
    hour = rng.choice(np.arange(7, 23), size=n, p=hour_w / hour_w.sum())

    # --- 기상 (계절 + 일변화 + 잡음) : 거제·울산 동구 해안 기후 가정 ---
    seasonal = 14.5 + 11.5 * np.sin(2 * np.pi * (doy - 105) / 365)
    diurnal = 4.0 * np.sin(np.pi * (hour - 9) / 12)
    temp_c = np.round(seasonal + diurnal + rng.normal(0, 2.5, n), 1)
    humidity = np.round(np.clip(60 + 18 * np.sin(2 * np.pi * (doy - 105) / 365)
                                + rng.normal(0, 10, n), 20, 98), 0)

    # --- 작업 높이 ---
    height = np.zeros(n)
    m = work_type == "족장(비계) 설치·해체"
    height[m] = rng.uniform(2, 25, m.sum())
    m = work_type == "의장품 설치"
    height[m] = rng.choice([0, 1], m.sum(), p=[0.4, 0.6]) * rng.uniform(1, 12, m.sum())
    m = np.isin(work_type, ["용접", "가스·플라즈마 절단", "도장", "사상(그라인딩)"])
    height[m] = rng.choice([0, 1], m.sum(), p=[0.55, 0.45]) * rng.uniform(0.5, 15, m.sum())
    m = work_type == "중량물 인양(크레인)"
    height[m] = rng.uniform(0, 3, m.sum())
    m = work_type == "탱크·이중저 내부 작업"
    height[m] = rng.uniform(0, 4, m.sum())
    height = np.round(height, 1)

    # --- 밀폐공간 여부 ---
    p_conf = np.select(
        [work_type == "탱크·이중저 내부 작업",
         np.isin(work_type, ["용접", "도장", "가스·플라즈마 절단"])],
        [1.0, 0.25], default=0.03)
    confined = (rng.random(n) < p_conf).astype(int)

    # --- 인력 특성 : 언어장벽(모국어 ≠ 작업지시 언어), 경력, 하청 ---
    is_foreign = rng.random(n) < 0.55
    lang = rng.choice(["vi", "ne", "uz", "th", "id", "en"], n,
                      p=[0.30, 0.22, 0.18, 0.14, 0.10, 0.06])
    language = np.where(is_foreign, lang, "ko")
    exp = np.where(is_foreign,
                   rng.lognormal(np.log(8), 0.9, n),
                   rng.lognormal(np.log(36), 1.0, n))
    exp = np.clip(exp, 0, 360).round(0)
    subcontractor = (rng.random(n) < np.where(is_foreign, 0.75, 0.45)).astype(int)

    df = pd.DataFrame({
        "record_id": [f"W{i:05d}" for i in range(n)],
        "date": dates.strftime("%Y-%m-%d"),
        "work_type": work_type,
        "height_m": height,
        "confined_space": confined,
        "temp_c": temp_c,
        "humidity": humidity,
        "hour": hour,
        "experience_months": exp,
        "language_barrier": is_foreign.astype(int),
        "language": language,
        "subcontractor": subcontractor,
    })
    feat = engineer_features(df)

    # --- 잠재 위험점수 → 위험성평가 등급 (라벨 1) ---
    wt_base = {"용접": .4, "가스·플라즈마 절단": .5, "도장": .4, "족장(비계) 설치·해체": .8,
               "중량물 인양(크레인)": .7, "탱크·이중저 내부 작업": .9,
               "의장품 설치": .3, "사상(그라인딩)": .2}
    score = (feat["work_type"].map(wt_base)
             + 0.09 * feat["height_m"]
             + 1.0 * feat["confined_space"]
             + 1.0 * feat["hot_x_confined"]
             + 0.18 * np.clip(feat["feels_like"] - 28, 0, None)
             + 0.9 * feat["is_novice"]
             + 0.35 * (feat["experience_months"] < 12)
             + 0.35 * feat["language_barrier"]
             + 0.25 * feat["subcontractor"]
             + 0.45 * feat["is_night"]
             + rng.normal(0, 0.55, n)).to_numpy()
    q1, q2 = np.quantile(score, [0.50, 0.85])
    df["risk_level"] = np.digitize(score, [q1, q2])          # 0/1/2

    # --- 지배적 재해유형 (라벨 2) : 조건별 가중치 → 확률적 샘플링 ---
    hot = feat["is_hot_work"].to_numpy()
    conf = feat["confined_space"].to_numpy()
    h = feat["height_m"].to_numpy()
    fl = feat["feels_like"].to_numpy()
    wt = feat["work_type"].to_numpy()
    sc = np.column_stack([
        0.28 * h,                                                                    # 추락
        np.where(np.isin(wt, ["중량물 인양(크레인)", "의장품 설치", "사상(그라인딩)"]), 1.8, 0.4),  # 끼임
        1.6 * hot + 1.4 * hot * conf + np.where(wt == "도장", 1.0, 0.0),            # 화재·폭발
        2.4 * conf * (1 - hot) + np.where((wt == "도장") & (conf == 1), 1.2, 0.0) + 0.6 * conf,  # 질식
        0.38 * np.clip(fl - 29, 0, None),                                            # 온열질환
        np.where(np.isin(wt, ["중량물 인양(크레인)", "족장(비계) 설치·해체"]), 1.6, 0.3),  # 낙하·맞음
    ])
    p = _softmax(sc * 1.3)
    idx = np.minimum((p.cumsum(axis=1) < rng.random(n)[:, None]).sum(axis=1), len(HAZARD_TYPES) - 1)
    df["hazard_type"] = np.array(HAZARD_TYPES)[idx]

    # --- 사고/아차사고 발생 여부 (KPI 계산용) ---
    p_inc = 1 / (1 + np.exp(-(1.1 * score - 5.0)))
    df["incident"] = (rng.random(n) < p_inc).astype(int)
    return df


# ---------------------------------------------------------------------
# 3. CSV 로딩 & 검증
# ---------------------------------------------------------------------
def read_csv_flexible(source) -> pd.DataFrame:
    """공공데이터는 cp949/euc-kr 인코딩이 흔하므로 순서대로 시도한다."""
    raw = source.read() if hasattr(source, "read") else Path(source).read_bytes()
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("CSV 인코딩을 판별할 수 없습니다 (utf-8 / cp949 / euc-kr 시도).")


def validate_records(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """스키마 검증 + 타입 변환 + 이상치 제거. (정제된 df, 처리 로그) 반환."""
    log: list[str] = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    out = df.copy()
    for col, default in OPTIONAL_DEFAULTS.items():
        if col not in out.columns:
            out[col] = default
            log.append(f"'{col}' 컬럼이 없어 기본값 {default!r}로 채움")

    num_cols = ["height_m", "confined_space", "temp_c", "humidity", "hour",
                "experience_months", "language_barrier", "subcontractor"]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # 위험등급이 '상/중/하' 같은 텍스트로 들어온 경우 숫자로 매핑
    if "risk_level" in out.columns and out["risk_level"].dtype == object:
        out["risk_level"] = out["risk_level"].astype(str).str.strip().str.lower().map(RISK_TEXT_TO_INT)
        log.append("risk_level 텍스트(상/중/하 등)를 0/1/2로 변환")

    before = len(out)
    out = out.dropna(subset=num_cols)
    out = out[(out["height_m"] >= 0) & out["humidity"].between(0, 100)
              & out["hour"].between(0, 23) & (out["experience_months"] >= 0)]
    out = out[out["work_type"].isin(WORK_TYPES)]
    if len(out) < before:
        log.append(f"결측·이상치·미정의 작업유형 {before - len(out)}행 제거")

    for c in ["confined_space", "language_barrier", "subcontractor", "hour"]:
        out[c] = out[c].astype(int)
    log.append(f"최종 {len(out):,}행 사용")
    return out.reset_index(drop=True), log


def load_records(source=None) -> tuple[pd.DataFrame, list[str]]:
    """작업 기록 로딩 → 검증 → 피처 엔지니어링까지 한 번에.

    source 가 None 이면 샘플 CSV를 쓰고, 없으면 합성 데이터를 생성해 저장한다.
    source 는 파일 경로 또는 Streamlit UploadedFile(파일 객체) 모두 가능.
    """
    if source is None:
        if not SAMPLE_CSV.exists():
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            generate_synthetic_records().to_csv(SAMPLE_CSV, index=False, encoding="utf-8-sig")
        raw = read_csv_flexible(SAMPLE_CSV)
    else:
        raw = read_csv_flexible(source)
    clean, log = validate_records(raw)
    return engineer_features(clean), log




# ---------------------------------------------------------------------
# 4. 데이터 리터러시 KPI
# ---------------------------------------------------------------------
def compute_kpis(df: pd.DataFrame) -> dict:
    """README의 'KPI 3종'을 계산한다. incident 라벨이 없으면 계산 불가.

    KPI 1. 폭염 상대위험도(RR) = P(사고 | 체감 33℃↑) / P(사고 | 체감 31℃ 미만)
    KPI 2. 신규자 집중도      = (사고 중 신규자 비중) / (전체 작업 중 신규자 비중)
    KPI 3. 위험 집중도(파레토) = 사고 상위 20% '작업유형×시간대' 셀이 차지하는 사고 비중
    """
    if "incident" not in df.columns or df["incident"].sum() == 0:
        return {"available": False}

    inc = df["incident"]
    hot = inc[df["feels_like"] >= 33].mean()
    normal = inc[df["feels_like"] < 31].mean()

    novice_rec = df["is_novice"].mean()
    novice_inc = df.loc[inc == 1, "is_novice"].mean()

    cells = (df.groupby(["work_type", "hour_band"], observed=True)["incident"]
               .sum().sort_values(ascending=False))
    top_n = max(1, int(np.ceil(len(cells) * 0.2)))

    return {
        "available": True,
        "n_records": len(df),
        "incident_rate": float(inc.mean()),
        "heat_rr": float(hot / normal) if normal > 0 and not np.isnan(hot) else float("nan"),
        "heat_rate_hot": float(hot) if not np.isnan(hot) else float("nan"),
        "heat_rate_normal": float(normal),
        "novice_concentration": float(novice_inc / novice_rec) if novice_rec > 0 else float("nan"),
        "novice_share_records": float(novice_rec),
        "novice_share_incidents": float(novice_inc),
        "pareto_share": float(cells.head(top_n).sum() / cells.sum()),
        "pareto_cells": top_n,
        "total_cells": len(cells),
    }


# ---------------------------------------------------------------------
# 5. RAG 문서 로딩 & 분할
# ---------------------------------------------------------------------
# '## 제42조(…)', '### 2. 용접 작업', 또는 줄 머리의 '제42조' 에서 섹션을 끊는다
SECTION_SPLIT = re.compile(r"(?m)^(?=#{2,3}\s|제\s*\d+\s*조)")
ARTICLE_RE = re.compile(r"제\s*\d+\s*조(?:의\s*\d+)?(?:\s*\([^)]*\))?")


def _read_pdf_bytes(data: bytes) -> str:
    from pypdf import PdfReader  # 선택 의존성
    reader = PdfReader(io.BytesIO(data))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def load_documents(doc_dir: Path = DOCS_DIR,
                   extra_files: Iterable[tuple[str, bytes]] = ()) -> list[Document]:
    """docs 폴더의 .md/.txt/.pdf + 업로드된 (파일명, bytes) 를 Document 리스트로."""
    docs: list[Document] = []
    items: list[tuple[str, bytes]] = []
    if doc_dir.exists():
        for p in sorted(doc_dir.iterdir()):
            if p.suffix.lower() in {".md", ".txt", ".pdf"}:
                items.append((p.name, p.read_bytes()))
    items.extend(extra_files)

    for name, data in items:
        try:
            text = _read_pdf_bytes(data) if name.lower().endswith(".pdf") \
                else data.decode("utf-8", errors="ignore")
        except Exception as e:  # PDF 파싱 실패 등은 건너뛰고 계속
            print(f"[load_documents] {name} 읽기 실패: {e}")
            continue
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": name}))
    return docs


def split_documents(docs: list[Document], chunk_size: int = 500,
                    chunk_overlap: int = 80) -> list[Document]:
    """2단계 분할 전략.

    1단계: 조항/섹션 경계에서 자른다 → 하나의 청크에 두 조항이 섞이지 않게
    2단계: 긴 섹션만 RecursiveCharacterTextSplitter 로 다시 자른다
    각 청크 앞에 '[문서명 > 섹션]' 머리말을 붙여 검색 정확도와 출처 표시를 동시에 확보.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "다. ", ". ", " ", ""],
    )
    chunks: list[Document] = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        doc_title = Path(source).stem.replace("_", " ")
        for section in SECTION_SPLIT.split(doc.page_content):
            section = section.strip()
            # 빈 줄·짧은 머리말, 그리고 '# 제목 + 면책문구' 서문(H1)은 검색 대상에서 제외
            if len(section) < 20 or section.startswith("# "):
                continue
            heading = section.splitlines()[0].lstrip("#").strip()[:60]
            art = ARTICLE_RE.search(heading)
            for piece in splitter.split_text(section):
                chunks.append(Document(
                    page_content=f"[{doc_title} > {heading}]\n{piece}",
                    metadata={"source": source, "section": heading,
                              "article": art.group(0) if art else ""},
                ))
    return chunks


if __name__ == "__main__":
    # 빠른 점검: python data_loader.py
    df, log = load_records()
    print("\n".join(log))
    print(df[MODEL_FEATURES + LABEL_COLUMNS].head())
    print(compute_kpis(df))
    ch = split_documents(load_documents())
    print(f"RAG 청크 {len(ch)}개, 예시:\n{ch[0].page_content[:200]}")

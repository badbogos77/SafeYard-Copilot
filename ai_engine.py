"""
=====================================================================
 ai_engine.py  —  SafeYard Copilot AI 엔진
---------------------------------------------------------------------
 [ML]  Scikit-learn
   - 위험등급 분류기 (낮음/보통/높음)  : RandomForest + OneHot 파이프라인
   - 재해유형 분류기 (6종, Top-3 확률)
   - 로컬 설명 : "이 요인을 안전 기준값으로 바꾸면 위험지수가 얼마나 내려가나" (Occlusion)

 [RAG] LangChain + FAISS
   - 임베딩 2가지 모드
       hashing     : 설치만 하면 바로 동작 (문자 n-gram 해싱, 오프라인)  ← 기본값
       huggingface : 다국어 의미 검색 (BAAI/bge-m3 등) — 외국어 질문 → 한국어 문서 검색 가능
   - LLM 2가지 모드
       API 키 있음 : Claude(Anthropic) 또는 OpenAI 로 근거 인용 답변 + 다국어 번역
       API 키 없음 : 검색된 근거를 그대로 보여주는 '추출형' 폴백 (데모가 멈추지 않게)
=====================================================================
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")
from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data_loader import (BASE_DIR, CATEGORICAL_FEATURES, DOCS_DIR, HAZARD_TYPES,
                         MODEL_FEATURES, NUMERIC_FEATURES, RISK_LABELS,
                         engineer_features, load_documents, split_documents)

try:  # .env 파일의 API 키 자동 로드 (선택)
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

MODEL_PATH = BASE_DIR / "models" / "model_bundle.joblib"
INDEX_DIR = BASE_DIR / "vector_store"


# =====================================================================
# Part 1. 머신러닝
# =====================================================================
@dataclass
class ModelBundle:
    """학습된 모델 2개 + 평가지표 + 피처 중요도를 한 객체로 묶어 저장/로드."""
    risk_model: Pipeline
    hazard_model: Pipeline
    metrics: dict = field(default_factory=dict)
    feature_importance: pd.DataFrame | None = None
    trained_rows: int = 0


def _build_pipeline(n_estimators: int = 300, seed: int = 42) -> Pipeline:
    """범주형은 One-Hot, 수치형은 그대로 → RandomForest.

    RandomForest를 고른 이유
      - 스케일링 불필요, 비선형 상호작용(화기×밀폐 등)을 잘 잡음
      - class_weight='balanced_subsample' 로 소수 클래스(높음) 학습 보정
    """
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("num", "passthrough", NUMERIC_FEATURES),
    ])
    clf = RandomForestClassifier(
        n_estimators=n_estimators, min_samples_leaf=3,
        class_weight="balanced_subsample", n_jobs=-1, random_state=seed,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def _aggregate_importance(pipe: Pipeline) -> pd.DataFrame:
    """One-Hot 으로 쪼개진 중요도를 원래 피처 이름 기준으로 합산."""
    names = pipe.named_steps["pre"].get_feature_names_out()
    imps = pipe.named_steps["clf"].feature_importances_
    agg: dict[str, float] = {}
    for n, v in zip(names, imps):
        kind, raw = n.split("__", 1)
        if kind == "cat":
            raw = next(c for c in CATEGORICAL_FEATURES if raw.startswith(c + "_"))
        agg[raw] = agg.get(raw, 0.0) + float(v)
    return (pd.DataFrame({"feature": list(agg), "importance": list(agg.values())})
              .sort_values("importance", ascending=False).reset_index(drop=True))


def train_models(df: pd.DataFrame, test_size: float = 0.2, n_estimators: int = 300,
                 seed: int = 42) -> ModelBundle:
    """위험등급·재해유형 분류기를 학습하고 홀드아웃 평가지표를 계산한다."""
    data = df.dropna(subset=["risk_level", "hazard_type"]).copy()
    if "feels_like" not in data.columns:
        data = engineer_features(data)
    X = data[MODEL_FEATURES]
    y_risk = data["risk_level"].astype(int)
    y_haz = data["hazard_type"].astype(str)

    X_tr, X_te, yr_tr, yr_te, yh_tr, yh_te = train_test_split(
        X, y_risk, y_haz, test_size=test_size, stratify=y_risk, random_state=seed)

    risk_model = _build_pipeline(n_estimators, seed).fit(X_tr, yr_tr)
    hazard_model = _build_pipeline(n_estimators, seed).fit(X_tr, yh_tr)

    # ---- 위험등급 평가 : '높음'을 놓치지 않는 것(Recall)이 가장 중요 ----
    yr_pred = risk_model.predict(X_te)
    labels = [0, 1, 2]
    # ---- 재해유형 평가 : Top-1 / Top-2 정확도 ----
    proba = hazard_model.predict_proba(X_te)
    classes = hazard_model.classes_
    top2 = classes[np.argsort(proba, axis=1)[:, -2:]]
    top1 = classes[proba.argmax(axis=1)]

    metrics = {
        "risk_accuracy": accuracy_score(yr_te, yr_pred),
        "risk_macro_f1": f1_score(yr_te, yr_pred, average="macro"),
        "risk_high_recall": recall_score(yr_te, yr_pred, labels=[2], average="macro", zero_division=0),
        "risk_confusion": confusion_matrix(yr_te, yr_pred, labels=labels).tolist(),
        "risk_report": classification_report(
            yr_te, yr_pred, labels=labels,
            target_names=[RISK_LABELS[i] for i in labels], output_dict=True, zero_division=0),
        "hazard_top1_acc": float((top1 == yh_te.to_numpy()).mean()),
        "hazard_top2_acc": float(np.mean([t in row for t, row in zip(yh_te.to_numpy(), top2)])),
        "n_train": len(X_tr), "n_test": len(X_te),
    }
    return ModelBundle(risk_model, hazard_model, metrics,
                       _aggregate_importance(risk_model), len(data))


def save_bundle(bundle: ModelBundle, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: Path = MODEL_PATH) -> ModelBundle | None:
    return joblib.load(path) if path.exists() else None


def _risk_index(bundle: ModelBundle, X: pd.DataFrame) -> np.ndarray:
    """확률을 0~100 '위험지수'로 환산: (P보통×1 + P높음×2) / 2 × 100"""
    proba = bundle.risk_model.predict_proba(X)
    weights = bundle.risk_model.classes_.astype(float)
    return proba @ weights / 2 * 100


def predict_risk(bundle: ModelBundle, conditions: dict) -> dict:
    """작업 조건 1건 → 위험등급, 등급별 확률, 위험지수, 재해유형 Top-3."""
    row = engineer_features(pd.DataFrame([conditions]))
    X = row[MODEL_FEATURES]

    r_proba = bundle.risk_model.predict_proba(X)[0]
    r_classes = bundle.risk_model.classes_
    level = int(r_classes[r_proba.argmax()])

    h_proba = bundle.hazard_model.predict_proba(X)[0]
    h_order = np.argsort(h_proba)[::-1]
    hazards = [(str(bundle.hazard_model.classes_[i]), float(h_proba[i])) for i in h_order]

    return {
        "risk_level": level,
        "risk_label": RISK_LABELS[level],
        "risk_proba": {RISK_LABELS[int(c)]: float(p) for c, p in zip(r_classes, r_proba)},
        "risk_index": float(_risk_index(bundle, X)[0]),
        "hazards": hazards,          # [(유형, 확률), …] 내림차순
        "top_hazards": [h for h, _ in hazards[:3]],
        "features": row.iloc[0].to_dict(),
    }


# '안전 기준값' : 각 요인을 이 값으로 바꿨을 때 위험지수가 얼마나 떨어지는지 측정
SAFE_BASELINE = {
    "작업 높이": {"height_m": 0.0},
    "밀폐공간": {"confined_space": 0},
    "체감온도(기온·습도)": {"temp_c": 22.0, "humidity": 55.0},
    "작업 시간대": {"hour": 10},
    "작업자 경력": {"experience_months": 36},
    "언어 장벽": {"language_barrier": 0},
    "하청 소속": {"subcontractor": 0},
    "작업 종류": {"work_type": "의장품 설치"},
}


def explain_local(bundle: ModelBundle, conditions: dict) -> pd.DataFrame:
    """Occlusion 방식의 로컬 설명 (SHAP 없이도 직관적인 '요인별 기여도').

    기여도 = 현재 위험지수 − (해당 요인만 안전 기준값으로 바꾼 위험지수)
    양수가 클수록 '이 요인 때문에 위험해졌다'는 뜻.
    """
    base = pd.DataFrame([conditions])
    variants = [base]
    for changes in SAFE_BASELINE.values():
        v = base.copy()
        for k, val in changes.items():
            v[k] = val
        variants.append(v)
    X = engineer_features(pd.concat(variants, ignore_index=True))[MODEL_FEATURES]
    idx = _risk_index(bundle, X)
    contrib = idx[0] - idx[1:]
    return (pd.DataFrame({"factor": list(SAFE_BASELINE), "contribution": contrib})
              .sort_values("contribution", ascending=False).reset_index(drop=True))


# =====================================================================
# Part 2. RAG — 임베딩 & 벡터스토어
# =====================================================================
class HashingKoreanEmbeddings(Embeddings):
    """설치 즉시 동작하는 오프라인 임베딩.

    한국어는 조사·어미 변화가 많아 '단어' 단위보다 '문자 2~4gram'이 견고하다.
    (예: '밀폐공간에서' ↔ '밀폐공간의' 가 '밀폐','폐공','공간' n-gram 을 공유)
    한계: 의미 유사도는 못 잡음, 외국어 질문 → 한국어 문서 검색 불가
          → 발표 전 huggingface 모드(bge-m3)로 전환 권장
    """

    def __init__(self, n_features: int = 2 ** 14):
        self.vectorizer = HashingVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), n_features=n_features,
            alternate_sign=False, norm="l2")

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self.vectorizer.transform(texts).toarray().astype(np.float32).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


def get_embeddings() -> tuple[Embeddings, str]:
    """환경변수 EMBEDDING_BACKEND=huggingface 이면 다국어 모델, 아니면 해싱."""
    backend = os.getenv("EMBEDDING_BACKEND", "hashing").lower()
    if backend == "huggingface":
        model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        try:
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError:
                from langchain_community.embeddings import HuggingFaceEmbeddings
            emb = HuggingFaceEmbeddings(model_name=model_name,
                                        encode_kwargs={"normalize_embeddings": True})
            return emb, f"huggingface:{model_name}"
        except Exception as e:
            print(f"[get_embeddings] HuggingFace 로딩 실패 → hashing 으로 대체: {e}")
    return HashingKoreanEmbeddings(), "hashing:char2-4gram"


def build_vector_store(extra_files=(), persist: bool = True) -> tuple[FAISS, dict]:
    """문서 로딩 → 조항 단위 분할 → 임베딩 → FAISS 인덱스 생성(+저장)."""
    embeddings, backend = get_embeddings()
    chunks = split_documents(load_documents(DOCS_DIR, extra_files))
    if not chunks:
        raise ValueError("RAG 문서가 없습니다. data/docs 폴더에 .md/.txt/.pdf 를 넣어주세요.")
    store = FAISS.from_documents(chunks, embeddings)
    info = {"backend": backend, "n_chunks": len(chunks),
            "sources": sorted({c.metadata["source"] for c in chunks})}
    if persist and not extra_files:          # 업로드 문서가 섞인 인덱스는 임시로만 사용
        store.save_local(str(INDEX_DIR))
        (INDEX_DIR / "meta.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    return store, info


def load_or_build_vector_store(extra_files=()) -> tuple[FAISS, dict]:
    """저장된 인덱스가 있고 임베딩 백엔드가 같으면 재사용, 아니면 새로 생성."""
    if not extra_files and (INDEX_DIR / "meta.json").exists():
        meta = json.loads((INDEX_DIR / "meta.json").read_text(encoding="utf-8"))
        embeddings, backend = get_embeddings()
        if meta.get("backend") == backend:   # 백엔드가 바뀌면 벡터 차원이 달라 재생성 필요
            store = FAISS.load_local(str(INDEX_DIR), embeddings,
                                     allow_dangerous_deserialization=True)  # 내가 만든 파일만 로드
            return store, meta
    return build_vector_store(extra_files)


def default_min_score(store: FAISS) -> float:
    """임베딩 백엔드별 '근거 없음' 판정 임계값."""
    return 0.07 if isinstance(store.embedding_function, HashingKoreanEmbeddings) else 0.35


def search(store: FAISS, query: str, k: int = 4) -> list[tuple[Document, float]]:
    """유사도 검색. FAISS(L2²) 거리를 정규화 벡터 기준 코사인 유사도로 변환해 반환."""
    results = store.similarity_search_with_score(query, k=k)
    return [(doc, float(1 - dist / 2)) for doc, dist in results]


# =====================================================================
# Part 3. RAG — LLM 생성
# =====================================================================
def get_llm():
    """API 키가 있는 LLM 을 반환. 없으면 None (추출형 폴백 사용)."""
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=os.getenv("LLM_MODEL", "claude-haiku-5-5"),
                                 temperature=0.1, max_tokens=1500)
        except ImportError:
            print("[get_llm] pip install langchain-anthropic 필요")
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0.1)
        except ImportError:
            print("[get_llm] pip install langchain-openai 필요")
    return None


SYSTEM_PROMPT = """너는 부울경 조선소 현장의 안전보건 전문가 'SafeYard Copilot'이다.
[규칙]
1. 반드시 아래 [근거 문서]에 있는 내용만 사용한다. 근거에 없으면 추측하지 말고
   "제공된 문서에서 근거를 찾지 못했습니다. 안전관리자에게 확인하세요."라고 답한다.
2. 각 문장 끝에 근거 번호를 [1], [2] 형식으로 붙인다.
3. 출력 언어: {language}. 한국어가 아니면 각 문장 바로 아래 줄에 '(KO) 한국어 원문'을 병기한다.
   (작업반장이 함께 확인할 수 있도록)
4. 짧고 명확한 명령형 문장. 높이·농도·시간 같은 숫자는 근거 그대로 정확히 쓴다.
5. 질식·추락·화상 등 생명이 위급한 상황이면 첫 줄에 '즉시 작업 중지 → 사내 비상연락/119'를 안내한다."""

TBM_INSTRUCTION = """아래 작업 조건과 AI 위험 예측 결과로 '작업 전 안전점검(TBM) 카드'를 작성하라.
형식 (마크다운):
### ⚠️ 오늘의 핵심 위험 (3개)
### ✅ 작업 전 필수 확인 (5개 체크리스트, '- [ ]' 형식)
### 🚫 절대 금지 행동 (3개)
### 🆘 비상 시 행동
모든 항목에 근거 번호를 달고, 근거 문서에 없는 내용은 쓰지 마라.

[작업 조건]
{conditions}

[AI 예측]
위험등급: {risk_label} (위험지수 {risk_index:.0f}/100)
예상 재해유형 Top-3: {hazards}"""


def _format_context(hits: list[tuple[Document, float]]) -> str:
    return "\n\n".join(f"[{i}] (출처: {d.metadata.get('source')} / {d.metadata.get('section')})\n{d.page_content}"
                       for i, (d, _) in enumerate(hits, 1))


def _sources(hits: list[tuple[Document, float]]) -> list[dict]:
    return [{"no": i, "source": d.metadata.get("source"), "section": d.metadata.get("section"),
             "score": round(s, 3), "text": d.page_content} for i, (d, s) in enumerate(hits, 1)]


def _invoke(llm, system: str, user: str) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    content = resp.content
    if isinstance(content, list):  # 일부 모델은 블록 리스트로 반환
        content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return content


def _extractive_answer(hits: list[tuple[Document, float]], header: str) -> str:
    """LLM 없이 동작하는 폴백: 검색된 근거의 핵심 줄을 출처와 함께 나열."""
    lines = [header, ""]
    for i, (d, _) in enumerate(hits, 1):
        body = [l.strip() for l in d.page_content.splitlines()[1:]
                if l.strip() and not l.lstrip().startswith("#")]
        snippet = " ".join(body)[:220]
        lines.append(f"**[{i}] {d.metadata.get('section')}** — {snippet}…")
    lines.append("\n> ℹ️ LLM 미연결 모드: 검색 근거만 표시합니다. `.env`에 API 키를 넣으면 "
                 "근거 기반 요약·다국어 번역 답변이 생성됩니다.")
    return "\n\n".join(lines)


def answer_question(store: FAISS, question: str, language: str = "한국어",
                    prediction: dict | None = None, k: int = 4,
                    min_score: float | None = None, llm=None) -> dict:
    """RAG 질의응답. prediction 이 있으면 예측 재해유형을 검색어에 보강한다.

    min_score(최소 코사인 유사도) 기본값은 임베딩 방식에 따라 자동 설정:
      hashing(문자 n-gram)은 점수 분포가 낮아 0.07, 의미 임베딩(bge-m3 등)은 0.35
      → 실제 문서로 교체한 뒤 '관련 질문 / 무관한 질문' 점수를 찍어 보고 재조정할 것
    """
    if min_score is None:
        min_score = default_min_score(store)
    query = question
    if prediction:
        query += " " + " ".join(prediction["top_hazards"][:2])
    hits = search(store, query, k=k)

    # 가드레일: 가장 유사한 근거조차 너무 멀면 답변 거부 (환각 방지)
    if not hits or hits[0][1] < min_score:
        return {"answer": "제공된 문서에서 근거를 찾지 못했습니다. 안전관리자에게 확인하세요.",
                "sources": _sources(hits), "mode": "no-evidence"}

    llm = llm if llm is not None else get_llm()
    if llm is None:
        return {"answer": _extractive_answer(hits, f"**질문:** {question}\n\n**관련 근거:**"),
                "sources": _sources(hits), "mode": "extractive"}

    pred_txt = ""
    if prediction:
        pred_txt = (f"\n\n[현재 작업 AI 예측] 위험등급 {prediction['risk_label']}, "
                    f"예상 재해유형 {', '.join(prediction['top_hazards'])}")
    user = f"[근거 문서]\n{_format_context(hits)}{pred_txt}\n\n[질문]\n{question}"
    try:
        answer = _invoke(llm, SYSTEM_PROMPT.format(language=language), user)
        return {"answer": answer, "sources": _sources(hits), "mode": "llm"}
    except Exception as e:
        return {"answer": _extractive_answer(hits, f"⚠️ LLM 호출 실패({e}) → 근거만 표시"),
                "sources": _sources(hits), "mode": "extractive"}


def build_tbm_query(conditions: dict, prediction: dict) -> str:
    """예측 결과 + 작업 조건 → RAG 검색어 생성 (조건별 키워드 보강)."""
    f = prediction["features"]
    parts = [conditions["work_type"], *prediction["top_hazards"][:2], "안전수칙"]
    if f["confined_space"]:
        parts += ["밀폐공간", "산소농도", "환기"]
    if f["is_hot_work"]:
        parts += ["화기작업", "화재감시자"]
    if f["is_high_work"]:
        parts += ["추락", "안전대", "작업발판"]
    if f["feels_like"] >= 31:
        parts += ["폭염", "휴식", "온열질환"]
    return " ".join(parts)


def generate_tbm_card(store: FAISS, conditions: dict, prediction: dict,
                      language: str = "한국어", k: int = 6, llm=None) -> dict:
    """예측 결과를 근거 문서와 결합해 작업 전 안전점검(TBM) 카드를 생성."""
    hits = search(store, build_tbm_query(conditions, prediction), k=k)
    llm = llm if llm is not None else get_llm()
    f = prediction["features"]
    cond_txt = (f"- 작업: {conditions['work_type']}\n- 높이: {f['height_m']}m\n"
                f"- 밀폐공간: {'예' if f['confined_space'] else '아니오'}\n"
                f"- 체감온도: {f['feels_like']}℃ ({f['heat_stage']})\n"
                f"- 작업자 경력: {int(f['experience_months'])}개월\n- 출력 언어: {language}")

    if llm is not None:
        user = TBM_INSTRUCTION.format(
            conditions=cond_txt, risk_label=prediction["risk_label"],
            risk_index=prediction["risk_index"],
            hazards=", ".join(f"{h}({p:.0%})" for h, p in prediction["hazards"][:3]))
        try:
            card = _invoke(llm, SYSTEM_PROMPT.format(language=language),
                           f"[근거 문서]\n{_format_context(hits)}\n\n{user}")
            return {"card": card, "sources": _sources(hits), "mode": "llm"}
        except Exception as e:
            print(f"[generate_tbm_card] LLM 실패 → 템플릿 폴백: {e}")

    # ---- 폴백: 근거 문서의 '- ' 항목을 뽑아 템플릿 카드 구성 (한국어) ----
    checklist = []
    for i, (d, _) in enumerate(hits, 1):           # 근거 다양성을 위해 문서당 최대 2개
        picked = 0
        for line in d.page_content.splitlines():
            line = line.strip()
            if line.startswith(("- ", "· ")) and picked < 2 and len(checklist) < 6:
                checklist.append(f"- [ ] {line[2:].strip()} [{i}]")
                picked += 1
    hz = "\n".join(f"- **{h}** (예측 확률 {p:.0%})" for h, p in prediction["hazards"][:3])
    card = (f"### ⚠️ 오늘의 핵심 위험\n{hz}\n\n"
            f"### ✅ 작업 전 필수 확인\n" + "\n".join(checklist) + "\n\n"
            f"### 🆘 비상 시 행동\n- 이상 징후 시 즉시 작업 중지 → 대피 → 사내 비상연락/119\n\n"
            f"> ℹ️ LLM 미연결 템플릿 카드입니다(한국어). API 키 연결 시 {language} 번역·요약 카드가 생성됩니다.")
    return {"card": card, "sources": _sources(hits), "mode": "template"}


if __name__ == "__main__":
    # 빠른 점검: python ai_engine.py
    from data_loader import load_records
    df, _ = load_records()
    b = train_models(df)
    save_bundle(b)
    print({k: v for k, v in b.metrics.items() if not isinstance(v, (list, dict))})
    cond = {"work_type": "탱크·이중저 내부 작업", "height_m": 1.5, "confined_space": 1,
            "temp_c": 32, "humidity": 70, "hour": 14, "experience_months": 2,
            "language_barrier": 1, "subcontractor": 1}
    p = predict_risk(b, cond)
    print(p["risk_label"], round(p["risk_index"]), p["top_hazards"])
    print(explain_local(b, cond).head(4))
    store, info = load_or_build_vector_store()
    print(info)
    print(generate_tbm_card(store, cond, p)["card"])

"""
=====================================================================
 app.py  —  SafeYard Copilot Streamlit UI
---------------------------------------------------------------------
 실행:  streamlit run app.py

 화면 구성
  [사이드바]  데이터 소스(샘플/CSV 업로드) · RAG 문서 업로드 · 오늘의 작업 조건 입력 · 출력 언어
  [Tab 1]    📊 데이터 분석 대시보드  — KPI 3종 + Plotly 시각화
  [Tab 2]    🎯 ML 위험 예측          — 위험지수 게이지, 재해유형 Top-3, 요인 기여도, TBM 카드 생성
  [Tab 3]    💬 RAG 안전 챗봇         — 근거 조항 인용 + 다국어 답변
=====================================================================
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ai_engine import (answer_question, explain_local, generate_tbm_card, get_llm,
                       load_or_build_vector_store, predict_risk, train_models)
from data_loader import (EXP_BANDS, HEAT_STAGES, HOUR_BANDS, LANGUAGES, RISK_LABELS,
                         WORK_TYPES, compute_kpis, load_records)

st.set_page_config(page_title="SafeYard Copilot", page_icon="🛠️", layout="wide")

RISK_COLORS = {"낮음": "#2e7d32", "보통": "#f9a825", "높음": "#c62828"}


# =====================================================================
# 캐시 계층 : 같은 입력이면 재계산하지 않음 (Streamlit은 위젯 조작마다 스크립트 전체 재실행)
#   st.cache_data     → DataFrame 같은 '값'
#   st.cache_resource → 모델·벡터스토어 같은 '무거운 객체'
# =====================================================================
@st.cache_data(show_spinner="작업 기록을 불러오는 중…")
def cached_records(file_bytes: bytes | None):
    import io
    return load_records(io.BytesIO(file_bytes) if file_bytes else None)


@st.cache_resource(show_spinner="ML 모델 학습 중… (최초 1회)")
def cached_models(data_key: str, _df: pd.DataFrame, n_estimators: int):
    # 인자 앞의 '_' 는 해시 대상에서 제외하라는 Streamlit 규칙 → data_key 로 캐시 구분
    return train_models(_df, n_estimators=n_estimators)


@st.cache_resource(show_spinner="RAG 벡터 인덱스 구축 중…")
def cached_store(extra_files: tuple[tuple[str, bytes], ...]):
    return load_or_build_vector_store(extra_files)


@st.cache_resource
def cached_llm():
    return get_llm()


# =====================================================================
# 사이드바
# =====================================================================
with st.sidebar:
    st.title("🛠️ SafeYard Copilot")
    st.caption("부울경 조선소 외국인 근로자 맞춤 · 근거 있는 다국어 TBM")

    st.subheader("① 데이터")
    data_mode = st.radio("작업 기록 데이터", ["샘플(합성) 데이터", "CSV 업로드"], horizontal=True)
    uploaded_csv = None
    if data_mode == "CSV 업로드":
        uploaded_csv = st.file_uploader(
            "작업 기록 CSV", type=["csv"],
            help="필수 컬럼: work_type, height_m, confined_space, temp_c, humidity, hour, "
                 "experience_months / 학습용 라벨: risk_level, hazard_type, incident")
    n_estimators = st.slider("RandomForest 트리 개수", 100, 600, 300, 50)

    st.subheader("② RAG 문서")
    uploaded_docs = st.file_uploader("법령·매뉴얼 추가 (md/txt/pdf)", type=["md", "txt", "pdf"],
                                     accept_multiple_files=True)
    top_k = st.slider("검색 근거 개수 (Top-k)", 2, 8, 4)

    st.subheader("③ 오늘의 작업 조건")
    work_type = st.selectbox("작업 종류", WORK_TYPES, index=WORK_TYPES.index("탱크·이중저 내부 작업"))
    height_m = st.slider("작업 높이 (m)", 0.0, 30.0, 1.5, 0.5)
    confined = st.toggle("밀폐공간 작업", value=True)
    temp_c = st.slider("기온 (℃)", -5.0, 40.0, 31.0, 0.5)
    humidity = st.slider("상대습도 (%)", 20, 100, 70, 5)
    hour = st.slider("작업 시각 (시)", 6, 23, 14)
    experience = st.slider("작업자 경력 (개월)", 0, 240, 2)
    language_barrier = st.toggle("언어 장벽 (모국어 ≠ 작업지시 언어)", value=True)
    subcontractor = st.toggle("협력사(하청) 소속", value=True)

    st.subheader("④ 출력 언어")
    language = st.selectbox("TBM 카드·챗봇 언어", list(LANGUAGES), index=1)

conditions = {
    "work_type": work_type, "height_m": height_m, "confined_space": int(confined),
    "temp_c": temp_c, "humidity": humidity, "hour": hour,
    "experience_months": experience, "language_barrier": int(language_barrier),
    "subcontractor": int(subcontractor),
}

# =====================================================================
# 데이터 · 모델 · 벡터스토어 준비
# =====================================================================
try:
    csv_bytes = uploaded_csv.getvalue() if uploaded_csv else None
    df, load_log = cached_records(csv_bytes)
except Exception as e:
    st.error(f"데이터 로딩 실패: {e}")
    st.stop()

has_labels = {"risk_level", "hazard_type"}.issubset(df.columns)
bundle = None
if has_labels:
    data_key = f"{len(df)}-{pd.util.hash_pandas_object(df, index=False).sum()}"
    bundle = cached_models(data_key, df, n_estimators)

extra = tuple((f.name, f.getvalue()) for f in (uploaded_docs or []))
try:
    store, store_info = cached_store(extra)
except Exception as e:
    store, store_info = None, {"error": str(e)}
llm = cached_llm()

st.title("SafeYard Copilot")
c1, c2, c3 = st.columns(3)
c1.info(f"📁 데이터: **{len(df):,}건** {'(합성 데모)' if uploaded_csv is None else '(업로드)'}")
c2.info(f"📚 RAG: **{store_info.get('n_chunks', 0)}개 청크** · `{store_info.get('backend', '-')}`")
c3.info(f"🤖 LLM: **{'연결됨' if llm else '미연결 (추출형 모드)'}**")
if uploaded_csv is None:
    st.caption("⚠️ 샘플 데이터는 파이프라인 검증용 **합성 데이터**입니다. 수치를 실제 통계로 해석하지 마세요.")

tab1, tab2, tab3 = st.tabs(["📊 데이터 분석 대시보드", "🎯 ML 위험 예측 · TBM", "💬 RAG 안전 챗봇"])

# =====================================================================
# Tab 1. 데이터 분석 대시보드
# =====================================================================
with tab1:
    kpi = compute_kpis(df)
    if not kpi["available"]:
        st.warning("incident(사고 여부) 컬럼이 없어 KPI를 계산할 수 없습니다.")
    else:
        k1, k2, k3 = st.columns(3)
        k1.metric("KPI 1 · 폭염 상대위험도", f"{kpi['heat_rr']:.1f}배",
                  help="체감 33℃↑ 사고율 ÷ 체감 31℃ 미만 사고율")
        k1.caption(f"33℃↑ {kpi['heat_rate_hot']:.1%} vs 31℃ 미만 {kpi['heat_rate_normal']:.1%}")
        k2.metric("KPI 2 · 신규자 사고 집중도", f"{kpi['novice_concentration']:.1f}배",
                  help="(사고 중 경력 3개월 미만 비중) ÷ (전체 작업 중 3개월 미만 비중)")
        k2.caption(f"작업 비중 {kpi['novice_share_records']:.1%} → 사고 비중 {kpi['novice_share_incidents']:.1%}")
        k3.metric("KPI 3 · 위험 집중도(파레토)", f"{kpi['pareto_share']:.0%}",
                  help="사고 상위 20% '작업유형×시간대' 셀이 전체 사고에서 차지하는 비중")
        k3.caption(f"상위 {kpi['pareto_cells']}개 / 전체 {kpi['total_cells']}개 셀")

        st.divider()
        left, right = st.columns(2)

        # (1) 체감온도 단계별 사고율
        heat = (df.groupby("heat_stage")["incident"].agg(["mean", "count"])
                  .reindex(HEAT_STAGES).dropna().reset_index())
        fig = px.bar(heat, x="heat_stage", y="mean", text="count",
                     labels={"heat_stage": "체감온도 단계", "mean": "사고율"},
                     title="체감온도 단계별 사고율 (막대 위 숫자 = 작업 건수)")
        fig.update_yaxes(tickformat=".0%")
        left.plotly_chart(fig, width="stretch")

        # (2) 경력 구간별 사고율
        exp = (df.groupby("exp_band")["incident"].mean().reindex(EXP_BANDS).reset_index())
        fig = px.bar(exp, x="exp_band", y="incident",
                     labels={"exp_band": "경력", "incident": "사고율"}, title="경력 구간별 사고율")
        fig.update_yaxes(tickformat=".0%")
        right.plotly_chart(fig, width="stretch")

        # (3) 작업유형 × 시간대 히트맵
        pivot = (df.pivot_table(index="work_type", columns="hour_band", values="incident",
                                aggfunc="mean", observed=False).reindex(columns=HOUR_BANDS))
        fig = px.imshow(pivot, text_auto=".0%", color_continuous_scale="Reds", aspect="auto",
                        labels={"x": "시간대", "y": "작업 종류", "color": "사고율"},
                        title="작업유형 × 시간대 사고율 히트맵 (TBM 집중 관리 구간 찾기)")
        st.plotly_chart(fig, width="stretch")

        # (4) 위험등급 분포
        if "risk_level" in df.columns:
            rl = df["risk_level"].map(RISK_LABELS).value_counts().reindex(list(RISK_COLORS)).reset_index()
            rl.columns = ["등급", "건수"]
            fig = px.pie(rl, names="등급", values="건수", hole=0.5, color="등급",
                         color_discrete_map=RISK_COLORS, title="위험성평가 등급 분포")
            left.plotly_chart(fig, width="stretch")

        # (5) 재해유형 분포
        if "hazard_type" in df.columns:
            hz = df["hazard_type"].value_counts().reset_index()
            hz.columns = ["재해유형", "건수"]
            fig = px.bar(hz, x="건수", y="재해유형", orientation="h", title="지배적 재해유형 분포")
            right.plotly_chart(fig, width="stretch")

    with st.expander("🔎 데이터 미리보기 · 전처리 로그"):
        st.write(" / ".join(load_log))
        st.dataframe(df.head(200), width="stretch")
        st.download_button("전처리된 데이터 CSV 다운로드",
                           df.to_csv(index=False).encode("utf-8-sig"), "work_records_clean.csv")

# =====================================================================
# Tab 2. ML 위험 예측 + TBM 카드
# =====================================================================
with tab2:
    if bundle is None:
        st.warning("risk_level·hazard_type 라벨이 있는 데이터가 필요합니다.")
    else:
        pred = predict_risk(bundle, conditions)
        f = pred["features"]

        g, h = st.columns([1, 1])
        # (1) 위험지수 게이지
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=pred["risk_index"],
            title={"text": f"위험지수 · 등급 <b>{pred['risk_label']}</b>"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": RISK_COLORS[pred["risk_label"]]},
                   "steps": [{"range": [0, 33], "color": "#e8f5e9"},
                             {"range": [33, 66], "color": "#fff8e1"},
                             {"range": [66, 100], "color": "#ffebee"}]}))
        gauge.update_layout(height=300, margin=dict(t=60, b=10))
        g.plotly_chart(gauge, width="stretch")
        g.caption(f"체감온도 **{f['feels_like']}℃** ({f['heat_stage']}) · "
                  f"고소 {'O' if f['is_high_work'] else 'X'} · 화기 {'O' if f['is_hot_work'] else 'X'} · "
                  f"밀폐 {'O' if f['confined_space'] else 'X'} · 신규자 {'O' if f['is_novice'] else 'X'}")

        # (2) 재해유형 Top 확률
        hz = pd.DataFrame(pred["hazards"], columns=["재해유형", "확률"])
        fig = px.bar(hz, x="확률", y="재해유형", orientation="h", title="예상 재해유형 확률")
        fig.update_layout(height=300, yaxis={"categoryorder": "total ascending"}, margin=dict(t=50, b=10))
        fig.update_xaxes(tickformat=".0%")
        h.plotly_chart(fig, width="stretch")

        if pred["risk_label"] == "높음":
            st.error(f"🚨 고위험 작업입니다. 예상 재해: **{', '.join(pred['top_hazards'])}** — TBM 카드를 생성해 작업 전 공유하세요.")

        # (3) 요인별 기여도 (Occlusion 설명)
        exp_df = explain_local(bundle, conditions)
        fig = px.bar(exp_df, x="contribution", y="factor", orientation="h",
                     color=exp_df["contribution"] > 0,
                     color_discrete_map={True: "#c62828", False: "#90a4ae"},
                     labels={"contribution": "위험지수 기여도 (점)", "factor": ""},
                     title="무엇이 위험을 올렸나? (해당 요인만 안전 기준값으로 바꿨을 때 감소폭)")
        fig.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")

        # (4) TBM 카드 생성 (ML → RAG 연결 지점)
        st.subheader(f"📋 작업 전 안전점검(TBM) 카드 · {language}")
        if store is None:
            st.error(f"RAG 인덱스가 없습니다: {store_info.get('error')}")
        elif st.button("TBM 카드 생성", type="primary"):
            with st.spinner("근거 조항 검색 → 카드 생성 중…"):
                tbm = generate_tbm_card(store, conditions, pred, language, llm=llm)
                tbm["key"] = (str(conditions), language)   # 어떤 조건으로 만든 카드인지 기록
                st.session_state["tbm"] = tbm
        # 사이드바 조건이 바뀌면 이전 카드는 숨긴다 (다른 작업의 카드를 보여주는 실수 방지)
        if st.session_state.get("tbm", {}).get("key") == (str(conditions), language):
            tbm = st.session_state["tbm"]
            st.markdown(tbm["card"])
            with st.expander(f"📎 근거 문서 {len(tbm['sources'])}건 (모드: {tbm['mode']})"):
                for s in tbm["sources"]:
                    st.markdown(f"**[{s['no']}] {s['source']} › {s['section']}** (유사도 {s['score']})")
                    st.caption(s["text"])
            st.download_button("TBM 카드 다운로드 (.md)", tbm["card"].encode("utf-8"), "tbm_card.md")

        # (5) 모델 성능
        with st.expander("📈 모델 성능 (홀드아웃 20%)"):
            m = bundle.metrics
            a, b_, c = st.columns(3)
            a.metric("고위험 Recall", f"{m['risk_high_recall']:.1%}", help="실제 '높음'을 놓치지 않은 비율 — 안전 도메인 핵심 지표")
            b_.metric("위험등급 Macro-F1", f"{m['risk_macro_f1']:.3f}")
            c.metric("재해유형 Top-2 정확도", f"{m['hazard_top2_acc']:.1%}")
            cm = pd.DataFrame(m["risk_confusion"], index=[f"실제 {v}" for v in RISK_LABELS.values()],
                              columns=[f"예측 {v}" for v in RISK_LABELS.values()])
            st.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                      title="위험등급 혼동행렬"), width="stretch")
            st.plotly_chart(px.bar(bundle.feature_importance.head(10), x="importance", y="feature",
                                   orientation="h", title="피처 중요도 Top 10 (RandomForest)")
                            .update_layout(yaxis={"categoryorder": "total ascending"}), width="stretch")

# =====================================================================
# Tab 3. RAG 안전 챗봇
# =====================================================================
with tab3:
    st.caption("법령·매뉴얼에 **근거가 있는 내용만** 답합니다. 근거가 없으면 '찾지 못했습니다'라고 답합니다.")
    use_ctx = st.toggle("현재 작업 예측 결과를 질문에 반영", value=True,
                        help="Tab 2의 예상 재해유형을 검색어에 보강합니다.")
    examples = ["탱크 안에서 용접하기 전에 무엇을 확인해야 하나요?",
                "체감온도 35도일 때 휴식은 어떻게 하나요?",
                "동료가 탱크 안에서 쓰러졌어요"]
    ex_cols = st.columns(len(examples))
    clicked = next((q for col, q in zip(ex_cols, examples) if col.button(q, width="stretch")), None)

    if "chat" not in st.session_state:
        st.session_state.chat = []
    for msg in st.session_state.chat:                       # 대화 기록 다시 그리기
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 근거 보기"):
                    for s in msg["sources"]:
                        st.caption(f"[{s['no']}] {s['source']} › {s['section']} (유사도 {s['score']})")

    question = st.chat_input("안전 관련 질문을 모국어로 입력하세요") or clicked
    if question:
        if store is None:
            st.error("RAG 인덱스가 없습니다.")
        else:
            st.session_state.chat.append({"role": "user", "content": question})
            ctx = predict_risk(bundle, conditions) if (use_ctx and bundle) else None
            with st.spinner("근거 검색 중…"):
                res = answer_question(store, question, language, prediction=ctx, k=top_k, llm=llm)
            st.session_state.chat.append({"role": "assistant", "content": res["answer"],
                                          "sources": res["sources"]})
            st.rerun()

    if st.session_state.chat and st.button("대화 초기화"):
        st.session_state.chat = []
        st.rerun()

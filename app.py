# -*- coding: utf-8 -*-
"""
공공도서관 설문 데이터 분석 및 시각화 대시보드

이 스크립트는 Streamlit을 사용하여 설문조사 데이터를 분석하고,
다양한 시각화와 GPT 기반의 인사이트를 제공하는 웹 애플리케이션입니다.

주요 기능:
- 기본 인구통계 및 만족도 문항 분석
- 중분류별, 세그먼트별 심화 분석
- 자연어 질의를 통한 자동 분석 및 시각화
- GPT를 활용한 키워드 추출 및 리포트 생성

개선 사항:
- 코드 구조화: 기능별(유틸리티, 데이터 처리, 시각화, GPT 연동 등) 모듈화
- 함수 재사용성 증대: 중복되는 시각화 및 데이터 처리 로직을 일반화된 함수로 통합
- 가독성 향상: 명확한 변수명 사용, 타입 힌트 추가, 상세한 주석 및 docstring 작성
- 안정성 강화: 구체적인 예외 처리 및 사용자 피드백 개선
- Streamlit UX 개선: st.expander 등을 활용하여 깔끔한 UI 구성
- 상수 관리: 하드코딩된 문자열 및 설정을 상수로 분리하여 관리 용이성 증대
"""

import time
import re
import json
import logging
from itertools import cycle
from typing import List, Dict, Any, Optional, Tuple

# 서드파티 라이브러리 임포트
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI

# -----------------------------------------------------------------------------
# 0. 설정 및 상수 (Configuration & Constants)
# -----------------------------------------------------------------------------

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# OpenAI 클라이언트 설정
# Streamlit secrets에서 API 키를 안전하게 로드합니다.
try:
    client = OpenAI(api_key=st.secrets["openai"]["api_key"])
except (KeyError, FileNotFoundError):
    st.error("OpenAI API 키를 찾을 수 없습니다. Streamlit secrets에 `openai.api_key`를 설정해주세요.")
    st.stop()


# --- 시각화 관련 상수 ---
DEFAULT_PALETTE = px.colors.qualitative.Plotly
LIKERT_COLORS = {
    1: "#d73027", 2: "#fc8d59", 3: "#fee090",
    4: "#dddddd", 5: "#91bfdb", 6: "#4575b4", 7: "#313695"
}

# --- 데이터 매핑 관련 상수 ---
# 중분류 매핑: 각 중분류 이름과 해당 컬럼을 식별하는 람다 함수를 연결합니다.
MIDDLE_CATEGORY_MAPPING = {
    "공간 및 이용편의성":       lambda col: str(col).startswith("Q1-"),
    "정보 획득 및 활용":       lambda col: str(col).startswith("Q2-"),
    "소통 및 정책 활용":       lambda col: str(col).startswith("Q3-"),
    "문화·교육 향유":         lambda col: str(col).startswith("Q4-"),
    "사회적 관계 형성":       lambda col: str(col).startswith("Q5-"),
    "개인의 삶과 역량":       lambda col: str(col).startswith("Q6-"),
    "도서관의 공익성 및 기여도": lambda col: (str(col).startswith("Q7-") or str(col).startswith("Q8")),
    "자치구 구성 문항":        lambda col: str(col).startswith("Q9-D-3"),
}

# KDC 주제 키워드 매핑
KDC_KEYWORD_MAP = {
    '000 총류': ["백과사전", "도서관", "독서", "문헌정보", "기록", "출판", "서지"],
    '100 철학': ["철학", "명상", "윤리", "논리학", "심리학"],
    '200 종교': ["종교", "기독교", "불교", "천주교", "신화", "신앙", "종교학"],
    '300 사회과학': ["사회", "정치", "경제", "법률", "행정", "교육", "복지", "여성", "노인", "육아", "아동복지", "사회문제", "노동", "환경문제", "인권"],
    '400 자연과학': ["수학", "물리", "화학", "생물", "지구과학", "과학", "천문", "기후", "의학", "생명과학"],
    '500 기술과학': ["건강", "의료", "요리", "간호", "공학", "컴퓨터", "AI", "IT", "농업", "축산", "산업", "기술", "미용"],
    '600 예술': ["미술", "음악", "무용", "사진", "영화", "연극", "디자인", "공예", "예술", "문화예술"],
    '700 언어': ["언어", "국어", "영어", "일본어", "중국어", "외국어", "한자", "문법"],
    '800 문학': ["소설", "시", "수필", "에세이", "희곡", "문학", "동화", "웹툰", "판타지", "문예"],
    '900 역사·지리': ["역사", "지리", "한국사", "세계사", "여행", "문화유산", "관광"],
}

# 세그먼트 분석 옵션
SEGMENT_OPTIONS = [
    {"label": "SQ1. 성별", "key": "SQ1"},
    {"label": "SQ2. 연령", "key": "SQ2"},
    {"label": "SQ3. 거주지", "key": "SQ3"},
    {"label": "SQ4. 주 이용 도서관", "key": "SQ4"},
    {"label": "SQ5. 주로 이용 서비스", "key": "SQ5"},
    {"label": "DQ1. 월평균 이용 빈도", "key": "DQ1"},
    {"label": "DQ2. 이용기간", "key": "DQ2"},
    {"label": "DQ4. (1순위)이용목적", "key": "DQ4"},
]

# -----------------------------------------------------------------------------
# 1. 유틸리티 함수 (Utility Functions)
# -----------------------------------------------------------------------------

def wrap_label(label: str, width: int = 10) -> str:
    """긴 레이블을 지정된 너비로 줄바꿈하여 반환합니다."""
    return '<br>'.join([label[i:i+width] for i in range(0, len(label), width)])

def remove_parentheses(text: str) -> str:
    """문자열에서 괄호와 그 안의 내용을 제거합니다."""
    return re.sub(r'\(.*?\)', '', text).strip()

def get_qualitative_colors(n: int) -> List[str]:
    """n개의 질적 색상 팔레트를 반환합니다."""
    return [color for _, color in zip(range(n), cycle(DEFAULT_PALETTE))]

def escape_tildes(text: str) -> str:
    """마크다운에서 '~'가 취소선으로 해석되는 것을 방지합니다."""
    return text.replace("~", "～")

def render_insight_card(title: str, content: str, key: str):
    """인사이트를 보기 좋은 카드 형태로 렌더링합니다."""
    if not content:
        content = "(분석 내용이 없습니다.)"
    
    content_html = escape_tildes(content).replace("\n", "<br>")
    
    html = f"""
    <div style="
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        background: #f8f9fa;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        font-family: 'Pretendard', sans-serif;
    ">
        <h4 style="margin:0 0 12px 0; font-size:1.1rem; color:#333; border-bottom: 2px solid #dee2e6; padding-bottom: 8px;">{title}</h4>
        <div style="font-size:0.95em; line-height:1.6em; color:#555;">{content_html}</div>
    </div>
    """
    components.html(html, height=min(800, 100 + content.count('\n') * 25), scrolling=True)


# -----------------------------------------------------------------------------
# 2. 데이터 처리 및 계산 함수 (Data Processing & Computation)
# -----------------------------------------------------------------------------

@st.cache_data
def load_data(uploaded_file) -> Optional[pd.DataFrame]:
    """업로드된 엑셀 파일을 읽어 데이터프레임으로 반환합니다."""
    try:
        df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        logging.error(f"파일 읽기 오류: {e}")
        st.error(f"엑셀 파일을 읽는 중 오류가 발생했습니다: {e}")
        return None

def scale_likert(series: pd.Series) -> pd.Series:
    """7점 척도 점수를 0-100점 척도로 변환합니다."""
    # 7점 척도이므로 (점수-1) / 6 을 하여 0~1 사이로 정규화 후 100을 곱함
    return 100 * (pd.to_numeric(series, errors='coerce') - 1) / 6

@st.cache_data(show_spinner="중분류 점수를 계산 중입니다...")
def compute_midcategory_scores(_df: pd.DataFrame) -> pd.Series:
    """데이터프레임에서 중분류별 평균 점수를 계산합니다."""
    results = {}
    for mid, predicate in MIDDLE_CATEGORY_MAPPING.items():
        cols = [c for c in _df.columns if predicate(c)]
        if not cols:
            continue
        
        # 각 문항을 100점 척도로 변환
        scaled = _df[cols].apply(scale_likert)
        # 모든 문항의 평균을 계산하여 중분류 점수로 삼음
        mid_mean = scaled.mean(axis=0, skipna=True).mean()
        if not pd.isna(mid_mean):
            results[mid] = mid_mean
            
    return pd.Series(results)

@st.cache_data(show_spinner="세부 문항 점수를 계산 중입니다...")
def compute_within_category_item_scores(_df: pd.DataFrame) -> Dict[str, pd.Series]:
    """중분류 내 각 세부 문항별 평균 점수를 계산합니다."""
    item_scores = {}
    for mid, predicate in MIDDLE_CATEGORY_MAPPING.items():
        cols = [c for c in _df.columns if predicate(c)]
        if not cols:
            continue
        
        scaled = _df[cols].apply(scale_likert)
        item_means = scaled.mean(axis=0, skipna=True)
        item_scores[mid] = item_means
        
    return item_scores

def apply_filters(df: pd.DataFrame, filters: List[Dict[str, Any]]) -> pd.DataFrame:
    """지정된 필터 목록을 데이터프레임에 적용합니다."""
    dff = df.copy()
    for f in filters:
        col, op, val = f.get("col"), f.get("op"), f.get("value")
        if not all([col, op, val]) or col not in dff.columns:
            continue
        
        try:
            if op in ("==", "="):
                dff = dff[dff[col].astype(str) == str(val)]
            elif op == "in" and isinstance(val, list):
                dff = dff[dff[col].astype(str).isin(map(str, val))]
            elif op == "contains":
                dff = dff[dff[col].astype(str).str.contains(str(val), na=False)]
        except Exception as e:
            logging.warning(f"필터 적용 실패: {col} {op} {val}. 오류: {e}")

    return dff

# -----------------------------------------------------------------------------
# 3. GPT-4 연동 함수 (OpenAI Integration)
# -----------------------------------------------------------------------------

def safe_chat_completion(**kwargs) -> Optional[str]:
    """안전하게 OpenAI Chat Completion API를 호출하고 결과를 반환합니다."""
    try:
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"OpenAI API 호출 실패: {e}")
        st.warning(f"AI 모델 호출에 실패했습니다. ({e})")
        return None

def parse_nl_query_to_spec(question: str) -> Dict[str, Any]:
    """자연어 질의를 분석하여 JSON 형태의 분석 명세로 변환합니다."""
    system_prompt = """
    당신은 설문 데이터 분석 전문가입니다. 사용자의 자연어 질문을 시각화 및 분석을 위한 JSON 명세로 변환하세요.
    출력은 반드시 JSON 객체 하나만, 코드 블록 없이 반환해야 합니다.
    알 수 없는 값은 null이나 빈 리스트로 처리하세요.

    JSON 필드 설명:
    - chart: 추천 차트 유형 ('bar', 'heatmap', 'radar', 'grouped_bar', 'delta_bar', 'none' 등)
    - x: 분석의 주요 축이 될 컬럼명 또는 '중분류'
    - y: y축에 해당하는 컬럼명 (주로 x와 함께 사용)
    - groupby: 비교 기준으로 사용할 그룹화 컬럼명
    - filters: 데이터 필터링 조건 배열 (e.g., [{"col": "SQ1. 성별", "op": "==", "value": "여"}])
    - focus: 사용자의 핵심 질문 의도를 요약한 문장

    예시:
    1. 질문: "혼자 이용하는 30대 여성들의 주 이용 도서관별 만족도 강점과 약점을 비교해줘."
       결과: {
            "chart": "radar",
            "x": "중분류",
            "groupby": "SQ4. 주 이용 도서관",
            "filters": [
                {"col": "이용형태", "op": "contains", "value": "혼자"},
                {"col": "SQ1. 성별", "op": "==", "value": "여"},
                {"col": "SQ2_GROUP", "op": "in", "value": ["30~34세", "35~39세"]}
            ],
            "focus": "30대 여성 단독 이용자의 주 이용 도서관별 만족도 프로파일 비교"
          }
    2. 질문: "전체 평균과 비교해서 어떤 중분류가 강점인지 보여줘."
       결과: {"chart": "radar", "x": "중분류", "groupby": null, "filters": [], "focus": "전체 평균 대비 중분류별 강점/약점 분석"}
    """
    content = safe_chat_completion(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.1,
        max_tokens=500,
        response_format={"type": "json_object"}
    )
    
    if content:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logging.error("GPT로부터 유효하지 않은 JSON 응답을 받았습니다.")
    
    # 실패 시 기본값 반환
    return {"chart": None, "x": None, "y": None, "groupby": None, "filters": [], "focus": question}


def generate_insight_from_data(prompt: str) -> str:
    """주어진 프롬프트를 바탕으로 GPT를 호출하여 데이터 분석 인사이트를 생성합니다."""
    system_prompt = "당신은 데이터 분석 전문가이자 전략 컨설턴트입니다. 주어진 데이터를 바탕으로 핵심적인 관찰, 명확한 결론, 그리고 실행 가능한 제안을 담은 비즈니스 보고서를 작성합니다. 간결하고 명확한 어조를 사용해주세요."
    
    content = safe_chat_completion(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1200
    )
    return escape_tildes(content) if content else "인사이트 생성에 실패했습니다."

# -----------------------------------------------------------------------------
# 4. 시각화 함수 (Plotting Functions)
# -----------------------------------------------------------------------------

def plot_categorical_bar(df: pd.DataFrame, col: str, title: str) -> Tuple[go.Figure, pd.DataFrame]:
    """범주형 데이터에 대한 막대 차트와 요약 테이블을 생성합니다."""
    counts = df[col].value_counts()
    percent = (counts / counts.sum() * 100).round(1)
    
    fig = px.bar(
        x=counts.index, 
        y=counts.values,
        text=counts.values,
        title=title,
        labels={'x': col, 'y': '응답 수'}
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(height=400, showlegend=False)
    
    table = pd.DataFrame({'응답 수': counts, '비율 (%)': percent}).T
    return fig, table

def plot_likert_stacked_bar(df: pd.DataFrame, col: str) -> Tuple[go.Figure, pd.DataFrame]:
    """리커트 척도 데이터에 대한 스택형 막대 차트와 요약 테이블을 생성합니다."""
    order = sorted(df[col].dropna().unique())
    counts = df[col].value_counts().reindex(order, fill_value=0)
    percent = (counts / counts.sum() * 100).round(1)
    
    fig = go.Figure()
    for val in order:
        fig.add_trace(go.Bar(
            x=[percent[val]], 
            y=[remove_parentheses(col)], 
            orientation='h', 
            name=f"{val}점",
            marker_color=LIKERT_COLORS.get(val, 'grey'),
            text=f"{percent[val]}%",
            textposition='inside',
            insidetextanchor='middle'
        ))
        
    fig.update_layout(
        barmode='stack',
        title=col,
        xaxis_title="비율 (%)",
        yaxis=dict(showticklabels=False),
        height=180,
        margin=dict(t=40, b=20, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
    
    table = pd.DataFrame([counts, percent], index=["응답 수", "비율 (%)"], columns=[f"{v}점" for v in order])
    return fig, table

def plot_midcategory_radar(df: pd.DataFrame, title: str = "중분류별 만족도 수준") -> Optional[go.Figure]:
    """중분류별 만족도 점수를 레이더 차트로 시각화합니다."""
    mid_scores = compute_midcategory_scores(df)
    if mid_scores.empty:
        return None

    categories = list(mid_scores.index)
    values = mid_scores.values.tolist()
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='만족도',
        hovertemplate="%{theta}: %{r:.1f}<extra></extra>"
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title=title,
        height=450,
        margin=dict(t=80, b=40)
    )
    return fig

def plot_grouped_bar(df: pd.DataFrame, x_col: str, y_col: str, group_col: str) -> Optional[go.Figure]:
    """그룹화된 막대 차트를 생성합니다."""
    if not all(c in df.columns for c in [x_col, y_col, group_col]):
        return None

    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        color=group_col,
        barmode="group",
        text_auto=".2s",
        title=f"{x_col} 및 {group_col}에 따른 {y_col} 비교"
    )
    fig.update_layout(height=450)
    return fig

def plot_heatmap(df: pd.DataFrame, title: str) -> Optional[go.Figure]:
    """데이터프레임을 히트맵으로 시각화합니다."""
    if df.empty:
        return None
        
    fig = px.imshow(
        df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title=title
    )
    fig.update_layout(height=max(400, len(df.index) * 30))
    return fig


# -----------------------------------------------------------------------------
# 5. Streamlit 페이지 렌더링 함수 (UI Rendering)
# -----------------------------------------------------------------------------

def render_basic_analysis_page(df: pd.DataFrame):
    """'기본 분석' 페이지를 렌더링합니다."""
    st.header("📊 기본 문항 분석")
    
    # 인구통계 및 기본 질문
    sq_cols = [c for c in df.columns if c.startswith("SQ") or c.startswith("BQ")]
    likert_cols = [c for c in df.columns if re.match(r"Q[1-9][\.-]", str(c))]
    
    with st.expander("👤 응답자 정보 (SQ, BQ 문항)", expanded=True):
        for col in sq_cols:
            fig, table = plot_categorical_bar(df, col, col)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(table)
            st.divider()

    with st.expander("📈 7점 척도 만족도 문항 (Q1 ~ Q8)", expanded=True):
        for col in likert_cols:
            fig, table = plot_likert_stacked_bar(df, col)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(table)
            st.divider()

def render_advanced_analysis_page(df: pd.DataFrame):
    """'심화 분석' 페이지를 렌더링합니다."""
    st.header("🔬 심화 분석")

    tabs = st.tabs(["전체 중분류 분석", "세부 항목별 편차 분석", "이용자 세그먼트 분석"])

    with tabs[0]:
        st.subheader("중분류별 전체 만족도")
        fig = plot_midcategory_radar(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            scores = compute_midcategory_scores(df).rename("평균 점수").round(2)
            st.dataframe(scores)
        else:
            st.warning("중분류 점수를 계산할 수 없습니다.")

    with tabs[1]:
        st.subheader("중분류 내 세부 문항별 만족도 편차")
        item_scores_by_mid = compute_within_category_item_scores(df)
        mid_scores = compute_midcategory_scores(df)
        
        for mid, item_scores in item_scores_by_mid.items():
            with st.expander(f"**{mid}** 내 문항별 비교", expanded=False):
                mid_mean = mid_scores.get(mid)
                if item_scores.empty or mid_mean is None:
                    st.write("데이터가 부족하여 분석할 수 없습니다.")
                    continue

                plot_df = item_scores.rename("점수").to_frame()
                plot_df['편차'] = plot_df['점수'] - mid_mean
                
                fig = px.bar(
                    plot_df, 
                    x='편차', 
                    y=plot_df.index, 
                    orientation='h',
                    title=f"'{mid}' 내 문항별 평균 대비 편차",
                    text=plot_df['점수'].round(1)
                )
                fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="red")
                fig.update_layout(height=max(300, len(plot_df) * 40), yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(plot_df.round(2))

    with tabs[2]:
        page_segment_analysis(df)


def page_segment_analysis(df: pd.DataFrame):
    """세그먼트 분석 UI 및 로직을 처리합니다."""
    st.subheader("👥 이용자 세그먼트 조합 분석")
    st.markdown("인구통계 및 이용행태 변수를 조합하여, 특정 이용자 그룹의 중분류별 만족도 프로파일을 심층 분석합니다.")

    sel_labels = st.multiselect(
        "분석에 사용할 세그먼트 변수 선택 (최대 3개)",
        [o['label'] for o in SEGMENT_OPTIONS],
        default=[SEGMENT_OPTIONS[0]['label'], SEGMENT_OPTIONS[1]['label']],
        max_selections=3
    )
    
    if len(sel_labels) < 1:
        st.info("분석할 세그먼트 변수를 1개 이상 선택해주세요.")
        return

    selected_keys = [o['key'] for o in SEGMENT_OPTIONS if o['label'] in sel_labels]
    
    # 파생 변수 추가 (예: 연령대 그룹)
    # 이 부분은 실제 데이터 컬럼명에 맞게 커스터마이징 필요
    if "SQ2" in selected_keys and "SQ2_GROUP" not in df.columns:
        # 예시: 'SQ2. 연령' 컬럼에서 숫자만 추출하여 연령대 그룹 생성
        age_series = pd.to_numeric(df['SQ2. 연령'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
        bins = [0, 19, 29, 39, 49, 59, 69, 120]
        labels = ['10대 이하', '20대', '30대', '40대', '50대', '60대', '70대 이상']
        df['SQ2_GROUP'] = pd.cut(age_series, bins=bins, labels=labels, right=True)

    segment_cols = []
    col_map = {"SQ2": "SQ2_GROUP"} # 원본 키를 파생변수 컬럼으로 매핑
    for key in selected_keys:
        # 실제 데이터에 있는 컬럼명을 찾아 추가
        mapped_key = col_map.get(key, key)
        cols = [c for c in df.columns if mapped_key in c]
        if cols:
            segment_cols.append(cols[0])

    if not segment_cols:
        st.warning("선택한 변수에 해당하는 데이터 컬럼을 찾을 수 없습니다.")
        return

    # 그룹별 중분류 점수 계산
    grouped = df.dropna(subset=segment_cols).groupby(segment_cols)
    
    results = []
    for name, group_df in grouped:
        if len(group_df) < 5:  # 최소 응답자 수 필터링
            continue
        scores = compute_midcategory_scores(group_df)
        if scores.empty:
            continue
        
        # 그룹 이름이 튜플일 경우 문자열로 변환
        group_name = " | ".join(map(str, name)) if isinstance(name, tuple) else str(name)
        
        result_row = scores.to_dict()
        result_row['세그먼트'] = group_name
        result_row['응답자수'] = len(group_df)
        results.append(result_row)

    if not results:
        st.warning("분석 가능한 세그먼트 그룹이 없습니다 (응답자 수 5명 이상).")
        return

    result_df = pd.DataFrame(results).set_index('세그먼트')
    
    # 시각화
    st.markdown("#### 세그먼트별 만족도 히트맵")
    heatmap_df = result_df.drop(columns=['응답자수'])
    fig = plot_heatmap(heatmap_df, "세그먼트별 중분류 만족도 평균")
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    # GPT 인사이트
    st.markdown("#### AI 기반 분석 리포트")
    prompt = f"""
    다음은 도서관 이용자 세그먼트별 중분류 만족도 분석 결과입니다.
    데이터를 기반으로 비즈니스 보고서를 작성해주세요.

    데이터 (인덱스: 세그먼트, 컬럼: 중분류, 값: 만족도 점수):
    {heatmap_df.to_markdown()}

    보고서에 포함할 내용:
    1.  **핵심 요약 (Overall Summary):** 가장 두드러지는 패턴이나 인사이트를 2-3문장으로 요약합니다.
    2.  **주요 강점 그룹 (High-Performing Segments):** 전반적으로 만족도가 높은 세그먼트 그룹들을 식별하고, 그들의 공통적인 강점 영역을 분석합니다.
    3.  **개선 필요 그룹 (Segments Needing Attention):** 만족도가 낮은 그룹들을 식별하고, 특히 어떤 중분류에서 개선이 시급한지 설명합니다.
    4.  **전략적 제언 (Strategic Recommendations):** 분석 결과를 바탕으로, 도서관이 우선적으로 고려해야 할 2-3가지 전략적 방향을 구체적으로 제안합니다. (예: '30대 여성 그룹의 '소통 및 정책 활용' 만족도 개선을 위한 프로그램 기획')
    """
    insight = generate_insight_from_data(prompt)
    render_insight_card("세그먼트 분석 AI 리포트", insight, "segment-insight")


def render_nl_query_page(df: pd.DataFrame):
    """'자연어 질의' 페이지를 렌더링합니다."""
    st.header("💬 자연어 질문 기반 자동 분석")
    st.markdown("궁금한 점을 자유롭게 질문하면, AI가 질문을 해석하여 자동으로 데이터를 분석하고 시각화합니다.")
    st.info("예시: '20대 남성의 공간 만족도는 어떤가요?' 또는 '주 이용 도서관별로 만족도 차이를 비교해줘.'")

    question = st.text_input("분석하고 싶은 내용을 질문해주세요:", key="nl_question")

    if question:
        with st.spinner("AI가 질문을 분석하고 있습니다..."):
            spec = parse_nl_query_to_spec(question)
        
        st.markdown("#### 🤖 AI 분석 설계")
        with st.expander("AI가 이해한 분석 명세 보기"):
            st.json(spec)

        # 1. 필터 적용
        df_filtered = apply_filters(df, spec.get("filters", []))
        st.write(f"총 {len(df)}개 응답 중, 필터 조건을 만족하는 **{len(df_filtered)}개**의 데이터를 분석합니다.")

        if df_filtered.empty:
            st.warning("질문 조건에 맞는 데이터가 없습니다.")
            return

        # 2. 분석 및 시각화
        chart_type = spec.get("chart")
        x_col = spec.get("x")
        y_col = spec.get("y")
        group_col = spec.get("groupby")

        # 분석 로직을 여기에 구현
        # 예: x가 '중분류'이면 레이더 차트, groupby가 있으면 그룹바 차트 등
        if x_col == '중분류':
            fig = plot_midcategory_radar(df_filtered, title=f"분석 결과: {spec.get('focus')}")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        elif x_col and y_col and group_col:
             # 데이터 집계
            agg_df = df_filtered.groupby([x_col, group_col])[y_col].mean().reset_index()
            fig = plot_grouped_bar(agg_df, x_col, y_col, group_col)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        elif x_col:
            fig, _ = plot_categorical_bar(df_filtered, x_col, title=f"분석 결과: {spec.get('focus')}")
            st.plotly_chart(fig, use_container_width=True)
        else:
             st.info("질문에 맞는 시각화를 생성하기 어렵습니다. 더 구체적으로 질문해주세요.")

        # 3. GPT 기반 설명 생성
        with st.spinner("AI가 분석 결과를 해석하고 있습니다..."):
            # 프롬프트 구성 시, 분석에 사용된 데이터 요약 정보를 포함하면 더 좋은 결과 생성 가능
            prompt = f"""
            사용자의 질문 "{question}"에 대해 다음과 같은 분석을 수행했습니다.
            - 분석 명세: {json.dumps(spec, ensure_ascii=False)}
            - 필터링된 데이터 수: {len(df_filtered)}

            이 분석 결과를 바탕으로, 사용자가 이해하기 쉽게 핵심 내용을 설명해주세요.
            """
            explanation = generate_insight_from_data(prompt)
            render_insight_card("AI 분석 결과 요약", explanation, "nl-explanation")


# -----------------------------------------------------------------------------
# 6. 메인 실행 로직 (Main Execution Logic)
# -----------------------------------------------------------------------------

def main():
    """애플리케이션의 메인 로직을 실행합니다."""
    st.set_page_config(page_title="도서관 설문 분석 대시보드", layout="wide")
    st.title("📚 공공도서관 설문 분석 대시보드")
    st.markdown("---")

    # 파일 업로드
    uploaded_file = st.sidebar.file_uploader("📂 엑셀(.xlsx) 파일 업로드", type=["xlsx"])
    
    if uploaded_file is None:
        st.info("👈 사이드바에서 분석할 설문 데이터 엑셀 파일을 업로드해주세요.")
        st.stop()

    df = load_data(uploaded_file)
    if df is None:
        st.stop()

    # 분석 모드 선택
    st.sidebar.title("분석 메뉴")
    mode = st.sidebar.radio(
        "원하는 분석 유형을 선택하세요:",
        ["기본 분석", "심화 분석", "자연어 질의(AI)"],
        captions=["전체 문항별 분포 확인", "변수 간 관계 및 그룹 비교", "AI에게 질문하여 자동 분석"]
    )

    # 선택된 모드에 따라 해당 페이지 렌더링
    if mode == "기본 분석":
        render_basic_analysis_page(df)
    elif mode == "심화 분석":
        render_advanced_analysis_page(df)
    elif mode == "자연어 질의(AI)":
        render_nl_query_page(df)

if __name__ == "__main__":
    main()

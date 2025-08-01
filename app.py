import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import re
import openai
import logging
from itertools import cycle

# 로깅 설정 (필요시 파일로도 남기게 조정 가능)
logging.basicConfig(level=logging.INFO)

openai.api_key = st.secrets["openai"]["api_key"]
client = openai.OpenAI(api_key=openai.api_key)

# ─────────────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────────────
def remove_parentheses(text):
    return re.sub(r'\(.*?\)', '', text).strip()

def wrap_label(label, width=10):
    return '<br>'.join([label[i:i+width] for i in range(0, len(label), width)])

def get_qualitative_colors(n):
    palette = px.colors.qualitative.Plotly
    return [c for _, c in zip(range(n), cycle(palette))]

def show_table(df, caption):
    st.dataframe(df)

def render_chart_and_table(bar, table, title, key_prefix=""):
    if bar is not None:
        st.plotly_chart(bar, use_container_width=True, key=f"{key_prefix}-bar-{title}")
    if isinstance(table, go.Figure):
        st.plotly_chart(table, use_container_width=True, key=f"{key_prefix}-tbl-fig-{title}")
    elif isinstance(table, pd.DataFrame):
        st.dataframe(table, key=f"{key_prefix}-tbl-df-{title}")
    elif table is not None:
        st.write(table, key=f"{key_prefix}-tbl-raw-{title}")
def _sanitize_dataframe_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()

    # object 컬럼들에 대해 모두 문자열화 (NaN 유지)
    for col in df2.select_dtypes(include=["object"]).columns:
        df2[col] = df2[col].apply(lambda x: str(x) if not pd.isna(x) else x)

    # 인덱스가 복잡하면 단순 문자열로 변환 (MultiIndex 포함)
    if isinstance(df2.index, pd.MultiIndex):
        df2.index = df2.index.map(lambda tup: " | ".join(map(str, tup)))
    else:
        df2.index = df2.index.map(lambda x: str(x))

    # 컬럼 이름도 비표준이면 문자열로
    df2.columns = [str(c) for c in df2.columns]

    return df2

def render_chart_and_table(bar, table, title, key_prefix=""):
    if bar is not None:
        st.plotly_chart(bar, use_container_width=True, key=f"{key_prefix}-bar-{title}")
    if isinstance(table, go.Figure):
        st.plotly_chart(table, use_container_width=True, key=f"{key_prefix}-tbl-fig-{title}")
    elif isinstance(table, pd.DataFrame):
        try:
            safe_tbl = _sanitize_dataframe_for_streamlit(table)
            st.dataframe(safe_tbl, key=f"{key_prefix}-tbl-df-{title}")
        except Exception as e:
            logging.warning(f"DataFrame rendering failed, showing head only: {e}")
            # 샘플로라도 보여줌
            try:
                safe_head = _sanitize_dataframe_for_streamlit(table.head(200))
                st.dataframe(safe_head, key=f"{key_prefix}-tbl-df-{title}-sample")
                st.warning(f"전체 테이블 렌더링에 실패하여 상위 200개만 보여줍니다: {e}")
            except Exception as e2:
                st.error(f"테이블 렌더링 불가: {e2}")
    elif table is not None:
        st.write(table, key=f"{key_prefix}-tbl-raw-{title}")

# ─────────────────────────────────────────────────────
# SQ2: 연령 히스토그램 + 테이블
# ─────────────────────────────────────────────────────
def plot_age_histogram_with_labels(df, question):
    data = df[question].dropna().astype(str).str.extract(r'(\d+)')
    data.columns = ['age']
    data['age'] = pd.to_numeric(data['age'], errors='coerce').dropna()

    def age_group(age):
        if age < 15:
            return '14세 이하'
        elif age >= 80:
            return '80세 이상'
        else:
            return f"{(age//5)*5}~{(age//5)*5+4}세"

    data['group'] = data['age'].apply(age_group)
    grouped = data['group'].value_counts().sort_index()
    percent = (grouped / grouped.sum() * 100).round(1)

    fig = go.Figure(go.Bar(
        x=grouped.index, y=grouped.values,
        text=grouped.values, textposition='outside',
        marker_color="#1f77b4"
    ))
    fig.update_layout(
        title=question, yaxis_title="응답 수",
        bargap=0.1, height=450, margin=dict(t=40, b=10)
    )

    table_df = pd.DataFrame({'응답 수': grouped, '비율 (%)': percent}).T
    return fig, table_df

# ─────────────────────────────────────────────────────
# BQ2: 직업군 Bar + Table
# ─────────────────────────────────────────────────────
def plot_bq2_bar(df, question):
    data = df[question].dropna().astype(str)
    counts_raw = data.value_counts()
    percent_raw = (counts_raw / counts_raw.sum() * 100).round(1)

    categories_raw = counts_raw.index.tolist()
    categories = [label.split('. ', 1)[-1] for label in categories_raw]
    counts = counts_raw.values
    percent = percent_raw.values

    wrapped_labels = [wrap_label(remove_parentheses(label), width=10) for label in categories]

    colors = get_qualitative_colors(len(categories))
    fig = go.Figure(go.Bar(
        x=categories,
        y=counts,
        text=counts,
        textposition='outside',
        marker_color=colors
    ))
    y_max = counts.max() + 20
    fig.update_layout(
        title=dict(text=question, font=dict(size=16)),
        yaxis=dict(title="응답 수", range=[0, y_max]),
        height=450,
        margin=dict(t=50, b=100),
        xaxis_tickangle=-30
    )

    table_df = pd.DataFrame(
        [counts, percent],
        index=["응답 수", "비율 (%)"],
        columns=wrapped_labels
    )

    return fig, table_df

# ─────────────────────────────────────────────────────
# SQ4: 커스텀 누적 가로 Bar + Table
# ─────────────────────────────────────────────────────
def plot_sq4_custom_bar(df, question):
    data = df[question].dropna().astype(str)
    cats = sorted(data.unique())
    counts = data.value_counts().reindex(cats).fillna(0).astype(int)
    percent = (counts / counts.sum() * 100).round(1)
    display_labels = [wrap_label(remove_parentheses(x), 10) for x in cats]

    fig = go.Figure()
    colors = get_qualitative_colors(len(cats))
    for i, cat in enumerate(cats):
        fig.add_trace(go.Bar(
            x=[percent[cat]], y=[question],
            orientation='h', name=remove_parentheses(cat),
            marker_color=colors[i],
            text=f"{percent[cat]}%", textposition='inside'
        ))
    fig.update_layout(
        barmode='stack', showlegend=True,
        legend=dict(orientation='h', y=-0.5, x=0.5, xanchor='center', traceorder='reversed'),
        title=dict(text=question, font=dict(size=16)),
        yaxis=dict(showticklabels=False),
        height=250, margin=dict(t=40, b=100)
    )

    table_df = pd.DataFrame(
        [counts.values, percent.values],
        index=["응답 수", "비율 (%)"],
        columns=display_labels
    )

    return fig, table_df

# ─────────────────────────────────────────────────────
# 일반 범주형 누적 Bar + Table (SQ5/SQ3 등)
# ─────────────────────────────────────────────────────
def plot_categorical_stacked_bar(df, question):
    data = df[question].dropna().astype(str)
    categories_raw = sorted(data.unique())
    display_labels = [label.split('. ', 1)[-1] for label in categories_raw]

    counts = data.value_counts().reindex(categories_raw).fillna(0).astype(int)
    percent = (counts / counts.sum() * 100).round(1)

    fig = go.Figure()
    for i, label in enumerate(reversed(display_labels)):
        raw_cat = categories_raw[display_labels[::-1].index(label)]
        fig.add_trace(go.Bar(
            x=[percent[raw_cat]],
            y=[question],
            orientation='h',
            name=label,
            marker=dict(color=get_qualitative_colors(len(display_labels))[i]),
            text=f"{percent[raw_cat]}%",
            textposition='inside',
            insidetextanchor='middle',
            hoverinfo='x+name'
        ))

    fig.update_layout(
        barmode='stack',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom', y=-1,
            xanchor='center', x=0.5,
            traceorder='reversed'
        ),
        title=dict(text=question, font=dict(size=16)),
        yaxis=dict(showticklabels=False),
        height=250, margin=dict(t=40, b=100)
    )

    table_df = pd.DataFrame(
        [counts.values, percent.values],
        index=["응답 수", "비율 (%)"],
        columns=display_labels
    )

    return fig, table_df

# ─────────────────────────────────────────────────────
# Q1~Q9-D: 7점 척도 스택형 바 + Table
# ─────────────────────────────────────────────────────
def plot_stacked_bar_with_table(df, question):
    data = pd.to_numeric(df[question].dropna(), errors='coerce').dropna().astype(int)
    order = [1,2,3,4,5,6,7]
    counts = data.value_counts().reindex(order, fill_value=0)
    percent = (counts / counts.sum() * 100).round(1)

    colors = {
        1: "#d73027", 2: "#fc8d59", 3: "#fee090",
        4: "#dddddd", 5: "#91bfdb", 6: "#4575b4", 7: "#313695"
    }
    fig = go.Figure()
    for v in order:
        fig.add_trace(go.Bar(
            x=[percent[v]], y=[question], orientation='h', name=f"{v}점",
            marker_color=colors[v], text=f"{percent[v]}%", textposition='inside'
        ))
    fig.update_layout(
        barmode='stack', showlegend=False,
        title=question, xaxis_title="매우 불만족 → 매우 만족",
        yaxis=dict(showticklabels=False), height=180, margin=dict(t=40,b=2)
    )

    table_df = pd.DataFrame(
        [counts.values, percent.values],
        index=["응답 수", "비율 (%)"],
        columns=[f"{v}점" for v in order]
    )
    return fig, table_df

#----------------------------------------------------------------------------- 
# 단문 분석 관련 유틸
#-----------------------------------------------------------------------------
# 🔧 KDC 매핑 및 분석 유틸
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
    '원서(영어)': ["원서", "영문도서", "영문판", "영어원서"],
    '연속간행물': ["잡지", "간행물", "연속간행물"],
    '해당없음': []
}

# 응답이 trivial 한지 검사
def is_trivial(text):
    text = str(text).strip()
    return text in ["", "X", "x", "감사합니다", "감사", "없음"]

# 주제범주 매핑
def map_keyword_to_category(keyword):
    for cat, kws in KDC_KEYWORD_MAP.items():
        if any(k in keyword for k in kws):
            return cat
    return "해당없음"

# 단순 분할(Fallback)
def split_keywords_simple(text):
    parts = re.split(r"[.,/\s]+", text)
    return [p.strip() for p in parts if len(p.strip()) > 1]

# 통합 추출: 키워드 + 대상범주
@st.cache_data(show_spinner=False)
def extract_keyword_and_audience(responses, batch_size=20):  # 배치 크기 증가로 호출 횟수 감소:  # 배치 크기 축소로 응답 지연 개선  # 배치 크기 축소로 응답 지연 개선
    results = []
    for i in range(0, len(responses), batch_size):
        batch = responses[i:i+batch_size]
        prompt = f"""
당신은 도서관 자유응답에서 아래 형식의 JSON 배열만 반환합니다.
각 객체는 응답, 키워드 목록(1~3개), 대상층(유아/아동/청소년/일반)을 포함해야 합니다.

예시Output:
[
  {{"response": "응답1", "keywords": ["키워드1","키워드2"], "audience": "청소년"}},
  ...
]

응답 목록:
{chr(10).join(f"{j+1}. {txt}" for j, txt in enumerate(batch))}
"""
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",  # 빠른 처리 위해 모델을 낮춰 사용,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.2,
            max_tokens=300  # 토큰 제한 축소로 처리 시간 단축
        )
        content = resp.choices[0].message.content.strip()
        try:
            data = pd.read_json(content)
        except Exception:
            # fallback: 수동 분할 + 기본 규칙
            data = []
            for txt in batch:
                kws = split_keywords_simple(txt)
                audience = '일반'
                for w in ['어린이','초등']:
                    if w in txt: audience='아동'
                for w in ['유아','미취학','그림책']:
                    if w in txt: audience='유아'
                for w in ['청소년','진로','자기계발']:
                    if w in txt: audience='청소년'
                data.append({
                    'response': txt,
                    'keywords': kws,
                    'audience': audience
                })
            data = pd.DataFrame(data)
        for _, row in data.iterrows():
            results.append((row['response'], row['keywords'], row['audience']))
    return results

# 전체 응답 처리
import math

@st.cache_data(show_spinner=False)
def process_answers(responses):
    # 콤마(,) 기준으로 다중 응답 분리
    expanded = []
    for ans in responses:
        # trivial 응답 제외 전처리
        if is_trivial(ans):
            continue
        parts = [p.strip() for p in ans.split(',') if p.strip()]
        # 단일 또는 다중 항목 처리
        if len(parts) > 1:
            expanded.extend(parts)
        else:
            expanded.append(ans)

    processed = []
    # 통합 호출 횟수 계산
    batches = extract_keyword_and_audience(expanded, batch_size=8)  # 호출 횟수 조정
    for resp, kws, aud in batches:
        if is_trivial(resp):
            continue
        if not kws:
            kws = split_keywords_simple(resp)
        for kw in kws:
            cat = map_keyword_to_category(kw)
            if cat=='해당없음' and aud=='일반':
                continue
            processed.append({
                '응답': resp,
                '키워드': kw,
                '주제범주': cat,
                '대상범주': aud
            })
    return pd.DataFrame(processed)



# 시각화 페이지 함수
def show_short_answer_keyword_analysis(df_result):
    st.subheader("📘 Q9-DS-4 단문 응답 키워드 분석")
    order = list(KDC_KEYWORD_MAP.keys())
    df_cat = df_result.groupby("주제범주")["키워드"].count().reindex(order, fill_value=0).reset_index(name="빈도수")
    fig = px.bar(df_cat, x="주제범주", y="빈도수", title="주제범주별 키워드 빈도", text="빈도수")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
    df_aud = df_result.groupby("대상범주")["키워드"].count().reset_index(name="빈도수")
    fig2 = px.bar(df_aud, x="대상범주", y="빈도수", title="대상범주별 키워드 빈도", text="빈도수", color="대상범주")
    fig2.update_traces(textposition="outside")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("#### 🔍 분석 결과 테이블")
    st.dataframe(df_result[["응답", "키워드", "주제범주", "대상범주"]])

# ------------------ Likert 계산 매핑 ------------------
def scale_likert(series):
    return 100 * (pd.to_numeric(series, errors='coerce') - 1) / 6

MIDDLE_CATEGORY_MAPPING = {
    "공간 및 이용편의성":       lambda col: str(col).startswith("Q1-"),
    "정보 획득 및 활용":       lambda col: str(col).startswith("Q2-"),
    "소통 및 정책 활용":       lambda col: str(col).startswith("Q3-"),
    "문화·교육 향유":         lambda col: str(col).startswith("Q4-"),
    "사회적 관계 형성":       lambda col: str(col).startswith("Q5-"),
    "개인의 삶과 역량":       lambda col: str(col).startswith("Q6-"),
    "도서관의 공익성 및 기여도": lambda col: (str(col).startswith("Q7-") or str(col).startswith("Q8")),
}

@st.cache_data(show_spinner=False)
def compute_midcategory_scores(df):
    results = {}
    for mid, predicate in MIDDLE_CATEGORY_MAPPING.items():
        cols = [c for c in df.columns if predicate(c)]
        if not cols:
            continue
        scaled = df[cols].apply(scale_likert)
        mid_mean = scaled.mean(axis=0, skipna=True).mean()
        results[mid] = mid_mean
    return pd.Series(results).dropna()

@st.cache_data(show_spinner=False)
def compute_within_category_item_scores(df):
    item_scores = {}
    for mid, predicate in MIDDLE_CATEGORY_MAPPING.items():
        cols = [c for c in df.columns if predicate(c)]
        if not cols:
            continue
        scaled = df[cols].apply(scale_likert)
        item_means = scaled.mean(axis=0, skipna=True)
        item_scores[mid] = item_means
    return item_scores

def midcategory_avg_table(df):
    s = compute_midcategory_scores(df)
    if s.empty:
        return pd.DataFrame()
    tbl = s.rename("평균 점수(0~100)").to_frame().reset_index().rename(columns={"index": "중분류"})
    tbl["평균 점수(0~100)"] = tbl["평균 점수(0~100)"].round(2)
    tbl = tbl.sort_values(by="평균 점수(0~100)", ascending=False).reset_index(drop=True)
    return tbl

def plot_midcategory_radar(df):
    mid_scores = compute_midcategory_scores(df)
    if mid_scores.empty:
        return None
    categories = list(mid_scores.index)
    values = mid_scores.values.tolist()
    categories_closed = categories + categories[:1]
    values_closed = values + values[:1]
    overall_mean = mid_scores.mean()
    avg_values_closed = [overall_mean] * len(categories_closed)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        name='중분류 만족도',
        hovertemplate="%{theta}: %{r:.1f}<extra></extra>"
    ))
    fig.add_trace(go.Scatterpolar(
        r=avg_values_closed,
        theta=categories_closed,
        fill=None,
        name=f"전체 평균 ({overall_mean:.1f})",
        line=dict(color='red', dash='solid'),
        hovertemplate=f"전체 평균: {overall_mean:.1f}<extra></extra>"
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0,100], tickformat=".0f")),
        title="중분류별 만족도 수준 (0~100 환산, 레이더 차트)",
        showlegend=True,
        height=450,
        margin=dict(t=40, b=20)
    )
    return fig


def wrap_label_fixed(label: str, width: int = 35) -> str:
    # 한 줄에 공백 포함 정확히 width 글자씩 자르고 <br>로 연결
    parts = [label[i:i+width] for i in range(0, len(label), width)]
    return "<br>".join(parts)

def plot_within_category_bar(df, midcategory):
    item_scores = compute_within_category_item_scores(df)
    if midcategory not in item_scores:
        return None, None
    predicate = MIDDLE_CATEGORY_MAPPING[midcategory]
    orig_cols = [c for c in df.columns if predicate(c)]
    if not orig_cols:
        return None, None

    series_plot = item_scores[midcategory].reindex(orig_cols[::-1])
    series_table = item_scores[midcategory].reindex(orig_cols)
    mid_scores = compute_midcategory_scores(df)
    mid_mean = mid_scores.get(midcategory, None)

    # 고정 너비(15자) 줄바꿈된 y축 라벨
    wrapped_labels = [wrap_label_fixed(label, width=35) for label in series_plot.index]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=series_plot.values,
        y=wrapped_labels,
        orientation='h',
        text=series_plot.round(1),
        textposition='outside',
        marker_color='steelblue',
        hovertemplate="<b>%{customdata}</b><br>평균 점수: %{x:.1f}<extra></extra>",
        customdata=series_plot.index
    ))
    if mid_mean is not None:
        fig.add_vline(
            x=mid_mean,
            line_color="red"
        )
    # y축 라벨이 몇 줄로 나뉘었는지 계산해서 최소 높이 보장
    max_lines = max(label.count("<br>") + 1 for label in wrapped_labels) if wrapped_labels else 1
    per_item_height = 50  # 한 항목당 기본 높이
    total_height = max(300, per_item_height * len(wrapped_labels))

    fig.update_layout(
        title=f"{midcategory} 내 문항별 평균 점수 비교 (0~100 환산)",
        xaxis_title=f"{midcategory} 평균 {mid_mean:.2f}" if mid_mean is not None else "평균 점수",
        margin=dict(t=40, b=60),
        height=total_height
    )

    if mid_mean is not None:
        diff = series_table - mid_mean
        table_df = pd.DataFrame({
            '평균 점수': series_table.round(2),
            '중분류 평균': [round(mid_mean,2)] * len(series_table),
            '편차 (문항 - 중분류 평균)': diff.round(2)
        }, index=series_table.index)
    else:
        table_df = pd.DataFrame({
            '평균 점수': series_table.round(2)
        }, index=series_table.index)
    return fig, table_df


# ------------------ DQ 관련 ------------------
def plot_dq1(df):
    cols = [c for c in df.columns if c.startswith("DQ1")]
    if not cols:
        return None, None, ""
    question = cols[0]
    data = df[question].dropna().astype(str).str.extract(r"(\d+\.?\d*)")[0]
    monthly = pd.to_numeric(data, errors='coerce')
    yearly = monthly * 12

    def categorize(f):
        try:
            f = float(f)
        except:
            return None
        if f < 12:
            return "0~11회: 연 1회 미만"
        elif f < 24:
            return "12~23회: 월 1회 정도"
        elif f < 48:
            return "24~47회: 월 2~4회 정도"
        elif f < 72:
            return "48~71회: 주 1회 정도"
        elif f < 144:
            return "72~143회: 주 2~3회"
        else:
            return "144회 이상: 거의 매일"

    cat = yearly.apply(categorize)
    order = ["0~11회: 연 1회 미만","12~23회: 월 1회 정도","24~47회: 월 2~4회 정도",
             "48~71회: 주 1회 정도","72~143회: 주 2~3회","144회 이상: 거의 매일"]
    grp = cat.value_counts().reindex(order, fill_value=0)
    pct = (grp/grp.sum()*100).round(1)

    fig = go.Figure(go.Bar(x=grp.index, y=grp.values, text=grp.values,
                            textposition='outside', marker_color="#1f77b4"))
    fig.update_layout(title=question, xaxis_title="이용 빈도 구간", yaxis_title="응답 수",
                      bargap=0.2, height=450, margin=dict(t=30,b=50), xaxis_tickangle=-15)

    tbl_df = pd.DataFrame({"응답 수":grp, "비율 (%)":pct}).T  # <- DataFrame 그대로 반환
    return fig, tbl_df, question


def plot_dq2(df):
    cols = [c for c in df.columns if c.startswith("DQ2")]
    if not cols:
        return None, None, ""
    question = cols[0]

    def parse(s):
        s = str(s).strip()
        m = re.match(r'^(\d+)\s*년\s*(\d+)\s*개월$', s)
        if m:
            return int(m.group(1)) + (1 if int(m.group(2))>0 else 0)
        m = re.match(r'^(\d+)\s*년$', s)
        if m:
            return int(m.group(1))
        m = re.match(r'^(\d+)\s*개월$', s)
        if m:
            return 1
        return None

    yrs = df[question].dropna().apply(parse)
    grp = yrs.value_counts().sort_index()
    pct = (grp/grp.sum()*100).round(1)
    labels = [f"{y}년" for y in grp.index]
    fig = go.Figure(go.Bar(x=labels, y=grp.values, text=grp.values,
                            textposition='outside', marker_color="#1f77b4"))
    fig.update_layout(title=question, xaxis_title="이용 기간 (년)", yaxis_title="응답 수",
                      bargap=0.2, height=450, margin=dict(t=30,b=50), xaxis_tickangle=-15)
    tbl_df = pd.DataFrame({"응답 수":grp, "비율 (%)":pct}).T
    return fig, tbl_df, question

def plot_dq3(df):
    cols = [c for c in df.columns if c.startswith("DQ3")]
    if not cols:
        return None, None, ""
    question = cols[0]

    # 기존 범주형 누적 스택 바 + 대응 테이블(DataFrame)
    bar, table_df = plot_categorical_stacked_bar(df[[question]].dropna().astype(str), question)


    # 기본적으로 bar + DataFrame을 반환. 필요하면 table_fig로 바꿔도 된다.
    return bar, table_df, question


def plot_dq4_bar(df):
    cols = [c for c in df.columns if c.startswith("DQ4")]
    if len(cols) < 2:
        return None, None, ""
    col1, col2 = cols[0], cols[1]
    question = f"{col1} vs {col2}"

    s1 = df[col1].dropna().astype(str)
    s2 = df[col2].dropna().astype(str)
    cats = sorted(set(s1.unique()).union(s2.unique()))
    labels = [c.split('. ',1)[-1] if '. ' in c else c for c in cats]

    counts1 = s1.value_counts().reindex(cats, fill_value=0)
    counts2 = s2.value_counts().reindex(cats, fill_value=0)
    pct1 = (counts1 / counts1.sum() * 100).round(1)
    pct2 = (counts2 / counts2.sum() * 100).round(1)

    order_idx = counts1.sort_values(ascending=False).index.tolist()
    sorted_labels = [lbl.split('. ',1)[-1] if '. ' in lbl else lbl for lbl in order_idx]
    sorted_counts1 = counts1.reindex(order_idx)
    sorted_counts2 = counts2.reindex(order_idx)
    sorted_pct1 = pct1.reindex(order_idx)
    sorted_pct2 = pct2.reindex(order_idx)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sorted_labels, y=sorted_counts1.values,
        name='1순위', marker_color='light blue', text=sorted_counts1.values, textposition='outside'
    ))
    fig.add_trace(go.Bar(
        x=sorted_labels, y=sorted_counts2.values,
        name='2순위', marker_color='light green', text=sorted_counts2.values, textposition='outside'
    ))
    fig.update_layout(
        barmode='stack',
        title="DQ4. 도서관 이용 주요 목적 1순위 vs 2순위",
        xaxis_title="이용 목적",
        yaxis_title="응답자 수",
        height=550,
        margin=dict(t=40, b=10),
        xaxis_tickangle=-23
    )

    table_df = pd.DataFrame({
        '1순위 응답 수': sorted_counts1.values,
        '1순위 비율(%)': sorted_pct1.values,
        '2순위 응답 수': sorted_counts2.values,
        '2순위 비율(%)': sorted_pct2.values
    }, index=sorted_labels).T  # <- DataFrame 형태로 반환
    return fig, table_df, question

def plot_dq5(df):
    cols = [c for c in df.columns if c.startswith("DQ5")]
    if not cols:
        return None, None, ""
    question = cols[0]
    temp_df = df[[question]].dropna().astype(str)
    fig, table_df = plot_categorical_stacked_bar(temp_df, question)
    return fig, table_df, question

def plot_likert_diverging(df, prefix="DQ7-E"):
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return None, None
    dist = {}
    for col in cols:
        counts = df[col].dropna().astype(int).value_counts().reindex(range(1,8), fill_value=0)
        pct = (counts / counts.sum() * 100).round(1)
        dist[col] = pct
    likert_df = pd.DataFrame(dist).T
    likert_df = likert_df.reindex(columns=range(1,8))

    fig = go.Figure()
    neg_scores = [4,3,2,1]
    neg_colors = ["#dddddd","#91bfdb","#4575b4","#313695"]
    for score, color in zip(neg_scores, neg_colors):
        fig.add_trace(go.Bar(
            y=likert_df.index,
            x=-likert_df[score],
            name=f"{score}점",
            orientation='h',
            marker_color=color
        ))
    fig.add_trace(go.Bar(
        y=likert_df.index,
        x=likert_df[5],
        name="5점",
        orientation='h',
        marker_color="#fee090"
    ))
    for score, color in zip([6,7],["#fc8d59","#d73027"]):
        fig.add_trace(go.Bar(
            y=likert_df.index,
            x=likert_df[score],
            name=f"{score}점",
            orientation='h',
            marker_color=color
        ))

    # **0 위치에 검은색 실선 추가**
    fig.add_vline(
        x=0,
        line_color="black",
        line_width=2,
        line_dash="solid"
    )

    fig.update_layout(
        barmode='relative',
        title="DQ7-E 도서관 이미지 분포 (다이버징 바)",
        xaxis=dict(visible=False),
        legend=dict(traceorder='normal'),
        height=250,
        margin=dict(t=30, b=5),
    )

    table_df = likert_df.copy()
    return fig, table_df



def plot_pair_bar(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix)]
    if len(cols) < 2:
        return None, None, ""
    col1, col2 = cols[0], cols[1]
    question = f"{col1} vs (2순위)"
    s1 = df[col1].dropna().astype(str)
    s2 = df[col2].dropna().astype(str)
    cats = sorted(set(s1.unique()).union(s2.unique()))
    labels = [c.split('. ',1)[-1] if '. ' in c else c for c in cats]

    counts1 = s1.value_counts().reindex(cats, fill_value=0)
    counts2 = s2.value_counts().reindex(cats, fill_value=0)
    pct1 = (counts1 / counts1.sum() * 100).round(1)
    pct2 = (counts2 / counts2.sum() * 100).round(1)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=counts1, name='1순위', marker_color='light blue', text=counts1, textposition='outside'))
    fig.add_trace(go.Bar(x=labels, y=counts2, name='2순위', marker_color='light green', text=counts2, textposition='outside'))
    fig.update_layout(
        barmode='stack',
        title=f"{question}",
        yaxis_title="응답자 수",
        height=550,
        margin=dict(t=50, b=70),
        xaxis_tickangle=-23
    )

    table_df = pd.DataFrame({
        '1순위 응답 수': counts1.values,
        '1순위 비율(%)': pct1.values,
        '2순위 응답 수': counts2.values,
        '2순위 비율(%)': pct2.values
    }, index=labels).T  # <- DataFrame 형태로
    return fig, table_df, question


# ─────────────────────────────────────────────────────
# 페이지 구조
# ─────────────────────────────────────────────────────
def page_home(df):
    st.subheader("👤 인구통계 문항 (SQ1 ~ 5 / BQ1 ~ 2)")
    soc_qs = [c for c in df.columns if c.startswith("SQ") or c.startswith("BQ")]
    for q in soc_qs:
        try:
            if q.startswith("SQ2"):
                bar, table_df = plot_age_histogram_with_labels(df, q)
            elif q.startswith("BQ2"):
                bar, table_df = plot_bq2_bar(df, q)
            elif q.startswith("SQ4"):
                bar, table_df = plot_sq4_custom_bar(df, q)
            else:
                bar, table_df = plot_categorical_stacked_bar(df, q)
            render_chart_and_table(bar, table_df, q, key_prefix="home")
            st.divider()
        except Exception as e:
            st.error(f"{q} 에러: {e}")

def page_basic_vis(df):
    st.subheader("📈 7점 척도 만족도 문항 (Q1 ~ Q8)")
    likert_qs = [
        col for col in df.columns
        if re.match(r"Q[1-9][\.-]", str(col))
    ]
    section_mapping = {
        "공간 및 이용편의성":       [q for q in likert_qs if q.startswith("Q1-")],
        "정보 획득 및 활용":       [q for q in likert_qs if q.startswith("Q2-")],
        "소통 및 정책 활용":       [q for q in likert_qs if q.startswith("Q3-")],
        "문화·교육 향유":         [q for q in likert_qs if q.startswith("Q4-")],
        "사회적 관계 형성":       [q for q in likert_qs if q.startswith("Q5-")],
        "개인의 삶과 역량":       [q for q in likert_qs if q.startswith("Q6-")],
        "도서관의 공익성 및 기여도": [
            q for q in likert_qs 
            if q.startswith("Q7-") or q.startswith("Q8")
        ]
    }
    tabs2 = st.tabs(list(section_mapping.keys()))
    for tab, section_name in zip(tabs2, section_mapping.keys()):
        with tab:
            st.markdown(f"### {section_name}")
            for q in section_mapping[section_name]:
                try:
                    bar, table_df = plot_stacked_bar_with_table(df, q)
                    render_chart_and_table(bar, table_df, q, key_prefix="basic")
                except Exception as e:
                    st.error(f"{q} 에러: {e}")
            st.divider()

def page_short_keyword(df):
    with st.spinner("🔍 GPT 기반 키워드 분석 중..."):
        target_cols = [col for col in df.columns if "Q9-DS-4" in col]
        if not target_cols:
            st.warning("Q9-DS-4 관련 문항을 찾을 수 없습니다.")
            return
        answers = df[target_cols[0]].dropna().astype(str).tolist()
        df_result = process_answers(answers)
        show_short_answer_keyword_analysis(df_result)

# ------------------------------------------
# Q1~Q6 중분류별 A/B/C (서비스 평가/효과/만족도) 평균값 계산 및 시각화
# ------------------------------------------

CATEGORY_MAP = {
    "공간 및 이용편의성": "Q1",
    "정보 획득 및 활용": "Q2",
    "소통 및 정책 활용": "Q3",
    "문화·교육 향유": "Q4",
    "사회적 관계 형성": "Q5",
    "개인의 삶과 역량": "Q6",
}
TYPE_MAP = {
    "A": "서비스 평가",
    "B": "서비스 효과",
    "C": "전반적 만족도",
}

def get_abc_category_means(df):
    result = []
    for cat, prefix in CATEGORY_MAP.items():
        for t in ["A", "B", "C"]:
            if t == "C":
                # "Q1-C" 또는 "Q1-C-"로 시작하는 모든 컬럼 포함
                cols = [c for c in df.columns if c.startswith(f"{prefix}-C")]
            else:
                cols = [c for c in df.columns if c.startswith(f"{prefix}-{t}-")]
            if not cols:
                mean_val = None
            else:
                vals = df[cols].apply(pd.to_numeric, errors='coerce')
                mean_val = 100 * (vals.mean(axis=1, skipna=True) - 1) / 6
                mean_val = mean_val.mean()
            result.append({
                "중분류": cat,
                "문항유형": TYPE_MAP[t],
                "평균값": round(mean_val, 2) if mean_val is not None else None
            })
    return pd.DataFrame(result)


def plot_abc_radar(df_mean):
    categories = df_mean['중분류'].unique().tolist()
    fig = go.Figure()
    color_map = {
        "서비스 평가": "#2ca02c",
        "서비스 효과": "#1f77b4",
        "전반적 만족도": "#d62728"
    }
    for t in TYPE_MAP.values():
        vals = df_mean[df_mean['문항유형'] == t].set_index('중분류').reindex(categories)['평균값'].tolist()
        fig.add_trace(go.Scatterpolar(
            r = vals + [vals[0]],
            theta = categories + [categories[0]],
            fill = 'none',
            name = t,
            line=dict(color=color_map.get(t, None)),
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[50, 100])),
        title="중분류별 서비스 평가/효과/만족도 (A/B/C) 레이더차트",
        showlegend=True,
        height=450
    )
    return fig

def plot_abc_grouped_bar(df_mean):
    fig = px.bar(
        df_mean,
        x='중분류',
        y='평균값',
        color='문항유형',
        barmode='group',
        text='평균값',
        height=450,
        title="중분류별 서비스 평가/효과/만족도 (A/B/C) 평균값"
    )
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig.update_yaxes(range=[0,100])
    return fig

#----------------------------
#이용자 세그먼트 분석
#------------------------------
# 1. 옵션 및 매핑(문항명 변경에 무관)
SEGMENT_OPTIONS = [
    {"label": "SQ1. 성별",        "key": "SQ1"},
    {"label": "SQ2. 연령",        "key": "SQ2"},
    {"label": "SQ3. 거주지",      "key": "SQ3"},
    {"label": "SQ4. 주 이용 도서관", "key": "SQ4"},
    {"label": "SQ5. 주로 이용 서비스", "key": "SQ5"},
    {"label": "DQ1. 월평균 이용 빈도", "key": "DQ1"},
    {"label": "DQ2. 이용기간", "key": "DQ2"},
    {"label": "DQ4. (1순위)이용목적", "key": "DQ4"},
]
MIDCAT_MAP = {
    "공간 및 이용편의성": "Q1-",
    "정보 획득 및 활용": "Q2-",
    "소통 및 정책 활용": "Q3-",
    "문화·교육 향유": "Q4-",
    "사회적 관계 형성": "Q5-",
    "개인의 삶과 역량": "Q6-",
    "자치구 구성 문항": "Q9-D-3",
    "공익성 및 기여도": ["Q7-", "Q8-"],   # <- Q7과 Q8을 하나로 합침!
}
COLOR_CYCLER = cycle(px.colors.qualitative.Plotly)

# 2. 동적으로 실제 세그먼트 컬럼 리스트 반환
def get_segment_columns(df, key):
    if key == "DQ2":
        if "DQ2_YEARS_GROUP" in df.columns:
            return ["DQ2_YEARS_GROUP"]
        elif "DQ2_YEARS" in df.columns:
            return ["DQ2_YEARS"]
        return [col for col in df.columns if "DQ2" in col]
    elif key == "DQ4":
        return [col for col in df.columns if ("DQ4" in col) and ("1순위" in col)]
    elif key == "DQ1":
        # 파생(범주화) 먼저
        if "DQ1_FREQ" in df.columns:
            return ["DQ1_FREQ"]
        return [col for col in df.columns if "DQ1" in col]
    elif key == "DQ2":
        if "DQ2_YEARS" in df.columns:
            return ["DQ2_YEARS"]
        return [col for col in df.columns if "DQ2" in col]
    else:
        return [col for col in df.columns if key in col]

# 3. DQ1/DQ2/DQ4 파생컬럼 전처리
def add_derived_columns(df):
    df = df.copy()
    # DQ1: 월평균 이용 → 연간 환산 후 범주화
    if "DQ1_FREQ" not in df.columns:
        dq1_cols = [c for c in df.columns if "DQ1" in c]
        if dq1_cols:
            dq1_col = dq1_cols[0]
            monthly = pd.to_numeric(df[dq1_col].astype(str).str.extract(r"(\d+\.?\d*)")[0], errors="coerce")
            yearly = monthly * 12
            bins = [0,12,24,48,72,144,1e10]
            labels = ["0~11회: 연 1회 미만", "12~23회: 월 1회", "24~47회: 월 2~4회", "48~71회: 주 1회", "72~143회: 주 2~3회", "144회 이상: 거의 매일"]
            df["DQ1_FREQ"] = pd.cut(yearly, bins=bins, labels=labels, right=False)
    # DQ2: 이용기간 → 년수로 통일 + 5년 단위 범주화
    if "DQ2_YEARS" not in df.columns or "DQ2_YEARS_GROUP" not in df.columns:
        dq2_cols = [c for c in df.columns if "DQ2" in c]
        if dq2_cols:
            dq2_col = dq2_cols[0]
            def parse_years(s):
                s = str(s)
                m = re.match(r'^(\d+)\s*년\s*(\d+)\s*개월$', s)
                if m: return int(m.group(1)) + (1 if int(m.group(2)) > 0 else 0)
                m = re.match(r'^(\d+)\s*년$', s)
                if m: return int(m.group(1))
                m = re.match(r'^(\d+)\s*개월$', s)
                if m: return 1
                return None
            years = df[dq2_col].dropna().apply(parse_years)
            df["DQ2_YEARS"] = years

        # 5년 단위 범주화
        def year_group(y):
            if pd.isna(y):
                return None
            y = int(y)
            if y < 5:
                return "1~4년"
            elif y < 10:
                return "5~9년"
            elif y < 15:
                return "10~14년"
            elif y < 20:
                return "15~19년"
            else:
                return "20년 이상"
        df["DQ2_YEARS_GROUP"] = df["DQ2_YEARS"].apply(year_group)



    # DQ4: (1순위)만 파생
    if "DQ4_1ST" not in df.columns:
        dq4_cols = [c for c in df.columns if ("DQ4" in c) and ("1순위" in c)]
        if dq4_cols:
            df["DQ4_1ST"] = df[dq4_cols[0]]

    # SQ2: 5세 단위 범주화 (SQ2_GROUP)
    if "SQ2_GROUP" not in df.columns:
        sq2_cols = [c for c in df.columns if "SQ2" in c]
        if sq2_cols:
            sq2_col = sq2_cols[0]
            # 숫자 추출 후 정수 변환
            data = df[sq2_col].dropna().astype(str).str.extract(r'(\d+)')
            data.columns = ['age']
            data['age'] = pd.to_numeric(data['age'], errors='coerce')
            def age_group(age):
                if pd.isna(age):
                    return None
                age = int(age)
                if age < 15:
                    return '14세 이하'
                elif age >= 80:
                    return '80세 이상'
                else:
                    base = (age // 5) * 5
                    return f"{base}~{base+4}세"
            df["SQ2_GROUP"] = data['age'].apply(age_group)
    return df



# 4. 메인 분석 및 시각화 함수
def page_segment_analysis(df):
    st.header("🧩 이용자 세그먼트 조합 분석")
    st.markdown("""
    - SQ1~5, DQ1, DQ2, DQ4(1순위) 중 **최대 3개** 문항 선택  
    - 선택한 보기 조합별(응답자 5명 이상)로 Q1~Q6, Q9-D-3, 공익성/기여도(Q7,Q8) 중분류별 만족도 평균을 **히트맵**으로 비교
    """)

    seg_labels = [o["label"] for o in SEGMENT_OPTIONS]
    sel = st.multiselect("세그먼트 조건 (최대 3개)", seg_labels, default=seg_labels[:2], max_selections=3)
    if not sel:
        st.info("최소 1개 이상을 선택하세요.")
        return
    selected_keys = [o["key"] for o in SEGMENT_OPTIONS if o["label"] in sel]

    df2 = add_derived_columns(df)

    # 동적으로 실제 컬럼 추출(복수 선택 시 모두 사용)
    segment_cols = []
    for key in selected_keys:
        segment_cols.extend(get_segment_columns(df2, key))
    segment_cols = list(dict.fromkeys(segment_cols))  # 중복 제거

    if not segment_cols:
        st.warning("선택한 세그먼트 조건에 해당하는 컬럼이 없습니다.")
        return

    # 분석 대상: Q1~Q6, Q9-D-3, 공익성/기여도(Q7,Q8)
    midcat_prefixes = list(MIDCAT_MAP.values())
    analysis_cols = []
    for p in midcat_prefixes:
        if isinstance(p, list):
            for sub_p in p:
                analysis_cols.extend([c for c in df2.columns if c.startswith(sub_p)])
        else:
            analysis_cols.extend([c for c in df2.columns if c.startswith(p)])
    seg_df = df2[segment_cols + analysis_cols].copy()
    seg_df = seg_df.dropna(subset=segment_cols, how='any')
    for c in segment_cols:
        seg_df[c] = seg_df[c].astype(str)

    group = seg_df.groupby(segment_cols, dropna=False)
    counts = group.size().reset_index(name="응답자수")
    counts = counts[counts["응답자수"] >= 5]
    if counts.empty:
        st.warning("응답자 5명 이상인 세그먼트 조합이 없습니다.")
        return

    # 1. 세그먼트별 중분류별 평균점수 집계
    midcats = list(MIDCAT_MAP.keys())
    group_means = []

    for idx, row in counts.iterrows():
        key = tuple(row[c] for c in segment_cols)
        gdf = group.get_group(key)
        means = {}
        for cat, prefix in MIDCAT_MAP.items():
            if isinstance(prefix, list):
                cols = []
                for p in prefix:
                    cols += [c for c in gdf.columns if c.startswith(p)]
            else:
                cols = [c for c in gdf.columns if c.startswith(prefix)]
            if not cols:
                means[cat] = None
                continue
            vals = gdf[cols].apply(pd.to_numeric, errors="coerce")
            mean_val = 100 * (vals.mean(axis=1, skipna=True) - 1) / 6
            means[cat] = round(mean_val.mean(), 2)
        seg_info = {col: row[col] for col in segment_cols}
        seg_info.update(means)
        group_means.append(seg_info)

    group_means = pd.DataFrame(group_means)

    # 2. 숫자 세그먼트 컬럼 제거(나이 등, SQ2, DQ2_YEARS 등)
    segment_cols_filtered = [
        c for c in segment_cols
        if not (c.startswith("SQ2") and "GROUP" not in c) and c != "DQ2_YEARS"
    ]

    # 3. 응답자수 merge
    merge_keys = segment_cols_filtered
    counts_merge = counts[merge_keys + ["응답자수"]]
    group_means = pd.merge(group_means, counts_merge, how='left', on=merge_keys)

    # 4. 중분류평균/전체평균대비편차 추가
    group_means["중분류평균"] = group_means[midcats].mean(axis=1).round(2)
    overall_means = group_means[midcats].mean(axis=0)
    overall_mean_of_means = overall_means.mean()
    group_means["전체평균대비편차"] = (group_means["중분류평균"] - overall_mean_of_means).round(2)

    # 5. 표 컬럼 순서
    table_cols = segment_cols_filtered + midcats + ["중분류평균", "전체평균대비편차", "응답자수"]
    table_with_stats = group_means[table_cols]


    # --- 응답자 수 기준 상위 10개 세그먼트 조합의 중분류 만족도 프로파일 비교 (단일 레이더) ---
    st.markdown("### 응답자 수 기준 상위 10개 세그먼트 조합의 중분류 만족도 프로파일 비교")
    top_n = 10
    top_df = group_means.nlargest(top_n, "응답자수").copy()
    midcats = list(MIDCAT_MAP.keys())

    # 전체 평균 프로파일 (reference)
    overall_profile = group_means[midcats].mean(axis=0)
    overall_vals = [overall_profile.get(mc, overall_profile.mean()) for mc in midcats]
    overall_closed = overall_vals + [overall_vals[0]]
    cats_closed = midcats + [midcats[0]]

    fig_radar = go.Figure()
    # 전체 평균
    fig_radar.add_trace(go.Scatterpolar(
        r=overall_closed,
        theta=cats_closed,
        fill=None,
        name="전체 평균",
        line=dict(dash="dash", width=4, color = "black"),
        opacity=0.5

    ))

    colors = px.colors.qualitative.Plotly
    for i, (_, row) in enumerate(top_df.iterrows()):
        combo_label = " | ".join([str(row[c]) for c in segment_cols_filtered])
        vals = [row[mc] if not pd.isna(row[mc]) else overall_profile.get(mc, overall_profile.mean()) for mc in midcats]
        vals_closed = vals + [vals[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=cats_closed,
            fill=None,
            name=f"{combo_label} (n={int(row['응답자수'])})",
            hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
            marker=dict(color=colors[i % len(colors)]),
            opacity=0.9
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(range=[50, 100])),
        title=f"상위 {min(top_n, len(top_df))}개 세그먼트 조합 중분류 만족도 프로파일 vs 전체 평균",
        height=500,
        showlegend=True,
        legend=dict(orientation="v", y=0.85, x=1.02)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # --- 추가 지표/편차 계산 ---
    # 전체 중분류별 평균 벡터
    overall_means = group_means[midcats].mean(axis=0)
    # 각 조합별 delta (전체 평균 대비)
    for mc in midcats:
        group_means[f"{mc}_delta"] = group_means[mc] - overall_means[mc]
    # 순위 변화 계산
    ref_rank = overall_means.rank(ascending=False)
    rank_df = group_means[[mc for mc in midcats]].rank(ascending=False, axis=1)
    rank_change = rank_df.subtract(ref_rank, axis=1)
    group_means["조합"] = group_means.apply(lambda r: " | ".join([str(r[c]) for c in segment_cols_filtered]), axis=1)

    

    # --- 히트맵 + Delta 히트맵 ---
    st.markdown("### 히트맵 + 전체 평균 대비 중분류별 편차 히트맵")
    # 원본 히트맵 재사용 (중분류 평균)
    heatmap_plot = group_means.set_index("조합")[midcats]
    fig_abs = px.imshow(
        heatmap_plot,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title="세그먼트 조합별 중분류 평균",
        labels=dict(x="중분류", y="세그먼트 조합", color="평균점수")
    )
    st.plotly_chart(fig_abs, use_container_width=True)
    # Delta 히트맵
    delta_plot = group_means.set_index("조합")[[f"{mc}_delta" for mc in midcats]]
    # 컬럼명 다시 원래로
    delta_plot.columns = midcats
    fig_delta = px.imshow(
        delta_plot,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="전체 평균 대비 편차 (Delta)",
        labels=dict(x="중분류", y="세그먼트 조합", color="편차")
    )
    st.plotly_chart(fig_delta, use_container_width=True)


    # --- 평균 차이 + 간이 신뢰구간 막대 (예: 특정 중분류별 상위 10개) ---
    st.markdown("### 전체 평균 대비 편차와 간이 신뢰구간 (중분류별)")
    import numpy as np
    for mc in midcats[:2]:  # 부담 줄이려고 첫 두 개만; 필요하면 반복 범위 확장
        subset = group_means.nlargest(10, "응답자수").copy()
        subset["delta"] = subset[mc] - overall_means[mc]
        # 근사 표준오차: p*(1-p)/n 형태를 변형 (점수 범위 0~100이므로 단순화)
        # 실제로는 개별 응답자 데이터를 bootstrap 하는 게 정확함
        subset["se"] = np.sqrt((subset[mc] * (100 - subset[mc]) / subset["응답자수"]).clip(lower=0))
        fig_ci = go.Figure()
        fig_ci.add_trace(go.Bar(
            x=subset["조합"],
            y=subset["delta"],
            error_y=dict(type="data", array=subset["se"]),
            name=f"{mc} 편차"
        ))
        fig_ci.add_hline(y=0, line_dash="dash", line_color="black")
        fig_ci.update_layout(
            title=f"{mc} 전체 평균 대비 편차 (신뢰구간, 상위 5개 조합)",
            yaxis_title="편차",
            height=350,
            margin=dict(t=40, b=60)
        )
        st.plotly_chart(fig_ci, use_container_width=True)


    # --- Small Multiples: 중분류별 상위 3개 조합 비교 ---
    st.markdown("### Small Multiples: 중분류별 세그먼트 조합 비교 (상위 10개)")
    top3 = group_means.nlargest(10, "응답자수").copy()
    for mc in midcats:
        tmp = top3[[*segment_cols_filtered, mc, "응답자수"]].copy()
        tmp["조합"] = tmp.apply(lambda r: " | ".join([str(r[c]) for c in segment_cols_filtered]), axis=1)
        fig_small = px.bar(
            tmp,
            x="조합",
            y=mc,
            text=mc,
            title=f"{mc} 비교 (상위 10개 세그먼트 조합)"
        )
        fig_small.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        st.plotly_chart(fig_small, use_container_width=True)

    # 8. 통합 표 한 번에 출력
    st.markdown("#### 세그먼트 조합별 중분류별 만족도 및 응답자수")
    st.dataframe(table_with_stats, use_container_width=True)

    # --- 유틸: DQ4(이용 목적) 컬럼 추론 -----------------
    def infer_dq4_primary_column(df):
        for c in df.columns:
            if "DQ4" in c and "1순위" in c:
                return c
        for c in df.columns:
            if "DQ4" in c:
                return c
        return None

def show_basic_strategy_insights(df):
    # 1. 이용 목적 × 전반 만족도 (중분류 레이더, 하나의 차트에 전체 + 상위 목적들)
    st.subheader("1. 이용 목적 (DQ4 계열) × 전반 만족도 (중분류 기준 레이더)")
    purpose_col = None
    for c in df.columns:
        if "DQ4" in c and "1순위" in c:
            purpose_col = c
            break
    if purpose_col is None:
        for c in df.columns:
            if "DQ4" in c:
                purpose_col = c
                break

    if purpose_col is None:
        st.warning("이용 목적 관련 컬럼(DQ4 계열)을 찾을 수 없어 전반 만족도 대비 레이더를 그릴 수 없습니다.")
    else:
        overall_mid_scores = compute_midcategory_scores(df)
        if overall_mid_scores.empty:
            st.warning("중분류 점수 계산에 필요한 문항이 부족합니다.")
        else:
            midcats = list(overall_mid_scores.index)
            purpose_counts = df[purpose_col].dropna().astype(str).value_counts()

            # 기본 top_n 값 (존재하는 목적 개수에 맞춰 클램프)
            default_n = min(5, len(purpose_counts))
            # 레이더 차트 생성 (현재 top_n 기준)
            top_n = st.session_state.get("strategy_radar_top_n_main", default_n)
            top_purposes = purpose_counts.nlargest(top_n).index.tolist()

            fig = go.Figure()
            overall_vals = [overall_mid_scores.get(m, 0) for m in midcats]
            fig.add_trace(go.Scatterpolar(
                r=overall_vals + [overall_vals[0]],
                theta=midcats + [midcats[0]],
                fill=None,
                name="전체 평균",
                line=dict(dash='dash', width=2),
                opacity=1
            ))

            colors = px.colors.qualitative.Plotly
            for i, purpose in enumerate(top_purposes):
                subset = df[df[purpose_col].astype(str) == purpose]
                if len(subset) < 5:
                    continue
                purpose_scores = compute_midcategory_scores(subset)
                vals = [purpose_scores.get(m, overall_mid_scores.get(m, 0)) for m in midcats]
                fig.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=midcats + [midcats[0]],
                    fill=None,
                    name=f"{purpose} (n={int(purpose_counts[purpose])})",
                    hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
                    marker=dict(color=colors[i % len(colors)]),
                    opacity=0.6
                ))

            fig.update_layout(
                polar=dict(radialaxis=dict(range=[50, 100])),
                title=f"상위 {len(top_purposes)}개 이용 목적별 중분류 만족도 vs 전체 평균",
                height=450,
                legend=dict(orientation="v", x=1.02, y=0.9)
            )
            st.plotly_chart(fig, use_container_width=True)

            # 여기서 슬라이더를 차트 아래에 둠: 변경되면 rerun 되면서 위 차트도 top_n 반영
            top_n = st.number_input(
                "레이더에 표시할 상위 이용 목적 개수",
                min_value=1,
                max_value=max(1, len(purpose_counts)),
                value=default_n,
                step=1,
                key="strategy_radar_top_n_main"
            )

            # 요약 테이블: 목적별 중분류 점수 + 전체 평균 (top_n 반영)
            top_purposes = purpose_counts.nlargest(top_n).index.tolist()
            summary_rows = []
            for purpose in top_purposes:
                subset = df[df[purpose_col].astype(str) == purpose]
                if len(subset) < 5:
                    continue
                purpose_scores = compute_midcategory_scores(subset)
                row = {"이용목적": purpose, "응답자수": int(purpose_counts[purpose])}
                for m in midcats:
                    row[f"{m} (목적)"] = round(purpose_scores.get(m, overall_mid_scores.get(m, 0)), 1)
                    row[f"{m} (전체 평균)"] = round(overall_mid_scores.get(m, 0), 1)
                summary_rows.append(row)
            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)
                st.markdown("#### 상위 이용 목적별 중분류 프로파일 요약")
                st.dataframe(summary_df)


    # 2. 이용 목적 × 세부 항목 효과 (Q6-B 계열)
    st.subheader("2. 이용 목적 (DQ4 계열) × 세부 항목 효과 (Q6-B 계열)")
    q6b_cols = [c for c in df.columns if c.startswith("Q6-B")]
    if purpose_col is None:
        st.warning("이용 목적 컬럼이 없어 Q6-B 계열 효과를 이용 목적별로 비교할 수 없습니다.")
    elif not q6b_cols:
        st.warning("Q6-B 계열 문항을 찾을 수 없습니다.")
    else:
        purpose_counts = df[purpose_col].dropna().astype(str).value_counts()
        effect_rows = []
        for purpose in purpose_counts.index:
            subset = df[df[purpose_col].astype(str) == purpose]
            if len(subset) < 5:
                continue
            vals = subset[q6b_cols].apply(pd.to_numeric, errors='coerce')
            scaled = 100 * (vals.mean(axis=1, skipna=True) - 1) / 6
            mean_effect = scaled.mean()
            effect_rows.append({
                "이용목적": purpose,
                "Q6-B 계열 효과 평균(0~100)": round(mean_effect, 2),
                "응답자수": len(scaled.dropna())
            })
        if effect_rows:
            effect_df = pd.DataFrame(effect_rows).sort_values("Q6-B 계열 효과 평균(0~100)", ascending=False)
            fig = px.bar(
                effect_df,
                x="이용목적",
                y="Q6-B 계열 효과 평균(0~100)",
                text="Q6-B 계열 효과 평균(0~100)",
                title="이용 목적별 Q6-B 계열 세부 효과 평균 비교",
                hover_data=["응답자수"]
            )
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(effect_df)
        else:
            st.info("이용 목적별로 충분한 응답이 없어 Q6-B 효과 비교를 할 수 없습니다.")

    # 3. 주이용서비스별 중분류 만족도 (SQ5)
    st.subheader("3. 주이용서비스별 중분류 만족도 (SQ5 기준)")
    service_col = None
    for candidate in df.columns:
        if "SQ5" in candidate or "주로 이용 서비스" in candidate:
            service_col = candidate
            break
    if service_col is None:
        st.warning("주이용서비스 관련 컬럼을 찾을 수 없어 중분류 만족도 비교를 할 수 없습니다.")
    else:
        services = df[service_col].dropna().astype(str).unique()
        overall_mid_scores = compute_midcategory_scores(df)
        midcats = list(overall_mid_scores.index)
        plot_data = []
        for service in services:
            subset = df[df[service_col].astype(str) == service]
            if len(subset) < 5:
                continue
            service_scores = compute_midcategory_scores(subset)
            for m in midcats:
                plot_data.append({
                    "주이용서비스": service,
                    "중분류": m,
                    "서비스별 만족도": service_scores.get(m, None),
                    "전체 평균": overall_mid_scores.get(m, None)
                })
        if plot_data:
            plot_df = pd.DataFrame(plot_data)
            # grouped bar: 각 서비스별 중분류 비교
            fig = px.bar(
                plot_df,
                x="중분류",
                y="서비스별 만족도",
                color="주이용서비스",
                barmode="group",
                title="주이용서비스별 중분류 만족도 비교",
                text="서비스별 만족도"
            )
            # overlay overall average as line per midcategory
            avg_df = plot_df.drop_duplicates(subset=["중분류"])[["중분류", "전체 평균"]]
            fig.add_trace(go.Scatter(
                x=avg_df["중분류"],
                y=avg_df["전체 평균"],
                mode="lines+markers",
                name="전체 평균",
                line=dict(dash="dash"),
                hovertemplate="%{x}: %{y:.1f}<extra></extra>"
            ))
            # bar 트레이스에만 라벨 포맷 적용
            for trace in fig.data:
                if getattr(trace, "type", None) == "bar":
                    trace.texttemplate = '%{text:.1f}'
                    trace.textposition = 'outside'
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("주이용서비스별로 비교할 충분한 데이터가 없습니다.")

    # 4. 불이용 사유 × 중분류 만족도
    st.subheader("4. 불이용 사유 × 중분류 만족도")
    reason_col = None
    for candidate in df.columns:
        low = candidate.lower()
        if "불이용" in candidate or "이용 안함" in low or "이용하지" in low or "사용 안함" in low:
            reason_col = candidate
            break
    if reason_col is None:
        st.warning("불이용 사유 관련 컬럼을 찾을 수 없어 분석할 수 없습니다.")
    else:
        reasons = df[reason_col].dropna().astype(str).unique()
        rows_exist = False
        for reason in reasons:
            subset = df[df[reason_col].astype(str) == reason]
            if len(subset) < 5:
                continue
            reason_scores = compute_midcategory_scores(subset)
            if reason_scores.empty:
                continue
            rows_exist = True
            plot_df = pd.DataFrame({
                "중분류": list(reason_scores.index),
                "만족도": [reason_scores.get(m, None) for m in reason_scores.index]
            })
            fig = px.bar(
                plot_df,
                x="중분류",
                y="만족도",
                text="만족도",
                title=f"불이용 사유 '{reason}' 별 중분류 만족도",
            )
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
        if not rows_exist:
            st.info("불이용 사유별로 비교할 충분한 데이터가 없습니다.")

    # 5. 이용 목적 × 운영시간 만족도 (Q7-D-7)
    st.subheader("5. 이용 목적 (DQ4 계열) × 운영시간 만족도 (Q7-D-7)")
    # infer time satisfaction column
    time_sat_col = None
    for c in df.columns:
        if c.upper().startswith("Q7-D-7"):
            time_sat_col = c
            break
        if "운영시간" in c or "시간 만족도" in c:
            time_sat_col = c
            break

    if purpose_col is None or time_sat_col is None:
        st.warning("이용 목적 또는 운영시간 만족도 문항을 찾을 수 없어 비교할 수 없습니다.")
    else:
        purpose_counts = df[purpose_col].dropna().astype(str).value_counts()
        rows = []
        for purpose in purpose_counts.index:
            subset = df[df[purpose_col].astype(str) == purpose]
            if subset.empty:
                continue
            vals = pd.to_numeric(subset[time_sat_col], errors='coerce').dropna().astype(float)
            if vals.empty:
                continue
            mean_score = vals.mean()
            rows.append({
                "이용목적": purpose,
                "운영시간 만족도 평균": round(mean_score, 2),
                "응답자수": len(vals)
            })
        if rows:
            time_df = pd.DataFrame(rows).sort_values("운영시간 만족도 평균", ascending=False)
            fig = px.bar(
                time_df,
                x="이용목적",
                y="운영시간 만족도 평균",
                text="운영시간 만족도 평균",
                title="이용 목적별 운영시간(기대 대비) 만족도",
                hover_data=["응답자수"]
            )
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(time_df)
        else:
            st.info("비교 가능한 이용목적별 운영시간 만족도 데이터가 충분하지 않습니다.")

# ─────────────────────────────────────────────────────
# 실행 엔트리
# ─────────────────────────────────────────────────────
# 페이지 설정은 가장 위에 한 번만
st.set_page_config(
    page_title="공공도서관 설문 시각화 대시보드",
    layout="wide"
)

# 사이드바 모드 선택
mode = st.sidebar.radio("분석 모드", ["기본 분석", "심화 분석", "전략 인사이트(기본)"])

# 업로드 처리
uploaded = st.file_uploader("📂 엑셀(.xlsx) 파일 업로드", type=["xlsx"])
if not uploaded:
    st.info("데이터 파일을 업로드해 주세요.")
    st.stop()

try:
    df = pd.read_excel(uploaded)
    st.success("✅ 업로드 완료")
except Exception as e:
    st.error(f"파일 읽기 실패: {e}")
    st.stop()

# 모드별로 탭/내용 분리
if mode == "기본 분석":
    tabs = st.tabs([
        "👤 응답자 정보",
        "📈 만족도 기본 시각화",
        "🗺️ 자치구 구성 문항",
        "📊 도서관 이용양태 분석",
        "🖼️ 도서관 이미지 분석",
        "🏋️ 도서관 강약점 분석",
    ])

    with tabs[0]:
        page_home(df)

    with tabs[1]:
        page_basic_vis(df)

    with tabs[2]:
        st.header("🗺️ 자치구 구성 문항 분석")
        sub_tabs = st.tabs([
            "7점 척도 시각화",
            "단문 응답 분석",
            "장문 서술형 분석"
        ])
        with sub_tabs[0]:
            st.subheader("자치구 구성 문항 (7점 척도)")
            subregion_cols = [c for c in df.columns if "Q9-D-" in c]
            if not subregion_cols:
                st.error("Q9-D- 로 시작하는 문항을 찾을 수 없습니다.")
            else:
                for idx, col in enumerate(subregion_cols):
                    bar, tbl = plot_stacked_bar_with_table(df, col)
                    st.markdown(f"##### {col}")
                    render_chart_and_table(bar, tbl, col, key_prefix=f"subregion-{idx}")
        with sub_tabs[1]:
            page_short_keyword(df)
        with sub_tabs[2]:
            st.subheader("장문 서술형 분석 (Q9-DS-5)")
            long_cols = [c for c in df.columns if "Q9-DS-5" in c]
            if not long_cols:
                st.warning("Q9-DS-5 관련 문항을 찾을 수 없습니다.")
            else:
                answers = df[long_cols[0]].dropna().astype(str).tolist()
                df_long = process_answers(answers)
                show_short_answer_keyword_analysis(df_long)

    with tabs[3]:
        st.header("📊 도서관 이용양태 분석")
        sub_tabs = st.tabs(["DQ1~5", "DQ6 계열"])
        with sub_tabs[0]:
            fig1, tbl1, q1 = plot_dq1(df)
            render_chart_and_table(fig1, tbl1, q1, key_prefix="dq1")

            fig2, tbl2, q2 = plot_dq2(df)
            render_chart_and_table(fig2, tbl2, q2, key_prefix="dq2")

            fig3, tbl3, q3 = plot_dq3(df)
            render_chart_and_table(fig3, tbl3, q3, key_prefix="dq3")

            fig4, tbl4, q4 = plot_dq4_bar(df)
            render_chart_and_table(fig4, tbl4, q4, key_prefix="dq4")

            fig5, tbl5, q5 = plot_dq5(df)
            render_chart_and_table(fig5, tbl5, q5, key_prefix="dq5")
        with sub_tabs[1]:
            st.subheader("DQ6 계열 문항 분석")
            dq6_cols = [c for c in df.columns if c.startswith("DQ6")]
            if not dq6_cols:
                st.warning("DQ6 계열 문항이 없습니다.")
            else:
                for col in dq6_cols:
                    st.markdown(f"### {col}")
                    if col == dq6_cols[0]:
                        multi = df[col].dropna().astype(str).str.split(',')
                        exploded = multi.explode().str.strip()
                        counts = exploded.value_counts()
                        percent = (counts / counts.sum() * 100).round(1)

                        fig = go.Figure(go.Bar(
                            x=counts.values, y=counts.index,
                            orientation='h', text=counts.values,
                            textposition='outside', marker_color=get_qualitative_colors(len(counts))
                        ))
                        fig.update_layout(
                            title=col,
                            xaxis_title="응답 수",
                            yaxis_title="서비스",
                            height=400,
                            margin=dict(t=50, b=100)
                        )
                        table_df = pd.DataFrame({
                            '응답 수': counts,
                            '비율 (%)': percent
                        }).T
                        render_chart_and_table(fig, table_df, col, key_prefix="dq6")
                    else:
                        bar, tbl = plot_categorical_stacked_bar(df, col)
                        render_chart_and_table(bar, tbl, col, key_prefix="dq6")

    with tabs[4]:
        st.header("🖼️ 도서관 이미지 분석")
        fig, tbl = plot_likert_diverging(df, prefix="DQ7-E")
        if fig is not None:
            render_chart_and_table(fig, tbl, "DQ7-E 이미지 분포", key_prefix="image-diverge")
        else:
            st.warning("DQ7-E 문항이 없습니다.")

    with tabs[5]:
        st.header("🏋️ 도서관 강약점 분석")
        fig8, tbl8, q8 = plot_pair_bar(df, "DQ8")
        if fig8 is not None:
            render_chart_and_table(fig8, tbl8, q8, key_prefix="strength")
        else:
            st.warning("DQ8 문항이 없습니다.")
        fig9, tbl9, q9 = plot_pair_bar(df, "DQ9")
        if fig9 is not None:
            render_chart_and_table(fig9, tbl9, q9, key_prefix="weakness")
        else:
            st.warning("DQ9 문항이 없습니다.")

elif mode == "심화 분석":
    tabs = st.tabs(["공통 심화 분석(전체)", "공통 심화 분석(영역)", "이용자 세그먼트 조합 분석"])
    with tabs[0]:
        st.header("🔍 공통 심화 분석(전체)")
        st.subheader("중분류별 전체 만족도 (레이더 차트 및 평균값)")
        radar = plot_midcategory_radar(df)
        if radar is not None:
            st.plotly_chart(radar, use_container_width=True)
            tbl_avg = midcategory_avg_table(df)
            if not tbl_avg.empty:
                show_table(tbl_avg, "중분류별 평균 점수")
                st.markdown("---")
            else:
                st.warning("중분류 평균을 계산할 수 없습니다.")
        else:
            st.warning("필요한 문항이 없어 중분류 점수를 계산할 수 없습니다.")

        st.subheader("중분류 내 문항별 편차")
        mid_scores = compute_midcategory_scores(df)
        if mid_scores.empty:
            st.warning("중분류 문항이 없어 편차를 계산할 수 없습니다.")
        else:
            for mid in mid_scores.index:
                fig, table_df = plot_within_category_bar(df, mid)
                if fig is None:
                    continue
                st.markdown(f"### {mid}")
                st.plotly_chart(fig, use_container_width=True)
                if table_df is not None:
                    show_table(
                        table_df.reset_index().rename(columns={"index": "문항"}),
                        f"{mid} 항목별 편차"
                    )
                    st.markdown("---")
    with tabs[1]:
        st.header("🔍 공통 심화 분석(영역별 A/B/C 비교)")
        df_mean = get_abc_category_means(df)
        radar_fig = plot_abc_radar(df_mean)
        bar_fig = plot_abc_grouped_bar(df_mean)

        st.subheader("중분류별 서비스 평가/효과/만족도 (A/B/C) 레이더 차트")
        st.plotly_chart(radar_fig, use_container_width=True)

        st.subheader("중분류별 서비스 평가/효과/만족도 (A/B/C) 묶음(bar) 차트")
        st.plotly_chart(bar_fig, use_container_width=True)

        st.markdown("#### 상세 데이터")
        st.dataframe(df_mean)
    with tabs[2]:
        page_segment_analysis(df)

elif mode == "전략 인사이트(기본)":
    st.header("🧠 전략 인사이트 (기본)")
    show_basic_strategy_insights(df)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import re
import openai

openai.api_key = st.secrets["openai"]["api_key"]
client = openai.OpenAI(api_key=openai.api_key)

# ─────────────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────────────
def remove_parentheses(text):
    return re.sub(r'\(.*?\)', '', text).strip()
def wrap_label(label, width=10):
    return '<br>'.join([label[i:i+width] for i in range(0, len(label), width)])

# ─────────────────────────────────────────────────────
# SQ2: 연령 히스토그램 + Table
# ─────────────────────────────────────────────────────
def plot_age_histogram_with_labels(df, question):
    data = df[question].dropna().astype(str).str.extract(r'(\d+)')
    data.columns = ['age']
    data['age'] = pd.to_numeric(data['age'], errors='coerce').dropna()

    def age_group(age):
        if age < 15: return '14세 이하'
        elif age >= 80: return '80세 이상'
        else: return f"{(age//5)*5}~{(age//5)*5+4}세"

    data['group'] = data['age'].apply(age_group)
    grouped = data['group'].value_counts().sort_index()
    percent = (grouped / grouped.sum() * 100).round(1)

    # Bar
    fig = go.Figure(go.Bar(
        x=grouped.index, y=grouped.values,
        text=grouped.values, textposition='outside',
        marker_color="#1f77b4"
    ))
    fig.update_layout(
        title=question, yaxis_title="응답 수",
        bargap=0.1, height=450, margin=dict(t=40, b=10)
    )

    # Table
    table_df = pd.DataFrame({'응답 수': grouped, '비율 (%)': percent}).T
    table_fig = go.Figure(go.Table(
        header=dict(values=[""]+list(table_df.columns)),
        cells=dict(values=[table_df.index] + [table_df[c].tolist() for c in table_df.columns])
    ))
    table_fig.update_layout(height=180, margin=dict(t=10, b=5))

    return fig, table_fig

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

    # 자동 줄바꿈 적용
    wrapped_labels = [wrap_label(remove_parentheses(label), width=10) for label in categories]

    colors = px.colors.qualitative.Plotly
    fig = go.Figure(go.Bar(
        x=categories,
        y=counts,
        text=counts,
        textposition='outside',
        marker_color=colors[:len(categories)]
    ))

    y_max = counts.max() + 20
    fig.update_layout(
        title=dict(text=question, font=dict(size=16)),
        yaxis=dict(title="응답 수", range=[0, y_max]),
        height=450,
        margin=dict(t=50, b=100),
        xaxis_tickangle=-30
    )

    # 자동 줄바꿈된 레이블을 표에 사용
    table_df = pd.DataFrame({'응답 수': counts, '비율 (%)': percent}, index=wrapped_labels).T
    table_fig = go.Figure(go.Table(
        header=dict(values=[""] + list(table_df.columns), align='center', height=36, font=dict(size=11)),
        cells=dict(values=[table_df.index] + [table_df[col].tolist() for col in table_df.columns], align='center', height=36, font=dict(size=11))
    ))
    table_fig.update_layout(height=150, margin=dict(t=10, b=5))

    return fig, table_fig

# ─────────────────────────────────────────────────────
# SQ4: 커스텀 누적 가로 Bar + Table
# ─────────────────────────────────────────────────────
def plot_sq4_custom_bar(df, question):
    data = df[question].dropna().astype(str)
    cats = sorted(data.unique())
    counts = data.value_counts().reindex(cats).fillna(0).astype(int)
    percent = (counts/counts.sum()*100).round(1)
    labels = [wrap_label(remove_parentheses(x),10) for x in cats]
    colors = px.colors.qualitative.Plotly

    fig = go.Figure()
    for i, cat in enumerate(cats):
        fig.add_trace(go.Bar(
            x=[percent[cat]], y=[question],
            orientation='h', name=remove_parentheses(cat),
            marker_color=colors[i%len(colors)],
            text=f"{percent[cat]}%", textposition='inside'
        ))
    fig.update_layout(
        barmode='stack', showlegend=True,
        legend=dict(orientation='h', y=-0.5, x=0.5, xanchor='center', traceorder='reversed'),
        title=question, yaxis=dict(showticklabels=False),
        height=250, margin=dict(t=40,b=100)
    )

    # 기존 table_df 생성
    table_df = pd.DataFrame({
        '응답 수': [counts[c] for c in cats],
        '비율 (%)': [percent[c] for c in cats]
    }, index=labels).T
    
    table_fig = go.Figure(go.Table(
        header=dict(values=[""] + list(table_df.columns), align='center'),
        cells=dict(values=[table_df.index] + [table_df[col].tolist() for col in table_df.columns], align='center')
    ))
    table_fig.update_layout(height=120, margin=dict(t=10, b=5))
    return fig, table_fig 

# ─────────────────────────────────────────────────────
# 일반 범주형 누적 Bar + Table SQ5/SQ3/SQ4
# ─────────────────────────────────────────────────────
def plot_categorical_stacked_bar(df, question):
    data = df[question].dropna().astype(str)
    categories_raw = sorted(data.unique())
    categories = [label.split('. ', 1)[-1] for label in categories_raw]

    counts = data.value_counts().reindex(categories_raw).fillna(0).astype(int)
    percent = (counts / counts.sum() * 100).round(1)
    colors = px.colors.qualitative.Plotly

    fig = go.Figure()
    for i, cat in enumerate(reversed(categories)):
        raw_cat = categories_raw[categories.index(cat)]
        fig.add_trace(go.Bar(
            x=[percent[raw_cat]],
            y=[question],
            orientation='h',
            name=cat,
            marker=dict(color=colors[i % len(colors)]),
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

    table_df = pd.DataFrame({
        '응답 수': [counts[c] for c in categories_raw],
        '비율 (%)': [percent[c] for c in categories_raw]
    }, index=categories).T

    # 역순으로 컬럼 뒤집기
    table_df = table_df[table_df.columns[::-1]]

    table_fig = go.Figure(go.Table(
        header=dict(values=[""] + list(table_df.columns), align='center'),
        cells=dict(values=[table_df.index] + [table_df[col].tolist() for col in table_df.columns], align='center')
    ))
    table_fig.update_layout(height=120, margin=dict(t=10, b=5))
    return fig, table_fig 

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

    table_df = pd.DataFrame({
        '응답 수': [int(counts[v]) for v in order],
        '비율 (%)': [percent[v] for v in order]
    }, index=[f"{v}점" for v in order]).T
    table_fig = go.Figure(go.Table(
        header=dict(values=[""] + list(table_df.columns), align='center'),
        cells=dict(values=[table_df.index] + [table_df[c].tolist() for c in table_df.columns], align='center')
    ))
    table_fig.update_layout(height=80, margin=dict(t=10,b=0))
    return fig, table_fig


#--------------------------------------------------------------------------
#단문 분석
#----------------------------------------------------------------------------
# ─────────────────────────────────────────────────────
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


#-----------------------------------------------------------------------------
#페이지 구분
def page_home(df):
    st.subheader("👤 인구통계 문항 (SQ1 ~ 5 / BQ1 ~ 2)")
    soc_qs = [c for c in df.columns if c.startswith("SQ") or c.startswith("BQ")]
    for q in soc_qs:
        try:
            if q.startswith("SQ2"):
                bar, tbl = plot_age_histogram_with_labels(df, q)
            elif q.startswith("BQ2"):
                bar, tbl = plot_bq2_bar(df, q)
            elif q.startswith("SQ4"):
                bar, tbl = plot_sq4_custom_bar(df, q)
            else:
                bar, tbl = plot_categorical_stacked_bar(df, q)
            st.plotly_chart(bar, use_container_width=True)
            st.plotly_chart(tbl, use_container_width=True)
            st.divider()
        except Exception as e:
            st.error(f"{q} 에러: {e}")

def page_basic_vis(df):
    st.subheader("📈 7점 척도 만족도 문항 (Q1 ~ Q8)")
    # ─── likert_qs 수정 ───
    likert_qs = [
        col for col in df.columns
        if (re.match(r"Q[1-9][\.-]", str(col)))  # Q1-, Q1. 모두 매칭
    ]
    # ─────────────────────

    section_mapping = {
        "공간 및 이용편의성":       [q for q in likert_qs if q.startswith("Q1-")],
        "정보 획득 및 활용":       [q for q in likert_qs if q.startswith("Q2-")],
        "소통 및 정책 활용":       [q for q in likert_qs if q.startswith("Q3-")],
        "문화·교육 향유":         [q for q in likert_qs if q.startswith("Q4-")],
        "사회적 관계 형성":       [q for q in likert_qs if q.startswith("Q5-")],
        "개인의 삶과 역량":       [q for q in likert_qs if q.startswith("Q6-")],
        "도서관의 공익성 및 기여도": [
            q for q in likert_qs 
            if q.startswith("Q7-") or q.startswith("Q8")  # 이제 Q8. 문항도 포함
        ]
    }

    tabs = st.tabs(list(section_mapping.keys()))
    for tab, section_name in zip(tabs, section_mapping.keys()):
        with tab:
            st.markdown(f"### {section_name}")
            for q in section_mapping[section_name]:
                bar, tbl = plot_stacked_bar_with_table(df, q)
                st.plotly_chart(bar, use_container_width=True)
                st.plotly_chart(tbl, use_container_width=True)

#------------- 단문분석
def page_short_keyword(df):

    with st.spinner("🔍 GPT 기반 키워드 분석 중..."):
        target_cols = [col for col in df.columns if "Q9-DS-4" in col]
        if not target_cols:
            st.warning("Q9-DS-4 관련 문항을 찾을 수 없습니다.")
            return
        answers = df[target_cols[0]].dropna().astype(str).tolist()
        df_result = process_answers(answers)
        show_short_answer_keyword_analysis(df_result)


# ─────────────────────────────────────────────────────
# DQ1: 세로 막대 + Table (자동 탐색)
# ─────────────────────────────────────────────────────
def plot_dq1(df):
    cols = [c for c in df.columns if c.startswith("DQ1")]
    if not cols:
        return None, None, ""
    question = cols[0]
    # 숫자 추출 및 연 환산
    data = df[question].dropna().astype(str).str.extract(r"(\d+\.?\d*)")[0]
    monthly = pd.to_numeric(data, errors='coerce')
    yearly = monthly * 12
    # 구간화 함수
    def categorize(f):
        try:
            f = float(f)
        except:
            return None
        if f < 12: return "0~11회: 연 1회 미만"
        elif f < 24: return "12~23회: 월 1회 정도"
        elif f < 48: return "24~47회: 월 2~4회 정도"
        elif f < 72: return "48~71회: 주 1회 정도"
        elif f < 144: return "72~143회: 주 2~3회"
        else: return "144회 이상: 거의 매일"
    cat = yearly.apply(categorize)
    order = ["0~11회: 연 1회 미만","12~23회: 월 1회 정도","24~47회: 월 2~4회 정도",
             "48~71회: 주 1회 정도","72~143회: 주 2~3회","144회 이상: 거의 매일"]
    grp = cat.value_counts().reindex(order, fill_value=0)
    pct = (grp/grp.sum()*100).round(1)
    # 그래프
    fig = go.Figure(go.Bar(x=grp.index, y=grp.values, text=grp.values,
                            textposition='outside', marker_color="#1f77b4"))
    fig.update_layout(title=question, xaxis_title="이용 빈도 구간", yaxis_title="응답 수",
                      bargap=0.2, height=450, margin=dict(t=30,b=50), xaxis_tickangle=-15)
    # 테이블
    tbl_df = pd.DataFrame({"응답 수":grp, "비율 (%)":pct}).T
    tbl = go.Figure(go.Table(header=dict(values=[""]+list(tbl_df.columns)),
                               cells=dict(values=[tbl_df.index]+[tbl_df[c].tolist() for c in tbl_df.columns])))
    tbl.update_layout(height=250, margin=dict(t=10,b=5))
    return fig, tbl, question

# ─────────────────────────────────────────────────────
# DQ2: 이용기간 (년 단위 올림) 자동 탐색
# ─────────────────────────────────────────────────────
def plot_dq2(df):
    cols = [c for c in df.columns if c.startswith("DQ2")]
    if not cols:
        return None, None, ""
    question = cols[0]
    # 파싱
    def parse(s):
        s = str(s).strip()
        m = re.match(r'^(\d+)\s*년\s*(\d+)\s*개월$', s)
        if m: return int(m.group(1)) + (1 if int(m.group(2))>0 else 0)
        m = re.match(r'^(\d+)\s*년$', s);
        if m: return int(m.group(1))
        m = re.match(r'^(\d+)\s*개월$', s)
        if m: return 1
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
    tbl = go.Figure(go.Table(header=dict(values=[""]+labels),
                               cells=dict(values=[tbl_df.index]+[tbl_df[c].tolist() for c in tbl_df.columns])))
    tbl.update_layout(height=250, margin=dict(t=10,b=5))
    return fig, tbl, question

def plot_dq3(df):
    # DQ3 문항 자동 탐색
    cols = [c for c in df.columns if c.startswith("DQ3")]
    if not cols:
        return None, None, ""
    question = cols[0]
    # 임시 DataFrame 생성
    temp_df = df[[question]].dropna().astype(str)
    # 기존 범주형 스택 바 호출
    fig, table_fig = plot_categorical_stacked_bar(temp_df, question)
    return fig, table_fig, question

    # 테이블
    table_df = pd.DataFrame({
        "응답 수": counts.values,
        "비율 (%)": percent.values
    }, index=display_labels).T
    table_fig = go.Figure(go.Table(
        header=dict(
            values=[""] + display_labels,
            align='center', font=dict(size=11), height=30
        ),
        cells=dict(
            values=[table_df.index] + [table_df[label].tolist() for label in display_labels],
            align='center', font=dict(size=10), height=28
        )
    ))
    table_fig.update_layout(height=250, margin=dict(t=10, b=5))

    return fig, table_fig, question


# ─────────────────────────────────────────────────────
# DQ4: 누적 세로 Bar 그래프 + Table (1순위 기준 내림차순 정렬)
# ─────────────────────────────────────────────────────
def plot_dq4_bar(df):
    cols = [c for c in df.columns if c.startswith("DQ4")]
    if len(cols) < 2:
        return None, None, ""
    col1, col2 = cols[0], cols[1]
    question = f"{col1} vs {col2}"

    s1 = df[col1].dropna().astype(str)
    s2 = df[col2].dropna().astype(str)
    # 원본 카테고리 집합
    cats = sorted(set(s1.unique()).union(s2.unique()))
    # prefix 제거용 라벨
    labels = [c.split('. ', 1)[-1] if '. ' in c else c for c in cats]

    # 응답 수 계산
    counts1 = s1.value_counts().reindex(cats, fill_value=0)
    counts2 = s2.value_counts().reindex(cats, fill_value=0)
    pct1 = (counts1 / counts1.sum() * 100).round(1)
    pct2 = (counts2 / counts2.sum() * 100).round(1)

    # 1순위 기준 내림차순으로 순서 정렬
    order_idx = counts1.sort_values(ascending=False).index.tolist()
    # 정렬된 display labels
    sorted_labels = [lbl.split('. ',1)[-1] if '. ' in lbl else lbl for lbl in order_idx]
    # 정렬된 counts
    sorted_counts1 = counts1.reindex(order_idx)
    sorted_counts2 = counts2.reindex(order_idx)
    # 테이블용 percent 재정렬
    sorted_pct1 = pct1.reindex(order_idx)
    sorted_pct2 = pct2.reindex(order_idx)

    # 누적 세로 Bar그래프 생성 (응답자 수)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sorted_labels, y=sorted_counts1.values,
        name='1순위', marker_color='blue', text=sorted_counts1.values, textposition='outside'
    ))
    fig.add_trace(go.Bar(
        x=sorted_labels, y=sorted_counts2.values,
        name='2순위', marker_color='green', text=sorted_counts2.values, textposition='outside'
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

    # 하단 테이블 생성 (응답 수 + 비율)
    table_df = pd.DataFrame({
        '1순위 응답 수': sorted_counts1.values,
        '1순위 비율(%)': sorted_pct1.values,
        '2순위 응답 수': sorted_counts2.values,
        '2순위 비율(%)': sorted_pct2.values
    }, index=sorted_labels).T
    table_fig = go.Figure(go.Table(
        header=dict(
            values=[""] + sorted_labels,
            align='center', height=30, font=dict(size=11)
        ),
        cells=dict(
            values=[table_df.index] + [table_df[label].tolist() for label in sorted_labels],
            align='center', height=28, font=dict(size=10)
        )
    ))
    table_fig.update_layout(height=250, margin=dict(t=10, b=5))

    return fig, table_fig, question

# ─────────────────────────────────────────────────────
# DQ5: 범주형 누적 가로 Bar + Table (plot_categorical_stacked_bar 재활용)
# ─────────────────────────────────────────────────────
def plot_dq5(df):
    # DQ5 문항 자동 탐색
    cols = [c for c in df.columns if c.startswith("DQ5")]
    if not cols:
        return None, None, ""
    question = cols[0]
    temp_df = df[[question]].dropna().astype(str)
    fig, table_fig = plot_categorical_stacked_bar(temp_df, question)
    return fig, table_fig, question


# ─────────────────────────────────────────────────────
# DQ7-E: 다이버징 스택형 바 차트 (Likert) 함수 정의
# ─────────────────────────────────────────────────────
def plot_likert_diverging(df, prefix="DQ7-E"):
    # 해당 prefix로 시작하는 문항들 탐색
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return None, None
    # 1~7 점수 분포 계산
    dist = {}
    for col in cols:
        counts = df[col].dropna().astype(int).value_counts().reindex(range(1,8), fill_value=0)
        pct = (counts / counts.sum() * 100).round(1)
        dist[col] = pct
    likert_df = pd.DataFrame(dist).T  # index: 문항, columns: 1~7
    # 명시적 컬럼 순서 보장
    likert_df = likert_df.reindex(columns=range(1,8))

        # 다이버징 스택 바
    fig = go.Figure()
    # 부정(1-3): 스택 순서 변경하여 1점이 가장 왼쪽(외곽)에 위치하도록
    neg_scores = [3,2,1]
    neg_colors = ["#91bfdb","#4575b4","#313695"]  # 1~3점 긍정 색상 (파랑 계열)  # 3점→2점→1점 순서
    for score, color in zip(neg_scores, neg_colors):
        fig.add_trace(go.Bar(
            y=likert_df.index,
            x=-likert_df[score],
            name=f"{score}점",
            orientation='h',
            marker_color=color
        ))
    # 중립(4)
    fig.add_trace(go.Bar(
        y=likert_df.index,
        x=likert_df[4],
        name="4점",
        orientation='h',
        marker_color="#dddddd"
    ))
    # 긍정(5-7)
    for score, color in zip([5,6,7],["#fee090","#fc8d59","#d73027"]):  # 5~7점 부정 색상 (빨강 계열)
        fig.add_trace(go.Bar(
            y=likert_df.index,
            x=likert_df[score],
            name=f"{score}점",
            orientation='h',
            marker_color=color
        ))
    fig.update_layout(
        barmode='relative',
        title="DQ7-E 도서관 이미지 분포 (다이버징 바)",
        xaxis=dict(visible=False),  # X축 레이블 및 눈금 표시 없음
        legend=dict(traceorder='normal'),
        height=250,
        margin=dict(t=30, b=5),
    )

    # 테이블: 명시적 컬럼 순서
    table_df = likert_df.copy()
    table_df = table_df.reindex(columns=range(1,8))
    table_fig = go.Figure(go.Table(
        header=dict(
            values=["문항"] + [f"{c}점" for c in table_df.columns],
            align='center'
        ),
        cells=dict(
            values=[table_df.index] + [table_df[c].tolist() for c in table_df.columns],
            align='center'
        )
    ))
    table_fig.update_layout(margin=dict(t=5, b=5))
    return fig, table_fig

# ─────────────────────────────────────────────────────
# DQ8 & DQ9: 1순위 vs 2순위 누적 세로 Bar 차트 공통 함수
# ─────────────────────────────────────────────────────
def plot_pair_bar(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix)]
    if len(cols) < 2:
        return None, None, ""
    col1, col2 = cols[0], cols[1]
    question = f"{col1} vs (2순위)"
    s1 = df[col1].dropna().astype(str)
    s2 = df[col2].dropna().astype(str)
    cats = sorted(set(s1.unique()).union(s2.unique()))
    # 번호 제거 라벨
    labels = [c.split('. ',1)[-1] if '. ' in c else c for c in cats]
    # 응답자 수
    counts1 = s1.value_counts().reindex(cats, fill_value=0)
    counts2 = s2.value_counts().reindex(cats, fill_value=0)
    # 막대 차트
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=counts1, name='1순위', marker_color='blue', text=counts1, textposition='outside'))
    fig.add_trace(go.Bar(x=labels, y=counts2, name='2순위', marker_color='green', text=counts2, textposition='outside'))
    fig.update_layout(
        barmode='stack',
        title=f"{question}",
        yaxis_title="응답자 수",
        height=550,
        margin=dict(t=50, b=70),
        xaxis_tickangle=-23
    )
    # 테이블
    pct1 = (counts1 / counts1.sum() * 100).round(1)
    pct2 = (counts2 / counts2.sum() * 100).round(1)
    table_df = pd.DataFrame({
        '1순위 응답 수': counts1.values,
        '1순위 비율(%)': pct1.values,
        '2순위 응답 수': counts2.values,
        '2순위 비율(%)': pct2.values
    }, index=labels).T 
    table_fig = go.Figure(go.Table(
        header=dict(values=[""] + labels, align='center'),
        cells=dict(values=[table_df.index] + [table_df[l].tolist() for l in labels], align='center')
    ))
    table_fig.update_layout(height=250, margin=dict(t=10, b=10))
    return fig, table_fig, question
# ------------------ Likert 스케일 변환 / 중분류 정의 ------------------
# 7점 척도 → 0~100 변환
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

# ------------------ 시각화: 심화 분석 ------------------

def plot_midcategory_radar(df):
    mid_scores = compute_midcategory_scores(df)
    if mid_scores.empty:
        return None
    categories = list(mid_scores.index)
    values = mid_scores.values.tolist()
    categories += categories[:1]
    values += values[:1]
    fig = go.Figure(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='중분류 만족도'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0,100], tickformat=".0f")),
        title="중분류별 만족도 수준 (0~100 환산, 레이더 차트)",
        showlegend=False,
        height=450,
        margin=dict(t=40, b=20)
    )
    return fig


def plot_within_category_bar(df, midcategory):
    item_scores = compute_within_category_item_scores(df)
    if midcategory not in item_scores:
        return None
    series = item_scores[midcategory].sort_values(ascending=False)
    fig = go.Figure(go.Bar(
        x=series.values,
        y=series.index,
        orientation='h',
        text=series.round(1),
        textposition='outside',
        marker_color='steelblue'
    ))
    fig.update_layout(
        title=f"{midcategory} 내 문항별 평균 점수 비교 (0~100 환산)",
        xaxis_title="평균 점수",
        height=350,
        margin=dict(t=40, b=60)
    )
    return fig

# ─────────────────────────────────────────────────────
# ▶️ Streamlit 실행
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="공공도서관 설문 시각화 대시보드",
    layout="wide"
)

uploaded = st.file_uploader("📂 엑셀(.xlsx) 파일 업로드", type=["xlsx"])
if not uploaded:
    st.info("데이터 파일을 업로드해 주세요.")
    st.stop()

df = pd.read_excel(uploaded)
st.success("✅ 업로드 완료")

# 상단 메인 탭 정의
main_tabs = st.tabs([
    "👤 응답자 정보",
    "📈 만족도 기본 시각화",
    "🗺️ 자치구 구성 문항",
    "📊도서관 이용양태 분석",
    "🖼️ 도서관 이미지 분석",
    "🏋️ 도서관 강약점 분석",
    "🔍 심화 분석"

])

# 1) 응답자 정보
with main_tabs[0]:
    page_home(df)

# 2) 기본 만족도 시각화 (Q1~Q8)
with main_tabs[1]:
    page_basic_vis(df)

# 3) 자치구 구성 문항 탭 안에 서브 탭 추가
with main_tabs[2]:
    st.header("🗺️ 자치구 구성 문항 분석")
    sub_tabs = st.tabs([
        "7점 척도 시각화",   # Q9-D-1~3
        "단문 응답 분석",     # Q9-DS-4
        "장문 서술형 분석"    # Q9-DS-5
    ])

    # 3-1) 7점 척도 시각화
    with sub_tabs[0]:
        st.subheader("자치구 구성 문항 (7점 척도)")
        subregion_cols = [c for c in df.columns if "Q9-D-" in c]
        if not subregion_cols:
            st.error("Q9-D- 로 시작하는 문항을 찾을 수 없습니다.")
        else:
            for idx, col in enumerate(subregion_cols):
                bar, tbl = plot_stacked_bar_with_table(df, col)
                st.markdown(f"##### {col}")
                st.plotly_chart(bar, use_container_width=True, key=f"bar-{idx}-{col}")
                st.plotly_chart(tbl, use_container_width=True, key=f"tbl-{idx}-{col}")

    # 3-2) 단문 응답 키워드 분석
    with sub_tabs[1]:
        page_short_keyword(df)

    # 3-3) 장문 서술형 분석
    with sub_tabs[2]:
        st.subheader("장문 서술형 분석 (Q9-DS-5)")
        # Q9-DS-5 컬럼 필터
        long_cols = [c for c in df.columns if "Q9-DS-5" in c]
        if not long_cols:
            st.warning("Q9-DS-5 관련 문항을 찾을 수 없습니다.")
        else:
            answers = df[long_cols[0]].dropna().astype(str).tolist()
            df_long = process_answers(answers)
            show_short_answer_keyword_analysis(df_long)
# 4) 도서관 이용양태 분석
with main_tabs[3]:
    st.header("📊 도서관 이용양태 분석")
    # 하위 탭: DQ1~5, DQ6 계열
    sub_tabs = st.tabs(["DQ1~5","DQ6 계열"])

    # --- DQ1~5 탭 ---
    with sub_tabs[0]:
        # DQ1~DQ2~DQ3~DQ4 기존 구현
        fig1, tbl1, q1 = plot_dq1(df)
        if fig1: st.subheader(q1); st.plotly_chart(fig1, use_container_width=True); st.plotly_chart(tbl1, use_container_width=True)
        fig2, tbl2, q2 = plot_dq2(df)
        if fig2: st.subheader(q2); st.plotly_chart(fig2, use_container_width=True); st.plotly_chart(tbl2, use_container_width=True)
        fig3, tbl3, q3 = plot_dq3(df)
        if fig3: st.subheader(q3); st.plotly_chart(fig3, use_container_width=True); st.plotly_chart(tbl3, use_container_width=True)
        fig4, tbl4, q4 = plot_dq4_bar(df)
        if fig4: st.subheader(q4); st.plotly_chart(fig4, use_container_width=True); st.plotly_chart(tbl4, use_container_width=True)
        fig5, tbl5, q5 = plot_dq5(df)
        if fig5: st.subheader(q5); st.plotly_chart(fig5, use_container_width=True); st.plotly_chart(tbl5, use_container_width=True)


    # --- DQ6 계열 탭 ---
    with sub_tabs[1]:
        st.subheader("DQ6 계열 문항 분석")
        # DQ6부터 DQ6-3까지 자동 탐색
        dq6_cols = [c for c in df.columns if c.startswith("DQ6")]
        if not dq6_cols:
            st.warning("DQ6 계열 문항이 없습니다.")
        else:
            for col in dq6_cols:
                st.markdown(f"### {col}")
                # 1) DQ6 (복수선택) -> 멀티 응답 explode 후 카운트
                if col == dq6_cols[0]:  # 첫번째 DQ6 문항
                    multi = df[col].dropna().astype(str).str.split(',')
                    exploded = multi.explode().str.strip()
                    counts = exploded.value_counts()
                    percent = (counts / counts.sum() * 100).round(1)
                    # 가로 막대 차트
                    fig = go.Figure(go.Bar(
                        x=counts.values, y=counts.index,
                        orientation='h', text=counts.values,
                        textposition='outside', marker_color=px.colors.qualitative.Plotly
                    ))
                    fig.update_layout(
                        title=col,
                        xaxis_title="응답 수",
                        yaxis_title="서비스",
                        height=400,
                        margin=dict(t=50, b=100)
                    )
                    # 테이블
                    table_df = pd.DataFrame({
                        '응답 수': counts,
                        '비율 (%)': percent
                    }).T
                    table_fig = go.Figure(go.Table(
                        header=dict(values=[""] + list(table_df.columns), align='center'),
                        cells=dict(values=[table_df.index] + [table_df[c].tolist() for c in table_df.columns], align='center')
                    ))
                    table_fig.update_layout(height=250, margin=dict(t=10,b=5))
                    st.plotly_chart(fig, use_container_width=True)
                    st.plotly_chart(table_fig, use_container_width=True)
                else:
                    # DQ6-1 ~ DQ6-3: 단일 선택 카테고리
                    bar, tbl = plot_categorical_stacked_bar(df, col)
                    st.plotly_chart(bar, use_container_width=True)
                    st.plotly_chart(tbl, use_container_width=True)
# 5) 도서관 이미지 분석 탭
with main_tabs[4]:
    st.header("🖼️ 도서관 이미지 분석")
    fig, tbl = plot_likert_diverging(df, prefix="DQ7-E")
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(tbl, use_container_width=True)
    else:
        st.warning("DQ7-E 문항이 없습니다.")

# 6) 도서관 강약점 분석 탭
with main_tabs[5]:
    st.header("🏋️ 도서관 강약점 분석")
    # DQ8: 강점
    fig8, tbl8, q8 = plot_pair_bar(df, "DQ8")
    if fig8 is not None:
        st.plotly_chart(fig8, use_container_width=True)
        st.plotly_chart(tbl8, use_container_width=True)
    else:
        st.warning("DQ8 문항이 없습니다.")
    # DQ9: 약점
    fig9, tbl9, q9 = plot_pair_bar(df, "DQ9")
    if fig9 is not None:
        st.plotly_chart(fig9, use_container_width=True)
        st.plotly_chart(tbl9, use_container_width=True)
    else:
        st.warning("DQ9 문항이 없습니다.")

# 7) 심화 분석 탭
with main_tabs[6]:
    st.header("🔍 심화 분석")
    advanced_tabs = st.tabs(["공통 심화 분석", "중분류 내 문항 편차"]);

    with advanced_tabs[0]:
        st.subheader("중분류별 전체 만족도 (레이더 차트)")
        radar = plot_midcategory_radar(df)
        if radar:
            st.plotly_chart(radar, use_container_width=True)
        else:
            st.warning("중분류 점수를 계산할 수 있는 문항이 부족합니다.")

    with advanced_tabs[1]:
        st.subheader("중분류 내 문항별 편차")
        mid_scores = compute_midcategory_scores(df)
        if mid_scores.empty:
            st.warning("중분류 문항이 없어 편차를 계산할 수 없습니다.")
        else:
            for mid in mid_scores.index:
                bar = plot_within_category_bar(df, mid)
                if bar:
                    st.markdown(f"### {mid}")
                    st.plotly_chart(bar, use_container_width=True)

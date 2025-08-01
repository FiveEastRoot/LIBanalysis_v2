import time
import numpy as np
import streamlit as st
import scipy.stats as stats
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import re
import openai
import streamlit.components.v1 as components  # 파일 상단에 위치해야 함
import json 
import math
import logging
from itertools import cycle



# 로깅 설정 (필요시 파일로도 남기게 조정 가능)
logging.basicConfig(level=logging.INFO)

openai.api_key = st.secrets["openai"]["api_key"]
client = openai.OpenAI(api_key=openai.api_key)

# 기본 색상 팔레트
DEFAULT_PALETTE = px.colors.qualitative.Plotly
COLOR_CYCLER = cycle(DEFAULT_PALETTE)

# ─────────────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────────────

def safe_chat_completion(*, model="gpt-4.1-nano", messages, temperature=0.2, max_tokens=300, retries=3, backoff_base=1.0):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return resp
        except Exception as e:
            logging.warning(f"OpenAI call failed (attempt {attempt}): {e}")
            last_exc = e
            time.sleep(backoff_base * (2 ** (attempt - 1)))
    logging.error("OpenAI call failed after retries")
    raise last_exc

def interpret_midcategory_scores(df):
    scores = compute_midcategory_scores(df)
    if scores.empty:
        return "중분류 점수를 계산할 충분한 데이터가 없습니다."
    overall = scores.mean()
    high = scores[scores >= overall + 5].index.tolist()
    low = scores[scores <= overall - 5].index.tolist()

    parts = []
    parts.append(f"전체 중분류 평균은 {overall:.1f}점입니다.")
    if high:
        parts.append(f"평균보다 높은 중분류: {', '.join(high)}.")
    if low:
        parts.append(f"평균보다 낮은 중분류: {', '.join(low)}.")
    if not high and not low:
        parts.append("모든 중분류가 전체 평균 수준과 비슷합니다.")
    return " ".join(parts)

# ─────────────────────────────────────────────────────
# 전처리/매핑 유틸
# ─────────────────────────────────────────────────────
def remove_parentheses(text):
    return re.sub(r'\(.*?\)', '', text).strip()

def wrap_label(label, width=10):
    return '<br>'.join([label[i:i+width] for i in range(0, len(label), width)])

def get_qualitative_colors(n):
    palette = DEFAULT_PALETTE
    return [c for _, c in zip(range(n), cycle(palette))]

def wrap_label_fixed(label: str, width: int = 35) -> str:
    # 한 줄에 공백 포함 정확히 width 글자씩 자르고 <br>로 연결
    parts = [label[i:i+width] for i in range(0, len(label), width)]
    return "<br>".join(parts)

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

def is_trivial(text):
    text = str(text).strip()
    return text in ["", "X", "x", "감사합니다", "감사", "없음"]

def split_keywords_simple(text):
    parts = re.split(r"[.,/\s]+", text)
    return [p.strip() for p in parts if len(p.strip()) > 1]

def map_keyword_to_category(keyword):
    for cat, kws in KDC_KEYWORD_MAP.items():
        if any(k in keyword for k in kws):
            return cat
    return "해당없음"

def escape_tildes(text: str, mode: str = "html") -> str:
    """
    mode="html": 카드처럼 HTML로 렌더링할 때 물결표 처리.
    mode="markdown": st.markdown 같은 마크다운 컨텍스트에서 취소선 방지.
    """
    if mode == "html":
        # '~~' 먼저 바꾸고 단일 ~도 엔티티로
        text = text.replace("~~", "&#126;&#126;")
        return text.replace("~", "&#126;")
    else:  # markdown
        text = text.replace("~~", r"\~\~")
        return text.replace("~", r"\~")

def safe_markdown(text, **kwargs):
    # 마크다운 취소선 방지를 위해 ~를 이스케이프
    safe = escape_tildes(text, mode="markdown")
    st.markdown(safe, **kwargs)


# ─────────────────────────────────────────────────────
# DataFrame & visualization helpers
# ─────────────────────────────────────────────────────

def _sanitize_dataframe_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()

    for col in df2.select_dtypes(include=["object"]).columns:
        df2[col] = df2[col].apply(lambda x: str(x) if not pd.isna(x) else x)

    if isinstance(df2.index, pd.MultiIndex):
        df2.index = df2.index.map(lambda tup: " | ".join(map(str, tup)))
    else:
        df2.index = df2.index.map(lambda x: str(x))

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
            try:
                safe_head = _sanitize_dataframe_for_streamlit(table.head(200))
                st.dataframe(safe_head, key=f"{key_prefix}-tbl-df-{title}-sample")
                st.warning(f"전체 테이블 렌더링에 실패하여 상위 200개만 보여줍니다: {e}")
            except Exception as e2:
                st.error(f"테이블 렌더링 불가: {e2}")
    elif table is not None:
        st.write(table, key=f"{key_prefix}-tbl-raw-{title}")

def show_table(df, caption):
    st.dataframe(df)


def render_insight_card(title: str, content: str, key: str = None):
    if content is None:
        content = "(내용 없음)"
    content = str(content)
    content_html = escape_tildes(content, mode="html").replace("\n", "<br>")
    line_count = content.count("\n") + 3
    height = min(800, 70 + 20 * line_count)
    html = f"""
    <div style="
        border:1px solid #e2e8f0;
        border-radius:12px;
        padding:16px;
        margin-bottom:16px;
        background: #ffffff;
        box-shadow: 0 8px 24px rgba(0,0,0,0.04);
        font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', Arial;
    ">
        <h4 style="margin:0 0 8px 0; font-size:1.1rem;">{title}</h4>
        <div style="font-size:0.95em; line-height:1.4em;">{content_html}</div>
    </div>
    """
    try:
        components.html(html, height=height, key=key)
    except Exception as e:
        logging.warning(f"components.html failed for key={key}: {e}")
        # Fallback so the user still sees something
        st.markdown(f"**{title}**\n\n{content}")

# ─────────────────────────────────────────────────────
# Likert / 중분류 점수 계산
# ─────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────
# GPT 관련 헬퍼
# ─────────────────────────────────────────────────────

def cohen_d(x, y):
    x = np.array(x.dropna(), dtype=float)
    y = np.array(y.dropna(), dtype=float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return None
    # pooled standard deviation
    pooled = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2)) if (nx + ny - 2) > 0 else 0
    if pooled == 0:
        return 0.0
    return (x.mean() - y.mean()) / pooled

def compare_midcategory_by_group(df, group_col):
    """
    group_col 기준으로 각 그룹의 중분류 만족도를 전체 나머지와 비교하여
    mean, delta, Welch t-test p-value, Cohen's d, sample size를 계산.
    반환 형태: {group_label: {midcategory: {mean, delta_vs_overall, p_value_vs_rest, cohen_d_vs_rest, n}}}
    """
    results = {}
    global_mid = compute_midcategory_scores(df)
    
    # per-row midcategory scores (각 응답자별로 중분류 점수 계산)
    def per_row_mid_scores(subdf):
        per_mid = {}
        for mid, predicate in MIDDLE_CATEGORY_MAPPING.items():
            cols = [c for c in subdf.columns if predicate(c)]
            if not cols:
                continue
            scaled = subdf[cols].apply(scale_likert)
            per_mid[mid] = scaled.mean(axis=1, skipna=True)
        return per_mid  # dict of Series

    # 전체 데이터 기준 per-row
    overall_per_row = per_row_mid_scores(df)

    for group_value, sub in df.groupby(df[group_col].astype(str)):
        group_per_row = per_row_mid_scores(sub)
        group_summary = {}
        for mid in global_mid.index:
            if mid not in group_per_row or mid not in overall_per_row:
                continue
            grp_scores = group_per_row[mid].dropna()
            rest_mask = df[group_col].astype(str) != str(group_value)
            rest = df[rest_mask]
            rest_per_row = per_row_mid_scores(rest).get(mid, pd.Series(dtype=float)).dropna()
            if grp_scores.empty or rest_per_row.empty:
                continue
            # Welch's t-test
            try:
                stat, p = stats.ttest_ind(grp_scores, rest_per_row, equal_var=False)
            except Exception:
                p = None
            d = cohen_d(grp_scores, rest_per_row)
            mean_group = compute_midcategory_scores(sub).get(mid, None)
            delta = None
            if mean_group is not None and mid in global_mid:
                delta = mean_group - global_mid.get(mid)
            group_summary[mid] = {
                "mean": round(float(mean_group), 1) if mean_group is not None else None,
                "delta_vs_overall": round(float(delta), 1) if delta is not None else None,
                "p_value_vs_rest": round(p, 4) if p is not None else None,
                "cohen_d_vs_rest": round(d, 2) if d is not None else None,
                "n": int(len(grp_scores))
            }
        results[str(group_value)] = group_summary
    return results


@st.cache_data(show_spinner=False)
def process_answers(responses):
    expanded = []
    for ans in responses:
        if is_trivial(ans):
            continue
        parts = [p.strip() for p in ans.split(',') if p.strip()]
        if len(parts) > 1:
            expanded.extend(parts)
        else:
            expanded.append(ans)

    processed = []
    batches = extract_keyword_and_audience(expanded, batch_size=8)
    for resp, kws, aud in batches:
        if is_trivial(resp):
            continue
        if not kws:
            kws = split_keywords_simple(resp)
        for kw in kws:
            cat = map_keyword_to_category(kw)
            if cat == '해당없음' and aud == '일반':
                continue
            processed.append({
                '응답': resp,
                '키워드': kw,
                '주제범주': cat,
                '대상범주': aud
            })
    return pd.DataFrame(processed)

@st.cache_data(show_spinner=False)
def extract_keyword_and_audience(responses, batch_size=20):
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
        try:
            resp = safe_chat_completion(
                model="gpt-4.1-nano",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.2,
                max_tokens=300
            )
            content = resp.choices[0].message.content.strip()
            try:
                data = pd.read_json(content)
            except Exception:
                raise ValueError("JSON 파싱 실패, fallback으로 전환")
        except Exception:
            # fallback
            data = []
            for txt in batch:
                kws = split_keywords_simple(txt)
                audience = '일반'
                for w in ['어린이', '초등']:
                    if w in txt:
                        audience = '아동'
                for w in ['유아', '미취학', '그림책']:
                    if w in txt:
                        audience = '유아'
                for w in ['청소년', '진로', '자기계발']:
                    if w in txt:
                        audience = '청소년'
                data.append({
                    'response': txt,
                    'keywords': kws,
                    'audience': audience
                })
            data = pd.DataFrame(data)
        for _, row in data.iterrows():
            results.append((row['response'], row['keywords'], row['audience']))
    return results

def call_gpt_for_insight(prompt, model="gpt-4.1-nano", temperature=0.2, max_tokens=1000):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "너는 전략 리포트 작성자이며, 주어진 데이터를 바탕으로 명확하고 간결한 인사이트를 제공해야 한다."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content.strip()
        return content
    except Exception as e:
        logging.warning(f"GPT 호출 실패: {e}")
        return f"GPT 해석 생성에 실패했습니다: {e}"

def build_radar_prompt(overall_profile: dict, combos: list):
    # combos: list of dicts with keys: label (e.g. "여성 | 30-34세"), n, profile (dict of midcat->score)
    overall_str = ", ".join(f"{k}: {v:.1f}" for k, v in overall_profile.items())
    combo_lines = []
    for c in combos:
        prof = ", ".join(f"{k}: {v:.1f}" for k, v in c["profile"].items())
        combo_lines.append(f"{c['label']} (응답자수={c['n']}): {prof}")
    combo_str = "\n".join(combo_lines)
    prompt = f"""
너는 전략 보고서 작성자다. 다음 데이터를 보고 '레이더 차트 해석' 섹션을 만들어줘.

입력:
- 전체 평균 중분류 만족도: {overall_str}
- 상위 세그먼트 조합별 중분류 만족도 프로파일:
{combo_str}

요청:
1. 각 조합명(예: '여성 | 30-34세')을 중심으로, 전체 평균과 비교해 그 조합이 어디에서 강점(만족)이고 어디에서 약점(불만족)인지 각각 2-3문장씩 설명해줘. 조합명을 반복해서 사용하고, 상대적으로 어떤지(예: “여성 | 30-34세는 정보 획득이 전체 평균보다 높아 강점이나, 소통 및 정책 활용에서는 낮아 보완 필요”) 표현해줘.
2. 서로 뚜렷히 대비되는 두 개의 조합명 쌍을 골라 비교 설명해줘. 각각 어떤 중분류에서 차이가 나는지, 그로부터 어떤 인사이트가 나오는지 구체적으로 서술.
3. 전체적으로 관찰되는 패턴 3개를 도출해줘. (예: “특정 연령대 조합들이 정보 획득은 높지만 소통에서 일관되게 낮음”처럼 조합 특성 포함)
4. 전략적 시사점 3개: 어떤 조합을 우선 공략/보완할지, 조합명을 명시하며 구체적 행동 방향을 제안해줘.
5. 전체 길이는 2000자로 제한돼. 그리고 작성간에 "~"는 모두 "-" 로 표시되어야해. 

스타일: 비즈니스 보고서 톤, 소제목 포함, 숫자는 한 자리 소수, 조합명(예: '여성 | 30-34세')을 모든 설명에 명시하고 ‘조합 1’ 같은 일반명은 쓰지 마."""
    return prompt.strip()

def build_heatmap_prompt(table_df: pd.DataFrame, midcats: list):
    # table_df: DataFrame with columns including "조합", "응답자수", and each midcat
    rows = []
    for _, r in table_df.iterrows():
        label = r.get("조합", "")
        n = int(r.get("응답자수", 0))
        scores = ", ".join(f"{mc}: {r.get(mc, 0):.1f}" for mc in midcats)
        rows.append(f"{label} (응답자수={n}): {scores}")
    table_str = "\n".join(rows)
    prompt = f"""
너는 전략 보고서 작성자다. 아래 히트맵용 데이터를 보고 인사이트를 정리해줘. 모든 설명에서 조합명(예: '여성 | 30-34세')을 중심으로 쓰고, 일반적인 '조합 A/B' 같은 표현은 피하고 구체적인 이름과 비교를 반복해줘.

입력:
- 세그먼트 조합별 중분류별 평균 만족도:
{table_str}

요청:
1. 점수가 몰려 있는 군집(예: 특정 조합명들이 여러 중분류에서 모두 높은 패턴 또는 낮은 패턴)을 이름(조합명)과 함께 정리해줘. 어떤 조합명들이 비슷한 강점/약점을 공유하는지도 묶어서 써줘.
2. 응답자 수가 충분한 조합명들에서 일관된 강점과 약점을 각각 요약해줘. 예: '여성 | 30-34세'는 정보 획득과 공익성에서 일관되게 높고, 소통 및 정책 활용이 낮음.
3. 전체적인 중분류별 경향: 어떤 중분류가 전반적으로 높은지/낮은지, 그리고 특정 조합명들이 그 흐름과 어떻게 다른지 설명해줘.
4. 요약 개요 (한 문단)과, 도출할 수 있는 3개의 구체적인 행동 권장점(조합명을 포함한 우선순위 포함)을 제시해줘.
5. 전체 길이는 2000자로 제한돼. 그리고 작성간에 "~"는 모두 "-" 로 표시되어야해. 


스타일: 비즈니스 리포트 톤, 소제목 포함, 숫자는 한 자리 소수, 조합명을 반복적으로 사용하여 비교·대조 중심으로 작성."""
    return prompt.strip()

def build_delta_prompt(delta_df: pd.DataFrame, midcats: list):
    # delta_df: DataFrame indexed by "조합", with columns like "<midcat>_delta"
    rows = []
    for _, r in delta_df.iterrows():
        combo = r.name  # 조합명
        diffs = ", ".join(f"{mc}: {r.get(f'{mc}_delta', 0):+.1f}" for mc in midcats)
        rows.append(f"{combo}: {diffs}")
    table_str = "\n".join(rows)
    prompt = f"""
너는 전략 보고서 작성자다. 아래 데이터는 세그먼트 조합명별 중분류 만족도가 전체 평균 대비 얼마나 벗어나는지(Delta)를 보여준다. 모든 설명에서 조합명(예: '여성 | 30-34세')을 반복 사용하여, 강하게 벗어난 영역과 그 반복적 패턴을 중심으로 서술해줘.

입력:
{table_str}

요청:
1. 각 조합명별로 전체 평균 대비 가장 과도하게 높은/낮은 중분류를 명확히 짚어줘. 예: '여성 | 30-34세는 정보 획득에서 +12.3으로 강점이나, 소통 및 정책 활용에서는 -9.5로 약점'처럼.
2. 여러 조합명이 반복해서 비슷한 편차 패턴(예: 동일하게 특정 중분류가 낮거나 높은)을 보이는지 그룹화하여 설명해줘. 구체적인 조합명들을 묶어 비교.
3. 개선/확장 우선순위를 조합명 기준으로 추천해줘. (예: '소통 및 정책 활용이 반복적으로 낮은 조합명들부터 개선해야 하며, 중분류 평균 대비 높고 빈번한 조합명들에 대해 확장 전략 고려' 등)
4. 전체 길이는 2000자로 제한돼. 그리고 작성간에 "~"는 모두 "-" 로 표시되어야해. 


스타일: 간결한 비즈니스 요약, 소제목 포함, 숫자는 한 자리 소수, 조합명 명시 중심."""
    return prompt.strip()

def build_ci_prompt(subset_df: pd.DataFrame, mc: str):
    # subset_df: contains columns '조합', 'delta', 'se' for given midcategory, and maybe 응답자수
    rows = []
    for _, r in subset_df.iterrows():
        combo = r.get("조합", "")
        delta = r.get("delta", 0)
        se = r.get("se", 0)
        rows.append(f"{combo}: 편차 {delta:.1f}, 표준오차 {se:.2f}")
    table_str = "\n".join(rows)
    prompt = f"""
너는 전략 보고서 작성자다. 아래 데이터는 중분류 '{mc}'에 대해 상위 세그먼트 조합명들이 전체 평균 대비 편차(delta)와 그 불확실성(표준오차)을 나타낸다. 모든 분석에서 조합명(예: '여성 | 30-34세')을 중심으로 비교하고, 의미 있는 vs 불확실한 케이스를 구분해서 서술해줘.

입력:
{table_str}

요청:
1. 어떤 조합명들의 편차가 0 기준선과 비교해 실질적으로 의미 있어 보이는지 설명해줘. (예: 표준오차에 비해 편차가 충분히 큰지 여부를 조합명별로 판단)
2. 편차가 크지만 불확실성이 큰 조합명과, 편차는 작지만 안정적인 조합명을 조합명 기준으로 구분하여 비교 설명해줘.
3. 우선 개입하거나 주목해야 할 조합명 3개를 추천해줘. 각각 왜 우선순위인지(편차/불확실성 관점) 구체적으로 써줘.
4. 전체 길이는 2000자로 제한돼. 그리고 작성간에 "~"는 모두 "-" 로 표시되어야해. 


스타일: 비즈니스 리포트 톤, 소제목 포함, 숫자는 한 자리 소수, 조합명을 반복 명시하여 읽는 사람이 바로 어떤 그룹인지 알 수 있게 작성."""
    return prompt.strip()

def build_small_multiple_prompt(top_df: pd.DataFrame, midcat: str, segment_cols_filtered: list):
    # top_df: DataFrame with a '조합' constructed from segment_cols_filtered and midcat scores
    rows = []
    for _, r in top_df.iterrows():
        combo = " | ".join(str(r[c]) for c in segment_cols_filtered)
        score = r.get(midcat, None)
        if pd.notna(score):
            rows.append(f"{combo}: {midcat} 점수 {score:.1f}")
        else:
            rows.append(f"{combo}: 데이터 없음")
    table_str = "\n".join(rows)
    prompt = f"""
너는 전략 보고서 작성자다. 아래 데이터는 중분류 '{midcat}'에 대해 응답자 수 상위 세그먼트 조합명들의 점수를 비교한 것이다. 모든 설명에서 조합명(예: '여성 | 30-34세')을 중심으로 하고, 일반적 표현 없이 구체적으로 어떤 조합이 눈에 띄는지, 순위 변동성과 outlier를 기술해줘.

입력:
{table_str}

요청:
1. 중분류 '{midcat}'에서 조합명들 간 점수 분포와 순위 변동성이 어떤지 요약해줘. (예: 어떤 조합명이 일관되게 상위인지, 어떤 조합명은 변동성이 크며 불안정한지)
2. 특징적인 outlier 조합명들을 지적하고, 그들이 왜 예외적인지 설명해줘. (예: 다른 조합명들보다 훨씬 높거나 낮은 경우)
3. 일관성 있는 강점/약점을 보이는 조합명 그룹이 있다면 묶어서 설명해줘. (예: '30~34세 여성' 계열이 정보 획득은 높지만 소통이 낮은 패턴)
4. 전체 길이는 2000자로 제한돼. 그리고 작성간에 "~"는 모두 "-" 로 표시되어야해. 


스타일: 짧고 명확한 비즈니스 요약, 소제목 포함, 숫자는 한 자리 소수, 조합명을 반복하여 비교 중심으로 작성."""
    return prompt.strip()


# ---------- 자연어 질의 기반 자동 시각화 + 인사이트 파이프라인 ----------

def parse_nl_query_to_spec(question: str):
    system_prompt = """
너는 설문 데이터에 대한 자연어 질의를 받아서 시각화/분석 스펙을 구조화된 JSON으로 바꾸는 파서야.
출력은 오직 JSON 객체 하나만, 코드블럭 없이 반환해. 모르면 null이나 빈 리스트로 만들어줘.

필드:
{
  "chart": "bar" / "line" / "heatmap" / "radar" / "delta_bar" / null,
  "x": "컬럼명 또는 중분류 이름",
  "y": null,
  "groupby": "컬럼명 (비교 기준)",
  "filters": [ {"col": "컬럼명", "op": "contains"|"=="|"in", "value": "값 또는 리스트"} ],
  "focus": "설명에서 중점적으로 다룰 포인트"
}

예시:
1. "혼자 이용하는 사람들의 연령대 분포 보여주고, 주로 가는 도서관별 중분류 만족도 강점/약점 비교해줘."
   -> {
        "chart": null,
        "x": "SQ2_GROUP",
        "groupby": "SQ4", 
        "filters": [{"col": "이용형태", "op": "contains", "value": "혼자"}],
        "focus": "혼자 이용자 연령대 분포 및 주 이용 도서관별 중분류 만족도 강약점 비교"
      }
2. "전체 평균 대비 어떤 중분류가 강점인지 레이더로 보여줘."
   -> {"chart": "radar", "focus": "전체 평균 대비 중분류 강점/약점 비교"}
3. "주이용서비스별 정보 획득과 소통 만족도 비교해줘."
   -> {"chart": "grouped_bar", "groupby": "SQ5", "x": "중분류", "focus": "정보 획득 vs 소통 비교"}

반드시 자연어에서 유추할 수 있는 필드들을 채우고, focus는 사용자 의도를 압축해서 짧게 작성해.
"""
    resp = safe_chat_completion(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.2,
        max_tokens=300
    )
    content = resp.choices[0].message.content.strip()
    content = re.sub(r"^```|```$", "", content).strip()
    try:
        spec = json.loads(content)
    except Exception:
        spec = {
            "chart": None,
            "x": None,
            "y": None,
            "groupby": None,
            "filters": [],
            "focus": question
        }
    return spec

def infer_chart_type(spec: dict, df_subset: pd.DataFrame):
    # chart이 주어져 있지 않으면 스펙 + 키워드로 추론
    chart = spec.get("chart")
    focus = spec.get("focus", "").lower() if spec.get("focus") else ""
    groupby = spec.get("groupby")
    x = spec.get("x")
    y = spec.get("y")

    if chart:
        return chart  # 명시적이면 그대로

    # 전체 평균 대비, 강점/약점 언급: radar + delta 우선
    if any(k in focus for k in ["전체 평균 대비", "강점", "약점", "비교"]):
        if groupby or x is None:
            return "radar"
        if x and not groupby:
            return "delta_bar"

    # groupby + 중분류 비교
    if groupby and (("중분류" in (x or "").lower()) or "만족도" in (focus or "").lower()):
        return "grouped_bar"

    # 분포 관련
    if any(k in focus for k in ["분포", "연령대", "비율", "많이"]):
        if x:
            return "bar"

    # 패턴, 군집 -> heatmap
    if any(k in focus for k in ["패턴", "군집", "비슷한", "차이"]):
        return "heatmap"

    # default fallback
    return "bar"

def generate_explanation_from_spec(df_subset: pd.DataFrame, spec: dict, computed_metrics: dict, extra_group_stats=None):
    focus = spec.get("focus", "기본 요약")
    parts = []
    if "overall_mid_scores" in computed_metrics:
        mids = computed_metrics["overall_mid_scores"]
        parts.append("전체 중분류 평균: " + ", ".join(f"{k} {v:.1f}" for k, v in mids.items()))
    if "deltas" in computed_metrics:
        deltas = computed_metrics["deltas"]
        delta_str = ", ".join(f"{k} {v:+.1f}" for k, v in deltas.items())
        parts.append("전체 평균 대비 편차: " + delta_str)
    if "top_segments" in computed_metrics:
        top = computed_metrics["top_segments"]
        parts.append("주요 세그먼트/조합: " + "; ".join(f"{t['label']} (n={t['n']})" for t in top))
    if extra_group_stats:
        summary_lines = []
        for group_label, mids in extra_group_stats.items():
            for mid, stats in mids.items():
                line = f"{group_label}의 '{mid}' 평균 {stats.get('mean')}, 전체 대비 {stats.get('delta_vs_overall'):+.1f}"
                if stats.get("p_value_vs_rest") is not None:
                    line += f", p={stats['p_value_vs_rest']}"
                if stats.get("cohen_d_vs_rest") is not None:
                    line += f", d={stats['cohen_d_vs_rest']}"
                summary_lines.append(line)
        parts.append("그룹 비교: " + " / ".join(summary_lines[:3]))  # 길이 제한 감안

    summary_context = "\n".join(parts)
    prompt = f"""
너는 전략 리포트 작성자다. 아래 컨텍스트와 사용자 질의 포커스를 참고해 명확한 인사이트를 만들어줘.

사용자 질의 포커스: {spec.get('focus', '')}

데이터 요약:
{summary_context}

요청:
1. 주요 관찰 패턴 2~3개를 기술해줘.
2. 강점과 약점을 구체적인 항목명이나 세그먼트명을 숫자와 함께 설명해줘.
3. 우선 개입/확장할만한 행동 제안 2개를 제시해줘.
4. 전체 길이 500~1000자, 비즈니스 톤, 숫자는 한 자리 소수, '-' 사용.

출력만 텍스트로 해줘.
"""
    explanation = call_gpt_for_insight(prompt)
    return explanation.replace("~", "-")


def apply_filters(df: pd.DataFrame, filters: list):
    dff = df.copy()
    for f in filters:
        col = f.get("col")
        op = f.get("op", "==")
        val = f.get("value")
        if col not in dff.columns or val is None:
            continue
        if op in ("==", "="):
            dff = dff[dff[col].astype(str) == str(val)]
        elif op == "in" and isinstance(val, list):
            dff = dff[dff[col].astype(str).isin([str(v) for v in val])]
        elif op == "contains":
            dff = dff[dff[col].astype(str).str.contains(str(val), na=False)]
    return dff

def handle_nl_question(df: pd.DataFrame, question: str):
    st.markdown("## 자연어 질의 결과")
    st.markdown(f"**질의:** {question}")

    spec = parse_nl_query_to_spec(question)
    df_filtered = apply_filters(df, spec.get("filters", []))

    if df_filtered.empty:
        st.warning("필터 적용 결과 데이터가 없습니다. 조건을 조정해보세요.")
        return

    # 중분류 관련 주요 지표
    overall_mid_scores = compute_midcategory_scores(df_filtered)
    overall_mid_dict = {k: float(v) for k, v in overall_mid_scores.items()} if not overall_mid_scores.empty else {}

    # 전체 평균 대비 (원래 전체 데이터 기준)
    global_mid_scores = compute_midcategory_scores(df)
    deltas = {}
    for k in overall_mid_dict:
        base = float(global_mid_scores.get(k, overall_mid_dict.get(k, 0)))
        deltas[k] = overall_mid_dict.get(k, 0) - base

    # 상위 조합/세그먼트 추출 (간단: groupby가 있으면 그 그룹별 개수)
    top_segments = []
    gb = spec.get("groupby")
    if gb and gb in df_filtered.columns:
        counts = df_filtered[gb].astype(str).value_counts().nlargest(3)
        for label, n in counts.items():
            subset = df_filtered[df_filtered[gb].astype(str) == label]
            profile = compute_midcategory_scores(subset)
            top_segments.append({"label": f"{gb}={label}", "n": int(n), "profile": {k: float(v) for k, v in profile.items()}})
    else:
        top_segments.append({"label": "필터된 전체", "n": len(df_filtered), "profile": overall_mid_dict})

    computed_metrics = {
        "overall_mid_scores": overall_mid_dict,
        "deltas": deltas,
        "top_segments": top_segments
    }

    # 그룹 비교 통계 (groupby가 있으면)
    extra_group_stats = None
    gb = spec.get("groupby")
    if gb and gb in df_filtered.columns:
        extra_group_stats = compare_midcategory_by_group(df_filtered, gb)

    # 설명 생성 (기존 호출을 아래로 교체)
    explanation = generate_explanation_from_spec(df_filtered, spec, computed_metrics, extra_group_stats=extra_group_stats)
    render_insight_card("자연어 기반 설명", explanation, key="nlq-insight")


    # 차트 유형 결정
    chart_type = infer_chart_type(spec, df_filtered)

    chart = None
    # 시각화 생성 (간단한 규칙 기반)
    if chart_type == "radar":
        # 전체 평균 + 필터된 세그먼트 레이더
        fig = go.Figure()
        midcats = list(overall_mid_scores.index)
        if not midcats:
            st.warning("중분류 점수를 계산할 수 없습니다.")
        else:
            # 전체 평균 (원본 전체)
            global_vals = [float(global_mid_scores.get(m, 0)) for m in midcats]
            global_closed = global_vals + [global_vals[0]]
            cats_closed = midcats + [midcats[0]]
            fig.add_trace(go.Scatterpolar(
                r=global_closed,
                theta=cats_closed,
                fill=None,
                name="전체 평균",
                line=dict(dash="dash", width=2),
                opacity=0.6
            ))
            # 필터된
            vals = [float(overall_mid_scores.get(m, 0)) for m in midcats]
            vals_closed = vals + [vals[0]]
            fig.add_trace(go.Scatterpolar(
                r=vals_closed,
                theta=cats_closed,
                fill='toself',
                name="질의 대상",
                hovertemplate="%{theta}: %{r:.1f}<extra></extra>"
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(range=[0, 100])),
                title="중분류 만족도 프로파일 비교 (전체 평균 vs 대상)",
                height=450
            )
            chart = fig

    elif chart_type == "heatmap":
        # 중분류 평균 히트맵 (필터된)
        midcat_prefixes = list(MIDCAT_MAP.values())
        heat_df = df_filtered.copy()
        matrix = {}
        for mc, prefix in MIDCAT_MAP.items():
            if isinstance(prefix, list):
                cols = sum([ [c for c in heat_df.columns if c.startswith(p)] for p in prefix], [])
            else:
                cols = [c for c in heat_df.columns if c.startswith(prefix)]
            if not cols:
                continue
            vals = heat_df[cols].apply(pd.to_numeric, errors="coerce")
            matrix[mc] = round(100 * (vals.mean(axis=1, skipna=True) - 1) / 6).mean() if not vals.empty else None
        if matrix:
            hm_df = pd.DataFrame.from_dict(matrix, orient="index", columns=["평균점수"])
            fig = px.imshow(
                hm_df.T,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Blues",
                title="중분류 평균 히트맵 (질의 대상 기준)",
                labels=dict(x="중분류", y="", color="점수")
            )
            chart = fig

    elif chart_type in ("grouped_bar", "bar"):
        # 기본 x 기준 막대 or groupby+중분류 비교
        x = spec.get("x")
        groupby = spec.get("groupby")
        if groupby and groupby in df_filtered.columns:
            # groupby별 중분류 만족도 비교
            rows = []
            midcats = list(MIDDLE_CATEGORY_MAPPING.keys())
            for val, sub in df_filtered.groupby(df_filtered[groupby].astype(str)):
                scores = compute_midcategory_scores(sub)
                for m in scores.index:
                    rows.append({
                        groupby: val,
                        "중분류": m,
                        "만족도": float(scores.get(m, 0)),
                        "전체 평균": float(global_mid_scores.get(m, 0))
                    })
            if rows:
                plot_df = pd.DataFrame(rows)
                fig = px.bar(
                    plot_df,
                    x="중분류",
                    y="만족도",
                    color=groupby,
                    barmode="group",
                    title=f"{groupby}별 중분류 만족도 비교",
                    text="만족도"
                )
                avg_df = plot_df.drop_duplicates(subset=["중분류"])[["중분류", "전체 평균"]]
                fig.add_trace(go.Scatter(
                    x=avg_df["중분류"],
                    y=avg_df["전체 평균"],
                    mode="lines+markers",
                    name="전체 평균",
                    line=dict(dash="dash"),
                    hovertemplate="%{x}: %{y:.1f}<extra></extra>"
                ))
                fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
                chart = fig
        elif x and x in df_filtered.columns:
            cnt = df_filtered[x].astype(str).value_counts().reset_index()
            cnt.columns = [x, "count"]
            fig = px.bar(cnt, x=x, y="count", title=f"{x} 분포", text="count")
            fig.update_traces(textposition="outside")
            chart = fig

    else:
        st.warning("자동으로 적절한 시각화를 추론하지 못했습니다.")
    
    if chart is not None:
        st.plotly_chart(chart, use_container_width=True)
    else:
        st.info("생성할 차트가 없습니다.")

    explanation = generate_explanation_from_spec(df_filtered, spec, computed_metrics)
    render_insight_card("자연어 기반 설명", explanation, key="nlq-insight")


# ─────────────────────────────────────────────────────
# 세그먼트 파생/매핑
# ─────────────────────────────────────────────────────
def add_derived_columns(df):
    df = df.copy()
    if "DQ1_FREQ" not in df.columns:
        dq1_cols = [c for c in df.columns if "DQ1" in c]
        if dq1_cols:
            dq1_col = dq1_cols[0]
            monthly = pd.to_numeric(df[dq1_col].astype(str).str.extract(r"(\d+\.?\d*)")[0], errors="coerce")
            yearly = monthly * 12
            bins = [0,12,24,48,72,144,1e10]
            labels = ["0~11회: 연 1회 미만", "12~23회: 월 1회", "24~47회: 월 2~4회", "48~71회: 주 1회", "72~143회: 주 2~3회", "144회 이상: 거의 매일"]
            df["DQ1_FREQ"] = pd.cut(yearly, bins=bins, labels=labels, right=False)
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

    if "DQ4_1ST" not in df.columns:
        dq4_cols = [c for c in df.columns if ("DQ4" in c) and ("1순위" in c)]
        if dq4_cols:
            df["DQ4_1ST"] = df[dq4_cols[0]]

    if "SQ2_GROUP" not in df.columns:
        sq2_cols = [c for c in df.columns if "SQ2" in c]
        if sq2_cols:
            sq2_col = sq2_cols[0]
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
        if "DQ1_FREQ" in df.columns:
            return ["DQ1_FREQ"]
        return [col for col in df.columns if "DQ1" in col]
    else:
        return [col for col in df.columns if key in col]

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
    "공익성 및 기여도": ["Q7-", "Q8-"],
}

# ─────────────────────────────────────────────────────
# 시각화 함수들
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
        marker_color=get_qualitative_colors(1)[0]
    ))
    fig.update_layout(
        title=question, yaxis_title="응답 수",
        bargap=0.1, height=450, margin=dict(t=40, b=10)
    )

    table_df = pd.DataFrame({'응답 수': grouped, '비율 (%)': percent}).T
    return fig, table_df

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

def plot_categorical_stacked_bar(df, question):
    data = df[question].dropna().astype(str)
    categories_raw = sorted(data.unique())
    display_labels = [label.split('. ', 1)[-1] for label in categories_raw]

    counts = data.value_counts().reindex(categories_raw).fillna(0).astype(int)
    percent = (counts / counts.sum() * 100).round(1)

    fig = go.Figure()
    colors = get_qualitative_colors(len(display_labels))
    for i, (raw_cat, label) in enumerate(zip(categories_raw, display_labels)):
        fig.add_trace(go.Bar(
            x=[percent[raw_cat]],
            y=[question],
            orientation='h',
            name=label,
            marker=dict(color=colors[i]),
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
        fig.add_vline(x=mid_mean, line_color="red")

    per_item_height = 50
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
                            textposition='outside', marker_color=get_qualitative_colors(1)[0]))
    fig.update_layout(title=question, xaxis_title="이용 빈도 구간", yaxis_title="응답 수",
                      bargap=0.2, height=450, margin=dict(t=30,b=50), xaxis_tickangle=-15)

    tbl_df = pd.DataFrame({"응답 수":grp, "비율 (%)":pct}).T
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
                            textposition='outside', marker_color=get_qualitative_colors(1)[0]))
    fig.update_layout(title=question, xaxis_title="이용 기간 (년)", yaxis_title="응답 수",
                      bargap=0.2, height=450, margin=dict(t=30,b=50), xaxis_tickangle=-15)
    tbl_df = pd.DataFrame({"응답 수":grp, "비율 (%)":pct}).T
    return fig, tbl_df, question

def plot_dq3(df):
    cols = [c for c in df.columns if c.startswith("DQ3")]
    if not cols:
        return None, None, ""
    question = cols[0]
    bar, table_df = plot_categorical_stacked_bar(df[[question]].dropna().astype(str), question)
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
        name='1순위', marker_color="#1f77b4", text=sorted_counts1.values, textposition='outside'
    ))
    fig.add_trace(go.Bar(
        x=sorted_labels, y=sorted_counts2.values,
        name='2순위', marker_color="#2ca02c", text=sorted_counts2.values, textposition='outside'
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
    }, index=sorted_labels).T
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
    fig.add_trace(go.Bar(x=labels, y=counts1, name='1순위', marker_color="#1f77b4", text=counts1, textposition='outside'))
    fig.add_trace(go.Bar(x=labels, y=counts2, name='2순위', marker_color="#2ca02c", text=counts2, textposition='outside'))
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
    }, index=labels).T
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
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            fill='none',
            name=t,
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

def page_segment_analysis(df):
    st.header("🧩 이용자 세그먼트 조합 분석")
    st.markdown("""
    - SQ1~5, DQ1, DQ2, DQ4(1순위) 중 **최대 3개** 문항 선택  
    - 선택한 보기 조합별(응답자 5명 이상)로 Q1~Q6, Q9-D-3, 공익성/기여도(Q7,Q8) 중분류별 만족도 평균을 **히트맵**으로 비교
    """)

    def safe_markdown(text, **kwargs):
        # 마크다운 해석에 따른 취소선 방지 (~ → \~ 또는 대체)
        escaped = escape_tildes(text, mode="markdown").replace("~~", r"\~\~")
        st.markdown(escaped, **kwargs)

    seg_labels = [o["label"] for o in SEGMENT_OPTIONS]
    sel = st.multiselect("세그먼트 조건 (최대 3개)", seg_labels, default=seg_labels[:2], max_selections=3)
    if not sel:
        st.info("최소 1개 이상을 선택하세요.")
        return
    selected_keys = [o["key"] for o in SEGMENT_OPTIONS if o["label"] in sel]

    df2 = add_derived_columns(df)

    segment_cols = []
    for key in selected_keys:
        segment_cols.extend(get_segment_columns(df2, key))
    segment_cols = list(dict.fromkeys(segment_cols))

    if not segment_cols:
        st.warning("선택한 세그먼트 조건에 해당하는 컬럼이 없습니다.")
        return

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

    segment_cols_filtered = [
        c for c in segment_cols
        if not (c.startswith("SQ2") and "GROUP" not in c) and c != "DQ2_YEARS"
    ]

    merge_keys = segment_cols_filtered
    counts_merge = counts[merge_keys + ["응답자수"]]
    group_means = pd.merge(group_means, counts_merge, how='left', on=merge_keys)

    # 중분류 평균 및 전체 평균 대비 편차
    group_means["중분류평균"] = group_means[midcats].mean(axis=1).round(2)
    overall_means = group_means[midcats].mean(axis=0)
    overall_mean_of_means = overall_means.mean()
    group_means["전체평균대비편차"] = (group_means["중분류평균"] - overall_mean_of_means).round(2)

    st.markdown("### 응답자 수 기준 상위 10개 세그먼트 조합의 중분류 만족도 프로파일 비교")
    top_n = 10
    top_df = group_means.nlargest(top_n, "응답자수").copy()

    overall_profile = group_means[midcats].mean(axis=0)
    overall_vals = [overall_profile.get(mc, overall_profile.mean()) for mc in midcats]
    overall_closed = overall_vals + [overall_vals[0]]
    cats_closed = midcats + [midcats[0]]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=overall_closed,
        theta=cats_closed,
        fill=None,
        name="전체 평균",
        line=dict(dash="dash", width=5, color="black"),
        opacity=0.5
    ))

    colors = DEFAULT_PALETTE
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

    # 룰 기반 요약
    st.markdown("#### 룰 기반 요약")
    overall_profile_dict = {mc: overall_profile.get(mc, 0) for mc in midcats}
    high_low_summary = interpret_midcategory_scores(df) if 'interpret_midcategory_scores' in globals() else ""
    safe_markdown(high_low_summary)

    # GPT 기반 해석 (레이더)
    st.markdown("#### GPT 생성형 해석")
    combos = []
    for _, row in top_df.iterrows():
        combo_label = " | ".join([str(row[c]) for c in segment_cols_filtered])
        profile = {mc: row.get(mc, overall_profile.get(mc, overall_profile.mean())) for mc in midcats}
        combos.append({"label": combo_label, "n": int(row["응답자수"]), "profile": profile})
    prompt = build_radar_prompt(overall_profile_dict, combos)
    insight_text = call_gpt_for_insight(prompt)
    insight_text = insight_text.replace("~", "-")
    render_insight_card("GPT 생성형 해석", insight_text, key="segment-radar")

    # 조합명 생성 및 delta 계산 (전체 평균 기준)
    group_means["조합"] = group_means.apply(lambda r: " | ".join([str(r[c]) for c in segment_cols_filtered]), axis=1)
    for mc in midcats:
        group_means[f"{mc}_delta"] = group_means[mc] - overall_means[mc]

    # 히트맵: 중분류 평균
    st.markdown("### 히트맵 + 전체 평균 대비 중분류별 편차 히트맵")
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

    st.markdown("#### 히트맵 룰 기반 요약")
    st.write("**전체 평균 대비 중분류 평균 프로파일**")

    st.markdown("#### GPT 생성형 해석 (히트맵)")
    heatmap_table = group_means[[*segment_cols_filtered, *midcats, "응답자수"]]
    prompt_heat = build_heatmap_prompt(heatmap_table.rename(columns={"응답자수": "응답자수"}), midcats)
    heat_insight = call_gpt_for_insight(prompt_heat)
    heat_insight = heat_insight.replace("~", "-")
    render_insight_card("GPT 생성형 해석 (히트맵)", heat_insight, key="heatmap-insight")

    # 델타 히트맵
    delta_plot = group_means.set_index("조합")[[f"{mc}_delta" for mc in midcats]]
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

    st.markdown("#### Delta 히트맵 룰 기반 요약")
    delta_summary_parts = []
    for mc in midcats:
        col_delta = f"{mc}_delta"
        if col_delta in group_means:
            top_pos = group_means.nlargest(1, col_delta)
            top_neg = group_means.nsmallest(1, col_delta)
            if not top_pos.empty:
                delta_summary_parts.append(f"{mc}에서 가장 높은 편차: {top_pos.iloc[0]['조합']} (+{top_pos.iloc[0][col_delta]:.1f})")
            if not top_neg.empty:
                delta_summary_parts.append(f"{mc}에서 가장 낮은 편차: {top_neg.iloc[0]['조합']} ({top_neg.iloc[0][col_delta]:.1f})")
    st.text("；".join(delta_summary_parts) if delta_summary_parts else "의미 있는 편차를 발견하지 못했습니다.")

    st.markdown("#### GPT 생성형 해석 (Delta)")
    delta_df_for_prompt = group_means.set_index("조합")
    prompt_delta = build_delta_prompt(delta_df_for_prompt, midcats)
    delta_insight = call_gpt_for_insight(prompt_delta)
    delta_insight = delta_insight.replace("~", "-")
    render_insight_card("GPT 생성형 해석 (델타 히트맵)", delta_insight, key="delta-heatmap-insight")

#신뢰구간 포함 편차 바 차트 해석

    st.markdown("### 전체 평균 대비 편차와 간이 신뢰구간 (중분류별)")
    for mc in midcats[:2]:
        subset = group_means.nlargest(10, "응답자수").copy()
        subset["delta"] = subset[mc] - overall_means[mc]
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
        st.markdown(f"#### '{mc}' 편차 신뢰도 해석")
        # 룰 기반: 0을 벗어나는지 체크
        ci_summary = []
        subset_local = subset  # 기존 변수
        for _, r in subset_local.iterrows():
            combo = r["조합"]
            delta = r["delta"]
            se = r["se"]
            ci_lower = delta - se
            ci_upper = delta + se
            signif = "유의미" if not (ci_lower <= 0 <= ci_upper) else "불확실"
            ci_summary.append(f"{combo}: 편차 {delta:.1f}, SE {se:.2f} ({signif})")
        safe_markdown("；".join(ci_summary))

        st.markdown("#### GPT 생성형 해석 (신뢰구간)")
        prompt_ci = build_ci_prompt(subset_local, mc)
        ci_insight = call_gpt_for_insight(prompt_ci)
        render_insight_card("GPT 생성형 해석 (신뢰구간)", ci_insight, key="ci-insight")

def show_basic_strategy_insights(df):
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

            default_n = min(5, len(purpose_counts))
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

            colors = DEFAULT_PALETTE
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

            top_n = st.number_input(
                "레이더에 표시할 상위 이용 목적 개수",
                min_value=1,
                max_value=max(1, len(purpose_counts)),
                value=default_n,
                step=1,
                key="strategy_radar_top_n_main"
            )

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
            fig = px.bar(
                plot_df,
                x="중분류",
                y="서비스별 만족도",
                color="주이용서비스",
                barmode="group",
                title="주이용서비스별 중분류 만족도 비교",
                text="서비스별 만족도"
            )
            avg_df = plot_df.drop_duplicates(subset=["중분류"])[["중분류", "전체 평균"]]
            fig.add_trace(go.Scatter(
                x=avg_df["중분류"],
                y=avg_df["전체 평균"],
                mode="lines+markers",
                name="전체 평균",
                line=dict(dash="dash"),
                hovertemplate="%{x}: %{y:.1f}<extra></extra>"
            ))
            for trace in fig.data:
                if getattr(trace, "type", None) == "bar":
                    trace.texttemplate = '%{text:.1f}'
                    trace.textposition = 'outside'
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("주이용서비스별로 비교할 충분한 데이터가 없습니다.")

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

    st.subheader("5. 이용 목적 (DQ4 계열) × 운영시간 만족도 (Q7-D-7)")
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
st.set_page_config(
    page_title="공공도서관 설문 시각화 대시보드",
    layout="wide"
)

mode = st.sidebar.radio("분석 모드", ["기본 분석", "심화 분석", "전략 인사이트(기본)", "자연어 질의"])

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
elif mode == "자연어 질의":
    st.header("🗣️ 자연어 질문 기반 자동 분석")
    st.markdown("예시: '혼자 이용하는 사람들의 연령대 분포 보여주고 주로 가는 도서관별 중분류 만족도 강점/약점 비교해줘.'")
    question = st.text_input("자연어 질문을 입력하세요", placeholder="예: 혼자 이용자들의 주 이용 도서관별 만족도 비교하고 강점 약점 알려줘")
    if question:
        handle_nl_question(df, question)
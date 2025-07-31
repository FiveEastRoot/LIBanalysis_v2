import math
import re
import pandas as pd
import streamlit as st
import openai
import plotly.express as px
import plotly.graph_objects as go

openai.api_key = st.secrets["openai"]["api_key"]
client = openai.OpenAI(api_key=openai.api_key)

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


def map_keyword_to_category(keyword):
    for cat, kws in KDC_KEYWORD_MAP.items():
        if any(k in keyword for k in kws):
            return cat
    return "해당없음"


def split_keywords_simple(text):
    parts = re.split(r"[.,/\s]+", text)
    return [p.strip() for p in parts if len(p.strip()) > 1]


@st.cache_data(show_spinner=False)
def extract_keyword_and_audience(responses, batch_size=20):
    results = []
    for i in range(0, len(responses), batch_size):
        batch = responses[i:i + batch_size]
        prompt = f"""당신은 도서관 자유응답에서 아래 형식의 JSON 배열만 반환합니다.
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
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.2,
            max_tokens=300
        )
        content = resp.choices[0].message.content.strip()
        try:
            data = pd.read_json(content)
        except Exception:
            data = []
            for txt in batch:
                kws = split_keywords_simple(txt)
                audience = '일반'
                for w in ['어린이','초등']:
                    if w in txt:
                        audience='아동'
                for w in ['유아','미취학','그림책']:
                    if w in txt:
                        audience='유아'
                for w in ['청소년','진로','자기계발']:
                    if w in txt:
                        audience='청소년'
                data.append({
                    'response': txt,
                    'keywords': kws,
                    'audience': audience
                })
            data = pd.DataFrame(data)
        for _, row in data.iterrows():
            results.append((row['response'], row['keywords'], row['audience']))
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


def page_short_keyword(df):
    with st.spinner("🔍 GPT 기반 키워드 분석 중..."):
        target_cols = [col for col in df.columns if "Q9-DS-4" in col]
        if not target_cols:
            st.warning("Q9-DS-4 관련 문항을 찾을 수 없습니다.")
            return
        answers = df[target_cols[0]].dropna().astype(str).tolist()
        df_result = process_answers(answers)
        show_short_answer_keyword_analysis(df_result)

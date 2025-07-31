import re
import streamlit as st

from plots.satisfaction import plot_stacked_bar_with_table


def page_basic_vis(df):
    st.subheader("📈 7점 척도 만족도 문항 (Q1 ~ Q8)")
    likert_qs = [
        col for col in df.columns
        if (re.match(r"Q[1-9][\.-]", str(col)))
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

    tabs = st.tabs(list(section_mapping.keys()))
    for tab, section_name in zip(tabs, section_mapping.keys()):
        with tab:
            st.markdown(f"### {section_name}")
            for q in section_mapping[section_name]:
                bar, tbl = plot_stacked_bar_with_table(df, q)
                st.plotly_chart(bar, use_container_width=True)
                st.plotly_chart(tbl, use_container_width=True)

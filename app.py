import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from pages.home import page_home
from pages.basic_vis import page_basic_vis
from pages.short_keyword import (
    page_short_keyword,
    show_short_answer_keyword_analysis,
    process_answers,
)
from plots.demographics import plot_categorical_stacked_bar
from plots.satisfaction import (
    plot_dq1,
    plot_dq2,
    plot_dq3,
    plot_dq4_bar,
    plot_dq5,
    plot_likert_diverging,
    plot_pair_bar,
    plot_stacked_bar_with_table,
)

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

main_tabs = st.tabs([
    "👤 응답자 정보",
    "📈 만족도 기본 시각화",
    "🗺️ 자치구 구성 문항",
    "📊도서관 이용양태 분석",
    "🖼️ 도서관 이미지 분석",
    "🏋️ 도서관 강약점 분석",
])

with main_tabs[0]:
    page_home(df)

with main_tabs[1]:
    page_basic_vis(df)

with main_tabs[2]:
    st.header("🗺️ 자치구 구성 문항 분석")
    sub_tabs = st.tabs([
        "7점 척도 시각화",
        "단문 응답 분석",
        "장문 서술형 분석",
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
                st.plotly_chart(bar, use_container_width=True, key=f"bar-{idx}-{col}")
                st.plotly_chart(tbl, use_container_width=True, key=f"tbl-{idx}-{col}")

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

with main_tabs[3]:
    st.header("📊 도서관 이용양태 분석")
    sub_tabs = st.tabs(["DQ1~5", "DQ6 계열"])

    with sub_tabs[0]:
        fig1, tbl1, q1 = plot_dq1(df)
        if fig1:
            st.subheader(q1)
            st.plotly_chart(fig1, use_container_width=True)
            st.plotly_chart(tbl1, use_container_width=True)
        fig2, tbl2, q2 = plot_dq2(df)
        if fig2:
            st.subheader(q2)
            st.plotly_chart(fig2, use_container_width=True)
            st.plotly_chart(tbl2, use_container_width=True)
        fig3, tbl3, q3 = plot_dq3(df)
        if fig3:
            st.subheader(q3)
            st.plotly_chart(fig3, use_container_width=True)
            st.plotly_chart(tbl3, use_container_width=True)
        fig4, tbl4, q4 = plot_dq4_bar(df)
        if fig4:
            st.subheader(q4)
            st.plotly_chart(fig4, use_container_width=True)
            st.plotly_chart(tbl4, use_container_width=True)
        fig5, tbl5, q5 = plot_dq5(df)
        if fig5:
            st.subheader(q5)
            st.plotly_chart(fig5, use_container_width=True)
            st.plotly_chart(tbl5, use_container_width=True)

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
                        textposition='outside', marker_color=px.colors.qualitative.Plotly
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
                    table_fig = go.Figure(go.Table(
                        header=dict(values=[""] + list(table_df.columns), align='center'),
                        cells=dict(values=[table_df.index] + [table_df[c].tolist() for c in table_df.columns], align='center')
                    ))
                    table_fig.update_layout(height=250, margin=dict(t=10, b=5))
                    st.plotly_chart(fig, use_container_width=True)
                    st.plotly_chart(table_fig, use_container_width=True)
                else:
                    bar, tbl = plot_categorical_stacked_bar(df, col)
                    st.plotly_chart(bar, use_container_width=True)
                    st.plotly_chart(tbl, use_container_width=True)

with main_tabs[4]:
    st.header("🖼️ 도서관 이미지 분석")
    fig, tbl = plot_likert_diverging(df, prefix="DQ7-E")
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(tbl, use_container_width=True)
    else:
        st.warning("DQ7-E 문항이 없습니다.")

with main_tabs[5]:
    st.header("🏋️ 도서관 강약점 분석")
    fig8, tbl8, q8 = plot_pair_bar(df, "DQ8")
    if fig8 is not None:
        st.plotly_chart(fig8, use_container_width=True)
        st.plotly_chart(tbl8, use_container_width=True)
    else:
        st.warning("DQ8 문항이 없습니다.")
    fig9, tbl9, q9 = plot_pair_bar(df, "DQ9")
    if fig9 is not None:
        st.plotly_chart(fig9, use_container_width=True)
        st.plotly_chart(tbl9, use_container_width=True)
    else:
        st.warning("DQ9 문항이 없습니다.")

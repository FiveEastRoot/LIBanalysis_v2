import streamlit as st

from plots.demographics import (
    plot_age_histogram_with_labels,
    plot_bq2_bar,
    plot_sq4_custom_bar,
    plot_categorical_stacked_bar,
)


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

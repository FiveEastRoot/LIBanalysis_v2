import re
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from utils.text import remove_parentheses, wrap_label
from .demographics import plot_categorical_stacked_bar


def plot_stacked_bar_with_table(df, question):
    data = pd.to_numeric(df[question].dropna(), errors='coerce').dropna().astype(int)
    order = [1, 2, 3, 4, 5, 6, 7]
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
        yaxis=dict(showticklabels=False), height=180, margin=dict(t=40, b=2)
    )

    table_df = pd.DataFrame({
        '응답 수': [int(counts[v]) for v in order],
        '비율 (%)': [percent[v] for v in order]
    }, index=[f"{v}점" for v in order]).T
    table_fig = go.Figure(go.Table(
        header=dict(values=[""] + list(table_df.columns), align='center'),
        cells=dict(values=[table_df.index] + [table_df[c].tolist() for c in table_df.columns], align='center')
    ))
    table_fig.update_layout(height=80, margin=dict(t=10, b=0))
    return fig, table_fig


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
        except Exception:
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
    order = [
        "0~11회: 연 1회 미만",
        "12~23회: 월 1회 정도",
        "24~47회: 월 2~4회 정도",
        "48~71회: 주 1회 정도",
        "72~143회: 주 2~3회",
        "144회 이상: 거의 매일"
    ]
    grp = cat.value_counts().reindex(order, fill_value=0)
    pct = (grp / grp.sum() * 100).round(1)

    fig = go.Figure(go.Bar(x=grp.index, y=grp.values, text=grp.values,
                            textposition='outside', marker_color="#1f77b4"))
    fig.update_layout(title=question, xaxis_title="이용 빈도 구간", yaxis_title="응답 수",
                      bargap=0.2, height=450, margin=dict(t=30, b=50), xaxis_tickangle=-15)
    tbl_df = pd.DataFrame({"응답 수": grp, "비율 (%)": pct}).T
    tbl = go.Figure(go.Table(header=dict(values=[""] + list(tbl_df.columns)),
                             cells=dict(values=[tbl_df.index] + [tbl_df[c].tolist() for c in tbl_df.columns])))
    tbl.update_layout(height=250, margin=dict(t=10, b=5))
    return fig, tbl, question


def plot_dq2(df):
    cols = [c for c in df.columns if c.startswith("DQ2")]
    if not cols:
        return None, None, ""
    question = cols[0]

    def parse(s):
        s = str(s).strip()
        m = re.match(r'^(\d+)\s*년\s*(\d+)\s*개월$', s)
        if m:
            return int(m.group(1)) + (1 if int(m.group(2)) > 0 else 0)
        m = re.match(r'^(\d+)\s*년$', s)
        if m:
            return int(m.group(1))
        m = re.match(r'^(\d+)\s*개월$', s)
        if m:
            return 1
        return None

    yrs = df[question].dropna().apply(parse)
    grp = yrs.value_counts().sort_index()
    pct = (grp / grp.sum() * 100).round(1)
    labels = [f"{y}년" for y in grp.index]
    fig = go.Figure(go.Bar(x=labels, y=grp.values, text=grp.values,
                            textposition='outside', marker_color="#1f77b4"))
    fig.update_layout(title=question, xaxis_title="이용 기간 (년)", yaxis_title="응답 수",
                      bargap=0.2, height=450, margin=dict(t=30, b=50), xaxis_tickangle=-15)
    tbl_df = pd.DataFrame({"응답 수": grp, "비율 (%)": pct}).T
    tbl = go.Figure(go.Table(header=dict(values=[""] + labels),
                             cells=dict(values=[tbl_df.index] + [tbl_df[c].tolist() for c in tbl_df.columns])))
    tbl.update_layout(height=250, margin=dict(t=10, b=5))
    return fig, tbl, question


def plot_dq3(df):
    cols = [c for c in df.columns if c.startswith("DQ3")]
    if not cols:
        return None, None, ""
    question = cols[0]
    temp_df = df[[question]].dropna().astype(str)
    fig, table_fig = plot_categorical_stacked_bar(temp_df, question)
    return fig, table_fig, question


def plot_dq4_bar(df):
    cols = [c for c in df.columns if c.startswith("DQ4")]
    if len(cols) < 2:
        return None, None, ""
    col1, col2 = cols[0], cols[1]
    question = f"{col1} vs {col2}"

    s1 = df[col1].dropna().astype(str)
    s2 = df[col2].dropna().astype(str)
    cats = sorted(set(s1.unique()).union(s2.unique()))
    labels = [c.split('. ', 1)[-1] if '. ' in c else c for c in cats]
    counts1 = s1.value_counts().reindex(cats, fill_value=0)
    counts2 = s2.value_counts().reindex(cats, fill_value=0)
    pct1 = (counts1 / counts1.sum() * 100).round(1)
    pct2 = (counts2 / counts2.sum() * 100).round(1)

    order_idx = counts1.sort_values(ascending=False).index.tolist()
    sorted_labels = [lbl.split('. ', 1)[-1] if '. ' in lbl else lbl for lbl in order_idx]
    sorted_counts1 = counts1.reindex(order_idx)
    sorted_counts2 = counts2.reindex(order_idx)
    sorted_pct1 = pct1.reindex(order_idx)
    sorted_pct2 = pct2.reindex(order_idx)

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


def plot_dq5(df):
    cols = [c for c in df.columns if c.startswith("DQ5")]
    if not cols:
        return None, None, ""
    question = cols[0]
    temp_df = df[[question]].dropna().astype(str)
    fig, table_fig = plot_categorical_stacked_bar(temp_df, question)
    return fig, table_fig, question


def plot_likert_diverging(df, prefix="DQ7-E"):
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return None, None
    dist = {}
    for col in cols:
        counts = df[col].dropna().astype(int).value_counts().reindex(range(1, 8), fill_value=0)
        pct = (counts / counts.sum() * 100).round(1)
        dist[col] = pct
    likert_df = pd.DataFrame(dist).T
    likert_df = likert_df.reindex(columns=range(1, 8))

    fig = go.Figure()
    neg_scores = [3, 2, 1]
    neg_colors = ["#91bfdb", "#4575b4", "#313695"]
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
        x=likert_df[4],
        name="4점",
        orientation='h',
        marker_color="#dddddd"
    ))
    for score, color in zip([5, 6, 7], ["#fee090", "#fc8d59", "#d73027"]):
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
        xaxis=dict(visible=False),
        legend=dict(traceorder='normal'),
        height=250,
        margin=dict(t=30, b=5),
    )

    table_df = likert_df.copy().reindex(columns=range(1, 8))
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


def plot_pair_bar(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix)]
    if len(cols) < 2:
        return None, None, ""
    col1, col2 = cols[0], cols[1]
    question = f"{col1} vs (2순위)"
    s1 = df[col1].dropna().astype(str)
    s2 = df[col2].dropna().astype(str)
    cats = sorted(set(s1.unique()).union(s2.unique()))
    labels = [c.split('. ', 1)[-1] if '. ' in c else c for c in cats]
    counts1 = s1.value_counts().reindex(cats, fill_value=0)
    counts2 = s2.value_counts().reindex(cats, fill_value=0)

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

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from utils.text import remove_parentheses, wrap_label


def plot_age_histogram_with_labels(df, question):
    data = df[question].dropna().astype(str).str.extract(r"(\d+)")
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
        x=grouped.index,
        y=grouped.values,
        text=grouped.values,
        textposition='outside',
        marker_color="#1f77b4"
    ))
    fig.update_layout(
        title=question,
        yaxis_title="응답 수",
        bargap=0.1,
        height=450,
        margin=dict(t=40, b=10)
    )

    table_df = pd.DataFrame({'응답 수': grouped, '비율 (%)': percent}).T
    table_fig = go.Figure(go.Table(
        header=dict(values=[""] + list(table_df.columns)),
        cells=dict(values=[table_df.index] + [table_df[c].tolist() for c in table_df.columns])
    ))
    table_fig.update_layout(height=180, margin=dict(t=10, b=5))

    return fig, table_fig


def plot_bq2_bar(df, question):
    data = df[question].dropna().astype(str)
    counts_raw = data.value_counts()
    percent_raw = (counts_raw / counts_raw.sum() * 100).round(1)

    categories_raw = counts_raw.index.tolist()
    categories = [label.split('. ', 1)[-1] for label in categories_raw]
    counts = counts_raw.values
    percent = percent_raw.values

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

    table_df = pd.DataFrame({'응답 수': counts, '비율 (%)': percent}, index=wrapped_labels).T
    table_fig = go.Figure(go.Table(
        header=dict(values=[""] + list(table_df.columns), align='center', height=36, font=dict(size=11)),
        cells=dict(values=[table_df.index] + [table_df[col].tolist() for col in table_df.columns], align='center', height=36, font=dict(size=11))
    ))
    table_fig.update_layout(height=150, margin=dict(t=10, b=5))

    return fig, table_fig


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
            x=[percent[cat]],
            y=[question],
            orientation='h',
            name=remove_parentheses(cat),
            marker_color=colors[i%len(colors)],
            text=f"{percent[cat]}%",
            textposition='inside'
        ))
    fig.update_layout(
        barmode='stack',
        showlegend=True,
        legend=dict(orientation='h', y=-0.5, x=0.5, xanchor='center', traceorder='reversed'),
        title=question,
        yaxis=dict(showticklabels=False),
        height=250,
        margin=dict(t=40,b=100)
    )

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
            yanchor='bottom',
            y=-1,
            xanchor='center',
            x=0.5,
            traceorder='reversed'
        ),
        title=dict(text=question, font=dict(size=16)),
        yaxis=dict(showticklabels=False),
        height=250,
        margin=dict(t=40, b=100)
    )

    table_df = pd.DataFrame({
        '응답 수': [counts[c] for c in categories_raw],
        '비율 (%)': [percent[c] for c in categories_raw]
    }, index=categories).T

    table_df = table_df[table_df.columns[::-1]]

    table_fig = go.Figure(go.Table(
        header=dict(values=[""] + list(table_df.columns), align='center'),
        cells=dict(values=[table_df.index] + [table_df[col].tolist() for col in table_df.columns], align='center')
    ))
    table_fig.update_layout(height=120, margin=dict(t=10, b=5))
    return fig, table_fig

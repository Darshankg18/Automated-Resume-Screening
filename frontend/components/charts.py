"""Interactive Visualization Components using Plotly.

Provides executive-tier, dark-glass charts styled with vibrant neon accents,
crisp typography, and seamless transparency matching the modern dashboard theme.
"""

from __future__ import annotations

from typing import Any, Dict, List
import plotly.graph_objects as go


CHART_FONT_FAMILY = "Plus Jakarta Sans, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"


def create_feature_contribution_chart(
    feature_names: List[str],
    contributions: List[float],
    base_score: float,
    final_score: float,
    method_title: str = "SHAP Feature Contributions"
) -> go.Figure:
    """Create a high-impact horizontal contribution bar chart of feature attributions."""
    colors = ["#10b981" if val >= 0 else "#f43f5e" for val in contributions]
    border_colors = ["#34d399" if val >= 0 else "#fb7185" for val in contributions]

    fig = go.Figure(go.Bar(
        x=contributions,
        y=feature_names,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(color=border_colors, width=1.5),
            opacity=0.9
        ),
        text=[f"{v:+.1f} pts" for v in contributions],
        textposition="auto",
        textfont=dict(family=CHART_FONT_FAMILY, size=12, color="#ffffff")
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>{method_title}</b><br><span style='font-size:12px;color:#94a3b8;'>Base Expectation: <b>{base_score:.1f}</b> → Final Calibrated Score: <b>{final_score:.1f}</b></span>",
            font=dict(family=CHART_FONT_FAMILY, size=16, color="#ffffff")
        ),
        xaxis=dict(
            title=dict(
                text="Score Impact (Points relative to baseline expectation)",
                font=dict(family=CHART_FONT_FAMILY, size=12, color="#94a3b8")
            ),
            zeroline=True,
            zerolinecolor="#6366f1",
            zerolinewidth=2,
            gridcolor="rgba(255, 255, 255, 0.06)",
            tickfont=dict(family=CHART_FONT_FAMILY, color="#94a3b8", size=11)
        ),
        yaxis=dict(
            title="",
            autorange="reversed",
            gridcolor="rgba(255, 255, 255, 0.06)",
            tickfont=dict(family=CHART_FONT_FAMILY, color="#e2e8f0", size=12)
        ),
        plot_bgcolor="rgba(15, 23, 42, 0.5)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(l=20, r=20, t=65, b=45),
        height=390,
    )
    return fig


def create_waterfall_chart(
    feature_names: List[str],
    contributions: List[float],
    base_score: float,
    final_score: float,
    method_title: str = "SHAP Waterfall Breakdown"
) -> go.Figure:
    """Create an executive Plotly waterfall chart decomposing base score into final match score."""
    x_labels = ["Base Baseline"] + feature_names + ["Final Match Score"]
    measures = ["absolute"] + ["relative"] * len(feature_names) + ["total"]
    y_values = [base_score] + contributions + [final_score]

    text_labels = [f"{base_score:.1f}"] + [f"{v:+.1f}" for v in contributions] + [f"{final_score:.1f}"]

    fig = go.Figure(go.Waterfall(
        name="Score Path",
        orientation="v",
        measure=measures,
        x=x_labels,
        y=y_values,
        text=text_labels,
        textposition="outside",
        textfont=dict(family=CHART_FONT_FAMILY, size=12, color="#f8fafc"),
        connector=dict(line=dict(color="rgba(148, 163, 184, 0.6)", width=1.5, dash="dot")),
        increasing=dict(
            marker=dict(
                color="#10b981",
                line=dict(color="#34d399", width=1.5)
            )
        ),
        decreasing=dict(
            marker=dict(
                color="#f43f5e",
                line=dict(color="#fb7185", width=1.5)
            )
        ),
        totals=dict(
            marker=dict(
                color="#06b6d4",
                line=dict(color="#38bdf8", width=1.5)
            )
        ),
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>{method_title}</b>",
            font=dict(family=CHART_FONT_FAMILY, size=16, color="#ffffff")
        ),
        yaxis=dict(
            title=dict(text="Match Points (0 - 100)", font=dict(family=CHART_FONT_FAMILY, size=12, color="#94a3b8")),
            range=[0, 110],
            gridcolor="rgba(255, 255, 255, 0.06)",
            tickfont=dict(family=CHART_FONT_FAMILY, color="#94a3b8", size=11)
        ),
        xaxis=dict(
            tickangle=-25,
            gridcolor="rgba(255, 255, 255, 0.06)",
            tickfont=dict(family=CHART_FONT_FAMILY, color="#cbd5e1", size=11)
        ),
        plot_bgcolor="rgba(15, 23, 42, 0.5)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(l=20, r=20, t=55, b=85),
        height=440,
    )
    return fig


def create_group_fairness_bar_chart(
    group_names: List[str],
    selection_rates: List[float],
    threshold_value: float = 0.8
) -> go.Figure:
    """Create a demographic group selection rate comparison chart with 80% rule guideline."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=group_names,
        y=[r * 100 for r in selection_rates],
        marker=dict(
            color="#8b5cf6",
            line=dict(color="#a78bfa", width=1.5),
            opacity=0.9
        ),
        text=[f"{r*100:.1f}%" for r in selection_rates],
        textposition="outside",
        textfont=dict(family=CHART_FONT_FAMILY, color="#ffffff", size=12),
        name="Selection Rate (%)"
    ))

    # Add 80% four-fifths rule reference threshold guideline
    fig.add_hline(
        y=80.0,
        line_dash="dash",
        line_color="#f59e0b",
        line_width=1.5,
        annotation_text="EEOC 80% Rule Parity Threshold",
        annotation_position="top right",
        annotation_font=dict(family=CHART_FONT_FAMILY, size=11, color="#fbbf24")
    )

    fig.update_layout(
        title=dict(
            text="<b>Demographic Cohort Selection Rates (%)</b>",
            font=dict(family=CHART_FONT_FAMILY, size=15, color="#ffffff")
        ),
        yaxis=dict(
            title=dict(text="Selection Rate (%)", font=dict(family=CHART_FONT_FAMILY, size=12, color="#94a3b8")),
            range=[0, 115],
            gridcolor="rgba(255, 255, 255, 0.06)",
            tickfont=dict(family=CHART_FONT_FAMILY, color="#94a3b8", size=11)
        ),
        xaxis=dict(
            title=dict(text="Demographic Group Cohort", font=dict(family=CHART_FONT_FAMILY, size=12, color="#94a3b8")),
            tickfont=dict(family=CHART_FONT_FAMILY, color="#e2e8f0", size=12)
        ),
        plot_bgcolor="rgba(15, 23, 42, 0.5)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(l=20, r=20, t=55, b=45),
        height=340,
    )
    return fig


def create_score_distribution_chart(scores: List[float]) -> go.Figure:
    """Create a sleek neon distribution histogram of candidate scores."""
    fig = go.Figure(go.Histogram(
        x=scores,
        nbinsx=10,
        marker=dict(
            color="#06b6d4",
            line=dict(color="#38bdf8", width=1.5),
            opacity=0.85
        )
    ))

    fig.update_layout(
        title=dict(
            text="<b>Candidate Score Distribution</b>",
            font=dict(family=CHART_FONT_FAMILY, size=15, color="#ffffff")
        ),
        xaxis=dict(
            title=dict(text="Calibrated Match Score (0 - 100)", font=dict(family=CHART_FONT_FAMILY, size=12, color="#94a3b8")),
            range=[0, 100],
            gridcolor="rgba(255, 255, 255, 0.06)",
            tickfont=dict(family=CHART_FONT_FAMILY, color="#94a3b8", size=11)
        ),
        yaxis=dict(
            title=dict(text="Number of Candidates", font=dict(family=CHART_FONT_FAMILY, size=12, color="#94a3b8")),
            gridcolor="rgba(255, 255, 255, 0.06)",
            tickfont=dict(family=CHART_FONT_FAMILY, color="#94a3b8", size=11)
        ),
        plot_bgcolor="rgba(15, 23, 42, 0.5)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(l=20, r=20, t=45, b=45),
        height=300,
    )
    return fig

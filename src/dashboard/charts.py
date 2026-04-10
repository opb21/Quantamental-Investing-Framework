"""
Plotly chart builders for the dashboard.

Each function returns a plotly Figure, passed to st.plotly_chart().
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#2E5E8E", "#A85C30", "#3A7A4A", "#8B2020", "#574A80",
]

_BASE_LAYOUT = dict(
    template="plotly_white",
    height=320,
    margin=dict(l=60, r=20, t=30, b=40),
    hovermode="x unified",
)


def _no_data_fig(message: str = "No data available for this run.") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False, font=dict(size=13, color="#888888"),
    )
    fig.update_layout(xaxis_visible=False, yaxis_visible=False,
                      height=200, template="plotly_white")
    return fig


def equity_curve(
    returns_df: pd.DataFrame,
    benchmark_returns: pd.Series | None,
    run_name: str,
) -> go.Figure:
    """Cumulative portfolio return vs benchmark, both indexed to 1.0 at start."""
    if "portfolio_returns_net" not in returns_df.columns:
        return _no_data_fig("No return data available.")

    port_net = returns_df["portfolio_returns_net"].dropna()
    port_cum = (1 + port_net).cumprod()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=port_cum.index, y=port_cum.values,
        name=run_name,
        line=dict(color=PALETTE[0], width=2),
        hovertemplate="%{y:.2f}x<extra>" + run_name + "</extra>",
    ))

    if benchmark_returns is not None:
        bench = benchmark_returns.reindex(port_cum.index).dropna()
        if not bench.empty:
            bench_cum = (1 + bench).cumprod()
            bench_name = str(benchmark_returns.name or "Benchmark")
            fig.add_trace(go.Scatter(
                x=bench_cum.index, y=bench_cum.values,
                name=bench_name,
                line=dict(color=PALETTE[7], width=1.5, dash="dash"),
                hovertemplate="%{y:.2f}x<extra>" + bench_name + "</extra>",
            ))

    fig.add_hline(y=1.0, line_color="#cccccc", line_width=1, line_dash="dot")
    fig.update_layout(
        **_BASE_LAYOUT,
        yaxis_title="Growth of £1",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def drawdown_chart(returns_df: pd.DataFrame) -> go.Figure:
    """Drawdown series as a filled area."""
    if "drawdown" not in returns_df.columns:
        return _no_data_fig("Drawdown data not available for this run.")

    dd = returns_df["drawdown"].dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        fill="tozeroy",
        fillcolor="rgba(196,78,82,0.35)",
        line=dict(color=PALETTE[3], width=1),
        name="Drawdown",
        hovertemplate="%{y:.1%}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#cccccc", line_width=0.8)
    fig.update_layout(
        **_BASE_LAYOUT,
        yaxis_title="Drawdown",
        yaxis_tickformat=".0%",
        showlegend=False,
    )
    return fig


def rolling_risk_chart(returns_df: pd.DataFrame) -> go.Figure:
    """Two-panel: rolling annualised vol (top) and rolling Sharpe (bottom)."""
    has_vol = "rolling_vol" in returns_df.columns
    has_sharpe = "rolling_sharpe_12m" in returns_df.columns

    if not has_vol and not has_sharpe:
        return _no_data_fig("Rolling risk data not available for this run.")

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("Rolling Volatility (12m)", "Rolling Sharpe (12m)"),
    )

    if has_vol:
        rv = returns_df["rolling_vol"].dropna()
        fig.add_trace(go.Scatter(
            x=rv.index, y=rv.values,
            line=dict(color=PALETTE[1], width=1.5),
            name="Rolling Vol",
            hovertemplate="%{y:.1%}<extra></extra>",
        ), row=1, col=1)
        fig.update_yaxes(tickformat=".0%", title_text="Ann. Vol", row=1, col=1)

    if has_sharpe:
        rs = returns_df["rolling_sharpe_12m"].dropna()
        fig.add_trace(go.Scatter(
            x=rs.index, y=rs.values,
            line=dict(color=PALETTE[0], width=1.5),
            name="Rolling Sharpe",
            hovertemplate="%{y:.2f}<extra></extra>",
        ), row=2, col=1)
        fig.add_hline(y=0, line_color="#cccccc", line_width=0.8, row=2, col=1)
        fig.add_hline(y=1, line_color="#cccccc", line_width=0.8, line_dash="dot", row=2, col=1)
        fig.update_yaxes(title_text="Sharpe", row=2, col=1)

    fig.update_layout(
        template="plotly_white",
        height=480,
        margin=dict(l=60, r=20, t=50, b=40),
        hovermode="x unified",
        showlegend=False,
    )
    return fig


def avg_corr_chart(avg_pairwise_corr: pd.Series) -> go.Figure:
    """Rolling average pairwise correlation of holdings."""
    clean = avg_pairwise_corr.dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=clean.index, y=clean.values,
        fill="tozeroy",
        fillcolor="rgba(129,114,178,0.2)",
        line=dict(color=PALETTE[4], width=1.5),
        name="Avg Correlation",
        hovertemplate="%{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#cccccc", line_width=0.8)
    fig.update_layout(
        **_BASE_LAYOUT,
        yaxis_title="Correlation",
        yaxis_range=[-1, 1],
        showlegend=False,
    )
    return fig


def _aggregate_by_group(df: pd.DataFrame, ticker_to_group: dict[str, str]) -> pd.DataFrame:
    """Sum DataFrame columns by group label (e.g. industry). Returns dates × groups."""
    groups: dict[str, pd.Series] = {}
    for col in df.columns:
        group = ticker_to_group.get(col, "Unknown")
        groups[group] = groups[group].add(df[col].fillna(0), fill_value=0) if group in groups else df[col].fillna(0).copy()
    result = pd.DataFrame(groups)
    # Sort groups by total descending so largest appear first in the stack
    return result[result.sum().sort_values(ascending=False).index]


def _top_n_plus_other(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """
    Keep the first top_n columns (caller is responsible for meaningful ordering)
    and aggregate the remainder into an 'Other' column.
    Ensures stacked area charts always sum to their true total (e.g. 100%).
    """
    if df.shape[1] <= top_n:
        return df
    result = df.iloc[:, :top_n].copy()
    other = df.iloc[:, top_n:].sum(axis=1)
    if (other.abs() > 1e-8).any():
        result["Other"] = other
    return result


def _stacked_area_fig(df: pd.DataFrame, yaxis_title: str) -> go.Figure:
    """Generic stacked area chart. df columns = groups/tickers, index = dates."""
    fig = go.Figure()
    for i, col in enumerate(df.columns):
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col],
            name=col,
            stackgroup="one",
            mode="none",
            fillcolor=PALETTE[i % len(PALETTE)],
            hovertemplate=f"{col}: %{{y:.1%}}<extra></extra>",
        ))
    fig.update_layout(
        **_BASE_LAYOUT,
        yaxis_title=yaxis_title,
        yaxis_tickformat=".0%",
        yaxis_range=[0, 1.05],
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=9)),
    )
    return fig


def contrib_to_vol_chart(contrib_to_vol: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Stacked area of per-ticker percentage contribution to volatility."""
    active = contrib_to_vol.dropna(how="all", axis=1)
    active = active.loc[:, (active > 0).any()]
    if active.empty:
        return _no_data_fig("No contribution to volatility data.")
    plot_df = _top_n_plus_other(active.dropna(how="all").fillna(0), top_n)
    return _stacked_area_fig(plot_df, "Vol Contribution")


def contrib_to_vol_industry_chart(
    contrib_to_vol: pd.DataFrame,
    ticker_to_industry: dict[str, str],
) -> go.Figure:
    """Stacked area of vol contribution aggregated by industry."""
    active = contrib_to_vol.dropna(how="all", axis=1)
    active = active.loc[:, (active > 0).any()]
    if active.empty:
        return _no_data_fig("No contribution to volatility data.")
    grouped = _aggregate_by_group(active.dropna(how="all").fillna(0), ticker_to_industry)
    return _stacked_area_fig(grouped, "Vol Contribution")


def weights_area_chart(weights: pd.DataFrame, active_tickers: list[str], top_n: int = 20) -> go.Figure:
    """Stacked area of portfolio weights over time. Top N by total weight + Other."""
    w = weights[active_tickers].fillna(0)
    # Sort by total weight across history so the most consistently held names
    # get individual lines rather than the most recently held.
    by_total = w.sum().sort_values(ascending=False).index.tolist()
    plot_df = _top_n_plus_other(w[by_total], top_n)
    return _stacked_area_fig(plot_df, "Weight")


def weights_industry_chart(
    weights: pd.DataFrame,
    active_tickers: list[str],
    ticker_to_industry: dict[str, str],
) -> go.Figure:
    """Stacked area of portfolio weights aggregated by industry."""
    grouped = _aggregate_by_group(weights[active_tickers].fillna(0), ticker_to_industry)
    return _stacked_area_fig(grouped, "Weight")


def attribution_bar_chart(attribution: pd.Series) -> go.Figure:
    """
    Horizontal bar chart of total return attribution, ranked best to worst.
    Positive bars green, negative bars red.
    """
    s = attribution.sort_values(ascending=True)
    colors = [PALETTE[2] if v >= 0 else PALETTE[3] for v in s.values]
    fig = go.Figure(go.Bar(
        x=s.values,
        y=s.index,
        orientation="h",
        marker_color=colors,
        opacity=0.8,
        hovertemplate="%{y}: %{x:.2%}<extra></extra>",
    ))
    fig.update_layout(
        **{**_BASE_LAYOUT, "height": max(280, len(s) * 26 + 80)},
        xaxis_title="Return Contribution",
        xaxis_tickformat=".1%",
        showlegend=False,
    )
    return fig


def contribution_vs_return_chart(
    contribution: pd.Series,
    stock_return: pd.Series,
) -> go.Figure:
    """
    Grouped horizontal bar chart comparing each stock's contribution to portfolio
    return (weight × return × time held) against its standalone return while held.
    Sorted by contribution descending (top of chart = biggest contributor).
    """
    # Align and sort by contribution
    both = pd.DataFrame({"Contribution": contribution, "Stock Return": stock_return}).dropna()
    both = both.sort_values("Contribution", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=both["Contribution"], y=both.index,
        name="Portfolio Contribution",
        orientation="h",
        marker_color=PALETTE[0], opacity=0.85,
        hovertemplate="%{y} contribution: %{x:.2%}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=both["Stock Return"], y=both.index,
        name="Stock Return (while held)",
        orientation="h",
        marker_color=PALETTE[1], opacity=0.85,
        hovertemplate="%{y} return: %{x:.2%}<extra></extra>",
    ))
    fig.update_layout(
        **{**_BASE_LAYOUT, "height": max(300, len(both) * 32 + 100)},
        barmode="group",
        xaxis_title="Return",
        xaxis_tickformat=".1%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def cumulative_attribution_chart(stock_attribution: pd.DataFrame, top_n: int = 8) -> go.Figure:
    """
    Line chart of cumulative return attribution over time.
    Shows top_n series by absolute total contribution to reduce clutter.
    """
    totals = stock_attribution.sum().abs().sort_values(ascending=False)
    cols = totals.head(top_n).index.tolist()
    cum = stock_attribution[cols].cumsum()

    fig = go.Figure()
    for i, col in enumerate(cum.columns):
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum[col],
            name=col,
            line=dict(color=PALETTE[i % len(PALETTE)], width=1.5),
            hovertemplate=f"{col}: %{{y:.2%}}<extra></extra>",
        ))
    fig.add_hline(y=0, line_color="#cccccc", line_width=0.8)
    fig.update_layout(
        **{**_BASE_LAYOUT, "height": 360},
        yaxis_title="Cumulative Contribution",
        yaxis_tickformat=".1%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=9)),
    )
    return fig


def multi_equity_curves(runs: dict) -> go.Figure:
    """
    Cumulative return curves for multiple runs on one chart.
    runs: {label: portfolio_returns_net Series}
    """
    if not runs:
        return _no_data_fig("No runs selected.")
    fig = go.Figure()
    for i, (label, returns) in enumerate(runs.items()):
        cum = (1 + returns.dropna()).cumprod()
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values,
            name=label,
            line=dict(color=PALETTE[i % len(PALETTE)], width=1.5),
            hovertemplate=f"{label}: %{{y:.2f}}x<extra></extra>",
        ))
    fig.add_hline(y=1.0, line_color="#cccccc", line_width=1, line_dash="dot")
    fig.update_layout(
        **{**_BASE_LAYOUT, "height": 400},
        yaxis_title="Growth of £1",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=9)),
    )
    return fig


def multi_turnover_chart(runs: dict) -> go.Figure:
    """
    Turnover over time for multiple runs on one chart.
    runs: {label: turnover Series}
    """
    if not runs:
        return _no_data_fig("No runs selected.")
    fig = go.Figure()
    for i, (label, turnover) in enumerate(runs.items()):
        series = turnover.dropna()
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values,
            name=label,
            line=dict(color=PALETTE[i % len(PALETTE)], width=1.5),
            hovertemplate=f"{label}: %{{y:.1%}}<extra></extra>",
        ))
    fig.update_layout(
        **{**_BASE_LAYOUT, "height": 320},
        yaxis_title="Turnover",
        yaxis_tickformat=".0%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=9)),
    )
    return fig


def holdings_cumulative_returns_chart(cum_returns: pd.DataFrame) -> go.Figure:
    """
    Line chart of cumulative returns for each current holding since first purchased.
    cum_returns: DataFrame with dates as index, tickers as columns, values as
                 cumulative return (0.0 = breakeven, 0.10 = +10%).
    """
    if cum_returns.empty:
        return _no_data_fig("No return data available for current holdings.")

    fig = go.Figure()
    for i, col in enumerate(cum_returns.columns):
        series = cum_returns[col].dropna()
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values,
            name=col,
            line=dict(color=PALETTE[i % len(PALETTE)], width=1.5),
            hovertemplate=f"{col}: %{{y:.1%}}<extra></extra>",
        ))
    fig.add_hline(y=0, line_color="#cccccc", line_width=1, line_dash="dot")
    fig.update_layout(
        **{**_BASE_LAYOUT, "height": 360},
        yaxis_title="Cumulative Return",
        yaxis_tickformat=".0%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=9)),
    )
    return fig


def turnover_bar_chart(returns_df: pd.DataFrame) -> go.Figure:
    """Bar chart of one-way turnover at each rebalance period."""
    if "turnover" not in returns_df.columns:
        return _no_data_fig("No turnover data available.")

    to = returns_df["turnover"].dropna()
    to = to[to > 0]  # rebalance periods only; zero between rebalances
    if to.empty:
        return _no_data_fig("No turnover recorded.")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=to.index, y=to.values,
        marker_color=PALETTE[0],
        opacity=0.7,
        name="Turnover",
        hovertemplate="%{y:.1%}<extra></extra>",
    ))
    fig.update_layout(
        **_BASE_LAYOUT,
        yaxis_title="One-way turnover",
        yaxis_tickformat=".0%",
        showlegend=False,
        bargap=0.3,
    )
    return fig


def cost_drag_chart(returns_df: pd.DataFrame) -> go.Figure:
    """
    Per-rebalance-period transaction cost (bars, bps) with cumulative drag (line, %).

    Costs are already embedded in portfolio_returns_net; this chart makes the
    drag explicit. Only rebalance periods with non-zero cost are shown as bars.
    """
    if "costs" not in returns_df.columns:
        return _no_data_fig("No cost data available.")

    costs = returns_df["costs"].dropna()
    costs_rebal = costs[costs != 0]
    if costs_rebal.empty:
        return _no_data_fig("No costs recorded (0 bps commission + slippage).")

    costs_bps = costs_rebal * 10_000  # convert to bps (negative values)
    cumulative_pct = costs.cumsum() * 100  # cumulative drag in %

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=costs_bps.index,
        y=costs_bps.values,
        name="Period cost (bps)",
        marker_color=PALETTE[1],
        opacity=0.7,
        hovertemplate="%{y:.1f} bps<extra></extra>",
        yaxis="y",
    ))
    fig.add_trace(go.Scatter(
        x=cumulative_pct.index,
        y=cumulative_pct.values,
        name="Cumulative drag (%)",
        line=dict(color=PALETTE[0], width=1.5),
        hovertemplate="%{y:.2f}%<extra></extra>",
        yaxis="y2",
    ))
    fig.update_layout(
        **{**_BASE_LAYOUT, "height": 320},
        yaxis=dict(title="Cost (bps)"),
        yaxis2=dict(
            title="Cumulative drag (%)",
            overlaying="y",
            side="right",
            tickformat=".1f",
            showgrid=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        bargap=0.3,
    )
    return fig


def ic_timeseries_chart(ic_series: pd.Series, rolling_window: int = 12) -> go.Figure:
    """
    IC over time — bars coloured by sign with rolling mean overlay.

    Green bars = positive IC (signal predicted returns correctly that month).
    Red bars = negative IC. The rolling mean line shows trend in signal quality.
    """
    if ic_series is None or ic_series.empty:
        return _no_data_fig("No IC data available.")

    ic = ic_series.dropna()
    colors = [PALETTE[2] if v >= 0 else PALETTE[3] for v in ic.values]
    rolling_mean = ic.rolling(rolling_window, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ic.index, y=ic.values,
        marker_color=colors,
        opacity=0.6,
        name="IC",
        hovertemplate="%{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=rolling_mean.index, y=rolling_mean.values,
        name=f"{rolling_window}m rolling mean",
        line=dict(color=PALETTE[0], width=2),
        hovertemplate="%{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_width=1, line_color="black", opacity=0.3)
    fig.update_layout(
        **{**_BASE_LAYOUT, "height": 340},
        yaxis_title="IC",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        bargap=0.1,
    )
    return fig


def ic_decay_chart(ic_decay: pd.Series) -> go.Figure:
    """
    Mean IC at each forward horizon — shows how quickly the signal decays.

    A high IC at horizon 1 that drops sharply by horizon 3 means the signal
    is only useful for short holding periods. Slow decay supports longer
    rebalance intervals.
    """
    if ic_decay is None or ic_decay.empty:
        return _no_data_fig("No IC decay data available.")

    decay = ic_decay.dropna()
    colors = [PALETTE[2] if v >= 0 else PALETTE[3] for v in decay.values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=decay.index.astype(str),
        y=decay.values,
        marker_color=colors,
        opacity=0.7,
        name="Mean IC",
        hovertemplate="Horizon %{x}m: %{y:.4f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_width=1, line_color="black", opacity=0.3)
    fig.update_layout(
        **{**_BASE_LAYOUT, "height": 300},
        xaxis_title="Forward horizon (months)",
        yaxis_title="Mean IC",
        showlegend=False,
        bargap=0.3,
    )
    return fig


def wf_scatter(paired_df: pd.DataFrame, metric: str = "sharpe") -> go.Figure:
    """
    Scatter plot of IS vs OOS metric for paired walk-forward configs.
    Each point = one parameter combination. Diagonal = IS predicts OOS perfectly.
    Annotated with Spearman rank correlation.
    """
    is_col = f"{metric}_is"
    oos_col = f"{metric}_oos"
    if is_col not in paired_df.columns or oos_col not in paired_df.columns:
        return _no_data_fig(f"No IS/OOS data for metric '{metric}'.")

    valid = paired_df[[is_col, oos_col] + [
        c for c in ["lookback_months", "n_positions", "rebalance", "weighting"]
        if c in paired_df.columns
    ]].dropna(subset=[is_col, oos_col])

    if valid.empty:
        return _no_data_fig("No paired IS/OOS runs found.")

    x = valid[is_col].values
    y = valid[oos_col].values

    # Spearman rank correlation via pandas (no scipy needed)
    rho = float(pd.Series(x).rank().corr(pd.Series(y).rank())) if len(x) > 1 else float("nan")

    config_cols = [c for c in ["lookback_months", "n_positions", "rebalance", "weighting"]
                   if c in valid.columns]
    hover_texts = [
        "<br>".join(f"{c}: {row[c]}" for c in config_cols)
        for _, row in valid.iterrows()
    ]

    all_vals = list(x) + list(y)
    val_min, val_max = min(all_vals), max(all_vals)
    pad = (val_max - val_min) * 0.12 if val_max != val_min else 0.1

    fig = go.Figure()
    # Reference diagonal (IS = OOS)
    fig.add_trace(go.Scatter(
        x=[val_min - pad, val_max + pad],
        y=[val_min - pad, val_max + pad],
        mode="lines",
        line=dict(color="#cccccc", dash="dash", width=1),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers",
        marker=dict(size=11, color=PALETTE[0], opacity=0.85, line=dict(width=1, color="white")),
        text=hover_texts,
        hovertemplate=f"IS: %{{x:.3f}}<br>OOS: %{{y:.3f}}<br>%{{text}}<extra></extra>",
        showlegend=False,
    ))
    rho_label = f"{rho:.2f}" if not pd.isna(rho) else "n/a"
    fig.add_annotation(
        text=f"Rank correlation: {rho_label}",
        x=0.03, y=0.97, xref="paper", yref="paper",
        showarrow=False, align="left",
        font=dict(size=12), bgcolor="white", bordercolor="#dddddd", borderwidth=1,
    )
    fig.update_layout(
        **{**_BASE_LAYOUT, "height": 380},
        xaxis_title=f"IS {metric.replace('_', ' ').title()}",
        yaxis_title=f"OOS {metric.replace('_', ' ').title()}",
    )
    return fig


def sweep_sensitivity_chart(
    runs_df: pd.DataFrame,
    param: str,
    metric: str,
    group_by: str | None = None,
) -> go.Figure:
    """
    Bar chart of median metric per parameter value, with IQR whiskers.
    Shows which values of a parameter consistently outperform.

    When group_by is provided (e.g. 'signal_name'), renders grouped bars —
    one colour per group — so strategies can be compared side by side.
    """
    if param not in runs_df.columns or metric not in runs_df.columns:
        return _no_data_fig(f"Column '{param}' or '{metric}' not found.")

    if group_by and group_by in runs_df.columns and runs_df[group_by].nunique() > 1:
        groups = sorted(runs_df[group_by].dropna().unique().tolist())
        color_map = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(groups)}
        fig = go.Figure()
        for grp in groups:
            subset = runs_df[runs_df[group_by] == grp][[param, metric]].dropna()
            agg = subset.groupby(param)[metric]
            if agg.ngroups == 0:
                continue
            labels = [str(v) for v in agg.groups.keys()]
            medians = agg.median().values
            q1 = agg.quantile(0.25).values
            q3 = agg.quantile(0.75).values
            fig.add_trace(go.Bar(
                name=str(grp),
                x=labels,
                y=medians,
                error_y=dict(type="data", symmetric=False,
                             array=q3 - medians, arrayminus=medians - q1),
                marker_color=color_map[grp],
                text=[f"{v:.3f}" for v in medians],
                textposition="outside",
            ))
        fig.update_layout(
            **{**_BASE_LAYOUT, "height": 360},
            barmode="group",
            xaxis_title=param.replace("_", " ").title(),
            yaxis_title=f"Median {metric.title()}",
            legend=dict(title=group_by.replace("_", " ").title()),
        )
        return fig

    # Single-group path (original behaviour)
    grouped = runs_df[[param, metric]].dropna().groupby(param)[metric]
    if grouped.ngroups == 0:
        return _no_data_fig("No data for selected parameter/metric.")

    labels = [str(v) for v in grouped.groups.keys()]
    medians = grouped.median().values
    q1 = grouped.quantile(0.25).values
    q3 = grouped.quantile(0.75).values

    fig = go.Figure(go.Bar(
        x=labels,
        y=medians,
        error_y=dict(
            type="data", symmetric=False,
            array=q3 - medians,
            arrayminus=medians - q1,
        ),
        marker_color=[PALETTE[2] if v >= 0 else PALETTE[3] for v in medians],
        text=[f"{v:.3f}" for v in medians],
        textposition="outside",
    ))
    fig.update_layout(
        **{**_BASE_LAYOUT, "height": 320},
        xaxis_title=param.replace("_", " ").title(),
        yaxis_title=f"Median {metric.title()}",
        showlegend=False,
    )
    return fig


def strategy_scatter(
    runs_df: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    color_by: str = "signal_name",
) -> go.Figure:
    """
    Scatter plot of any two performance metrics, one dot per run.
    Dots coloured by a categorical parameter (default: signal_name).
    Default axes (volatility vs cagr) produce a risk-return / efficient frontier view.
    """
    cols = [x_metric, y_metric, color_by, "run_name"]
    valid = runs_df[[c for c in cols if c in runs_df.columns]].dropna(subset=[x_metric, y_metric])
    if valid.empty:
        return _no_data_fig("No data for selected metrics.")

    categories = valid[color_by].fillna("unknown").unique().tolist() if color_by in valid.columns else ["all"]
    cat_col = valid[color_by].fillna("unknown") if color_by in valid.columns else pd.Series(["all"] * len(valid))
    color_map = {cat: PALETTE[i % len(PALETTE)] for i, cat in enumerate(sorted(categories))}

    fig = go.Figure()
    for cat in sorted(categories):
        mask = cat_col == cat
        subset = valid[mask]
        hover = (
            subset["run_name"].fillna("")
            if "run_name" in subset.columns
            else pd.Series([""] * len(subset))
        )
        fig.add_trace(go.Scatter(
            x=subset[x_metric],
            y=subset[y_metric],
            mode="markers",
            name=str(cat),
            marker=dict(size=9, color=color_map[cat], opacity=0.8,
                        line=dict(width=0.5, color="white")),
            text=hover,
            hovertemplate=(
                f"<b>%{{text}}</b><br>"
                f"{x_metric}: %{{x:.3f}}<br>"
                f"{y_metric}: %{{y:.3f}}<extra>{cat}</extra>"
            ),
        ))
        fig.add_trace(go.Scatter(
            x=[subset[x_metric].mean()],
            y=[subset[y_metric].mean()],
            mode="markers",
            name=f"{cat} (avg)",
            showlegend=False,
            marker=dict(size=18, color=color_map[cat], opacity=1.0,
                        symbol="diamond",
                        line=dict(width=1.5, color="white")),
            hovertemplate=(
                f"<b>{cat} average</b><br>"
                f"{x_metric}: %{{x:.3f}}<br>"
                f"{y_metric}: %{{y:.3f}}<extra></extra>"
            ),
        ))

    _pct = {"cagr", "volatility", "max_drawdown", "excess_cagr", "avg_turnover"}
    xfmt = ".0%" if x_metric in _pct else ".3f"
    yfmt = ".0%" if y_metric in _pct else ".3f"
    fig.update_layout(
        **{**_BASE_LAYOUT, "height": 420},
        xaxis=dict(title=x_metric.replace("_", " ").title(), tickformat=xfmt),
        yaxis=dict(title=y_metric.replace("_", " ").title(), tickformat=yfmt),
        legend=dict(title=color_by.replace("_", " ").title()),
    )
    return fig


# ── Stock deep-dive charts ────────────────────────────────────────────────────

def stock_cumulative_return_chart(
    ticker: str,
    asset_returns: pd.DataFrame,
    weights: pd.DataFrame,
) -> go.Figure:
    """
    Cumulative return of a single stock with holding periods shaded.

    The cumulative return is computed from the first date the stock appears
    in asset_returns. Shaded bands mark intervals when the portfolio held the
    stock (weight > 0).
    """
    if ticker not in asset_returns.columns:
        return _no_data_fig(f"No return data for {ticker}.")

    rets = asset_returns[ticker].dropna()
    if rets.empty:
        return _no_data_fig(f"No return data for {ticker}.")

    cum = (1 + rets).cumprod()

    fig = go.Figure()

    # Shade holding periods
    if ticker in weights.columns:
        held = (weights[ticker].reindex(rets.index).fillna(0) > 0)
        # Find contiguous blocks where held == True
        in_block = False
        block_start = None
        for date, is_held in held.items():
            if is_held and not in_block:
                block_start = date
                in_block = True
            elif not is_held and in_block:
                fig.add_vrect(
                    x0=block_start, x1=date,
                    fillcolor=PALETTE[2], opacity=0.15,
                    layer="below", line_width=0,
                )
                in_block = False
        if in_block:
            fig.add_vrect(
                x0=block_start, x1=rets.index[-1],
                fillcolor=PALETTE[2], opacity=0.15,
                layer="below", line_width=0,
            )

    fig.add_trace(go.Scatter(
        x=cum.index, y=cum.values,
        name=ticker,
        line=dict(color=PALETTE[0], width=2),
        hovertemplate="%{y:.2f}x<extra></extra>",
    ))

    # Invisible trace for legend entry for shaded regions
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(size=10, color=PALETTE[2], symbol="square", opacity=0.4),
        name="Held in portfolio",
    ))

    fig.update_layout(
        **{**_BASE_LAYOUT, "height": 280},
        yaxis_title="Cumulative Return (×)",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    return fig


def stock_signal_chart(
    ticker: str,
    scores: pd.DataFrame,
    weights: pd.DataFrame,
) -> go.Figure:
    """
    Signal score for a single stock over time, with holding periods shaded.
    Points where the stock was held are coloured distinctly.
    """
    if ticker not in scores.columns:
        return _no_data_fig(f"No signal data for {ticker}.")

    s = scores[ticker].dropna()
    if s.empty:
        return _no_data_fig(f"No signal data for {ticker}.")

    held_mask = pd.Series(False, index=s.index)
    if ticker in weights.columns:
        held_mask = weights[ticker].reindex(s.index).fillna(0) > 0

    fig = go.Figure()

    # Baseline line
    fig.add_trace(go.Scatter(
        x=s.index, y=s.values,
        mode="lines",
        name="Signal",
        line=dict(color="#cccccc", width=1.5),
        showlegend=False,
    ))

    # Dots: held (green) vs not-held (grey)
    for is_held, color, label in [(False, "#cccccc", "Not held"), (True, PALETTE[2], "Held")]:
        mask = held_mask == is_held
        sub = s[mask]
        if not sub.empty:
            fig.add_trace(go.Scatter(
                x=sub.index, y=sub.values,
                mode="markers",
                marker=dict(color=color, size=6),
                name=label,
                hovertemplate=f"%{{y:.3f}}<extra>{label}</extra>",
            ))

    fig.update_layout(
        **{**_BASE_LAYOUT, "height": 240},
        yaxis_title="Signal Score",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    return fig


def stock_contribution_chart(
    ticker: str,
    stock_attr: pd.DataFrame,
) -> go.Figure:
    """
    Bar chart of per-period portfolio contribution for a single stock.
    Bars are coloured green (positive) / red (negative).
    """
    if ticker not in stock_attr.columns:
        return _no_data_fig(f"No attribution data for {ticker}.")

    attr = stock_attr[ticker].dropna()
    attr = attr[attr != 0]
    if attr.empty:
        return _no_data_fig(f"No contribution data for {ticker}.")

    colors = [PALETTE[2] if v >= 0 else PALETTE[3] for v in attr.values]

    fig = go.Figure(go.Bar(
        x=attr.index, y=attr.values,
        marker_color=colors,
        hovertemplate="%{x|%Y-%m}: %{y:.3%}<extra></extra>",
        name="Contribution",
    ))
    fig.update_layout(
        **{**_BASE_LAYOUT, "height": 240},
        yaxis=dict(title="Contribution", tickformat=".2%"),
        showlegend=False,
    )
    return fig


def sweep_heatmap(
    runs_df: pd.DataFrame,
    x_param: str,
    y_param: str,
    metric: str,
    agg: str = "median",
) -> go.Figure:
    """
    2D heatmap of a metric across all (x_param, y_param) combinations.
    Cell value = median (or mean) of that metric across all other parameters.
    """
    if x_param not in runs_df.columns or y_param not in runs_df.columns or metric not in runs_df.columns:
        return _no_data_fig("Selected columns not found in data.")
    if x_param == y_param:
        return _no_data_fig("X and Y parameters must be different.")

    pivot = (
        runs_df[[x_param, y_param, metric]].dropna()
        .groupby([y_param, x_param])[metric]
        .agg(agg)
        .unstack(x_param)
    )
    if pivot.empty:
        return _no_data_fig("No data for selected parameters.")

    z = pivot.values
    x_labels = [str(v) for v in pivot.columns.tolist()]
    y_labels = [str(v) for v in pivot.index.tolist()]

    text = [[f"{v:.3f}" if not pd.isna(v) else "" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=x_labels, y=y_labels,
        text=text, texttemplate="%{text}",
        colorscale="RdYlGn", showscale=True,
        hovertemplate=f"{y_param}=%{{y}}, {x_param}=%{{x}}<br>{metric}=%{{z:.3f}}<extra></extra>",
    ))
    fig.update_layout(
        **{**_BASE_LAYOUT, "height": max(280, len(y_labels) * 50 + 80)},
        xaxis_title=x_param.replace("_", " ").title(),
        yaxis_title=y_param.replace("_", " ").title(),
    )
    return fig


def annual_returns_bar(
    port_returns: pd.Series,
    bench_returns: pd.Series | None = None,
) -> go.Figure:
    """
    Grouped bar chart of calendar-year returns.
    Portfolio bars coloured green/red by sign; benchmark bars in grey.
    """
    annual_port = (1 + port_returns).resample("YE").prod() - 1
    years = annual_port.index.year.tolist()

    traces = [go.Bar(
        x=years,
        y=annual_port.values,
        name="Portfolio",
        marker_color=[PALETTE[2] if v >= 0 else PALETTE[3] for v in annual_port.values],
        text=[f"{v:.1%}" for v in annual_port.values],
        textposition="outside",
        hovertemplate="%{x}: %{y:.1%}<extra>Portfolio</extra>",
    )]
    if bench_returns is not None:
        annual_bench = (1 + bench_returns).resample("YE").prod() - 1
        annual_bench = annual_bench.reindex(annual_port.index)
        traces.append(go.Bar(
            x=years,
            y=annual_bench.values,
            name="Benchmark",
            marker_color="#95a5a6",
            text=[f"{v:.1%}" if not pd.isna(v) else "" for v in annual_bench.values],
            textposition="outside",
            hovertemplate="%{x}: %{y:.1%}<extra>Benchmark</extra>",
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        **{**_BASE_LAYOUT, "height": 340},
        barmode="group",
        yaxis=dict(tickformat=".0%"),
        xaxis=dict(tickmode="linear", dtick=1),
        bargap=0.2,
    )
    return fig


def monthly_returns_heatmap(port_returns: pd.Series) -> go.Figure:
    """
    Year × month heatmap of monthly returns.
    Cells annotated with the return %; diverging RdYlGn colour scale capped at ±10%.
    """
    df = port_returns.to_frame("ret")
    df["year"] = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot(index="year", columns="month", values="ret")

    _month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    cols = sorted(pivot.columns.tolist())
    col_labels = [_month_labels[c - 1] for c in cols]

    z = pivot[cols].values
    text = [
        [f"{v:.1%}" if not pd.isna(v) else "" for v in row]
        for row in z
    ]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=col_labels,
        y=pivot.index.tolist(),
        text=text,
        texttemplate="%{text}",
        colorscale="RdYlGn",
        zmid=0,
        zmin=-0.10,
        zmax=0.10,
        showscale=False,
        hoverongaps=False,
        hovertemplate="%{y} %{x}: %{z:.1%}<extra></extra>",
    ))
    fig.update_layout(
        **{**_BASE_LAYOUT, "height": max(200, len(pivot) * 32 + 60)},
        xaxis=dict(side="top"),
        yaxis=dict(autorange="reversed"),
    )
    return fig

"""
Quantamental Research Dashboard.

Run from the project root:
    streamlit run src/dashboard/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml
import pandas as pd
import streamlit as st

from src.io.trading212 import fetch_current_prices, parse_t212_csv
from src.dashboard.loader import (
    fetch_ticker_info,
    get_active_tickers,
    list_runs,
    load_run_data,
    load_run_returns,
)
from src.dashboard.charts import (
    annual_returns_bar,
    attribution_bar_chart,
    wf_scatter,
    avg_corr_chart,
    contrib_to_vol_chart,
    contrib_to_vol_industry_chart,
    contribution_vs_return_chart,
    cumulative_attribution_chart,
    cost_drag_chart,
    drawdown_chart,
    ic_decay_chart,
    ic_timeseries_chart,
    equity_curve,
    monthly_returns_heatmap,
    multi_equity_curves,
    multi_turnover_chart,
    rolling_risk_chart,
    stock_contribution_chart,
    stock_cumulative_return_chart,
    stock_signal_chart,
    strategy_scatter,
    sweep_heatmap,
    sweep_sensitivity_chart,
    turnover_bar_chart,
    weights_area_chart,
    weights_industry_chart,
)

# ── Metric formatting ──────────────────────────────────────────────────────────

_PCT_KEYS = {
    "CAGR", "Volatility", "MaxDrawdown", "ExcessCAGR", "BenchmarkCAGR",
    "TrackingError", "HitRate", "AvgWin", "AvgLoss", "AvgVol", "VolOfVol", "AvgTurnover",
}
_RATIO_KEYS = {"Sharpe", "Sortino", "Calmar", "InformationRatio", "Beta"}


def _fmt(key: str, value) -> str:
    if value is None:
        return "N/A"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if key in _PCT_KEYS:
        return f"{v:.1%}"
    if key in _RATIO_KEYS:
        return f"{v:.2f}"
    return f"{v:.4f}"


# ── Fundamental data formatting ────────────────────────────────────────────────

_FUND_PCT_COLS = {"dividendYield", "returnOnEquity", "revenueGrowth", "profitMargins", "earningsGrowth"}
_FUND_RATIO_COLS = {"trailingPE", "priceToBook", "enterpriseToEbitda", "debtToEquity", "trailingEps", "currentRatio"}


def _fmt_fundamental(col: str, value) -> str:
    import math
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    v = float(value)
    if col == "marketCap":
        if v >= 1e9:
            return f"£{v/1e9:.1f}bn"
        if v >= 1e6:
            return f"£{v/1e6:.0f}m"
        return f"£{v:,.0f}"
    if col in _FUND_PCT_COLS:
        return f"{v:.1%}"
    if col in _FUND_RATIO_COLS:
        return f"{v:.1f}"
    return str(value)


# ── Sidebar ────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run_pipeline(config_path: str) -> None:
    from src.main import run_pipeline, read_text, resolve_config_paths
    resolved = Path(config_path)
    if not resolved.is_absolute():
        resolved = _PROJECT_ROOT / resolved
    cfg_text = read_text(resolved)
    cfg = yaml.safe_load(cfg_text)
    resolve_config_paths(cfg, _PROJECT_ROOT)
    result = run_pipeline(cfg, cfg_text=cfg_text)
    m = result["metrics"]
    st.sidebar.success(
        f"Run complete — CAGR {m.get('CAGR', 0):.1%}  Sharpe {m.get('Sharpe', 0):.2f}"
    )
    list_runs.clear()


def _run_sweep(sweep_config_path: str) -> int:
    from src.experiments.run_sweep import run_sweep
    results = run_sweep(sweep_config_path)
    list_runs.clear()
    return len(results)


def _run_walk_forward(wf_config_path: str) -> tuple[int, int]:
    from src.experiments.walk_forward import run_walk_forward
    result = run_walk_forward(wf_config_path)
    list_runs.clear()
    return len(result["is_results"]), len(result["oos_results"])


def _resolve_path(p: str) -> str:
    path = Path(p)
    return str(_PROJECT_ROOT / path) if not path.is_absolute() else p


def _list_configs(subdir: str = "") -> list[str]:
    """Return relative paths of all YAML files under config/ (or a subdirectory)."""
    search_dir = _PROJECT_ROOT / "config" / subdir if subdir else _PROJECT_ROOT / "config"
    return sorted(
        str(p.relative_to(_PROJECT_ROOT)).replace("\\", "/")
        for p in search_dir.glob("*.yaml")
    )


def _render_sidebar() -> tuple[str, str]:
    st.sidebar.title("Quantamental")
    db_path = st.sidebar.text_input("Database path", value="reports/runs.db")
    db_path_abs = _resolve_path(db_path)

    # ── Deep Dive Run Selector ─────────────────────────────────────────────────
    runs = list_runs(db_path_abs)
    run_dir = ""
    if not runs.empty and "run_dir" in runs.columns:
        dir_map = {
            _run_label(row): row.run_dir
            for row in runs.itertuples()
            if pd.notna(row.run_dir)
        }
        label_options = list(dir_map.keys())
        if label_options:
            prev = st.session_state.get("drill_run_dir", "")
            prev_labels = [l for l, d in dir_map.items() if d == prev]
            default_idx = label_options.index(prev_labels[0]) if prev_labels else 0
            selected_label = st.sidebar.selectbox(
                "Deep dive run", label_options, index=default_idx, key="drill_select"
            )
            run_dir = dir_map[selected_label]
            st.session_state["drill_run_dir"] = run_dir

    st.sidebar.divider()
    with st.sidebar.expander("Run Experiments"):
        st.subheader("Single Run")
        single_configs = _list_configs()
        default_single = next((c for c in single_configs if "uk_smallcap.yaml" in c), single_configs[0] if single_configs else "")
        config_path = st.selectbox("Config file", single_configs, index=single_configs.index(default_single) if default_single in single_configs else 0)
        if st.button("▶  Run", use_container_width=True):
            with st.spinner("Running pipeline..."):
                try:
                    _run_pipeline(config_path)
                except Exception as e:
                    st.error(f"Pipeline failed: {e}")

        st.divider()
        st.subheader("Sweep")
        sweep_configs = _list_configs("experiments")
        default_sweep = next((c for c in sweep_configs if "momentum_sweep.yaml" in c), sweep_configs[0] if sweep_configs else "")
        sweep_config_path = st.selectbox("Sweep config", sweep_configs, index=sweep_configs.index(default_sweep) if default_sweep in sweep_configs else 0)
        if st.button("Run Sweep", use_container_width=True):
            with st.spinner("Running sweep (may take several minutes)..."):
                try:
                    n = _run_sweep(sweep_config_path)
                    st.success(f"Sweep complete — {n} variants run")
                except Exception as e:
                    st.error(f"Sweep failed: {e}")

        st.divider()
        st.subheader("Walk-Forward")
        wf_configs = _list_configs("experiments")
        default_wf = next((c for c in wf_configs if "walk_forward.yaml" in c), wf_configs[0] if wf_configs else "")
        wf_config_path = st.selectbox("Walk-forward config", wf_configs, index=wf_configs.index(default_wf) if default_wf in wf_configs else 0)
        if st.button("Run Walk-Forward", use_container_width=True):
            with st.spinner("Running walk-forward (IS sweep + OOS validation)..."):
                try:
                    n_is, n_oos = _run_walk_forward(wf_config_path)
                    st.success(f"Done — {n_is} IS variants, {n_oos} OOS runs")
                except Exception as e:
                    st.error(f"Walk-forward failed: {e}")

    return db_path_abs, run_dir


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _get_ticker_info(tickers: list[str]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    try:
        return fetch_ticker_info(tuple(tickers))
    except Exception:
        return pd.DataFrame()


def _to_sector_map(ticker_info: pd.DataFrame) -> dict[str, str]:
    if ticker_info.empty or "sector" not in ticker_info.columns:
        return {}
    return ticker_info["sector"].to_dict()


def _group_by_sector(df: pd.DataFrame, ticker_to_sector: dict[str, str]) -> pd.DataFrame:
    """Aggregate DataFrame columns by sector via sum."""
    renamed = df.copy()
    renamed.columns = [ticker_to_sector.get(c, "Unknown") for c in renamed.columns]
    return renamed.T.groupby(level=0).sum().T


# ── Tab renderers ──────────────────────────────────────────────────────────────

def _render_overview(data: dict, run_name: str) -> None:
    m = data["metrics"]

    cols = st.columns(5)
    for col, (label, key) in zip(cols, [
        ("CAGR", "CAGR"), ("Sharpe", "Sharpe"), ("Max Drawdown", "MaxDrawdown"),
        ("Excess CAGR", "ExcessCAGR"), ("Info Ratio", "InformationRatio"),
    ]):
        col.metric(label, _fmt(key, m.get(key)))

    st.divider()
    st.subheader("Equity Curve")
    st.plotly_chart(
        equity_curve(data["returns"], data["benchmark_returns"], run_name),
        use_container_width=True,
    )

    st.divider()
    st.subheader("Calendar Returns")
    col_ann, col_heat = st.columns(2)
    port_r = data["returns"]["portfolio_returns_net"].dropna()
    with col_ann:
        st.caption("Annual returns (portfolio vs benchmark)")
        st.plotly_chart(
            annual_returns_bar(port_r, data["benchmark_returns"]),
            use_container_width=True,
        )
    with col_heat:
        st.caption("Monthly returns heatmap")
        st.plotly_chart(
            monthly_returns_heatmap(port_r),
            use_container_width=True,
        )

    with st.expander("All Metrics"):
        _GROUPS = {
            "Performance": ["CAGR", "Volatility", "Sharpe", "Sortino", "Calmar",
                            "MaxDrawdown", "HitRate", "AvgWin", "AvgLoss"],
            "vs Benchmark": ["BenchmarkCAGR", "ExcessCAGR", "Beta",
                             "TrackingError", "InformationRatio"],
            "Risk & Signal": ["AvgVol", "VolOfVol", "MeanIC", "ICIR", "ICHitRate", "ICTStat"],
        }
        g_cols = st.columns(len(_GROUPS))
        for gc, (group_name, keys) in zip(g_cols, _GROUPS.items()):
            rows = [{"Metric": k, "Value": _fmt(k, m[k])} for k in keys if k in m]
            if rows:
                gc.caption(group_name)
                gc.dataframe(rows, hide_index=True, use_container_width=True)


def _render_risk(data: dict) -> None:
    st.subheader("Drawdown")
    st.plotly_chart(drawdown_chart(data["returns"]), use_container_width=True)

    st.subheader("Rolling Volatility & Sharpe")
    st.plotly_chart(rolling_risk_chart(data["returns"]), use_container_width=True)

    if data["avg_pairwise_corr"] is not None:
        st.subheader("Average Pairwise Correlation of Holdings")
        st.plotly_chart(avg_corr_chart(data["avg_pairwise_corr"]), use_container_width=True)
    else:
        st.info("Average pairwise correlation not available for this run.")

    if data["contrib_to_vol"] is not None:
        ctv = data["contrib_to_vol"]
        ctv_tickers = [c for c in ctv.columns if (ctv[c] > 0).any()]
        t2s = _to_sector_map(_get_ticker_info(ctv_tickers))

        st.subheader("Contribution to Volatility")
        col1, col2 = st.columns(2)
        with col1:
            st.caption("By holding")
            st.plotly_chart(contrib_to_vol_chart(ctv), use_container_width=True)
        with col2:
            st.caption("By sector")
            if t2s:
                st.plotly_chart(contrib_to_vol_industry_chart(ctv, t2s), use_container_width=True)
            else:
                st.info("Sector data not available.")
    else:
        st.info("Contribution to volatility not available for this run.")


def _render_holdings_table(data: dict, ticker_info: pd.DataFrame) -> None:
    holdings = data.get("holdings_stats")

    if holdings is None:
        leaders = data["leaders"]
        fmt = {"score": "{:.3f}"} if "score" in leaders.columns else {}
        st.dataframe(leaders.style.format(fmt), hide_index=True, use_container_width=True)
        return

    display = holdings.copy()

    # Add signal score from last valid row of scores DataFrame
    scores_df = data.get("scores")
    if scores_df is not None and not scores_df.empty:
        valid_rows = scores_df.dropna(how="all")
        if not valid_rows.empty:
            last_scores = valid_rows.iloc[-1]
            display["score"] = last_scores.reindex(display.index)

    # Sort by signal score descending (NaN to bottom)
    if "score" in display.columns:
        display = display.sort_values("score", ascending=False, na_position="last")

    # Enrich with name, sector, and website from ticker_info
    if not ticker_info.empty:
        for col in ["name", "sector", "website"]:
            if col in ticker_info.columns:
                display[col] = ticker_info[col]

    # Identify the Nm return column
    return_nm_col = next((c for c in display.columns if c.startswith("return_") and c != "return_1m"), None)

    col_order = ["name", "sector", "score", return_nm_col, "return_1m", "volatility", "website"]
    display = display[[c for c in col_order if c and c in display.columns]]
    display.index.name = "Ticker"

    rename = {}
    if return_nm_col:
        n = return_nm_col.replace("return_", "").replace("m", "")
        rename[return_nm_col] = f"Return ({n}m)"
    rename["return_1m"] = "Return (1m)"
    rename["volatility"] = "Vol (ann.)"
    rename["score"] = "Signal Score"
    rename["website"] = "Website"

    pct_cols = {rename.get(c, c) for c in [return_nm_col, "return_1m", "volatility"] if c in display.columns}

    col_config = {}
    if "Website" in rename.values() and "website" in display.columns:
        col_config["Website"] = st.column_config.LinkColumn(display_text="Visit site")

    fmt = {c: "{:.1%}" for c in pct_cols}
    if "Signal Score" in rename.values() and "score" in holdings.columns or (scores_df is not None):
        fmt["Signal Score"] = "{:.3f}"

    st.dataframe(
        display.rename(columns=rename).reset_index()
        .style.format(fmt, na_rep="—"),
        hide_index=True,
        use_container_width=True,
        column_config=col_config if col_config else None,
    )


def _render_fundamentals_expander(ticker_info: pd.DataFrame) -> None:
    if ticker_info.empty:
        st.info("Fundamental data not available.")
        return

    _SECTIONS = [
        ("Company", [
            ("name", "Name"), ("sector", "Sector"), ("industry", "Industry"),
            ("marketCap", "Mkt Cap"),
        ]),
        ("Valuation", [
            ("trailingPE", "P/E"), ("priceToBook", "P/B"),
            ("enterpriseToEbitda", "EV/EBITDA"),
        ]),
        ("Profitability", [
            ("trailingEps", "EPS"), ("earningsGrowth", "EPS Growth"),
            ("revenueGrowth", "Rev Growth"), ("profitMargins", "Margin"),
            ("returnOnEquity", "ROE"),
        ]),
        ("Balance Sheet", [
            ("debtToEquity", "Debt/Equity"), ("currentRatio", "Current Ratio"),
            ("dividendYield", "Div Yield"),
        ]),
    ]

    _STRING_COLS = {"name", "sector", "industry"}

    for section_label, col_defs in _SECTIONS:
        present = [(col, label) for col, label in col_defs if col in ticker_info.columns]
        if not present:
            continue
        cols, labels = zip(*present)
        display = ticker_info[list(cols)].copy().reset_index()
        display.columns = ["Ticker"] + list(labels)
        label_map = {label: col for col, label in present}
        fmt = {
            label: (lambda v, c=label_map[label]: _fmt_fundamental(c, v))
            for col, label in present if col not in _STRING_COLS
        }
        st.caption(section_label)
        st.dataframe(
            display.style.format(fmt, na_rep="—"),
            hide_index=True,
            use_container_width=True,
        )


def _holdings_cum_returns(data: dict, current_tickers: list[str]) -> pd.DataFrame:
    """Cumulative returns of current holdings from asset_returns artifact."""
    asset_returns = data.get("asset_returns")
    if asset_returns is None or not current_tickers:
        return pd.DataFrame()
    cols = [t for t in current_tickers if t in asset_returns.columns]
    if not cols:
        return pd.DataFrame()
    return (1 + asset_returns[cols].fillna(0)).cumprod() - 1


def _render_stock_deepdive(data: dict, ticker_info: pd.DataFrame) -> None:
    """Stock-level deep dive section at the bottom of the Portfolio tab."""
    weights = data.get("weights")
    asset_returns = data.get("asset_returns")
    scores = data.get("scores")
    stock_attr = data.get("stock_attribution")

    if weights is None or asset_returns is None:
        st.info("Asset return data not available for this run — please re-run the pipeline.")
        return

    ever_held = sorted(
        [t for t in weights.columns if (weights[t].fillna(0) > 0).any()]
    )
    if not ever_held:
        st.info("No held tickers found in weights.")
        return

    # Build display labels: "TICKER — Company Name" where available
    def _ticker_label(t: str) -> str:
        if not ticker_info.empty and t in ticker_info.index:
            name = ticker_info.loc[t, "name"] if "name" in ticker_info.columns else ""
            if name and str(name) not in ("nan", "None", ""):
                return f"{t} — {name}"
        return t

    label_to_ticker = {_ticker_label(t): t for t in ever_held}
    labels = list(label_to_ticker.keys())

    selected_label = st.selectbox("Select stock", labels, key="stock_drill_select")
    ticker = label_to_ticker[selected_label]

    # ── Summary stats ──────────────────────────────────────────────────────────
    held_series = weights[ticker].fillna(0) if ticker in weights.columns else pd.Series(dtype=float)
    periods_held = int((held_series > 0).sum())
    total_periods = len(held_series)
    pct_time = periods_held / total_periods if total_periods > 0 else 0.0
    avg_weight = float(held_series[held_series > 0].mean()) if periods_held > 0 else 0.0

    # Total return while held (compound)
    total_return_held = float("nan")
    if ticker in asset_returns.columns and ticker in weights.columns:
        lagged_w = weights[ticker].shift(1).reindex(asset_returns.index).fillna(0)
        held_mask = lagged_w > 0
        period_rets = asset_returns[ticker][held_mask]
        if not period_rets.empty:
            total_return_held = float((1 + period_rets.dropna()).prod() - 1)

    # Total contribution to portfolio
    total_contrib = float("nan")
    if stock_attr is not None and ticker in stock_attr.columns:
        total_contrib = float(stock_attr[ticker].sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Periods held", f"{periods_held} ({pct_time:.0%})")
    m2.metric("Avg weight (when held)", f"{avg_weight:.1%}" if avg_weight else "—")
    m3.metric("Return while held", f"{total_return_held:.1%}" if not pd.isna(total_return_held) else "—")
    m4.metric("Total contribution", f"{total_contrib:.2%}" if not pd.isna(total_contrib) else "—")

    # ── Charts ─────────────────────────────────────────────────────────────────
    st.plotly_chart(
        stock_cumulative_return_chart(ticker, asset_returns, weights),
        use_container_width=True,
    )

    col_sig, col_contrib = st.columns(2)
    with col_sig:
        st.caption("Signal score over time")
        if scores is not None:
            st.plotly_chart(stock_signal_chart(ticker, scores, weights), use_container_width=True)
        else:
            st.info("Signal scores not saved for this run.")
    with col_contrib:
        st.caption("Per-period contribution to portfolio")
        if stock_attr is not None:
            st.plotly_chart(stock_contribution_chart(ticker, stock_attr), use_container_width=True)
        else:
            st.info("Attribution data not available for this run.")


def _render_portfolio(data: dict) -> None:
    active = get_active_tickers(data["weights"])
    ticker_info = _get_ticker_info(active)
    t2s = _to_sector_map(ticker_info)

    st.subheader("Portfolio Weights Over Time")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("By holding")
        if active:
            st.plotly_chart(weights_area_chart(data["weights"], active), use_container_width=True)
        else:
            st.info("No non-zero weights found.")
    with col2:
        st.caption("By sector")
        if active and t2s:
            st.plotly_chart(weights_industry_chart(data["weights"], active, t2s), use_container_width=True)
        else:
            st.info("Sector data not available.")

    st.divider()
    st.subheader("Current Holdings")

    holdings = data.get("holdings_stats")
    current_tickers = holdings.index.tolist() if holdings is not None else []
    holdings_info = _get_ticker_info(current_tickers) if current_tickers else ticker_info
    _render_holdings_table(data, holdings_info)

    with st.expander("Fundamentals"):
        _render_fundamentals_expander(holdings_info)

    st.divider()
    col_to, col_cost = st.columns(2)
    with col_to:
        st.subheader("Turnover")
        st.plotly_chart(turnover_bar_chart(data["returns"]), use_container_width=True)
    with col_cost:
        st.subheader("Transaction Cost Drag")
        st.plotly_chart(cost_drag_chart(data["returns"]), use_container_width=True)

    st.divider()
    st.subheader("Stock Deep Dive")
    _render_stock_deepdive(data, ticker_info)


def _stock_returns_while_held(stock_attr: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    """
    For each ticker, compound its per-period standalone return (contribution / lagged weight)
    across all periods it was held, yielding total return while held.
    """
    lagged_w = weights.shift(1)
    results = {}
    for t in stock_attr.columns:
        held_mask = lagged_w[t] > 0 if t in lagged_w.columns else pd.Series(False, index=stock_attr.index)
        period_rets = stock_attr[t][held_mask] / lagged_w[t][held_mask]
        period_rets = period_rets.replace([float("inf"), float("-inf")], float("nan")).dropna()
        if period_rets.empty:
            results[t] = float("nan")
        else:
            results[t] = float((1 + period_rets).prod() - 1)
    return pd.Series(results)


def _render_attribution(data: dict) -> None:
    stock_attr = data.get("stock_attribution")
    if stock_attr is None:
        st.info("Attribution data not available for this run — please re-run the pipeline.")
        return

    held = stock_attr.columns.tolist()
    ticker_info = _get_ticker_info(held)
    t2s = _to_sector_map(ticker_info)

    total_by_stock = stock_attr.sum().sort_values(ascending=False)
    stock_returns = _stock_returns_while_held(stock_attr, data["weights"])

    # ── Contribution vs standalone return ─────────────────────────────────────
    st.subheader("Contribution to Portfolio vs Stock Return While Held")
    st.plotly_chart(
        contribution_vs_return_chart(total_by_stock, stock_returns),
        use_container_width=True,
    )

    # ── Attribution by sector ─────────────────────────────────────────────────
    if t2s:
        st.divider()
        st.subheader("Total Return Attribution by Sector")
        sector_attr = _group_by_sector(stock_attr, t2s)
        total_by_sector = sector_attr.sum().sort_values(ascending=False)
        st.plotly_chart(attribution_bar_chart(total_by_sector), use_container_width=True)

    # ── Leaderboard tables ────────────────────────────────────────────────────
    st.divider()
    st.subheader("Attribution Leaderboard")
    col1, col2 = st.columns(2)

    with col1:
        st.caption("By stock")
        tbl = total_by_stock.rename("Contribution").reset_index()
        tbl.columns = ["Ticker", "Contribution"]
        tbl["Return While Held"] = tbl["Ticker"].map(stock_returns)
        if not ticker_info.empty:
            tbl = tbl.merge(
                ticker_info[["sector"]].reset_index().rename(columns={"ticker": "Ticker"}),
                on="Ticker", how="left",
            )
        st.dataframe(
            tbl.style.format(
                {"Contribution": "{:.2%}", "Return While Held": "{:.2%}"}, na_rep="—"
            ),
            hide_index=True, use_container_width=True,
        )

    with col2:
        st.caption("By sector")
        if t2s:
            tbl_s = total_by_sector.rename("contribution").reset_index()
            tbl_s.columns = ["Sector", "Contribution"]
            st.dataframe(
                tbl_s.style.format({"Contribution": "{:.2%}"}, na_rep="—"),
                hide_index=True, use_container_width=True,
            )
        else:
            st.info("Sector data not available.")

    # ── Cumulative attribution over time ──────────────────────────────────────
    st.divider()
    st.subheader("Cumulative Attribution Over Time")
    col1, col2 = st.columns(2)

    with col1:
        st.caption("Top contributors by stock")
        st.plotly_chart(cumulative_attribution_chart(stock_attr, top_n=8), use_container_width=True)

    with col2:
        st.caption("By sector")
        if t2s:
            st.plotly_chart(
                cumulative_attribution_chart(_group_by_sector(stock_attr, t2s), top_n=20),
                use_container_width=True,
            )
        else:
            st.info("Sector data not available.")


def _run_label(row) -> str:
    parts = [row.rebalance, f"{row.n_positions}pos"]
    lk = getattr(row, "blend_lookbacks", None)
    if lk and str(lk) not in ("None", "nan", ""):
        parts.append(f"lk[{lk}]")
    else:
        lk_m = getattr(row, "lookback_months", None)
        if lk_m and str(lk_m) not in ("None", "nan", ""):
            parts.append(f"lk{int(lk_m)}")
    w = getattr(row, "weighting", None)
    if w and str(w) not in ("None", "nan", "", "equal"):
        parts.append(str(w))
    t = getattr(row, "n_tranches", None)
    if t and str(t) not in ("None", "nan", "", "1"):
        parts.append(f"{int(t)}tr")
    return f"{row.run_name} ({', '.join(parts)})"


def _render_compare(db_path: str) -> None:
    runs = list_runs(db_path)
    if runs.empty:
        st.info("No runs in database.")
        return

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        signal_options = sorted(runs["signal_name"].dropna().unique().tolist())
        selected_signals = st.multiselect("Strategies", signal_options, default=signal_options)
    with col2:
        sort_col = st.selectbox(
            "Sort by",
            ["sharpe", "cagr", "volatility", "sortino", "excess_cagr", "information_ratio", "max_drawdown"],
        )
    with col3:
        period_options = ["All", "IS only", "OOS only"]
        period_filter = st.selectbox("Period", period_options)

    active_signals = selected_signals if selected_signals else signal_options
    filtered = runs[runs["signal_name"].isin(active_signals)]
    if period_filter == "IS only" and "wf_period" in filtered.columns:
        filtered = filtered[filtered["wf_period"] == "is"]
    elif period_filter == "OOS only" and "wf_period" in filtered.columns:
        filtered = filtered[filtered["wf_period"] == "oos"]
    if sort_col in filtered.columns:
        filtered = filtered.sort_values(sort_col, ascending=False).reset_index(drop=True)

    display_cols = [
        "run_name", "created_at",
        "wf_period", "wf_group",
        "signal_name", "lookback_months", "skip_recent_months",
        "n_positions", "rebalance", "weighting", "n_tranches", "blend_lookbacks",
        "cagr", "volatility", "sharpe", "sortino",
        "max_drawdown", "excess_cagr", "information_ratio",
        "avg_turnover", "annual_cost_drag", "mean_ic", "icir", "ic_tstat",
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]
    pct_cols = {"cagr", "volatility", "max_drawdown", "excess_cagr", "avg_turnover", "annual_cost_drag"}
    ratio_cols = {"sharpe", "sortino", "calmar", "information_ratio", "icir"}
    ic_cols = {"mean_ic", "ic_tstat"}
    fmt = {}
    for c in display_cols:
        if c in pct_cols:
            fmt[c] = "{:.1%}"
        elif c in ratio_cols:
            fmt[c] = "{:.2f}"
        elif c in ic_cols:
            fmt[c] = "{:.4f}"

    top_n = st.slider(
        "Show top N runs",
        min_value=5,
        max_value=max(5, len(filtered)),
        value=min(25, len(filtered)),
        step=5,
    )
    filtered_display = filtered.head(top_n)

    selection = st.dataframe(
        filtered_display[display_cols].style.format(fmt, na_rep="—"),
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
    )
    selected_rows = selection.selection.rows if selection and selection.selection else []
    if selected_rows and "run_dir" in filtered_display.columns:
        chosen_dir = filtered_display.iloc[selected_rows[0]]["run_dir"]
        if pd.notna(chosen_dir):
            st.session_state["drill_run_dir"] = chosen_dir
    current_drill = st.session_state.get("drill_run_dir", "")
    if current_drill:
        drill_label = next(
            (_run_label(r) for r in filtered_display.itertuples()
             if pd.notna(getattr(r, "run_dir", None)) and r.run_dir == current_drill),
            Path(current_drill).name,
        )
        st.caption(f"Deep dive loaded: **{drill_label}** — switch to Overview / Risk / Portfolio tabs to explore.")

    # ── Strategy Scatter ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("Strategy Scatter")
    _all_metrics = [c for c in ["cagr", "volatility", "sharpe", "sortino", "max_drawdown",
                                 "excess_cagr", "avg_turnover", "annual_cost_drag", "information_ratio", "mean_ic", "ic_tstat"]
                    if c in filtered.columns]
    _cat_cols = [c for c in ["signal_name", "rebalance", "weighting", "n_tranches"]
                 if c in filtered.columns and filtered[c].nunique() > 1]
    if _all_metrics:
        sc1, sc2, sc3 = st.columns(3)
        x_metric = sc1.selectbox(
            "X axis", _all_metrics,
            index=_all_metrics.index("volatility") if "volatility" in _all_metrics else 0,
            key="sc_x",
        )
        y_metric = sc2.selectbox(
            "Y axis", _all_metrics,
            index=_all_metrics.index("cagr") if "cagr" in _all_metrics else 0,
            key="sc_y",
        )
        color_by = sc3.selectbox("Colour by", _cat_cols, key="sc_color") if _cat_cols else "signal_name"
        st.plotly_chart(strategy_scatter(filtered, x_metric, y_metric, color_by), use_container_width=True)

    # ── Sweep Analysis ─────────────────────────────────────────────────────────
    _sweep_params = [c for c in ["signal_name", "lookback_months", "n_positions", "rebalance",
                                  "weighting", "n_tranches", "blend_lookbacks"]
                     if c in filtered.columns and filtered[c].nunique() > 1]
    _sweep_metrics = [c for c in ["sharpe", "cagr", "volatility", "sortino", "max_drawdown",
                                   "excess_cagr", "information_ratio", "avg_turnover", "annual_cost_drag", "mean_ic", "icir", "ic_tstat"]
                      if c in filtered.columns]

    if len(filtered) >= 2 and _sweep_params and _sweep_metrics:
        st.divider()
        st.subheader("Sweep Analysis")

        sa_left, sa_right = st.columns(2)

        with sa_left:
            st.caption("Parameter Sensitivity")
            sens_param = st.selectbox("Parameter", _sweep_params, key="sens_param")
            sens_metric = st.selectbox("Metric", _sweep_metrics, key="sens_metric")
            multi_signal = "signal_name" in filtered.columns and filtered["signal_name"].nunique() > 1
            st.plotly_chart(
                sweep_sensitivity_chart(
                    filtered, sens_param, sens_metric,
                    group_by="signal_name" if multi_signal else None,
                ),
                use_container_width=True,
            )

        with sa_right:
            st.caption("Performance Heatmap")
            x_param = st.selectbox("X axis", _sweep_params, key="hm_x")
            y_param = st.selectbox(
                "Y axis", _sweep_params,
                index=min(1, len(_sweep_params) - 1),
                key="hm_y",
            )
            hm_metric = st.selectbox("Metric", _sweep_metrics, key="hm_metric")
            st.plotly_chart(
                sweep_heatmap(filtered, x_param, y_param, hm_metric),
                use_container_width=True,
            )

    # ── Walk-forward diagnostic ────────────────────────────────────────────────
    if "wf_period" in runs.columns and runs["wf_period"].notna().any():
        st.divider()
        st.subheader("Walk-Forward Analysis")
        st.caption("Compares in-sample (IS) optimisation results against the held-out out-of-sample (OOS) period. "
                   "A high rank correlation means the sweep is finding genuine signal, not overfit parameters.")

        wf_runs = runs[runs["wf_period"].notna()]
        wf_groups = sorted(wf_runs["wf_group"].dropna().unique().tolist(), reverse=True)
        selected_group = st.selectbox(
            "Walk-forward group",
            wf_groups,
            format_func=lambda g: f"Group {g}",
        )
        group_runs = wf_runs[wf_runs["wf_group"] == selected_group]
        is_df = group_runs[group_runs["wf_period"] == "is"].copy()
        oos_df = group_runs[group_runs["wf_period"] == "oos"].copy()

        key_cols = [c for c in ["lookback_months", "n_positions", "rebalance", "weighting"]
                    if c in is_df.columns and c in oos_df.columns]

        paired = (
            is_df.merge(oos_df, on=key_cols, suffixes=("_is", "_oos"), how="inner")
            if key_cols else pd.DataFrame()
        )

        scatter_metric = st.selectbox(
            "Metric to compare",
            ["sharpe", "cagr", "sortino", "excess_cagr", "information_ratio", "max_drawdown"],
            key="wf_scatter_metric",
        )

        col_scatter, col_table = st.columns(2)
        with col_scatter:
            st.caption(f"IS vs OOS {scatter_metric} — each dot is one parameter config")
            if not paired.empty:
                st.plotly_chart(wf_scatter(paired, metric=scatter_metric), use_container_width=True)
            else:
                st.info("No paired IS/OOS configs found for this group.")

        with col_table:
            st.caption("IS vs OOS — key metrics")
            if not paired.empty:
                _show_metrics = [m for m in ["sharpe", "cagr", "max_drawdown", "excess_cagr"]
                                 if f"{m}_is" in paired.columns]
                comp_rows = []
                for _, r in paired.iterrows():
                    label = " | ".join(f"{c}={r[c]}" for c in key_cols if c in r.index)
                    row: dict = {"Config": label}
                    for mc in _show_metrics:
                        is_v = r.get(f"{mc}_is")
                        oos_v = r.get(f"{mc}_oos")
                        row[f"IS {mc}"] = is_v
                        row[f"OOS {mc}"] = oos_v
                        if pd.notna(is_v) and pd.notna(oos_v):
                            row[f"Δ {mc}"] = oos_v - is_v
                    comp_rows.append(row)
                comp_df = pd.DataFrame(comp_rows)
                _pct = {"cagr", "max_drawdown", "excess_cagr"}
                comp_fmt = {}
                for col in comp_df.columns:
                    if col == "Config":
                        continue
                    mc = col.split(" ", 1)[-1] if " " in col else col
                    comp_fmt[col] = "{:.1%}" if mc in _pct else "{:.2f}"
                st.dataframe(
                    comp_df.style.format(comp_fmt, na_rep="—"),
                    hide_index=True, use_container_width=True,
                )
            else:
                st.info("No paired IS/OOS data available.")

    # ── Visual comparison ──────────────────────────────────────────────────────
    if filtered.empty or "run_dir" not in filtered.columns:
        return

    label_to_dir = {
        _run_label(row): row.run_dir
        for row in filtered.itertuples()
        if pd.notna(row.run_dir)
    }
    all_labels = list(label_to_dir.keys())
    default_labels = all_labels[:5]

    selected_labels = st.multiselect(
        "Select runs to compare",
        options=all_labels,
        default=default_labels,
    )

    if not selected_labels:
        return

    eq_runs: dict = {}
    to_runs: dict = {}
    for label in selected_labels:
        ret_df = load_run_returns(label_to_dir[label])
        if ret_df is None:
            continue
        if "portfolio_returns_net" in ret_df.columns:
            eq_runs[label] = ret_df["portfolio_returns_net"]
        if "turnover" in ret_df.columns:
            to_runs[label] = ret_df["turnover"]

    if eq_runs:
        col_eq, col_to = st.columns(2)
        with col_eq:
            st.subheader("Equity Curves")
            st.plotly_chart(multi_equity_curves(eq_runs), use_container_width=True)
        with col_to:
            st.subheader("Turnover Over Time")
            if to_runs:
                st.plotly_chart(multi_turnover_chart(to_runs), use_container_width=True)
            else:
                st.info("Turnover data not available.")



def _render_signal(data: dict) -> None:
    ic_series = data.get("ic_series")
    ic_decay = data.get("ic_decay")

    if ic_series is None:
        st.info("Signal analysis not available for this run. Re-run the pipeline to generate IC artifacts.")
        return

    metrics = data["metrics"]
    mean_ic   = metrics.get("MeanIC")
    icir      = metrics.get("ICIR")
    hit_rate  = metrics.get("ICHitRate")
    t_stat    = metrics.get("ICTStat")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean IC",   f"{mean_ic:.4f}"  if mean_ic  is not None else "—")
    c2.metric("ICIR",      f"{icir:.3f}"     if icir     is not None else "—",
              help="IC Information Ratio = Mean IC / Std(IC). Higher = more consistent signal.")
    c3.metric("IC Hit Rate", f"{hit_rate:.1%}" if hit_rate is not None else "—",
              help="Fraction of months where IC > 0.")
    c4.metric("IC t-stat", f"{t_stat:.2f}"   if t_stat   is not None else "—",
              help="t-statistic testing Mean IC ≠ 0. |t| > 2 suggests statistically significant predictive power.")

    st.subheader("IC Over Time")
    st.plotly_chart(ic_timeseries_chart(ic_series), use_container_width=True)

    if ic_decay is not None:
        st.subheader("IC Decay")
        st.caption("How far ahead does the signal predict? A fast decay suggests shorter holding periods are appropriate.")
        st.plotly_chart(ic_decay_chart(ic_decay), use_container_width=True)


# ── My Portfolio ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def _cached_current_prices(tickers: tuple[str, ...]) -> pd.Series:
    """Hourly-cached wrapper around fetch_current_prices (tuple for hashability)."""
    return fetch_current_prices(list(tickers))


def _render_my_portfolio(db_path: str) -> None:
    """Actual portfolio tracker: T212 CSV → positions, P&L, rebalancing guidance."""

    # ── Upload ─────────────────────────────────────────────────────────────────
    with st.expander("How to export from Trading212", expanded=False):
        st.markdown(
            "1. Open the Trading212 app (web or mobile)\n"
            "2. Go to **Invest** → **History**\n"
            "3. Tap the menu icon (top-right) → **Export CSV**\n"
            "4. Choose **All time** and download\n"
            "5. Upload the file below"
        )

    uploaded = st.file_uploader("Upload Trading212 export CSV", type="csv", key="t212_upload")
    if uploaded is not None:
        try:
            positions = parse_t212_csv(uploaded)
            st.session_state["t212_positions"] = positions
        except ValueError as e:
            st.error(f"Could not parse CSV: {e}")
            return

    positions: pd.DataFrame | None = st.session_state.get("t212_positions")

    if positions is None or positions.empty:
        st.info("Upload your Trading212 export CSV above to get started.")
        return

    tickers = list(positions.index)

    # ── Live prices ────────────────────────────────────────────────────────────
    prices = _cached_current_prices(tuple(tickers))

    # ── Current Positions table ────────────────────────────────────────────────
    st.subheader("Current Positions")

    ticker_info = _get_ticker_info(tickers)

    pos_rows = []
    for t in tickers:
        shares = positions.loc[t, "shares"]
        avg_cost = positions.loc[t, "avg_cost"]
        price = float(prices.get(t, float("nan")))
        market_value = shares * price if not pd.isna(price) else float("nan")
        invested = positions.loc[t, "total_invested"]
        pnl_gbp = market_value - invested if not pd.isna(market_value) else float("nan")
        pnl_pct = pnl_gbp / invested if (not pd.isna(pnl_gbp) and invested > 0) else float("nan")
        name = (
            ticker_info.loc[t, "name"]
            if not ticker_info.empty and t in ticker_info.index and "name" in ticker_info.columns
            else t
        )
        pos_rows.append({
            "Ticker": t,
            "Name": name,
            "Shares": round(shares, 4),
            "Avg Cost": avg_cost,
            "Current Price": price,
            "Market Value (£)": market_value,
            "P&L (£)": pnl_gbp,
            "P&L (%)": pnl_pct,
        })

    pos_df = pd.DataFrame(pos_rows)
    total_value = pos_df["Market Value (£)"].sum()
    total_pnl = pos_df["P&L (£)"].sum()

    st.dataframe(
        pos_df.style.format({
            "Avg Cost": "{:.2f}",
            "Current Price": "{:.2f}",
            "Market Value (£)": "£{:,.2f}",
            "P&L (£)": "£{:,.2f}",
            "P&L (%)": "{:.1%}",
        }, na_rep="—"),
        hide_index=True,
        use_container_width=True,
    )
    c1, c2 = st.columns(2)
    c1.metric("Total Portfolio Value", f"£{total_value:,.2f}" if not pd.isna(total_value) else "—")
    c2.metric("Total P&L", f"£{total_pnl:,.2f}" if not pd.isna(total_pnl) else "—")

    # ── Rebalancing Guidance ───────────────────────────────────────────────────
    st.divider()
    st.subheader("Rebalancing Guidance")

    runs = list_runs(db_path)
    if runs.empty or "run_dir" not in runs.columns:
        st.info("No model runs in database. Run a strategy first.")
        return

    dir_map = {
        _run_label(row): row.run_dir
        for row in runs.itertuples()
        if pd.notna(getattr(row, "run_dir", None))
    }
    if not dir_map:
        st.info("No runs with saved artifacts found.")
        return

    col_sel, col_thresh = st.columns([3, 1])
    with col_sel:
        selected_label = st.selectbox("Compare against model run", list(dir_map.keys()), key="t212_model_run")
    with col_thresh:
        threshold_pct = st.slider("Rebalancing threshold", 0.5, 5.0, 2.0, step=0.5, key="t212_threshold") / 100

    model_run_dir = dir_map[selected_label]
    try:
        model_data = load_run_data(model_run_dir)
    except FileNotFoundError:
        st.error("Run artifacts not found for selected model. Re-run the pipeline.")
        return

    weights_df = model_data["weights"]
    # Last row with any non-zero weight = most recent rebalance
    nonzero_rows = weights_df[(weights_df.fillna(0) > 0).any(axis=1)]
    if nonzero_rows.empty:
        st.warning("No non-zero weights found in selected model run.")
        return
    last_weights = nonzero_rows.iloc[-1].fillna(0)
    model_weights = {t: float(w) for t, w in last_weights.items() if w > 0}

    # Actual weights
    portfolio_total = float(total_value) if not pd.isna(total_value) and total_value > 0 else None
    if portfolio_total is None:
        st.warning("Could not compute portfolio total — some prices may be missing.")
        return

    actual_weights: dict[str, float] = {}
    for t in tickers:
        price = float(prices.get(t, float("nan")))
        if not pd.isna(price):
            actual_weights[t] = (positions.loc[t, "shares"] * price) / portfolio_total

    all_tickers = sorted(set(model_weights) | set(actual_weights))

    action_rows = []
    for t in all_tickers:
        in_model = t in model_weights
        in_actual = t in actual_weights
        mw = model_weights.get(t, 0.0)
        aw = actual_weights.get(t, 0.0)
        price = float(prices.get(t, float("nan")))
        drift = mw - aw

        if in_model and not in_actual:
            action = "Buy"
            target_value = mw * portfolio_total
            shares_to_trade = target_value / price if not pd.isna(price) and price > 0 else float("nan")
            est_value = target_value
        elif in_actual and not in_model:
            action = "Exit"
            shares_to_trade = -positions.loc[t, "shares"] if t in positions.index else float("nan")
            est_value = aw * portfolio_total
        elif drift > threshold_pct:
            action = "Add"
            delta_value = drift * portfolio_total
            shares_to_trade = delta_value / price if not pd.isna(price) and price > 0 else float("nan")
            est_value = delta_value
        elif drift < -threshold_pct:
            action = "Trim"
            delta_value = abs(drift) * portfolio_total
            shares_to_trade = -delta_value / price if not pd.isna(price) and price > 0 else float("nan")
            est_value = delta_value
        else:
            action = "Hold"
            shares_to_trade = float("nan")
            est_value = float("nan")

        name = (
            ticker_info.loc[t, "name"]
            if not ticker_info.empty and t in ticker_info.index and "name" in ticker_info.columns
            else t
        )
        action_rows.append({
            "Action": action,
            "Ticker": t,
            "Name": name,
            "Actual Wt": aw,
            "Model Wt": mw,
            "Shares to Trade": round(shares_to_trade, 4) if not pd.isna(shares_to_trade) else float("nan"),
            "Est. Value (£)": round(est_value, 2) if not pd.isna(est_value) else float("nan"),
        })

    action_df = pd.DataFrame(action_rows)

    trades_needed = int((action_df["Action"] != "Hold").sum())
    matching = int((action_df["Action"] == "Hold").sum())
    model_n = len(model_weights)
    largest_drift = float(
        (action_df["Actual Wt"] - action_df["Model Wt"]).abs().max()
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Portfolio Value", f"£{portfolio_total:,.0f}")
    m2.metric("Matching model", f"{matching} / {model_n}")
    m3.metric("Largest drift", f"{largest_drift:.1%}")
    m4.metric("Trades needed", str(trades_needed))

    _ACTION_COLORS = {
        "Buy":  "background-color: #d4edda",
        "Add":  "background-color: #d4edda",
        "Exit": "background-color: #f8d7da",
        "Trim": "background-color: #f8d7da",
        "Hold": "",
    }

    def _color_row(row):
        color = _ACTION_COLORS.get(row["Action"], "")
        return [color] * len(row)

    styled = (
        action_df.style
        .apply(_color_row, axis=1)
        .format({
            "Actual Wt": "{:.1%}",
            "Model Wt": "{:.1%}",
            "Est. Value (£)": "£{:,.0f}",
        }, na_rep="—")
    )
    st.dataframe(styled, hide_index=True, use_container_width=True)

    # ── Holdings Overlap ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("Holdings Overlap")

    in_both = sorted(set(model_weights) & set(actual_weights))
    model_only = sorted(set(model_weights) - set(actual_weights))
    actual_only = sorted(set(actual_weights) - set(model_weights))

    ov1, ov2, ov3 = st.columns(3)
    with ov1:
        st.caption(f"In both ({len(in_both)})")
        for t in in_both:
            st.write(t)
    with ov2:
        st.caption(f"Model only — not yet bought ({len(model_only)})")
        for t in model_only:
            st.write(t)
    with ov3:
        st.caption(f"Actual only — not in model ({len(actual_only)})")
        for t in actual_only:
            st.write(t)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Quantamental Research Dashboard", layout="wide")
    st.title("Quantamental Research Dashboard")

    db_path, run_dir = _render_sidebar()

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        ["Strategy Comparison", "Overview", "Risk", "Portfolio", "Attribution", "Signal", "My Portfolio"]
    )

    with tab1:
        _render_compare(db_path)

    with tab7:
        _render_my_portfolio(db_path)

    if not run_dir:
        for tab in (tab2, tab3, tab4, tab5, tab6):
            with tab:
                st.info("Select a run from the sidebar to explore it here.")
        return

    try:
        data = load_run_data(run_dir)
    except FileNotFoundError as e:
        st.error(f"Run artifacts not found: {e}")
        return

    run_name = Path(run_dir).name

    with tab2:
        st.caption(f"Analysing: {run_name}")
        _render_overview(data, run_name)
    with tab3:
        st.caption(f"Analysing: {run_name}")
        _render_risk(data)
    with tab4:
        st.caption(f"Analysing: {run_name}")
        _render_portfolio(data)
    with tab5:
        st.caption(f"Analysing: {run_name}")
        _render_attribution(data)
    with tab6:
        st.caption(f"Analysing: {run_name}")
        _render_signal(data)


main()

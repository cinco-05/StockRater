"""Fundamental equity valuation model V5.

Rebuilt from V4 with a focus on valuation correctness rather than feature count.
The headline output is a per-share economic fair value, a margin-of-safety
adjusted BUY-BELOW price, and an over/undervaluation verdict.

What changed versus V4 (and why it matters for accuracy)
--------------------------------------------------------
1.  FCFF/WACC discounting. V4 discounted OCF-capex (a *levered* cash flow, it
    is already net of interest paid) at a cost of equity and then *added* net
    cash. That double-counts the capital structure and systematically
    overvalues leveraged companies. V5 builds unlevered FCFF
    = EBIT(1-t) + D&A - capex - increase in working capital, discounts at WACC,
    then adds cash and subtracts debt. A levered FCFE cross-check is computed
    separately and reconciled.
2.  Reverse DCF. Solves for the growth rate the current price implies. This is
    the most falsifiable number in the report: it can be compared directly with
    realized 3Y/5Y growth.
3.  Correlation-aware method blending. V4's "method agreement" mostly measured
    shared inputs (two FCF methods, two earnings methods). V5 groups methods
    into families, averages within family, and computes agreement across
    families only.
4.  Forensic scores with published out-of-sample evidence: Piotroski F-Score,
    Altman Z-Score, Beneish M-Score. Reported separately, never blended into a
    hand-tuned composite.
5.  Statement integrity checks (balance-sheet identity, cash-flow articulation,
    sign sanity). A failed identity drops metric reliability instead of silently
    propagating a bad row.
6.  Rate-aware assumptions. Risk-free rate is pulled from the 10Y Treasury;
    cost of equity, cost of debt, and terminal growth all key off it instead of
    hardcoded constants that bake in one rate regime.
7.  Diluted weighted-average share count for per-share earnings math.
8.  Period-average FX for flow items, period-end FX for stock items.
9.  Full stock-based compensation deduction (no unprincipled partial factor);
    buybacks are assessed separately under capital allocation.
10. Gross margin, R&D intensity, DSO / DIO / DPO / cash conversion cycle.
11. JSON cache instead of pickle (pickle-from-disk is a code-execution vector).
12. Threaded peer fetch, bootstrap confidence intervals in the backtester, and
    a --self-test mode that exercises the pure functions with no network.

What changed in V5.1 (and why it matters for accuracy)
------------------------------------------------------
13. Cross-source validation. The highest-weight inputs are no longer trusted on
    one provider's word. Price is checked against Stooq's independent feed, and
    for USD-reporting US filers the core statement lines (revenue, net income,
    OCF, capex, assets, equity, diluted shares) are checked against SEC EDGAR
    companyfacts - the primary source, straight from the XBRL filings. A
    confirmed input earns a small reliability bump; a conflict cuts that
    input's reliability hard and is reported, so a mangled provider row lowers
    confidence instead of silently steering the fair value.
14. Backtest-calibrated curve shapes and DCF bands, not just weights.
    --calibrate-out now emits a full calibration file: category weights (as
    before), a per-category curve exponent (gamma) that steepens scoring
    curves for categories with demonstrated rank-correlation to forward excess
    returns and flattens them toward neutral for categories without it, and a
    DCF band scale that widens or narrows every method's bear/bull range to
    match the realized dispersion of (predicted upside - realized return).
    Apply a validated file with --calibration-json. The same data-sufficiency
    gates apply: no calibration is emitted from a sample too small to mean
    anything.
15. Rate-sensitive sector anchors. The hardcoded sector multiples are treated
    as observations at a 4.2% 10Y baseline and re-derived in yield space for
    the live risk-free rate (a 26x P/E anchor is an earnings-yield spread
    claim, not a constant). Optionally, each anchor also blends against the
    live trailing P/E of the matching SPDR sector ETF, cached daily, so the
    anchor drifts with the market instead of with the file's edit date.
16. Peer matching that degrades instead of discarding. A user-supplied peer in
    a different sector is now included at half weight with a note (you named
    it; the model shouldn't pretend it doesn't exist) instead of being dropped,
    and within-sector matching is normalization-tolerant. Medians are
    weight-aware, so peer_medians stops coming back empty when it shouldn't.

Install:
    pip install yfinance pandas numpy

Run:
    python stock_analyzer_v5.py --ticker AAPL
    python stock_analyzer_v5.py --ticker AAPL --peers MSFT,GOOGL --detail detailed
    python stock_analyzer_v5.py --ticker AAPL --json
    python stock_analyzer_v5.py --ticker AAPL --mos 0.30      # demand a 30% discount
    python stock_analyzer_v5.py --ticker AAPL --no-secondary  # skip Stooq/EDGAR checks
    python stock_analyzer_v5.py --self-test                   # no network required
    python stock_analyzer_v5.py --backtest-csv history.csv --calibrate-out calib.json
    python stock_analyzer_v5.py --ticker AAPL --calibration-json calib.json

Set SAV5_EDGAR_UA to "your-app-name your@email" to identify yourself to the
SEC's API per their fair-access policy (a generic default is used otherwise).

Educational quantitative screen. Not investment advice.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ModuleNotFoundError:  # keep pure functions and --self-test usable
    yf = None  # type: ignore[assignment]


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.\-^=]{0,19}$")
FLOW_PERIODS = 4
REPORT_DETAILS = ("compact", "standard", "detailed")

# Valuation regime defaults. Every one of these is overridden by live data when
# available; they exist so the model degrades to something sane, not so it
# pretends a 2021 rate environment is permanent.
DEFAULT_RISK_FREE = 0.042
EQUITY_RISK_PREMIUM = 0.048          # Damodaran-style implied ERP, long-run mid
MIN_COST_OF_EQUITY = 0.070
MAX_COST_OF_EQUITY = 0.180
MIN_WACC = 0.055
MAX_WACC = 0.160
TERMINAL_SPREAD_TO_RF = -0.010       # terminal g = rf - 1.0%, floored/capped
MIN_TERMINAL_GROWTH = 0.005
MAX_TERMINAL_GROWTH = 0.030
EXPLICIT_YEARS = 10                  # two-stage: 10y fade, then perpetuity
FADE_YEARS = 5                       # years of high growth before fade begins
STATUTORY_TAX_FALLBACK = 0.23

# Reliability gates
MIN_INPUT_CONFIDENCE = 60.0
MIN_CATEGORY_COVERAGE = 0.40
MAX_FAMILY_RATIO = 3.0               # across-family high/low above this -> no point estimate
MAX_TERMINAL_SHARE = 0.80            # terminal value share of EV above this -> flagged

CACHE_DIR = Path(os.environ.get("SAV5_CACHE", Path(tempfile.gettempdir()) / "stock_analyzer_v5"))
CACHE_TTL_SECONDS = int(os.environ.get("SAV5_CACHE_TTL", 6 * 60 * 60))
CACHE_ENABLED = True

CATEGORY_MAXIMUMS = {
    "profitability": 20.0,
    "growth": 18.0,
    "financial_health": 18.0,
    "valuation": 20.0,
    "cash_accounting": 16.0,
    "risk_data": 8.0,
}

# Independent raw inputs only. Derived ratios must not manufacture confidence.
CONFIDENCE_INPUT_WEIGHTS = {
    "current_price": 7, "revenue": 7, "cost_of_revenue": 4, "net_income": 6,
    "operating_income": 5, "ocf": 6, "capex": 5, "depreciation": 4,
    "working_capital_change": 3, "cash": 5, "debt": 5, "equity": 4,
    "assets": 3, "diluted_shares": 6, "shares": 3, "current_assets": 3,
    "current_liabilities": 3, "interest_expense": 3, "ebitda": 3, "sbc": 3,
    "inventory": 2, "receivables": 2, "retained_earnings": 2,
    "forward_eps": 4, "analyst_count": 2, "beta": 1,
}

# Sector anchors are starting points for relative valuation, not claims about
# permanent fair multiples. Peer medians and the company's own usable history
# blend against them, and everything is clamped to a band around the anchor.
SECTOR_PROFILES: dict[str, dict[str, float]] = {
    "technology":             {"pe": 26, "forward_pe": 24, "pfcf": 25, "ps": 5.5, "ev_ebitda": 17, "ev_ebit": 21},
    "communication services": {"pe": 22, "forward_pe": 20, "pfcf": 21, "ps": 3.6, "ev_ebitda": 13, "ev_ebit": 17},
    "consumer cyclical":      {"pe": 20, "forward_pe": 18, "pfcf": 19, "ps": 1.9, "ev_ebitda": 12, "ev_ebit": 16},
    "consumer defensive":     {"pe": 21, "forward_pe": 20, "pfcf": 20, "ps": 1.9, "ev_ebitda": 13, "ev_ebit": 17},
    "industrials":            {"pe": 20, "forward_pe": 19, "pfcf": 19, "ps": 1.9, "ev_ebitda": 12, "ev_ebit": 16},
    "healthcare":             {"pe": 22, "forward_pe": 20, "pfcf": 21, "ps": 3.6, "ev_ebitda": 14, "ev_ebit": 18},
    "energy":                 {"pe": 13, "forward_pe": 12, "pfcf": 11, "ps": 1.4, "ev_ebitda": 7,  "ev_ebit": 10},
    "basic materials":        {"pe": 15, "forward_pe": 14, "pfcf": 14, "ps": 1.7, "ev_ebitda": 9,  "ev_ebit": 12},
    "utilities":              {"pe": 18, "forward_pe": 17, "pfcf": 17, "ps": 2.4, "ev_ebitda": 11, "ev_ebit": 15},
    "real estate":            {"pe": 21, "forward_pe": 19, "pfcf": 19, "ps": 4.8, "ev_ebitda": 16, "ev_ebit": 22},
    "financial services":     {"pe": 13, "forward_pe": 12, "pfcf": 13, "ps": 2.4, "ev_ebitda": 10, "ev_ebit": 12},
    "default":                {"pe": 20, "forward_pe": 18, "pfcf": 19, "ps": 2.8, "ev_ebitda": 12, "ev_ebit": 16},
}

# The SECTOR_PROFILES table is an observation made at a specific rate regime,
# not a set of timeless constants. Anchors are re-derived in yield space for
# the live risk-free rate: earnings yield anchor = 1/multiple, and only a
# fraction (ANCHOR_RATE_BETA) of a rate move passes through, because equity
# multiples empirically underreact to bond yields.
ANCHOR_BASE_RF = 0.042               # the 10Y level the table above was set at
ANCHOR_RATE_BETA = 0.60              # pass-through of rate moves into earnings yields
ANCHOR_RATE_CLAMP = (0.72, 1.30)     # rate adjustment can move an anchor at most this much
LIVE_ANCHOR_TTL = int(os.environ.get("SAV5_ANCHOR_TTL", 24 * 60 * 60))

# SPDR sector ETFs used to refresh P/E anchors from live index data.
SECTOR_ETFS = {
    "technology": "XLK", "communication services": "XLC", "consumer cyclical": "XLY",
    "consumer defensive": "XLP", "industrials": "XLI", "healthcare": "XLV",
    "energy": "XLE", "basic materials": "XLB", "utilities": "XLU",
    "real estate": "XLRE", "financial services": "XLF",
}

# Cross-source validation of the highest-weight inputs. Each entry maps a
# metric key to (relative gap that still counts as confirmed, gap beyond which
# it is a conflict). Statement lines compare same-fiscal-year values (EDGAR FY
# versus this model's annual series), so tolerances can be tight; small gaps
# remain legitimate (restatements, segment definitions, share-class rollups).
SECONDARY_ENABLED = True
VALIDATION_TOLERANCES: dict[str, tuple[float, float]] = {
    "current_price": (0.03, 0.10),   # Stooq is typically the prior close
    "revenue": (0.03, 0.12),
    "net_income": (0.05, 0.20),
    "ocf": (0.05, 0.20),
    "capex": (0.08, 0.30),           # EDGAR tag excludes some intangible capex
    "assets": (0.03, 0.12),
    "equity": (0.05, 0.20),
    "diluted_shares": (0.03, 0.12),
}
VALIDATION_CONFIRM_BONUS = 1.04      # multiplicative, capped at 0.99
VALIDATION_CONFLICT_PENALTY = 0.55

# Calibration clamps: how far the backtester is allowed to bend the model.
CURVE_GAMMA_RANGE = (0.60, 1.60)     # per-category scoring-curve exponent
DCF_BAND_SCALE_RANGE = (0.60, 2.00)  # bear/bull half-width multiplier


class StockDataError(RuntimeError):
    """Raised when no defensible analysis can be produced for a symbol."""


# --------------------------------------------------------------------------
# Small numeric helpers
# --------------------------------------------------------------------------

def safe_number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, (str, bytes, bool)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def first_number(*values: Any) -> Optional[float]:
    for value in values:
        number = safe_number(value)
        if number is not None:
            return number
    return None


def divide(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or abs(b) < 1e-12:
        return None
    value = a / b
    return value if math.isfinite(value) else None


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def interpolate(value: float, points: Sequence[tuple[float, float]]) -> float:
    ordered = sorted(points)
    if value <= ordered[0][0]:
        return ordered[0][1]
    if value >= ordered[-1][0]:
        return ordered[-1][1]
    for (x1, y1), (x2, y2) in zip(ordered, ordered[1:]):
        if x1 <= value <= x2:
            return y1 + (value - x1) * (y2 - y1) / (x2 - x1) if x2 != x1 else y1
    return 0.0


def normalize_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def validate_ticker(symbol: str) -> str:
    cleaned = symbol.strip().upper()
    if not cleaned or not TICKER_PATTERN.fullmatch(cleaned):
        raise ValueError("Use a valid 1-20 character ticker symbol.")
    return cleaned


def weighted_median(values: Sequence[tuple[float, float]]) -> float:
    ordered = sorted(values)
    cutoff = sum(weight for _, weight in ordered) / 2
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= cutoff:
            return value
    return ordered[-1][0]


def winsorized_median(values: Sequence[float]) -> Optional[float]:
    """Median after trimming the extreme tail. Robust to one bad provider row."""
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if not clean:
        return None
    if len(clean) < 4:
        return float(np.median(clean))
    low, high = np.percentile(clean, [10, 90])
    trimmed = [min(max(v, low), high) for v in clean]
    return float(np.median(trimmed))


# --------------------------------------------------------------------------
# Cache (JSON, not pickle: a shared temp dir must never be an exec vector)
# --------------------------------------------------------------------------

def _cache_path(namespace: str, key: str) -> Path:
    digest = hashlib.sha1(f"{namespace}:{key}".encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{namespace}_{digest}.json"


def _frame_to_jsonable(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "__frame__": True,
        "index": [str(i) for i in frame.index],
        "columns": [str(c) for c in frame.columns],
        "data": [[None if pd.isna(v) else safe_number(v) for v in row] for row in frame.to_numpy()],
    }


def _frame_from_jsonable(payload: dict[str, Any]) -> pd.DataFrame:
    if not payload.get("index"):
        return pd.DataFrame()
    return pd.DataFrame(payload["data"], index=payload["index"], columns=payload["columns"])


def _encode(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return _frame_to_jsonable(value)
    if isinstance(value, dict):
        return {k: _encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return {"__ts__": value.isoformat()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _decode(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("__frame__"):
            return _frame_from_jsonable(value)
        if "__ts__" in value:
            return pd.Timestamp(value["__ts__"])
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v) for v in value]
    return value


def cache_get(namespace: str, key: str) -> Optional[Any]:
    if not CACHE_ENABLED:
        return None
    path = _cache_path(namespace, key)
    try:
        if not path.exists() or (time.time() - path.stat().st_mtime) > CACHE_TTL_SECONDS:
            return None
        with path.open("r", encoding="utf-8") as handle:
            return _decode(json.load(handle))
    except Exception:
        return None


def cache_put(namespace: str, key: str, value: Any) -> None:
    if not CACHE_ENABLED:
        return
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _cache_path(namespace, key)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(_encode(value), handle)
        tmp.replace(path)
    except Exception:
        pass  # cache is best-effort and must never break analysis


def clear_cache() -> int:
    removed = 0
    if CACHE_DIR.exists():
        for item in CACHE_DIR.glob("*.json"):
            try:
                item.unlink()
                removed += 1
            except OSError:
                pass
    return removed


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class MetricMeta:
    source: str
    period: str
    as_of: Optional[pd.Timestamp]
    reliability: float
    derived_from: tuple[str, ...] = ()


@dataclass
class IntegrityCheck:
    name: str
    passed: bool
    detail: str
    severity: str = "warning"      # "warning" | "critical"


@dataclass
class ValidationCheck:
    """One primary-versus-secondary comparison on a high-weight input."""
    key: str
    primary: float
    secondary: float
    source: str
    gap: float                     # abs relative difference
    verdict: str                   # "confirmed" | "review" | "conflict"
    detail: str = ""


@dataclass
class Calibration:
    """Backtest-derived adjustments. Every field is optional so a plain
    category-weight file (the old --weights-json format) still loads."""
    category_weights: Optional[dict[str, float]] = None
    curve_gamma: Optional[dict[str, float]] = None
    dcf_band_scale: Optional[float] = None
    provenance: str = ""


@dataclass
class ForensicScores:
    piotroski: Optional[int] = None
    piotroski_detail: list[str] = field(default_factory=list)
    altman_z: Optional[float] = None
    altman_zone: str = "unavailable"
    beneish_m: Optional[float] = None
    beneish_flag: str = "unavailable"
    beneish_components: dict[str, float] = field(default_factory=dict)


@dataclass
class MarketAssumptions:
    risk_free: float = DEFAULT_RISK_FREE
    equity_risk_premium: float = EQUITY_RISK_PREMIUM
    beta: float = 1.0
    cost_of_equity: float = 0.09
    cost_of_debt: float = 0.05
    tax_rate: float = STATUTORY_TAX_FALLBACK
    wacc: float = 0.09
    terminal_growth: float = 0.025
    equity_weight: float = 1.0
    debt_weight: float = 0.0
    source: str = "default"


@dataclass
class StockData:
    symbol: str
    company_name: str
    sector: str
    industry: str
    quote_type: str
    currency: str
    financial_currency: str
    currency_compatible: bool
    retrieved_at: datetime
    price_timestamp: Optional[datetime]
    metrics: dict[str, Optional[float]] = field(default_factory=dict)
    meta: dict[str, MetricMeta] = field(default_factory=dict)
    annual: dict[str, pd.Series] = field(default_factory=dict)
    price_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: list[str] = field(default_factory=list)
    integrity: list[IntegrityCheck] = field(default_factory=list)
    validation: list[ValidationCheck] = field(default_factory=list)
    forensics: ForensicScores = field(default_factory=ForensicScores)
    special_model: Optional[str] = None
    peers_used: list[str] = field(default_factory=list)
    peer_medians: dict[str, float] = field(default_factory=dict)


@dataclass
class CategoryResult:
    points: float
    maximum: float
    coverage: float
    details: list[str] = field(default_factory=list)


@dataclass
class ValuationMethod:
    name: str
    family: str                    # "cash flow" | "earnings" | "asset" | "market"
    value: float
    bear: float
    bull: float
    base_weight: float
    reliability: float
    effective_weight: float = 0.0
    status: str = "in range"
    note: str = ""


@dataclass
class DCFDetail:
    starting_fcff: Optional[float] = None
    fcfe_crosscheck: Optional[float] = None
    reconciliation_gap: Optional[float] = None
    growth: Optional[float] = None
    wacc: Optional[float] = None
    terminal_growth: Optional[float] = None
    explicit_pv: Optional[float] = None
    terminal_pv: Optional[float] = None
    terminal_share: Optional[float] = None
    enterprise_value: Optional[float] = None
    net_debt: Optional[float] = None
    equity_value: Optional[float] = None
    per_share: Optional[float] = None
    implied_growth: Optional[float] = None        # reverse DCF
    implied_vs_actual: Optional[float] = None     # implied minus realized 3Y


@dataclass
class FairValueResult:
    low: Optional[float] = None
    base: Optional[float] = None
    high: Optional[float] = None
    methods: list[ValuationMethod] = field(default_factory=list)
    family_values: dict[str, float] = field(default_factory=dict)
    analyst_reference: Optional[float] = None
    dollar_gap: Optional[float] = None
    upside_downside: Optional[float] = None
    discount_premium: Optional[float] = None
    status: str = "UNKNOWN"
    family_agreement: Optional[float] = None
    family_ratio: Optional[float] = None
    dispersion: Optional[float] = None
    confidence: float = 0.0
    margin_of_safety: Optional[float] = None
    buy_below: Optional[float] = None
    strong_buy_below: Optional[float] = None
    action: str = "INCONCLUSIVE"
    decision_basis: str = ""
    dcf: DCFDetail = field(default_factory=DCFDetail)
    assumptions_used: MarketAssumptions = field(default_factory=MarketAssumptions)
    owner_earnings: Optional[float] = None
    normalized_sbc: Optional[float] = None
    diagnostics: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    categories: dict[str, CategoryResult]
    overall: float
    score_low: float
    score_high: float
    business_quality: float
    confidence: float
    data_coverage: float
    data_freshness: float
    model_confidence: float
    model_fit: float
    integrity_score: float
    fair_value: FairValueResult
    reliable: bool
    strengths: list[str]
    concerns: list[str]
    conclusion: str


# --------------------------------------------------------------------------
# Statement series extraction
# --------------------------------------------------------------------------

def safe_frame(ticker: Any, attribute: str, warnings: list[str]) -> pd.DataFrame:
    try:
        value = getattr(ticker, attribute)
        return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame()
    except Exception as exc:
        warnings.append(f"Could not retrieve {attribute}: {type(exc).__name__}.")
        return pd.DataFrame()


def dated_series(frame: pd.DataFrame, aliases: Iterable[str]) -> pd.Series:
    """Numeric values indexed by fiscal period end, newest first."""
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    lookup = {normalize_label(index): index for index in frame.index}
    matched = None
    for alias in aliases:
        candidate = lookup.get(normalize_label(alias))
        if candidate is not None:
            matched = candidate
            break
    if matched is None:
        return pd.Series(dtype=float)
    row = frame.loc[matched]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    values: dict[pd.Timestamp, float] = {}
    for column, raw in row.items():
        value = safe_number(raw)
        if value is None:
            continue
        try:
            date = pd.Timestamp(column)
            if date.tz is not None:
                date = date.tz_localize(None)
        except Exception:
            continue
        values[date] = value
    if not values:
        return pd.Series(dtype=float)
    return pd.Series(values, dtype=float).sort_index(ascending=False)


def latest(series: pd.Series) -> tuple[Optional[float], Optional[pd.Timestamp]]:
    clean = series.dropna()
    if clean.empty:
        return None, None
    return safe_number(clean.iloc[0]), pd.Timestamp(clean.index[0])


def ttm(series: pd.Series) -> tuple[Optional[float], Optional[pd.Timestamp]]:
    clean = series.dropna().sort_index(ascending=False)
    if len(clean) < FLOW_PERIODS:
        return None, None
    window = clean.iloc[:FLOW_PERIODS]
    return safe_number(window.sum()), pd.Timestamp(window.index[0])


def sign_aware_change(new: Optional[float], old: Optional[float]) -> tuple[Optional[float], Optional[str]]:
    """Percentage change that refuses to produce nonsense off a negative base."""
    if new is None or old is None:
        return None, None
    if old > 0:
        return divide(new - old, abs(old)), "ordinary"
    if old <= 0 < new:
        return None, "turnaround"
    if old < 0 and new <= 0:
        return divide(new - old, abs(old)), "loss_change"
    return None, "zero_base"


def cagr(series: pd.Series, years: int) -> Optional[float]:
    clean = series.dropna().sort_index(ascending=False)
    if len(clean) < years + 1:
        return None
    newest, oldest = safe_number(clean.iloc[0]), safe_number(clean.iloc[years])
    if newest is None or oldest is None or newest <= 0 or oldest <= 0:
        return None
    return (newest / oldest) ** (1.0 / years) - 1.0


def consistency(series: pd.Series, positive_values: bool = False) -> Optional[float]:
    clean = series.dropna().sort_index()
    if len(clean) < 3:
        return None
    if positive_values:
        return float((clean > 0).mean())
    changes = clean.diff().dropna()
    return float((changes > 0).mean()) if not changes.empty else None


def coefficient_of_variation(series: pd.Series) -> Optional[float]:
    clean = series.dropna()
    if len(clean) < 3 or abs(clean.mean()) < 1e-12:
        return None
    return safe_number(clean.std(ddof=1) / abs(clean.mean()))


def put_metric(
    data: StockData,
    key: str,
    value: Optional[float],
    source: str,
    period: str,
    as_of: Optional[pd.Timestamp],
    reliability: float,
    derived_from: tuple[str, ...] = (),
) -> None:
    data.metrics[key] = safe_number(value)
    data.meta[key] = MetricMeta(source, period, as_of, clamp(reliability), derived_from)


def info_number(info: Mapping[str, Any], *keys: str) -> Optional[float]:
    return first_number(*(info.get(key) for key in keys))


def fast_value(fast_info: Any, key: str) -> Any:
    try:
        return fast_info.get(key) if hasattr(fast_info, "get") else getattr(fast_info, key, None)
    except Exception:
        return None


def analysis_value(frame: pd.DataFrame, row: str, column: str) -> Optional[float]:
    try:
        return safe_number(frame.loc[row, column])
    except Exception:
        return None


def detect_special_model(quote_type: str, name: str, sector: str, industry: str) -> Optional[str]:
    text = " ".join((name, sector, industry)).lower()
    if quote_type.upper() in {"ETF", "MUTUALFUND", "INDEX", "CURRENCY", "CRYPTOCURRENCY"}:
        return "fund or non-operating instrument"
    if "closed-end" in text or "closed end" in text:
        return "closed-end fund"
    if "bank" in text or "thrift" in text:
        return "bank"
    if "insurance" in text or "reinsur" in text:
        return "insurer"
    if "reit" in text or "real estate investment trust" in text:
        return "REIT"
    if "biotechnology" in text or "biotech" in text:
        return "clinical-stage biotechnology"
    if "shell compan" in text or "blank check" in text:
        return "shell/SPAC"
    return None


# --------------------------------------------------------------------------
# Market environment: risk-free rate and FX
# --------------------------------------------------------------------------

def fetch_risk_free_rate(warnings: list[str]) -> tuple[float, str]:
    """10-year Treasury yield. Falls back to a documented constant."""
    if yf is None:
        return DEFAULT_RISK_FREE, "default constant (yfinance unavailable)"
    cached = cache_get("rate", "TNX")
    if cached is not None:
        return float(cached["rate"]), str(cached["source"])
    for symbol, scale in (("^TNX", 0.01), ("^TYX", 0.01)):
        try:
            history = yf.Ticker(symbol).history(period="1mo", interval="1d", auto_adjust=False)
            if "Close" not in history:
                continue
            close = pd.to_numeric(history["Close"], errors="coerce").dropna()
            if close.empty:
                continue
            rate = float(close.iloc[-1]) * scale
            if 0.0 < rate < 0.20:
                source = f"{symbol} last close"
                cache_put("rate", "TNX", {"rate": rate, "source": source})
                return rate, source
        except Exception:
            continue
    warnings.append("Could not retrieve a live risk-free rate; the default constant was used.")
    return DEFAULT_RISK_FREE, "default constant"


def fetch_fx_series(source: str, destination: str, warnings: list[str]) -> tuple[Optional[float], Optional[pd.Series]]:
    """Return (spot rate, daily close series) in destination units per source unit."""
    source, destination = source.upper(), destination.upper()
    if source == destination:
        return 1.0, None
    if yf is None:
        return None, None
    cached = cache_get("fx", f"{source}->{destination}")
    if cached is not None and cached.get("spot"):
        series = None
        if cached.get("series"):
            try:
                series = pd.Series(
                    {pd.Timestamp(k): float(v) for k, v in cached["series"].items()}
                ).sort_index()
            except Exception:
                series = None
        return float(cached["spot"]), series
    for pair, inverse in ((f"{source}{destination}=X", False), (f"{destination}{source}=X", True)):
        try:
            history = yf.Ticker(pair).history(period="6y", interval="1d", auto_adjust=False)
            if "Close" not in history:
                continue
            close = pd.to_numeric(history["Close"], errors="coerce").dropna()
            close = close[close > 0]
            if close.empty:
                continue
            if inverse:
                close = 1.0 / close
            try:
                index = pd.DatetimeIndex(close.index)
                close.index = index.tz_convert(None) if index.tz is not None else index.tz_localize(None)
            except (TypeError, ValueError):
                pass
            spot = float(close.iloc[-1])
            cache_put("fx", f"{source}->{destination}", {
                "spot": spot,
                "series": {k.isoformat(): float(v) for k, v in close.tail(1600).items()},
            })
            return spot, close
        except Exception:
            continue
    warnings.append(
        f"Could not convert {source} statements into the {destination} quote currency; "
        "per-share fair value was disabled."
    )
    return None, None


def average_fx_for_period(fx_series: Optional[pd.Series], period_end: pd.Timestamp, spot: float) -> float:
    """Average rate over the 12 months ending at period_end (correct for flows)."""
    if fx_series is None or fx_series.empty:
        return spot
    try:
        start = period_end - pd.Timedelta(days=365)
        window = fx_series.loc[(fx_series.index > start) & (fx_series.index <= period_end)]
        if len(window) >= 30:
            return float(window.mean())
    except Exception:
        pass
    return spot


def spot_fx_for_period(fx_series: Optional[pd.Series], period_end: pd.Timestamp, spot: float) -> float:
    """Rate at period_end (correct for balance-sheet stock items)."""
    if fx_series is None or fx_series.empty:
        return spot
    try:
        eligible = fx_series.loc[:period_end]
        if not eligible.empty:
            return float(eligible.iloc[-1])
    except Exception:
        pass
    return spot


# --------------------------------------------------------------------------
# Provider download
# --------------------------------------------------------------------------

def _fast_info_to_dict(fast_info: Any) -> dict[str, Any]:
    keys = ("lastPrice", "currency", "marketCap", "shares", "previousClose")
    result: dict[str, Any] = {}
    for key in keys:
        value = fast_value(fast_info, key)
        number = safe_number(value)
        if number is not None:
            result[key] = number
        elif isinstance(value, str):
            result[key] = value
    return result


def download_raw_payload(symbol: str, warnings: list[str]) -> dict[str, Any]:
    cached = cache_get("raw", symbol)
    if cached is not None:
        warnings.extend(cached.get("warnings", []))
        return cached

    ticker = yf.Ticker(symbol)
    local: list[str] = []
    try:
        info = ticker.get_info()
        info = info if isinstance(info, dict) else {}
    except Exception as exc:
        info = {}
        local.append(f"Could not retrieve the company summary: {type(exc).__name__}.")
    try:
        fast_info = _fast_info_to_dict(ticker.fast_info)
    except Exception:
        fast_info = {}
    try:
        prices = ticker.history(period="6y", interval="1d", auto_adjust=False, actions=False)
    except Exception as exc:
        prices = pd.DataFrame()
        local.append(f"Could not retrieve price history: {type(exc).__name__}.")

    payload = {
        "info": {k: v for k, v in info.items() if isinstance(v, (str, int, float, bool)) or v is None},
        "fast_info": fast_info,
        "prices": prices[["Close"]] if "Close" in getattr(prices, "columns", []) else pd.DataFrame(),
        "income_a": safe_frame(ticker, "income_stmt", local),
        "balance_a": safe_frame(ticker, "balance_sheet", local),
        "cash_a": safe_frame(ticker, "cashflow", local),
        "income_q": safe_frame(ticker, "quarterly_income_stmt", local),
        "balance_q": safe_frame(ticker, "quarterly_balance_sheet", local),
        "cash_q": safe_frame(ticker, "quarterly_cashflow", local),
        "earnings_estimate": safe_frame(ticker, "earnings_estimate", local),
        "revenue_estimate": safe_frame(ticker, "revenue_estimate", local),
        "earnings_history": safe_frame(ticker, "earnings_history", local),
        "eps_revisions": safe_frame(ticker, "eps_revisions", local),
        "warnings": local,
    }
    cache_put("raw", symbol, payload)
    warnings.extend(local)
    return payload


# --------------------------------------------------------------------------
# Secondary sources: Stooq price check and SEC EDGAR statement check
# --------------------------------------------------------------------------
# The single biggest failure mode of a one-provider model is a silently wrong
# input. These functions add an independent read on the handful of inputs
# that carry the most confidence weight. They are strictly best-effort: any
# failure leaves the model exactly as it was, minus the bonus a confirmation
# would have earned.

_EDGAR_UA = os.environ.get("SAV5_EDGAR_UA", "stock-analyzer-v5 research (set SAV5_EDGAR_UA)")

# us-gaap XBRL tags checked in order for each metric; first match wins.
EDGAR_TAGS: dict[str, tuple[str, ...]] = {
    "revenue": ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"),
    "net_income": ("NetIncomeLoss",),
    "ocf": ("NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
    "capex": ("PaymentsToAcquirePropertyPlantAndEquipment",
              "PaymentsToAcquireProductiveAssets"),
    "assets": ("Assets",),
    "equity": ("StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    "diluted_shares": ("WeightedAverageNumberOfDilutedSharesOutstanding",),
}


def _http_get(url: str, timeout: float = 10.0) -> Optional[bytes]:
    try:
        request = urllib.request.Request(url, headers={"User-Agent": _EDGAR_UA,
                                                       "Accept-Encoding": "identity"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError):
        return None


def fetch_stooq_price(symbol: str) -> Optional[float]:
    """Independent daily close from Stooq. US common tickers only; anything
    with exchange suffixes, carets, or equals signs is out of scope."""
    if not re.fullmatch(r"[A-Z]{1,5}", symbol):
        return None
    cached = cache_get("stooq", symbol)
    if cached is not None:
        return safe_number(cached.get("close"))
    raw = _http_get(f"https://stooq.com/q/l/?s={symbol.lower()}.us&f=sd2t2ohlcv&h&e=csv")
    if raw is None:
        return None
    try:
        lines = raw.decode("utf-8", errors="replace").strip().splitlines()
        if len(lines) < 2:
            return None
        header = [h.strip().lower() for h in lines[0].split(",")]
        row = lines[1].split(",")
        close = safe_number(row[header.index("close")]) if "close" in header else None
        if close is not None and close > 0:
            cache_put("stooq", symbol, {"close": close})
            return close
    except (IndexError, ValueError):
        pass
    return None


def _edgar_cik(symbol: str) -> Optional[str]:
    """Zero-padded 10-digit CIK for a ticker, from the SEC's public mapping."""
    mapping = cache_get("edgar", "cik_map")
    if mapping is None:
        raw = _http_get("https://www.sec.gov/files/company_tickers.json", timeout=15.0)
        if raw is None:
            return None
        try:
            payload = json.loads(raw)
            mapping = {str(entry["ticker"]).upper(): int(entry["cik_str"])
                       for entry in payload.values()
                       if isinstance(entry, dict) and "ticker" in entry and "cik_str" in entry}
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
            return None
        cache_put("edgar", "cik_map", mapping)
    cik = mapping.get(symbol.upper())
    return f"{int(cik):010d}" if cik is not None else None


def fetch_edgar_annual_facts(symbol: str) -> dict[str, list[tuple[pd.Timestamp, float]]]:
    """Fiscal-year values per metric from SEC companyfacts (10-K entries, USD
    or share units), newest first. Returns {} on any failure."""
    cached = cache_get("edgar", f"facts_{symbol}")
    if cached is not None:
        try:
            return {k: [(pd.Timestamp(d), float(v)) for d, v in rows]
                    for k, rows in cached.items()}
        except Exception:
            pass
    cik = _edgar_cik(symbol)
    if cik is None:
        return {}
    raw = _http_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", timeout=15.0)
    if raw is None:
        return {}
    try:
        gaap = json.loads(raw).get("facts", {}).get("us-gaap", {})
    except (json.JSONDecodeError, AttributeError):
        return {}
    results: dict[str, list[tuple[pd.Timestamp, float]]] = {}
    for key, tags in EDGAR_TAGS.items():
        wanted_units = ("shares",) if key == "diluted_shares" else ("USD",)
        for tag in tags:
            units = gaap.get(tag, {}).get("units", {}) if isinstance(gaap.get(tag), dict) else {}
            rows: dict[pd.Timestamp, float] = {}
            for unit in wanted_units:
                for item in units.get(unit, []):
                    if not isinstance(item, dict) or item.get("form") != "10-K":
                        continue
                    if item.get("fp") not in (None, "FY"):
                        continue
                    value = safe_number(item.get("val"))
                    end = item.get("end")
                    start = item.get("start")
                    if value is None or end is None:
                        continue
                    try:
                        end_ts = pd.Timestamp(end)
                        # Flow tags must cover a full year; a 10-K also reports
                        # quarterly stubs under the same tag.
                        if start is not None and key not in ("assets", "equity"):
                            span = (end_ts - pd.Timestamp(start)).days
                            if not 300 <= span <= 400:
                                continue
                    except (TypeError, ValueError):
                        continue
                    rows[end_ts] = value  # later filings supersede earlier ones
            if rows:
                results[key] = sorted(rows.items(), key=lambda kv: kv[0], reverse=True)
                break
    cache_put("edgar", f"facts_{symbol}",
              {k: [(d.isoformat(), v) for d, v in rows] for k, rows in results.items()})
    return results


def _apply_validation(data: StockData, key: str, primary: Optional[float],
                      secondary: Optional[float], source: str) -> None:
    if primary is None or secondary is None or key not in VALIDATION_TOLERANCES:
        return
    denom = max(abs(primary), abs(secondary), 1e-9)
    gap = abs(primary - secondary) / denom
    confirm_at, conflict_at = VALIDATION_TOLERANCES[key]
    if gap <= confirm_at:
        verdict, detail = "confirmed", f"within {confirm_at:.0%} of {source}"
        if key in data.meta:
            data.meta[key].reliability = min(0.99, data.meta[key].reliability * VALIDATION_CONFIRM_BONUS)
    elif gap >= conflict_at:
        verdict, detail = "conflict", f"differs {gap:.0%} from {source}; input reliability was cut"
        if key in data.meta:
            data.meta[key].reliability *= VALIDATION_CONFLICT_PENALTY
        data.warnings.append(
            f"Cross-source conflict on {key}: the primary value differs {gap:.0%} from {source}. "
            "Verify the filing before trusting any output that depends on it."
        )
    else:
        verdict, detail = "review", f"differs {gap:.0%} from {source}; plausibly definitional"
    data.validation.append(ValidationCheck(key, float(primary), float(secondary),
                                           source, float(gap), verdict, detail))


def cross_validate_secondary(data: StockData) -> None:
    """Check the highest-weight inputs against independent sources.

    - Price: Stooq daily close versus the live quote.
    - Statements: SEC EDGAR companyfacts (the filings themselves) versus this
      model's annual series, matched on fiscal year end, for USD US filers.

    Confirmed inputs earn a small reliability bump; conflicts are penalized
    hard and surfaced. FY-level validation deliberately backs the same series
    the TTM figures are built from, so the reliability adjustment lands on the
    metric that actually feeds the valuation."""
    if not SECONDARY_ENABLED:
        return
    price = data.metrics.get("current_price")
    stooq = fetch_stooq_price(data.symbol)
    if price is not None and stooq is not None:
        _apply_validation(data, "current_price", price, stooq, "Stooq")

    if data.financial_currency != "USD" or data.currency != "USD":
        return
    facts = fetch_edgar_annual_facts(data.symbol)
    if not facts:
        return
    for key, rows in facts.items():
        ours = data.annual.get(key, pd.Series(dtype=float)).dropna()
        if key == "capex":
            ours = ours.abs()
        if ours.empty:
            continue
        # Match the newest EDGAR fiscal year to our series on period end.
        for end_ts, value in rows[:2]:
            candidates = [(abs((pd.Timestamp(idx) - end_ts).days), idx) for idx in ours.index]
            if not candidates:
                continue
            days, idx = min(candidates)
            if days <= 15:
                primary = safe_number(ours.loc[idx])
                secondary = abs(value) if key == "capex" else value
                _apply_validation(data, key, primary, secondary, "SEC EDGAR")
                break


# Statement row aliases. Order matters: the first match wins.
LINE_ALIASES: dict[str, tuple[str, ...]] = {
    # income statement
    "revenue": ("Total Revenue", "Operating Revenue"),
    "cost_of_revenue": ("Cost Of Revenue", "Reconciled Cost Of Revenue"),
    "gross_profit": ("Gross Profit",),
    "sga": ("Selling General And Administration", "Selling General And Administrative"),
    "rnd": ("Research And Development",),
    "operating_income": ("Operating Income", "Total Operating Income As Reported"),
    "ebit": ("EBIT", "Operating Income"),
    "ebitda": ("EBITDA", "Normalized EBITDA"),
    "net_income": ("Net Income", "Net Income Common Stockholders"),
    "pretax_income": ("Pretax Income", "Income Before Tax"),
    "tax_expense": ("Tax Provision", "Income Tax Expense"),
    "interest_expense": ("Interest Expense", "Interest Expense Non Operating"),
    "diluted_eps": ("Diluted EPS", "Basic EPS"),
    "diluted_shares": ("Diluted Average Shares", "Basic Average Shares"),
    "restructuring": ("Restructuring And Mergern Acquisition", "Restructuring Charges"),
    # cash flow statement
    "ocf": ("Operating Cash Flow", "Total Cash From Operating Activities"),
    "capex": ("Capital Expenditure", "Capital Expenditures"),
    "depreciation": ("Depreciation Amortization Depletion", "Depreciation And Amortization", "Depreciation"),
    "sbc": ("Stock Based Compensation",),
    "working_capital_change": ("Change In Working Capital",),
    "share_repurchase": ("Repurchase Of Capital Stock",),
    "dividends_paid": ("Cash Dividends Paid", "Common Stock Dividend Paid"),
    "acquisitions": ("Net Business Purchase And Sale", "Purchase Of Business"),
    "net_change_cash": ("Changes In Cash", "Change In Cash"),
    "investing_cash_flow": ("Investing Cash Flow", "Total Cashflows From Investing Activities"),
    "financing_cash_flow": ("Financing Cash Flow", "Total Cash From Financing Activities"),
    # balance sheet
    "assets": ("Total Assets",),
    "liabilities": ("Total Liabilities Net Minority Interest", "Total Liabilities"),
    "equity": ("Stockholders Equity", "Common Stock Equity", "Total Stockholder Equity"),
    "retained_earnings": ("Retained Earnings",),
    "cash": ("Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"),
    "short_term_investments": ("Other Short Term Investments", "Short Term Investments"),
    "debt": ("Total Debt",),
    "long_term_debt": ("Long Term Debt",),
    "current_assets": ("Current Assets", "Total Current Assets"),
    "current_liabilities": ("Current Liabilities", "Total Current Liabilities"),
    "receivables": ("Accounts Receivable", "Receivables"),
    "payables": ("Accounts Payable", "Payables"),
    "inventory": ("Inventory",),
    "net_ppe": ("Net PPE", "Property Plant And Equipment Net"),
    "gross_ppe": ("Gross PPE",),
    "goodwill": ("Goodwill And Other Intangible Assets", "Goodwill"),
    "lease_liabilities": ("Capital Lease Obligations", "Operating Lease Liability"),
    "shares": ("Ordinary Shares Number", "Share Issued"),
}

INCOME_KEYS = {"revenue", "cost_of_revenue", "gross_profit", "sga", "rnd", "operating_income", "ebit",
               "ebitda", "net_income", "pretax_income", "tax_expense", "interest_expense",
               "diluted_eps", "diluted_shares", "restructuring"}
CASH_KEYS = {"ocf", "capex", "depreciation", "sbc", "working_capital_change", "share_repurchase",
             "dividends_paid", "acquisitions", "net_change_cash", "investing_cash_flow",
             "financing_cash_flow"}
BALANCE_KEYS = set(LINE_ALIASES) - INCOME_KEYS - CASH_KEYS
FLOW_KEYS = (INCOME_KEYS | CASH_KEYS) - {"diluted_eps", "diluted_shares"}
NON_CURRENCY_KEYS = {"diluted_eps", "diluted_shares", "shares"}


def fetch_stock_data(symbol: str) -> StockData:
    if yf is None:
        raise StockDataError("yfinance is not installed. Run: pip install yfinance pandas numpy")
    warnings: list[str] = []
    payload = download_raw_payload(symbol, warnings)
    info, fast_info, prices = payload["info"], payload["fast_info"], payload["prices"]

    frames_a = {"income": payload["income_a"], "balance": payload["balance_a"], "cash": payload["cash_a"]}
    frames_q = {"income": payload["income_q"], "balance": payload["balance_q"], "cash": payload["cash_q"]}

    current_price = first_number(
        info.get("currentPrice"), info.get("regularMarketPrice"),
        fast_value(fast_info, "lastPrice"),
        prices["Close"].dropna().iloc[-1] if not prices.empty and "Close" in prices else None,
    )
    if current_price is None and not info.get("quoteType"):
        raise StockDataError(f"No usable market data was found for {symbol}.")

    name = str(info.get("longName") or info.get("shortName") or symbol)
    sector = str(info.get("sector") or "Not available")
    industry = str(info.get("industry") or "Not available")
    quote_type = str(info.get("quoteType") or "Unknown")

    price_timestamp = None
    market_time = safe_number(info.get("regularMarketTime"))
    if market_time is not None:
        try:
            price_timestamp = datetime.fromtimestamp(market_time, tz=timezone.utc)
        except (ValueError, OSError, OverflowError):
            pass

    quote_currency = str(info.get("currency") or fast_value(fast_info, "currency") or "USD").upper()
    financial_currency = str(info.get("financialCurrency") or quote_currency).upper()
    fx_spot, fx_series = fetch_fx_series(financial_currency, quote_currency, warnings)

    data = StockData(
        symbol=symbol, company_name=name, sector=sector, industry=industry,
        quote_type=quote_type, currency=quote_currency, financial_currency=financial_currency,
        currency_compatible=(financial_currency == quote_currency or fx_spot is not None),
        retrieved_at=datetime.now().astimezone(), price_timestamp=price_timestamp,
        price_history=prices, warnings=warnings,
        special_model=detect_special_model(quote_type, name, sector, industry),
    )

    def frame_for(key: str, quarterly: bool) -> pd.DataFrame:
        group = frames_q if quarterly else frames_a
        if key in INCOME_KEYS:
            return group["income"]
        if key in CASH_KEYS:
            return group["cash"]
        return group["balance"]

    def convert(series: pd.Series, key: str) -> pd.Series:
        """Period-average FX for flows, period-end FX for balance-sheet stocks."""
        if key in NON_CURRENCY_KEYS or fx_spot is None or fx_spot == 1.0 or series.empty:
            return series
        converter = average_fx_for_period if key in FLOW_KEYS else spot_fx_for_period
        return pd.Series(
            {date: value * converter(fx_series, pd.Timestamp(date), fx_spot) for date, value in series.items()},
            dtype=float,
        ).sort_index(ascending=False)

    # Annual history (used for trends, CAGRs, forensic scores).
    for key, aliases in LINE_ALIASES.items():
        data.annual[key] = convert(dated_series(frame_for(key, False), aliases), key)

    # Decide the flow basis once, for every flow line, so a TTM numerator can
    # never be divided by a fiscal-year denominator.
    core_quarterly = [
        dated_series(frames_q["income"], LINE_ALIASES["revenue"]),
        dated_series(frames_q["income"], LINE_ALIASES["net_income"]),
        dated_series(frames_q["income"], LINE_ALIASES["operating_income"]),
        dated_series(frames_q["cash"], LINE_ALIASES["ocf"]),
        dated_series(frames_q["cash"], LINE_ALIASES["capex"]),
    ]
    use_ttm = all(len(series.dropna()) >= FLOW_PERIODS for series in core_quarterly)

    summary_fallback = {
        "revenue": info_number(info, "totalRevenue"),
        "net_income": info_number(info, "netIncomeToCommon"),
        "ocf": info_number(info, "operatingCashflow"),
        "ebitda": info_number(info, "ebitda"),
        "cash": info_number(info, "totalCash"),
        "debt": info_number(info, "totalDebt"),
        "shares": first_number(info.get("sharesOutstanding"), fast_value(fast_info, "shares")),
    }
    if fx_spot is not None and fx_spot != 1.0:
        for key in ("revenue", "net_income", "ocf", "ebitda", "cash", "debt"):
            if summary_fallback[key] is not None:
                summary_fallback[key] *= fx_spot

    for key, aliases in LINE_ALIASES.items():
        quarterly = convert(dated_series(frame_for(key, True), aliases), key)
        if key in FLOW_KEYS or key == "diluted_shares":
            if use_ttm:
                if key == "diluted_shares":
                    # Share counts are levels, not flows: average the four quarters.
                    clean = quarterly.dropna().iloc[:FLOW_PERIODS]
                    value = float(clean.mean()) if len(clean) == FLOW_PERIODS else None
                    date = pd.Timestamp(clean.index[0]) if len(clean) else None
                else:
                    value, date = ttm(quarterly)
                source, period, reliability = "quarterly statements", "TTM", 0.95
                if value is None:
                    value, date = latest(data.annual[key])
                    source, period, reliability = "annual statements", "FY", 0.88
            else:
                value, date = latest(data.annual[key])
                source, period, reliability = "annual statements", "FY", 0.90
            if value is None and key in summary_fallback:
                value, date = summary_fallback[key], None
                source, period, reliability = "provider summary", "approximate", 0.68
        else:
            value, date = latest(quarterly)
            source, period, reliability = "quarterly statements", "latest quarter", 0.95
            if value is None:
                value, date = latest(data.annual[key])
                source, period, reliability = "annual statements", "FY-end", 0.90
            if value is None and key in summary_fallback:
                value, date = summary_fallback[key], None
                source, period, reliability = "provider summary", "approximate", 0.68
        put_metric(data, key, value, source, period, date, reliability)

    # Capex is always an outflow magnitude from here on.
    if data.metrics.get("capex") is not None:
        meta = data.meta["capex"]
        put_metric(data, "capex", abs(data.metrics["capex"]), meta.source, meta.period, meta.as_of, meta.reliability)
    data.annual["capex"] = data.annual.get("capex", pd.Series(dtype=float)).abs()

    # Market and consensus inputs.
    put_metric(data, "current_price", current_price, "market quote", "current",
               pd.Timestamp(price_timestamp).tz_localize(None) if price_timestamp else None, 0.98)
    put_metric(data, "reported_market_cap",
               first_number(info.get("marketCap"), fast_value(fast_info, "marketCap")),
               "provider summary", "current", None, 0.85)
    put_metric(data, "beta", info_number(info, "beta"), "provider summary", "5Y monthly", None, 0.60)
    put_metric(data, "dividend_yield", info_number(info, "dividendYield"), "provider summary", "current", None, 0.80)

    earnings_estimate = payload["earnings_estimate"]
    revenue_estimate = payload["revenue_estimate"]
    forward_eps = first_number(analysis_value(earnings_estimate, "+1y", "avg"), info_number(info, "forwardEps"))
    if forward_eps is not None and fx_spot is not None and financial_currency != quote_currency:
        pass  # forward EPS from the provider is already quoted in the trading currency
    put_metric(data, "forward_eps", forward_eps, "analyst consensus", "+1Y", None, 0.76)
    put_metric(data, "analyst_count",
               first_number(analysis_value(earnings_estimate, "+1y", "numberOfAnalysts"),
                            info_number(info, "numberOfAnalystOpinions")),
               "analyst consensus", "+1Y", None, 0.80)
    put_metric(data, "forward_revenue_growth", analysis_value(revenue_estimate, "+1y", "growth"),
               "analyst consensus", "+1Y", None, 0.72)
    put_metric(data, "forward_eps_growth", analysis_value(earnings_estimate, "+1y", "growth"),
               "analyst consensus", "+1Y", None, 0.72)
    put_metric(data, "analyst_target", info_number(info, "targetMeanPrice", "targetMedianPrice"),
               "analyst consensus", "12-month", None, 0.45)

    earnings_history = payload["earnings_history"]
    surprise = None
    if not earnings_history.empty and "surprisePercent" in earnings_history.columns:
        values = pd.to_numeric(earnings_history["surprisePercent"], errors="coerce").dropna()
        if not values.empty:
            surprise = float(values.tail(4).mean())
    put_metric(data, "earnings_surprise", surprise, "earnings history", "last 4 quarters", None, 0.70)

    eps_revisions = payload["eps_revisions"]
    revision = None
    if not eps_revisions.empty:
        numeric = eps_revisions.apply(pd.to_numeric, errors="coerce")
        up_cols = [c for c in numeric.columns if "up" in normalize_label(c)]
        down_cols = [c for c in numeric.columns if "down" in normalize_label(c)]
        ups = float(numeric[up_cols].sum().sum()) if up_cols else 0.0
        downs = float(numeric[down_cols].sum().sum()) if down_cols else 0.0
        if ups + downs > 0:
            revision = (ups - downs) / (ups + downs)
    put_metric(data, "revision_balance", revision, "estimate revisions", "recent", None, 0.68)

    run_integrity_checks(data)
    cross_validate_secondary(data)   # adjust input reliability before anything derives from it
    derive_metrics(data)
    data.forensics = compute_forensics(data)
    return data


# --------------------------------------------------------------------------
# Statement integrity
# --------------------------------------------------------------------------

def run_integrity_checks(data: StockData) -> None:
    """Verify the statements articulate. A failure degrades reliability rather
    than silently propagating a mangled provider row."""
    m = data.metrics
    checks: list[IntegrityCheck] = []

    assets, liabilities, equity = m.get("assets"), m.get("liabilities"), m.get("equity")
    if None not in (assets, liabilities, equity) and assets:
        gap = abs((liabilities + equity) / assets - 1.0)
        checks.append(IntegrityCheck(
            "Balance-sheet identity", gap <= 0.02,
            f"Liabilities + equity is within {gap:.2%} of total assets.",
            "critical" if gap > 0.05 else "warning",
        ))

    ocf, icf, fcf_fin, net_change = m.get("ocf"), m.get("investing_cash_flow"), m.get("financing_cash_flow"), m.get("net_change_cash")
    if None not in (ocf, icf, fcf_fin, net_change) and abs(net_change) > 1e-6:
        implied = ocf + icf + fcf_fin
        gap = abs(divide(implied - net_change, max(abs(net_change), abs(ocf) * 0.1)) or 0.0)
        checks.append(IntegrityCheck(
            "Cash-flow articulation", gap <= 0.10,
            f"Operating + investing + financing reconciles to the change in cash within {gap:.1%}.",
        ))

    revenue, gross = m.get("revenue"), m.get("gross_profit")
    if None not in (revenue, gross) and revenue:
        checks.append(IntegrityCheck(
            "Gross profit bound", gross <= revenue * 1.001,
            "Gross profit does not exceed revenue.", "critical",
        ))
    if revenue is not None:
        checks.append(IntegrityCheck("Revenue sign", revenue >= 0, "Revenue is non-negative.", "critical"))

    market_cap_calc = None
    if None not in (m.get("current_price"), m.get("shares")):
        market_cap_calc = m["current_price"] * m["shares"]
    reported_cap = m.get("reported_market_cap")
    if market_cap_calc is not None and reported_cap:
        gap = abs(market_cap_calc / reported_cap - 1.0)
        checks.append(IntegrityCheck(
            "Share-count scale", gap <= 0.25,
            f"Price times share count is within {gap:.0%} of the reported market capitalisation.",
            "critical" if gap > 0.35 else "warning",
        ))

    ocf_a = data.annual.get("ocf", pd.Series(dtype=float)).dropna()
    if len(ocf_a) >= 3:
        checks.append(IntegrityCheck(
            "Operating cash-flow history", True,
            f"{len(ocf_a)} fiscal years of operating cash flow are available.",
        ))

    data.integrity = checks
    critical_failures = [c for c in checks if not c.passed and c.severity == "critical"]
    for check in checks:
        if not check.passed:
            data.warnings.append(f"Integrity check failed - {check.name}: {check.detail}")
    if critical_failures:
        # Broadly degrade confidence in every statement-derived number.
        for key, meta in data.meta.items():
            if "statement" in meta.source:
                meta.reliability *= 0.55


def integrity_score(data: StockData) -> float:
    if not data.integrity:
        return 70.0
    weights = {"critical": 2.0, "warning": 1.0}
    total = sum(weights[c.severity] for c in data.integrity)
    earned = sum(weights[c.severity] for c in data.integrity if c.passed)
    return 100.0 * earned / total if total else 70.0


# --------------------------------------------------------------------------
# Derived metrics
# --------------------------------------------------------------------------

def aligned_fcfe_history(data: StockData) -> pd.Series:
    """Levered FCF = OCF - capex, joined on identical fiscal dates."""
    ocf = data.annual.get("ocf", pd.Series(dtype=float)).rename("ocf")
    capex = data.annual.get("capex", pd.Series(dtype=float)).abs().rename("capex")
    joined = pd.concat([ocf, capex], axis=1, join="inner").dropna()
    if joined.empty:
        return pd.Series(dtype=float)
    return (joined["ocf"] - joined["capex"]).sort_index(ascending=False)


def aligned_fcff_history(data: StockData, tax_rate: float) -> pd.Series:
    """Unlevered FCF = EBIT(1-t) + D&A - capex + change-in-working-capital.

    The cash-flow statement reports the working-capital line already signed as a
    cash effect (a build in working capital is negative), so it is added, not
    subtracted."""
    ebit = data.annual.get("ebit", pd.Series(dtype=float))
    if ebit.dropna().empty:
        ebit = data.annual.get("operating_income", pd.Series(dtype=float))
    parts = {
        "ebit": ebit,
        "dep": data.annual.get("depreciation", pd.Series(dtype=float)),
        "capex": data.annual.get("capex", pd.Series(dtype=float)).abs(),
        "dwc": data.annual.get("working_capital_change", pd.Series(dtype=float)),
    }
    joined = pd.concat([s.rename(k) for k, s in parts.items()], axis=1, join="inner").dropna(subset=["ebit", "capex"])
    if joined.empty:
        return pd.Series(dtype=float)
    joined = joined.fillna(0.0)
    fcff = joined["ebit"] * (1 - tax_rate) + joined["dep"] - joined["capex"] + joined["dwc"]
    return fcff.sort_index(ascending=False)


def effective_tax_rate(data: StockData) -> float:
    """Three-year median effective rate, bounded, with a statutory fallback."""
    tax = data.annual.get("tax_expense", pd.Series(dtype=float))
    pretax = data.annual.get("pretax_income", pd.Series(dtype=float))
    joined = pd.concat([tax.rename("t"), pretax.rename("p")], axis=1, join="inner").dropna()
    joined = joined[joined["p"] > 0].iloc[:4]
    if not joined.empty:
        rates = (joined["t"] / joined["p"]).tolist()
        candidate = winsorized_median(rates)
        if candidate is not None and 0.0 <= candidate <= 0.50:
            return float(clamp(candidate, 0.05, 0.35))
    single = divide(data.metrics.get("tax_expense"), data.metrics.get("pretax_income"))
    if single is not None and 0.0 <= single <= 0.50:
        return float(clamp(single, 0.05, 0.35))
    return STATUTORY_TAX_FALLBACK


def derive_metrics(data: StockData) -> None:
    m = data.metrics
    anchor_date = data.meta.get("revenue", MetricMeta("", "", None, 0.0)).as_of

    def derived(key: str, value: Optional[float], inputs: tuple[str, ...], period: str = "derived") -> None:
        available = [data.meta[i].reliability for i in inputs if i in data.meta and m.get(i) is not None]
        reliability = min(available) if len(available) == len(inputs) and available else 0.0
        put_metric(data, key, value, "derived", period, anchor_date, reliability, inputs)

    price = m.get("current_price")
    shares = m.get("shares")
    diluted = m.get("diluted_shares") or shares
    m["effective_shares"] = diluted
    data.meta["effective_shares"] = MetricMeta(
        "diluted weighted average" if m.get("diluted_shares") else "period-end share count",
        "TTM" if m.get("diluted_shares") else "latest", anchor_date,
        0.92 if m.get("diluted_shares") else 0.75,
    )

    market_cap = price * shares if None not in (price, shares) and shares else m.get("reported_market_cap")
    derived("market_cap", market_cap, ("current_price", "shares") if price and shares else ("reported_market_cap",), "current")

    debt, cash = m.get("debt"), m.get("cash")
    net_debt = debt - cash if None not in (debt, cash) else None
    derived("net_debt", net_debt, ("debt", "cash"), "current")
    enterprise_value = market_cap + net_debt if None not in (market_cap, net_debt) else None
    derived("enterprise_value", enterprise_value, ("market_cap", "debt", "cash"), "current")

    # --- margins and returns -------------------------------------------------
    gross_profit = m.get("gross_profit")
    if gross_profit is None and None not in (m.get("revenue"), m.get("cost_of_revenue")):
        gross_profit = m["revenue"] - m["cost_of_revenue"]
        m["gross_profit"] = gross_profit
    derived("gross_margin", divide(gross_profit, m.get("revenue")), ("revenue",), "aligned period")
    derived("operating_margin", divide(m.get("operating_income"), m.get("revenue")), ("operating_income", "revenue"), "aligned period")
    derived("profit_margin", divide(m.get("net_income"), m.get("revenue")), ("net_income", "revenue"), "aligned period")
    derived("rnd_intensity", divide(m.get("rnd"), m.get("revenue")), ("rnd", "revenue"), "aligned period")
    derived("sga_intensity", divide(m.get("sga"), m.get("revenue")), ("sga", "revenue"), "aligned period")

    tax_rate = effective_tax_rate(data)
    m["tax_rate"] = tax_rate
    data.meta["tax_rate"] = MetricMeta("derived", "3Y median effective", anchor_date, 0.85)

    ebit = m.get("ebit") if m.get("ebit") is not None else m.get("operating_income")
    m["ebit"] = ebit
    invested_capital = None
    if None not in (debt, m.get("equity"), cash):
        invested_capital = debt + m["equity"] - cash
    roic = None
    if ebit is not None and invested_capital and invested_capital > 0:
        roic = ebit * (1 - tax_rate) / invested_capital
    derived("invested_capital", invested_capital, ("debt", "equity", "cash"))
    derived("roic", roic, ("operating_income", "debt", "equity", "cash"), "NOPAT / invested capital")

    equity_a = data.annual.get("equity", pd.Series(dtype=float)).dropna()
    avg_equity = float(equity_a.iloc[:2].mean()) if len(equity_a) >= 2 else m.get("equity")
    derived("return_on_equity",
            divide(m.get("net_income"), avg_equity) if avg_equity and avg_equity > 0 else None,
            ("net_income", "equity"), "income / average equity")
    assets_a = data.annual.get("assets", pd.Series(dtype=float)).dropna()
    avg_assets = float(assets_a.iloc[:2].mean()) if len(assets_a) >= 2 else m.get("assets")
    derived("return_on_assets", divide(m.get("net_income"), avg_assets), ("net_income", "assets"), "income / average assets")
    derived("asset_turnover", divide(m.get("revenue"), avg_assets), ("revenue", "assets"))

    # --- cash flow -----------------------------------------------------------
    fcfe = None
    if None not in (m.get("ocf"), m.get("capex")):
        fcfe = m["ocf"] - m["capex"]
    derived("fcfe", fcfe, ("ocf", "capex"), "levered FCF")
    fcff = None
    if ebit is not None and m.get("capex") is not None:
        fcff = ebit * (1 - tax_rate) + (m.get("depreciation") or 0.0) - m["capex"] + (m.get("working_capital_change") or 0.0)
    derived("fcff", fcff, ("operating_income", "capex"), "unlevered FCF")
    derived("fcf_margin", divide(fcfe, m.get("revenue")), ("fcfe", "revenue"))
    derived("ocf_to_net_income", divide(m.get("ocf"), m.get("net_income")), ("ocf", "net_income"))
    derived("capex_to_ocf", divide(m.get("capex"), m.get("ocf")), ("capex", "ocf"))
    derived("capex_to_depreciation", divide(m.get("capex"), m.get("depreciation")), ("capex", "depreciation"))

    # --- balance sheet -------------------------------------------------------
    derived("cash_to_debt", 4.0 if debt == 0 and cash is not None else divide(cash, debt), ("cash", "debt"))
    derived("debt_to_equity", divide(debt, m.get("equity")), ("debt", "equity"))
    derived("net_debt_to_ebitda", divide(net_debt, m.get("ebitda")), ("debt", "cash", "ebitda"))
    derived("current_ratio", divide(m.get("current_assets"), m.get("current_liabilities")), ("current_assets", "current_liabilities"))
    quick = None
    if None not in (m.get("current_assets"), m.get("inventory"), m.get("current_liabilities")):
        quick = divide(m["current_assets"] - m["inventory"], m["current_liabilities"])
    derived("quick_ratio", quick, ("current_assets", "inventory", "current_liabilities"))
    derived("interest_coverage",
            divide(ebit, abs(m["interest_expense"]) if m.get("interest_expense") else None),
            ("operating_income", "interest_expense"))
    derived("goodwill_to_assets", divide(m.get("goodwill"), m.get("assets")), ("goodwill", "assets"))
    derived("lease_to_debt", divide(m.get("lease_liabilities"), debt), ("lease_liabilities", "debt"))

    # --- working-capital efficiency (classic pre-restatement tells) ----------
    revenue, cogs = m.get("revenue"), m.get("cost_of_revenue")
    derived("days_sales_outstanding", divide(m.get("receivables"), divide(revenue, 365.0)), ("receivables", "revenue"))
    derived("days_inventory", divide(m.get("inventory"), divide(cogs, 365.0)), ("inventory", "cost_of_revenue"))
    derived("days_payable", divide(m.get("payables"), divide(cogs, 365.0)), ("payables", "cost_of_revenue"))
    if None not in (m.get("days_sales_outstanding"), m.get("days_inventory"), m.get("days_payable")):
        m["cash_conversion_cycle"] = m["days_sales_outstanding"] + m["days_inventory"] - m["days_payable"]
        data.meta["cash_conversion_cycle"] = MetricMeta("derived", "days", anchor_date, 0.80)

    # --- accounting quality --------------------------------------------------
    accruals = divide(
        (m["net_income"] - m["ocf"]) if None not in (m.get("net_income"), m.get("ocf")) else None,
        avg_assets,
    )
    derived("accrual_ratio", accruals, ("net_income", "ocf", "assets"))
    derived("sbc_to_revenue", divide(m.get("sbc"), m.get("revenue")), ("sbc", "revenue"))
    derived("sbc_to_fcf", divide(m.get("sbc"), fcfe), ("sbc", "fcfe"))
    derived("restructuring_to_revenue",
            divide(abs(m["restructuring"]) if m.get("restructuring") else None, revenue),
            ("restructuring", "revenue"))
    derived("buybacks_to_fcf",
            divide(abs(m["share_repurchase"]) if m.get("share_repurchase") else None, fcfe),
            ("share_repurchase", "fcfe"))
    derived("dividend_payout",
            divide(abs(m["dividends_paid"]) if m.get("dividends_paid") else None, m.get("net_income")),
            ("dividends_paid", "net_income"))

    # --- growth and stability ------------------------------------------------
    fcf_history = aligned_fcfe_history(data)
    data.annual["fcfe"] = fcf_history
    data.annual["fcff"] = aligned_fcff_history(data, tax_rate)
    histories = {
        "revenue": data.annual.get("revenue", pd.Series(dtype=float)),
        "income": data.annual.get("net_income", pd.Series(dtype=float)),
        "fcf": fcf_history,
        "gross_profit": data.annual.get("gross_profit", pd.Series(dtype=float)),
        "shares": data.annual.get("diluted_shares", data.annual.get("shares", pd.Series(dtype=float))),
    }
    for key, series in histories.items():
        clean = series.dropna().sort_index(ascending=False)
        newest = safe_number(clean.iloc[0]) if len(clean) else None
        prior = safe_number(clean.iloc[1]) if len(clean) > 1 else None
        growth, growth_type = sign_aware_change(newest, prior)
        reliability = 0.90 if len(clean) >= 4 else 0.75 if len(clean) >= 2 else 0.0
        as_of = pd.Timestamp(clean.index[0]) if len(clean) else None
        m[f"{key}_growth"] = growth
        data.meta[f"{key}_growth"] = MetricMeta("date-aligned annual history", "latest FY YoY", as_of, reliability)
        m[f"{key}_turnaround"] = 1.0 if growth_type == "turnaround" else (0.0 if growth_type else None)
        for label, value in (
            ("cagr3", cagr(series, 3)),
            ("cagr5", cagr(series, 5)),
            ("consistency", consistency(series, positive_values=key in {"income", "fcf"})),
            ("volatility", coefficient_of_variation(series)),
        ):
            m[f"{key}_{label}"] = value
            data.meta[f"{key}_{label}"] = MetricMeta("date-aligned annual history", label, as_of, reliability)

    # --- per-share and multiples (price-dependent: not confidence inputs) -----
    eps = divide(m.get("net_income"), diluted)
    fcf_per_share = divide(fcfe, diluted)
    derived("eps", eps, ("net_income",), "TTM diluted")
    derived("fcf_per_share", fcf_per_share, ("ocf", "capex"), "TTM diluted")
    derived("book_per_share", divide(m.get("equity"), diluted), ("equity",))
    derived("trailing_pe", divide(price, eps), ("current_price", "eps"))
    derived("forward_pe", divide(price, m.get("forward_eps")), ("current_price", "forward_eps"))
    derived("price_to_fcf", divide(price, fcf_per_share), ("current_price", "fcf_per_share"))
    derived("price_to_sales", divide(market_cap, revenue), ("market_cap", "revenue"))
    derived("price_to_book", divide(market_cap, m.get("equity")), ("market_cap", "equity"))
    derived("ev_to_ebitda", divide(enterprise_value, m.get("ebitda")), ("enterprise_value", "ebitda"))
    derived("ev_to_ebit", divide(enterprise_value, ebit), ("enterprise_value", "operating_income"))
    derived("fcf_yield", divide(fcfe, market_cap), ("fcfe", "market_cap"))
    derived("earnings_yield", divide(ebit, enterprise_value), ("operating_income", "enterprise_value"))

    # --- price-based risk ----------------------------------------------------
    if not data.price_history.empty and "Close" in data.price_history:
        close = pd.to_numeric(data.price_history["Close"], errors="coerce").dropna()
        close = close[close > 0]
        returns = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        put_metric(data, "annual_volatility",
                   float(returns.std() * math.sqrt(252)) if len(returns) >= 60 else None,
                   "price history", "daily", None, 0.90)
        downside = returns[returns < 0]
        put_metric(data, "downside_volatility",
                   float(downside.std() * math.sqrt(252)) if len(downside) >= 30 else None,
                   "price history", "daily", None, 0.90)
        put_metric(data, "max_drawdown",
                   float((close / close.cummax() - 1).min()) if not close.empty else None,
                   "price history", "daily", None, 0.90)
        if len(close) > 30:
            put_metric(data, "price_vs_52w_high",
                       float(close.iloc[-1] / close.tail(252).max() - 1),
                       "price history", "52 week", None, 0.90)


# --------------------------------------------------------------------------
# Forensic scores (published, out-of-sample validated; reported separately)
# --------------------------------------------------------------------------

def _pair(series: pd.Series) -> tuple[Optional[float], Optional[float]]:
    """Newest and prior fiscal-year values."""
    clean = series.dropna().sort_index(ascending=False) if series is not None else pd.Series(dtype=float)
    newest = safe_number(clean.iloc[0]) if len(clean) >= 1 else None
    prior = safe_number(clean.iloc[1]) if len(clean) >= 2 else None
    return newest, prior


def piotroski_f_score(data: StockData) -> tuple[Optional[int], list[str]]:
    """Nine binary fundamental-momentum tests (Piotroski 2000)."""
    a = data.annual
    assets_now, assets_prior = _pair(a.get("assets", pd.Series(dtype=float)))
    if assets_now is None or assets_prior is None or assets_now <= 0:
        return None, []
    ni_now, ni_prior = _pair(a.get("net_income", pd.Series(dtype=float)))
    ocf_now, _ = _pair(a.get("ocf", pd.Series(dtype=float)))
    ltd_now, ltd_prior = _pair(a.get("long_term_debt", pd.Series(dtype=float)))
    ca_now, ca_prior = _pair(a.get("current_assets", pd.Series(dtype=float)))
    cl_now, cl_prior = _pair(a.get("current_liabilities", pd.Series(dtype=float)))
    sh_now, sh_prior = _pair(a.get("diluted_shares", a.get("shares", pd.Series(dtype=float))))
    gp_now, gp_prior = _pair(a.get("gross_profit", pd.Series(dtype=float)))
    rev_now, rev_prior = _pair(a.get("revenue", pd.Series(dtype=float)))

    score = 0
    detail: list[str] = []

    def award(label: str, condition: Optional[bool]) -> None:
        nonlocal score
        if condition is None:
            detail.append(f"  {label}: not testable")
            return
        score += 1 if condition else 0
        detail.append(f"  {'PASS' if condition else 'fail'}  {label}")

    roa_now = divide(ni_now, assets_now)
    roa_prior = divide(ni_prior, assets_prior)
    award("Positive return on assets", None if roa_now is None else roa_now > 0)
    award("Positive operating cash flow", None if ocf_now is None else ocf_now > 0)
    award("Return on assets improving", None if None in (roa_now, roa_prior) else roa_now > roa_prior)
    award("Operating cash flow exceeds net income", None if None in (ocf_now, ni_now) else ocf_now > ni_now)
    lev_now, lev_prior = divide(ltd_now, assets_now), divide(ltd_prior, assets_prior)
    award("Leverage not increasing", None if None in (lev_now, lev_prior) else lev_now <= lev_prior)
    cr_now, cr_prior = divide(ca_now, cl_now), divide(ca_prior, cl_prior)
    award("Current ratio improving", None if None in (cr_now, cr_prior) else cr_now > cr_prior)
    award("No net share issuance", None if None in (sh_now, sh_prior) else sh_now <= sh_prior * 1.005)
    gm_now, gm_prior = divide(gp_now, rev_now), divide(gp_prior, rev_prior)
    award("Gross margin improving", None if None in (gm_now, gm_prior) else gm_now > gm_prior)
    at_now, at_prior = divide(rev_now, assets_now), divide(rev_prior, assets_prior)
    award("Asset turnover improving", None if None in (at_now, at_prior) else at_now > at_prior)

    testable = sum(1 for line in detail if "not testable" not in line)
    if testable < 6:
        return None, detail
    return score, detail


def altman_z_score(data: StockData) -> tuple[Optional[float], str]:
    """Original public-manufacturer Z. Meaningless for banks and insurers."""
    m = data.metrics
    if data.special_model in {"bank", "insurer", "closed-end fund", "fund or non-operating instrument"}:
        return None, "not applicable to this business model"
    assets = m.get("assets")
    if not assets or assets <= 0:
        return None, "unavailable"
    working_capital = None
    if None not in (m.get("current_assets"), m.get("current_liabilities")):
        working_capital = m["current_assets"] - m["current_liabilities"]
    liabilities = m.get("liabilities")
    if liabilities is None and None not in (assets, m.get("equity")):
        liabilities = assets - m["equity"]
    parts = {
        "wc": divide(working_capital, assets),
        "re": divide(m.get("retained_earnings"), assets),
        "ebit": divide(m.get("ebit"), assets),
        "mve": divide(m.get("market_cap"), liabilities),
        "sales": divide(m.get("revenue"), assets),
    }
    if sum(1 for v in parts.values() if v is None) > 1:
        return None, "unavailable"
    filled = {k: (v if v is not None else 0.0) for k, v in parts.items()}
    z = (1.2 * filled["wc"] + 1.4 * filled["re"] + 3.3 * filled["ebit"]
         + 0.6 * filled["mve"] + 1.0 * filled["sales"])
    if z >= 2.99:
        zone = "safe zone"
    elif z >= 1.81:
        zone = "grey zone"
    else:
        zone = "distress zone"
    return float(z), zone


def beneish_m_score(data: StockData) -> tuple[Optional[float], str, dict[str, float]]:
    """Eight-variable earnings-manipulation probability model (Beneish 1999)."""
    a = data.annual
    rev_now, rev_prior = _pair(a.get("revenue", pd.Series(dtype=float)))
    if None in (rev_now, rev_prior) or rev_prior <= 0 or rev_now <= 0:
        return None, "unavailable", {}
    rec_now, rec_prior = _pair(a.get("receivables", pd.Series(dtype=float)))
    cogs_now, cogs_prior = _pair(a.get("cost_of_revenue", pd.Series(dtype=float)))
    assets_now, assets_prior = _pair(a.get("assets", pd.Series(dtype=float)))
    ppe_now, ppe_prior = _pair(a.get("net_ppe", pd.Series(dtype=float)))
    ca_now, ca_prior = _pair(a.get("current_assets", pd.Series(dtype=float)))
    sti_now, sti_prior = _pair(a.get("short_term_investments", pd.Series(dtype=float)))
    dep_now, dep_prior = _pair(a.get("depreciation", pd.Series(dtype=float)))
    sga_now, sga_prior = _pair(a.get("sga", pd.Series(dtype=float)))
    liab_now, liab_prior = _pair(a.get("liabilities", pd.Series(dtype=float)))
    ni_now, _ = _pair(a.get("net_income", pd.Series(dtype=float)))
    ocf_now, _ = _pair(a.get("ocf", pd.Series(dtype=float)))

    components: dict[str, float] = {}

    def ratio_index(num_now, den_now, num_prior, den_prior, name, default=1.0) -> float:
        now, prior = divide(num_now, den_now), divide(num_prior, den_prior)
        if now is None or prior is None or abs(prior) < 1e-12:
            return default
        value = now / prior
        if not math.isfinite(value):
            return default
        value = float(clamp(value, 0.2, 5.0))
        components[name] = value
        return value

    dsri = ratio_index(rec_now, rev_now, rec_prior, rev_prior, "DSRI")
    gm_now = divide(rev_now - (cogs_now or 0.0), rev_now)
    gm_prior = divide(rev_prior - (cogs_prior or 0.0), rev_prior)
    gmi = float(clamp(gm_prior / gm_now, 0.2, 5.0)) if None not in (gm_now, gm_prior) and gm_now else 1.0
    components["GMI"] = gmi

    def soft_asset_share(ca, ppe, assets) -> Optional[float]:
        if assets is None or assets <= 0:
            return None
        return 1.0 - ((ca or 0.0) + (ppe or 0.0)) / assets

    aq_now = soft_asset_share(ca_now, ppe_now, assets_now)
    aq_prior = soft_asset_share(ca_prior, ppe_prior, assets_prior)
    aqi = float(clamp(aq_now / aq_prior, 0.2, 5.0)) if None not in (aq_now, aq_prior) and aq_prior else 1.0
    components["AQI"] = aqi

    sgi = float(clamp(rev_now / rev_prior, 0.2, 5.0))
    components["SGI"] = sgi

    def dep_rate(dep, ppe) -> Optional[float]:
        if dep is None or ppe is None or (dep + ppe) <= 0:
            return None
        return dep / (dep + ppe)

    dr_now, dr_prior = dep_rate(dep_now, ppe_now), dep_rate(dep_prior, ppe_prior)
    depi = float(clamp(dr_prior / dr_now, 0.2, 5.0)) if None not in (dr_now, dr_prior) and dr_now else 1.0
    components["DEPI"] = depi

    sgai = ratio_index(sga_now, rev_now, sga_prior, rev_prior, "SGAI")
    lvgi = ratio_index(liab_now, assets_now, liab_prior, assets_prior, "LVGI")

    tata = divide((ni_now - ocf_now) if None not in (ni_now, ocf_now) else None, assets_now)
    tata = float(clamp(tata, -1.0, 1.0)) if tata is not None else 0.0
    components["TATA"] = tata

    if len(components) < 5:
        return None, "unavailable", components

    m_score = (-4.84 + 0.920 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi
               + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi)
    if m_score > -1.78:
        flag = "elevated manipulation risk"
    elif m_score > -2.22:
        flag = "borderline"
    else:
        flag = "no elevated risk indicated"
    return float(m_score), flag, components


def compute_forensics(data: StockData) -> ForensicScores:
    piotroski, detail = piotroski_f_score(data)
    z, zone = altman_z_score(data)
    m_score, flag, components = beneish_m_score(data)
    return ForensicScores(
        piotroski=piotroski, piotroski_detail=detail,
        altman_z=z, altman_zone=zone,
        beneish_m=m_score, beneish_flag=flag, beneish_components=components,
    )


# --------------------------------------------------------------------------
# Peers and relative-multiple targets
# --------------------------------------------------------------------------

def _peer_row(symbol: str, expected_sector: str,
              expected_industry: str) -> tuple[str, Optional[dict[str, float]], float, Optional[str]]:
    """Return (symbol, multiples, weight, note).

    V5 dropped any peer whose sector string didn't match exactly, which meant
    peer_medians came back empty for perfectly reasonable comparables ("Not
    available" sectors, cross-listed names, near-miss classifications). The
    user chose these peers on purpose, so mismatches now DEGRADE the peer's
    weight instead of discarding it:

        same sector or same industry .......... weight 1.0
        sector unknown on either side ......... weight 0.75
        genuinely different sector ............ weight 0.5, with a note

    Only non-equity instruments and peers with no usable multiples are still
    excluded outright."""
    try:
        info = cache_get("peer_info", symbol)
        if info is None:
            info = yf.Ticker(symbol).get_info()
            info = info if isinstance(info, dict) else {}
            info = {k: v for k, v in info.items() if isinstance(v, (str, int, float, bool)) or v is None}
            cache_put("peer_info", symbol, info)
        quote_type = str(info.get("quoteType") or "EQUITY").upper()
        if quote_type in {"ETF", "MUTUALFUND", "INDEX", "CURRENCY", "CRYPTOCURRENCY"}:
            return symbol, None, 0.0, f"Peer {symbol} was excluded: it is a {quote_type.lower()}, not an operating company."

        peer_sector = normalize_label(info.get("sector") or "")
        peer_industry = normalize_label(info.get("industry") or "")
        want_sector = normalize_label(expected_sector) if expected_sector.lower() not in ("", "not available") else ""
        want_industry = normalize_label(expected_industry) if expected_industry.lower() not in ("", "not available") else ""
        note = None
        if not peer_sector or not want_sector:
            weight = 0.75
        elif peer_sector == want_sector or (want_industry and peer_industry == want_industry):
            weight = 1.0
        else:
            weight = 0.5
            note = (f"Peer {symbol} is in a different sector "
                    f"({info.get('sector')}); it was kept at half weight.")

        same_currency = str(info.get("currency") or "").upper() == str(
            info.get("financialCurrency") or info.get("currency") or "").upper()
        row = {
            "pe": info_number(info, "trailingPE"),
            "forward_pe": info_number(info, "forwardPE"),
            "pfcf": divide(info_number(info, "marketCap"), info_number(info, "freeCashflow")) if same_currency else None,
            "ps": info_number(info, "priceToSalesTrailing12Months"),
            "ev_ebitda": info_number(info, "enterpriseToEbitda"),
        }
        usable = {k: v for k, v in row.items() if v is not None and 0 < v < 500}
        if not usable:
            return symbol, None, 0.0, f"Peer {symbol} returned no usable multiples."
        return symbol, usable, weight, note
    except Exception as exc:
        return symbol, None, 0.0, f"Peer {symbol} could not be retrieved: {type(exc).__name__}."


def fetch_peer_medians(symbols: list[str], warnings: list[str], expected_sector: str,
                       expected_industry: str = "") -> tuple[dict[str, float], list[str]]:
    if yf is None or not symbols:
        return {}, []
    rows: list[tuple[dict[str, float], float]] = []
    used: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(symbols))) as pool:
        for symbol, row, weight, warning in pool.map(
                lambda s: _peer_row(s, expected_sector, expected_industry), symbols[:8]):
            if warning:
                warnings.append(warning)
            if row and weight > 0:
                rows.append((row, weight))
                used.append(symbol)
    medians: dict[str, float] = {}
    for key in ("pe", "forward_pe", "pfcf", "ps", "ev_ebitda"):
        pairs = [(row[key], weight) for row, weight in rows if key in row]
        # One full-weight peer, or two of any weight, is enough for a usable
        # (if weak) central tendency; the anchor clamp in valuation_targets
        # bounds the damage a thin peer set can do.
        if len(pairs) >= 2 or any(weight >= 1.0 for _, weight in pairs):
            if pairs:
                medians[key] = float(weighted_median(pairs))
    return medians, sorted(used)


def sector_profile(data: StockData) -> dict[str, float]:
    return SECTOR_PROFILES.get(data.sector.lower(), SECTOR_PROFILES["default"]).copy()


def rate_adjusted_anchors(anchors: dict[str, float], risk_free: float) -> dict[str, float]:
    """Re-derive the multiple anchors for the live rate regime.

    A P/E anchor is an earnings-yield claim: 26x is a 3.85% earnings yield
    that made sense against ANCHOR_BASE_RF. When the 10Y moves, the required
    earnings yield moves with it - but only partially (ANCHOR_RATE_BETA),
    because equity multiples do not track bond yields one-for-one. Working in
    yield space keeps the adjustment convex the right way: high-multiple
    (long-duration) sectors move more than low-multiple ones, which is what
    the data shows. P/S has no yield interpretation, so it gets the square
    root of the earnings-multiple adjustment as a duration proxy."""
    delta_yield = (risk_free - ANCHOR_BASE_RF) * ANCHOR_RATE_BETA
    adjusted: dict[str, float] = {}
    for key, anchor in anchors.items():
        if key == "ps":
            reference = anchors.get("pe", 20.0)
            factor = _rate_factor(reference, delta_yield) ** 0.5
        else:
            factor = _rate_factor(anchor, delta_yield)
        adjusted[key] = float(anchor * clamp(factor, *ANCHOR_RATE_CLAMP))
    return adjusted


def _rate_factor(anchor_multiple: float, delta_yield: float) -> float:
    if anchor_multiple <= 0:
        return 1.0
    base_yield = 1.0 / anchor_multiple
    new_yield = max(base_yield + delta_yield, 0.005)
    return base_yield / new_yield


def fetch_live_anchor_pe(sector: str) -> Optional[float]:
    """Trailing P/E of the matching SPDR sector ETF, cached for a day, so the
    P/E anchor tracks the market instead of this file's last edit. Clamped to
    a band around the static anchor downstream, like every other component."""
    etf = SECTOR_ETFS.get(sector.lower())
    if etf is None or yf is None:
        return None
    cached = cache_get("anchor", etf)
    if cached is not None:
        stamp = safe_number(cached.get("at")) or 0.0
        if time.time() - stamp <= LIVE_ANCHOR_TTL:
            return safe_number(cached.get("pe"))
    try:
        info = yf.Ticker(etf).get_info()
        pe = info_number(info if isinstance(info, dict) else {}, "trailingPE")
        if pe is not None and 3 < pe < 60:
            cache_put("anchor", etf, {"pe": pe, "at": time.time()})
            return pe
    except Exception:
        pass
    return None


def historical_multiple(data: StockData, kind: str = "pe") -> Optional[float]:
    """The company's own median trading multiple, computed point-in-time."""
    if data.price_history.empty or "Close" not in data.price_history:
        return None
    if kind == "pe":
        per_share = data.annual.get("diluted_eps", pd.Series(dtype=float)).dropna()
    else:
        fcf = data.annual.get("fcfe", pd.Series(dtype=float))
        shares = data.annual.get("diluted_shares", data.annual.get("shares", pd.Series(dtype=float)))
        joined = pd.concat([fcf.rename("f"), shares.rename("s")], axis=1, join="inner").dropna()
        joined = joined[joined["s"] > 0]
        per_share = (joined["f"] / joined["s"]) if not joined.empty else pd.Series(dtype=float)
    if per_share.empty:
        return None
    close = pd.to_numeric(data.price_history["Close"], errors="coerce").dropna()
    try:
        index = pd.DatetimeIndex(close.index)
        close.index = index.tz_convert(None) if index.tz is not None else index.tz_localize(None)
    except (TypeError, ValueError):
        return None
    multiples: list[float] = []
    for date, value in per_share.items():
        if value is None or value <= 0:
            continue
        try:
            target = pd.Timestamp(date)
            if target.tz is not None:
                target = target.tz_localize(None)
        except (TypeError, ValueError):
            continue
        eligible = close.loc[:target]
        if eligible.empty:
            continue
        candidate = float(eligible.iloc[-1] / value)
        if 3 <= candidate <= 80:
            multiples.append(candidate)
    return float(np.median(multiples)) if len(multiples) >= 3 else None


def valuation_targets(data: StockData, peer_medians: dict[str, float],
                      risk_free: float = ANCHOR_BASE_RF,
                      live_pe: Optional[float] = None) -> dict[str, float]:
    """Blend the RATE-ADJUSTED sector anchor, the live sector-ETF multiple,
    peer medians, and the company's own history, then clamp hard so one bad
    input cannot run away with the valuation.

    The static table is first translated into the current rate regime (see
    rate_adjusted_anchors), which matches the rate-awareness of the DCF side:
    the same 10Y print that sets the WACC now also sets the relative-multiple
    yardstick, instead of the multiples quietly assuming a 4.2% world."""
    anchors = rate_adjusted_anchors(sector_profile(data), risk_free)
    own_pe = historical_multiple(data, "pe")
    own_pfcf = historical_multiple(data, "pfcf")
    live_scale = None
    if live_pe is not None and anchors.get("pe"):
        # The ETF P/E re-anchors the earnings multiples; other multiples move
        # with the same market-level scale factor, damped.
        live_scale = clamp(live_pe / anchors["pe"], 0.60, 1.60)
    targets: dict[str, float] = {}
    for key, anchor in anchors.items():
        components: list[tuple[float, float]] = [(anchor, 1.0)]
        if live_scale is not None:
            damp = 1.0 if key in ("pe", "forward_pe") else 0.5
            components.append((anchor * (1.0 + (live_scale - 1.0) * damp), 1.0))
        if key in peer_medians:
            components.append((clamp(peer_medians[key], anchor * 0.50, anchor * 1.50), 2.0))
        if key == "pe" and own_pe is not None:
            components.append((clamp(own_pe, anchor * 0.50, anchor * 1.50), 1.5))
        if key == "pfcf" and own_pfcf is not None:
            components.append((clamp(own_pfcf, anchor * 0.50, anchor * 1.50), 1.5))
        blended = sum(v * w for v, w in components) / sum(w for _, w in components)
        targets[key] = float(clamp(blended, anchor * 0.65, anchor * 1.35))
    return targets


# --------------------------------------------------------------------------
# Cost of capital
# --------------------------------------------------------------------------

def build_assumptions(data: StockData, risk_free: float, rate_source: str) -> MarketAssumptions:
    m = data.metrics
    beta = m.get("beta")
    if beta is None or not (0.2 <= beta <= 3.0):
        beta = 1.0
    tax_rate = float(m.get("tax_rate") or STATUTORY_TAX_FALLBACK)

    # Size premium: small caps genuinely carry a higher required return.
    market_cap = m.get("market_cap") or 0.0
    if market_cap and market_cap < 3e8:
        size_premium = 0.035
    elif market_cap and market_cap < 2e9:
        size_premium = 0.020
    elif market_cap and market_cap < 1e10:
        size_premium = 0.008
    else:
        size_premium = 0.0

    cost_of_equity = clamp(risk_free + beta * EQUITY_RISK_PREMIUM + size_premium,
                           MIN_COST_OF_EQUITY, MAX_COST_OF_EQUITY)

    # Cost of debt from actual interest paid, floored at the risk-free rate.
    debt = m.get("debt") or 0.0
    interest = abs(m.get("interest_expense") or 0.0)
    implied = divide(interest, debt)
    if implied is None or not (0.005 <= implied <= 0.25):
        implied = risk_free + 0.020
    cost_of_debt = float(clamp(implied, risk_free, 0.20))

    equity_value = m.get("market_cap") or 0.0
    total_capital = equity_value + debt
    equity_weight = (equity_value / total_capital) if total_capital > 0 else 1.0
    debt_weight = 1.0 - equity_weight
    wacc = clamp(equity_weight * cost_of_equity + debt_weight * cost_of_debt * (1 - tax_rate),
                 MIN_WACC, MAX_WACC)

    terminal_growth = float(clamp(risk_free + TERMINAL_SPREAD_TO_RF,
                                  MIN_TERMINAL_GROWTH, MAX_TERMINAL_GROWTH))
    # A perpetuity growth rate must stay well below the discount rate.
    terminal_growth = min(terminal_growth, wacc - 0.035)

    return MarketAssumptions(
        risk_free=risk_free, equity_risk_premium=EQUITY_RISK_PREMIUM, beta=beta,
        cost_of_equity=cost_of_equity, cost_of_debt=cost_of_debt, tax_rate=tax_rate,
        wacc=wacc, terminal_growth=terminal_growth,
        equity_weight=equity_weight, debt_weight=debt_weight, source=rate_source,
    )


# --------------------------------------------------------------------------
# Discounted cash flow
# --------------------------------------------------------------------------

def fade_path(growth: float, terminal_growth: float, years: int = EXPLICIT_YEARS,
              high_years: int = FADE_YEARS) -> list[float]:
    """High growth for `high_years`, then a linear fade to terminal growth."""
    path: list[float] = []
    for year in range(1, years + 1):
        if year <= high_years:
            path.append(growth)
        else:
            progress = (year - high_years) / max(1, years - high_years)
            path.append(growth + (terminal_growth - growth) * progress)
    return path


def dcf_enterprise_value(starting_fcff: float, growth: float, wacc: float,
                         terminal_growth: float) -> tuple[Optional[float], dict[str, float]]:
    if wacc <= terminal_growth or starting_fcff is None:
        return None, {}
    cash_flow = starting_fcff
    explicit_pv = 0.0
    for year, rate in enumerate(fade_path(growth, terminal_growth), start=1):
        cash_flow *= (1 + rate)
        explicit_pv += cash_flow / ((1 + wacc) ** year)
    terminal_value = cash_flow * (1 + terminal_growth) / (wacc - terminal_growth)
    terminal_pv = terminal_value / ((1 + wacc) ** EXPLICIT_YEARS)
    enterprise_value = explicit_pv + terminal_pv
    return enterprise_value, {
        "explicit_pv": explicit_pv,
        "terminal_pv": terminal_pv,
        "terminal_share": (terminal_pv / enterprise_value) if enterprise_value else 0.0,
    }


def dcf_per_share(data: StockData, starting_fcff: float, growth: float,
                  assumptions: MarketAssumptions) -> tuple[Optional[float], dict[str, float]]:
    shares = data.metrics.get("effective_shares")
    if not shares or shares <= 0:
        return None, {}
    enterprise_value, parts = dcf_enterprise_value(
        starting_fcff, growth, assumptions.wacc, assumptions.terminal_growth)
    if enterprise_value is None:
        return None, {}
    net_debt = data.metrics.get("net_debt") or 0.0
    equity_value = enterprise_value - net_debt
    parts.update({
        "enterprise_value": enterprise_value,
        "net_debt": net_debt,
        "equity_value": equity_value,
    })
    return divide(equity_value, shares), parts


def reverse_dcf(data: StockData, starting_fcff: float, assumptions: MarketAssumptions) -> Optional[float]:
    """Solve for the 5-year growth rate the current price implies."""
    price = data.metrics.get("current_price")
    shares = data.metrics.get("effective_shares")
    if not price or not shares or shares <= 0 or starting_fcff is None or starting_fcff <= 0:
        return None
    target_equity = price * shares
    net_debt = data.metrics.get("net_debt") or 0.0
    target_ev = target_equity + net_debt
    if target_ev <= 0:
        return None

    def value_at(growth: float) -> Optional[float]:
        enterprise_value, _ = dcf_enterprise_value(
            starting_fcff, growth, assumptions.wacc, assumptions.terminal_growth)
        return enterprise_value

    low, high = -0.30, 0.60
    if (value_at(low) or 0) > target_ev:
        return low
    if (value_at(high) or 0) < target_ev:
        return high
    for _ in range(80):
        mid = (low + high) / 2
        current = value_at(mid)
        if current is None:
            return None
        if current < target_ev:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def normalized_owner_earnings(data: StockData, assumptions: MarketAssumptions) -> tuple[
        Optional[float], Optional[float], Optional[float], list[str]]:
    """Return (starting FCFF, starting FCFE, normalized SBC, diagnostics).

    Both cash-flow bases are normalized over up to four years, stock-based
    compensation is deducted in full (it is a real economic cost regardless of
    how buybacks are financed), and the two are reconciled."""
    m = data.metrics
    diagnostics: list[str] = []

    def normalize(current: Optional[float], history: pd.Series) -> Optional[float]:
        clean = history.dropna().sort_index(ascending=False).iloc[:4]
        candidates: list[tuple[float, float]] = []
        if current is not None:
            candidates.append((current, 0.35))
        if len(clean) >= 2:
            candidates.append((float(clean.median()), 0.40))
            candidates.append((float(clean.mean()), 0.25))
        elif len(clean) == 1:
            candidates.append((float(clean.iloc[0]), 0.30))
        if not candidates:
            return None
        return weighted_median(candidates)

    fcff = normalize(m.get("fcff"), data.annual.get("fcff", pd.Series(dtype=float)))
    fcfe = normalize(m.get("fcfe"), data.annual.get("fcfe", pd.Series(dtype=float)))

    sbc_history = data.annual.get("sbc", pd.Series(dtype=float)).dropna().abs().iloc[:4]
    normalized_sbc = float(sbc_history.median()) if not sbc_history.empty else abs(m.get("sbc") or 0.0)

    # Stock-based compensation is added back inside operating cash flow. It is a
    # real cost to existing owners, so it is deducted here in full.
    if fcff is not None:
        fcff -= normalized_sbc
    if fcfe is not None:
        fcfe -= normalized_sbc
    if normalized_sbc and m.get("revenue"):
        share = normalized_sbc / m["revenue"]
        if share > 0.08:
            diagnostics.append(
                f"Stock-based compensation is {share:.1%} of revenue and is deducted in full from owner cash flow."
            )

    if None not in (fcff, fcfe) and fcff and abs(fcff) > 1e-6:
        gap = abs(fcfe / fcff - 1.0)
        if gap > 0.35:
            diagnostics.append(
                f"Unlevered and levered cash-flow bases differ by {gap:.0%}; "
                "check interest, leases, and working-capital classification."
            )
    if m.get("fcff") is not None and fcff and abs(m["fcff"] - normalized_sbc - fcff) / max(abs(fcff), 1.0) > 0.25:
        diagnostics.append("The latest period's cash flow differs materially from the normalized level; the normalized figure is used.")
    return fcff, fcfe, normalized_sbc, diagnostics


# --------------------------------------------------------------------------
# Fair value: build methods, blend by family, produce the buy-below price
# --------------------------------------------------------------------------

FAMILY_WEIGHTS = {"cash flow": 0.50, "earnings": 0.30, "asset": 0.10, "market": 0.10}


def valuation_status(upside: Optional[float]) -> str:
    if upside is None:
        return "UNKNOWN"
    if upside >= 0.35:
        return "DEEPLY UNDERVALUED"
    if upside >= 0.15:
        return "UNDERVALUED"
    if upside >= 0.05:
        return "MODESTLY UNDERVALUED"
    if upside > -0.05:
        return "NEAR FAIR VALUE"
    if upside > -0.15:
        return "MODESTLY OVERVALUED"
    if upside > -0.30:
        return "OVERVALUED"
    return "DEEPLY OVERVALUED"


def required_margin_of_safety(
    quality: float, confidence: float, volatility: Optional[float],
    forensics: ForensicScores, override: Optional[float],
) -> float:
    """A better business with cleaner data earns a smaller required discount."""
    if override is not None:
        return float(clamp(override, 0.0, 0.60))
    margin = 0.25
    margin -= 0.10 * clamp((quality - 55) / 45)          # high quality -> less discount
    margin += 0.12 * clamp((75 - confidence) / 40)       # weak data -> more discount
    if volatility is not None:
        margin += 0.08 * clamp((volatility - 0.25) / 0.55)
    if forensics.piotroski is not None and forensics.piotroski <= 3:
        margin += 0.06
    if forensics.altman_zone == "distress zone":
        margin += 0.08
    if forensics.beneish_flag == "elevated manipulation risk":
        margin += 0.08
    return float(clamp(margin, 0.10, 0.50))


def estimate_fair_value(
    data: StockData,
    targets: dict[str, float],
    assumptions: MarketAssumptions,
    input_confidence: float,
    data_coverage: float,
    data_freshness: float,
    quality: float,
    model_fit: float,
    integrity: float,
    mos_override: Optional[float] = None,
    band_scale: float = 1.0,
) -> FairValueResult:
    m = data.metrics
    analyst_reference = m.get("analyst_target") if (m.get("analyst_target") or 0) > 0 else None
    starting_fcff, starting_fcfe, normalized_sbc, diagnostics = normalized_owner_earnings(data, assumptions)

    notes = [
        "Discounted cash flow uses unlevered free cash flow (FCFF) discounted at WACC; "
        "net debt is subtracted once, at the end.",
        "Stock-based compensation is deducted in full from owner cash flow.",
        "Analyst price targets are shown for reference only and carry zero weight.",
        f"Risk-free rate {assumptions.risk_free:.2%} ({assumptions.source}); "
        f"WACC {assumptions.wacc:.2%}; terminal growth {assumptions.terminal_growth:.2%}.",
        f"Sector multiple anchors are rate-adjusted in yield space from a {ANCHOR_BASE_RF:.1%} "
        f"baseline to the live {assumptions.risk_free:.2%} risk-free rate.",
    ]

    def blocked(status: str, action: str, extra: str) -> FairValueResult:
        return FairValueResult(
            analyst_reference=analyst_reference, status=status, action=action,
            diagnostics=diagnostics + [extra], assumptions=notes,
            assumptions_used=assumptions, owner_earnings=starting_fcff,
            normalized_sbc=normalized_sbc,
        )

    if data.special_model:
        return blocked("SPECIALIZED MODEL REQUIRED", "SPECIALIZED MODEL REQUIRED",
                       f"A general corporate model is not valid for a {data.special_model}; "
                       "the required inputs are structurally different.")
    if not data.currency_compatible:
        return blocked("INCONCLUSIVE", "INCONCLUSIVE - CURRENCY MISMATCH",
                       "Statement currency could not be reconciled to the quote currency.")
    if any(not c.passed and c.severity == "critical" for c in data.integrity):
        failed = ", ".join(c.name for c in data.integrity if not c.passed and c.severity == "critical")
        return blocked("INCONCLUSIVE", "INCONCLUSIVE - STATEMENT INTEGRITY FAILURE",
                       f"Critical integrity check(s) failed: {failed}.")

    shares = m.get("effective_shares")
    methods: list[ValuationMethod] = []
    dcf = DCFDetail(starting_fcff=starting_fcff, fcfe_crosscheck=starting_fcfe,
                    wacc=assumptions.wacc, terminal_growth=assumptions.terminal_growth)
    if None not in (starting_fcff, starting_fcfe) and starting_fcff:
        dcf.reconciliation_gap = abs(starting_fcfe / starting_fcff - 1.0)

    base_factor = ((0.70 + 0.30 * clamp(input_confidence / 100))
                   * (0.70 + 0.30 * clamp(model_fit / 100))
                   * (0.80 + 0.20 * clamp(integrity / 100)))

    # ---- Family: cash flow --------------------------------------------------
    if starting_fcff is not None and starting_fcff > 0 and shares:
        growth_inputs = [v for v in (m.get("revenue_cagr3"), m.get("revenue_cagr5"),
                                     m.get("fcf_cagr3"), m.get("forward_revenue_growth")) if v is not None]
        growth = float(np.median(growth_inputs)) if growth_inputs else 0.04
        growth = float(clamp(growth, -0.03, 0.18))
        dcf.growth = growth

        base, parts = dcf_per_share(data, starting_fcff, growth, assumptions)
        bear_assumptions = MarketAssumptions(**{**asdict(assumptions),
                                                "wacc": min(MAX_WACC, assumptions.wacc + 0.015),
                                                "terminal_growth": max(MIN_TERMINAL_GROWTH, assumptions.terminal_growth - 0.005)})
        bull_assumptions = MarketAssumptions(**{**asdict(assumptions),
                                                "wacc": max(MIN_WACC, assumptions.wacc - 0.010),
                                                "terminal_growth": min(MAX_TERMINAL_GROWTH, assumptions.terminal_growth + 0.005)})
        bear, _ = dcf_per_share(data, starting_fcff * 0.90, max(-0.05, growth - 0.05), bear_assumptions)
        bull, _ = dcf_per_share(data, starting_fcff * 1.05, min(0.22, growth + 0.04), bull_assumptions)

        if base is not None and base > 0 and bear is not None and bull is not None:
            dcf.explicit_pv = parts.get("explicit_pv")
            dcf.terminal_pv = parts.get("terminal_pv")
            dcf.terminal_share = parts.get("terminal_share")
            dcf.enterprise_value = parts.get("enterprise_value")
            dcf.net_debt = parts.get("net_debt")
            dcf.equity_value = parts.get("equity_value")
            dcf.per_share = base
            stability = m.get("fcf_consistency") if m.get("fcf_consistency") is not None else 0.40
            reliability = 0.92 * base_factor * (0.60 + 0.40 * stability)
            note = ""
            if (dcf.terminal_share or 0) > MAX_TERMINAL_SHARE:
                reliability *= 0.75
                note = f"terminal value is {dcf.terminal_share:.0%} of enterprise value"
                diagnostics.append(
                    f"Terminal value is {dcf.terminal_share:.0%} of the discounted enterprise value; "
                    "the estimate leans heavily on perpetuity assumptions."
                )
            if dcf.reconciliation_gap is not None and dcf.reconciliation_gap > 0.35:
                reliability *= 0.85
            methods.append(ValuationMethod("DCF (FCFF / WACC)", "cash flow", base, min(bear, base), max(bull, base),
                                           1.0, clamp(reliability), note=note))

        dcf.implied_growth = reverse_dcf(data, starting_fcff, assumptions)
        realized = m.get("revenue_cagr3")
        if dcf.implied_growth is not None and realized is not None:
            dcf.implied_vs_actual = dcf.implied_growth - realized

    # Earnings power value: no growth at all, capitalized in perpetuity. This is
    # the floor a cash-generative business is worth if it never grows again.
    if starting_fcff is not None and starting_fcff > 0 and shares:
        epv_enterprise = starting_fcff / assumptions.wacc
        epv_equity = epv_enterprise - (m.get("net_debt") or 0.0)
        epv = divide(epv_equity, shares)
        if epv is not None and epv > 0:
            reliability = 0.78 * base_factor
            methods.append(ValuationMethod("Earnings power (no growth)", "cash flow", epv, epv * 0.85, epv * 1.15,
                                           0.55, clamp(reliability),
                                           note="zero-growth perpetuity floor"))

    # ---- Family: earnings ---------------------------------------------------
    analyst_count = m.get("analyst_count") or 0.0
    if (m.get("forward_eps") or 0) > 0:
        sustainable = float(clamp(first_number(m.get("forward_eps_growth"), m.get("income_cagr3"), 0.04) or 0.04, 0.0, 0.22))
        # Multiple caps stack: the lowest supported ceiling wins.
        formula_pe = 9.0 + sustainable * 45 + (quality - 60) * 0.04
        own_pe = historical_multiple(data, "pe")
        caps = [formula_pe, targets["forward_pe"], 9.0 + sustainable * 55]
        if own_pe is not None:
            caps.append(own_pe * 1.10)
        caps.append(32.0 if sustainable > 0.15 and model_fit >= 75 else 24.0)
        selected = max(7.0, min(caps))
        value = m["forward_eps"] * selected
        analyst_factor = 0.45 if analyst_count < 3 else 0.65 if analyst_count < 5 else 0.85 if analyst_count < 10 else 1.0
        reliability = 0.82 * base_factor * analyst_factor
        jump = divide(m.get("forward_eps"), m.get("eps"))
        if jump is not None and jump > 1.40:
            reliability *= 0.60
            diagnostics.append(
                f"Forward EPS sits {jump - 1:.0%} above trailing EPS; verify GAAP versus adjusted comparability."
            )
        methods.append(ValuationMethod("Forward earnings", "earnings", value, value * 0.80, value * 1.18,
                                       1.0, clamp(reliability), note=f"{selected:.1f}x forward"))
        notes.append(f"Forward earnings applies the lowest supported P/E ceiling of {selected:.1f}x.")

    if (m.get("eps") or 0) > 0:
        selected = min(targets["pe"], 24.0)
        value = m["eps"] * selected
        stability = m.get("income_consistency") if m.get("income_consistency") is not None else 0.50
        methods.append(ValuationMethod("Normalized trailing earnings", "earnings", value, value * 0.80, value * 1.18,
                                       0.65, clamp(0.70 * base_factor * (0.60 + 0.40 * stability)),
                                       note=f"{selected:.1f}x trailing"))

    # ---- Family: asset ------------------------------------------------------
    ebit = m.get("ebit")
    if None not in (ebit, m.get("net_debt"), shares) and ebit and ebit > 0 and shares:
        multiple_cap = min(targets["ev_ebit"], sector_profile(data)["ev_ebit"] * 1.10)
        equity_value = ebit * multiple_cap - (m.get("net_debt") or 0.0)
        value = divide(equity_value, shares)
        if value is not None and value > 0:
            methods.append(ValuationMethod("EV / EBIT", "asset", value, value * 0.82, value * 1.15,
                                           1.0, clamp(0.66 * base_factor), note=f"{multiple_cap:.1f}x EBIT"))
    book = m.get("book_per_share")
    roe = m.get("return_on_equity")
    if book is not None and book > 0 and roe is not None and roe > 0:
        # Justified price-to-book from sustainable ROE against cost of equity.
        justified_pb = clamp((roe - assumptions.terminal_growth) /
                             max(assumptions.cost_of_equity - assumptions.terminal_growth, 0.01), 0.3, 8.0)
        value = book * justified_pb
        methods.append(ValuationMethod("Justified price/book", "asset", value, value * 0.80, value * 1.20,
                                       0.55, clamp(0.55 * base_factor), note=f"{justified_pb:.2f}x book"))

    # ---- Family: market -----------------------------------------------------
    if (m.get("revenue") or 0) > 0 and shares:
        margin_adjustment = 1.0
        if m.get("operating_margin") is not None:
            sector_typical = 0.13
            margin_adjustment = float(clamp(m["operating_margin"] / sector_typical, 0.35, 2.2))
        value = divide(m["revenue"] * targets["ps"] * margin_adjustment, shares)
        if value is not None and value > 0:
            methods.append(ValuationMethod("Margin-adjusted price/sales", "market", value, value * 0.75, value * 1.25,
                                           1.0, clamp(0.48 * base_factor),
                                           note=f"{targets['ps'] * margin_adjustment:.2f}x sales"))

    if len(methods) < 2:
        return blocked("INSUFFICIENT DATA", "INSUFFICIENT DATA",
                       "Fewer than two usable valuation methods could be constructed.")

    # ---- Calibrated sensitivity bands ---------------------------------------
    # Every method's bear/bull half-width above is a hand-set sensitivity
    # guess (-10% FCFF here, +/-18% there). band_scale is the backtester's
    # verdict on those guesses: if realized (predicted upside - actual return)
    # dispersion exceeded the published bands, they widen; if the bands were
    # padded relative to realized error, they narrow. Applied uniformly so
    # relative method sensitivities - which ARE structural - are preserved.
    band_scale = clamp(band_scale, *DCF_BAND_SCALE_RANGE)
    if band_scale != 1.0:
        for method in methods:
            method.bear = max(method.value - (method.value - method.bear) * band_scale,
                              method.value * 0.10)
            method.bull = method.value + (method.bull - method.value) * band_scale
        notes.append(f"Bear/bull bands are scaled {band_scale:.2f}x by backtest calibration.")

    # ---- Blend: within family first, then across families -------------------
    families: dict[str, list[ValuationMethod]] = {}
    for method in methods:
        families.setdefault(method.family, []).append(method)

    family_values: dict[str, float] = {}
    family_weight: dict[str, float] = {}
    family_bear: dict[str, float] = {}
    family_bull: dict[str, float] = {}
    for family, group in families.items():
        weights = [max(method.base_weight * method.reliability, 1e-9) for method in group]
        total = sum(weights)
        family_values[family] = sum(mth.value * w for mth, w in zip(group, weights)) / total
        family_bear[family] = sum(mth.bear * w for mth, w in zip(group, weights)) / total
        family_bull[family] = sum(mth.bull * w for mth, w in zip(group, weights)) / total
        # A family's weight reflects both its prior and the reliability inside it.
        family_weight[family] = FAMILY_WEIGHTS.get(family, 0.10) * (total / sum(m2.base_weight for m2 in group))

    # Downweight a family that disagrees violently with the cross-family median.
    median_value = float(np.median(list(family_values.values())))
    for family, value in family_values.items():
        ratio = max(value, median_value) / max(min(value, median_value), 1e-9)
        if ratio > 2.5:
            family_weight[family] *= 0.25
            for method in families[family]:
                method.status = "extreme outlier"
        elif ratio > 1.7:
            family_weight[family] *= 0.60
            for method in families[family]:
                method.status = "outlier - review"

    total_family_weight = sum(family_weight.values())
    if total_family_weight <= 0:
        return blocked("INCONCLUSIVE", "INCONCLUSIVE - NO WEIGHTED METHOD SURVIVED",
                       "All valuation families were downweighted to zero.")

    base = sum(family_values[f] * family_weight[f] for f in family_values) / total_family_weight
    low = sum(family_bear[f] * family_weight[f] for f in family_values) / total_family_weight
    high = sum(family_bull[f] * family_weight[f] for f in family_values) / total_family_weight

    # Effective per-method weights, for display.
    for family, group in families.items():
        inner = [max(mth.base_weight * mth.reliability, 1e-9) for mth in group]
        inner_total = sum(inner)
        share = family_weight[family] / total_family_weight
        for method, weight in zip(group, inner):
            method.effective_weight = share * weight / inner_total

    values = list(family_values.values())
    family_ratio = max(values) / max(min(values), 1e-9) if len(values) > 1 else 1.0
    dispersion = (max(values) - min(values)) / base if base else None

    if family_ratio <= 1.20:
        agreement = 0.95
    elif family_ratio <= 1.60:
        agreement = 0.82
    elif family_ratio <= 2.20:
        agreement = 0.62
    elif family_ratio < 3.00:
        agreement = 0.38
    else:
        agreement = 0.15

    if len(values) == 1:
        agreement = min(agreement, 0.50)
        diagnostics.append("Only one valuation family was available; cross-family agreement could not be tested.")
    if family_ratio > 2.0:
        diagnostics.append(
            f"The highest valuation family sits at {family_ratio:.1f}x the lowest; treat the point estimate loosely."
        )

    confidence = (0.26 * data_coverage + 0.16 * data_freshness + 0.18 * model_fit
                  + 0.16 * integrity + 0.24 * agreement * 100)

    if family_ratio >= MAX_FAMILY_RATIO:
        return FairValueResult(
            methods=methods, family_values=family_values, analyst_reference=analyst_reference,
            status="INCONCLUSIVE", family_agreement=agreement, family_ratio=family_ratio,
            dispersion=dispersion, confidence=min(confidence, 45.0),
            action="INCONCLUSIVE - VALUATION METHODS DISAGREE", decision_basis="No point estimate issued",
            dcf=dcf, assumptions_used=assumptions, owner_earnings=starting_fcff,
            normalized_sbc=normalized_sbc, assumptions=notes,
            diagnostics=diagnostics + ["Independent valuation families disagree too widely to publish a single fair value."],
        )

    margin = required_margin_of_safety(quality, confidence, m.get("annual_volatility"), data.forensics, mos_override)
    buy_below = base * (1 - margin)
    strong_buy_below = min(low, base * (1 - min(0.55, margin + 0.12)))

    price = m.get("current_price")
    dollar_gap = base - price if price is not None else None
    upside = divide(dollar_gap, price)
    discount_premium = divide(dollar_gap, base)

    if price is None:
        action, basis = "INCONCLUSIVE - NO PRICE", "No current price available"
    elif confidence < 45:
        action, basis = "INSUFFICIENT CONFIDENCE", "Low-confidence family blend"
    elif price <= strong_buy_below and quality >= 65:
        action, basis = "STRONG BUY ZONE - LARGE MARGIN OF SAFETY", "Cross-family reliability-weighted blend"
    elif price <= buy_below and quality >= 55:
        action, basis = "BUY ZONE - BELOW REQUIRED MARGIN OF SAFETY", "Cross-family reliability-weighted blend"
    elif price <= buy_below:
        action, basis = "CHEAP BUT LOW BUSINESS QUALITY - VERIFY", "Cross-family reliability-weighted blend"
    elif price < base:
        action, basis = "BELOW FAIR VALUE - INSUFFICIENT MARGIN OF SAFETY", "Cross-family reliability-weighted blend"
    elif price <= base * 1.10:
        action, basis = "NEAR FAIR VALUE - HOLD / WATCH", "Cross-family reliability-weighted blend"
    elif price <= high:
        action, basis = "ABOVE FAIR VALUE - WATCHLIST ONLY", "Cross-family reliability-weighted blend"
    else:
        action, basis = "ABOVE THE ENTIRE ESTIMATED RANGE", "Cross-family reliability-weighted blend"

    return FairValueResult(
        low=min(low, base), base=base, high=max(high, base), methods=methods,
        family_values=family_values, analyst_reference=analyst_reference,
        dollar_gap=dollar_gap, upside_downside=upside, discount_premium=discount_premium,
        status=valuation_status(upside), family_agreement=agreement, family_ratio=family_ratio,
        dispersion=dispersion, confidence=confidence, margin_of_safety=margin,
        buy_below=buy_below, strong_buy_below=strong_buy_below, action=action, decision_basis=basis,
        dcf=dcf, assumptions_used=assumptions, owner_earnings=starting_fcff,
        normalized_sbc=normalized_sbc, diagnostics=diagnostics, assumptions=notes,
    )


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def score_relative_multiple(value: float, target: float) -> float:
    """Cheap is good, but absurdly cheap is usually a trap, not a gift."""
    if value <= 0 or target <= 0:
        return 0.0
    relative = value / target
    return interpolate(relative, [(0.15, 0.55), (0.45, 0.85), (0.70, 1.0),
                                  (1.0, 0.78), (1.35, 0.50), (1.80, 0.22), (2.60, 0.0)])


def metric_score(data: StockData, key: str, scorer: Callable[[float], float]) -> Optional[float]:
    value = data.metrics.get(key)
    if value is None:
        return None
    reliability = data.meta[key].reliability if key in data.meta else 0.70
    if reliability < 0.20:
        return None
    return clamp(scorer(value)) * (0.85 + 0.15 * reliability)


def grouped_category(maximum: float, groups: list[tuple[str, float, list[Optional[float]]]],
                     gamma: float = 1.0) -> CategoryResult:
    """gamma is a backtest-calibrated curve exponent applied to the category's
    normalized score (score**gamma). The hand-drawn interpolation breakpoints
    fix each curve's SHAPE from intuition; gamma is the one degree of freedom
    the backtester is allowed to bend, from out-of-sample evidence only:

        gamma > 1  the category's ranking predicted excess returns, so the
                   curve steepens - mediocre readings are punished harder and
                   only genuinely strong readings earn full credit;
        gamma < 1  the category showed little or no predictive ordering, so
                   the curve flattens toward indifference and the category's
                   internal distinctions matter less.

    Clamped to CURVE_GAMMA_RANGE so no sample can invert or degenerate a
    curve. Identity (1.0) until a validated calibration file says otherwise."""
    earned = 0.0
    available_weight = 0.0
    details: list[str] = []
    total_weight = sum(weight for _, weight, _ in groups)
    for name, weight, scores in groups:
        available = [s for s in scores if s is not None]
        if not available:
            details.append(f"{name}: unavailable")
            continue
        group_score = float(np.mean(available))
        earned += group_score * weight
        available_weight += weight
        details.append(f"{name}: {group_score * 100:.0f}%")
    if available_weight == 0 or total_weight == 0:
        return CategoryResult(0.0, maximum, 0.0, details)
    coverage = available_weight / total_weight
    normalized = earned / available_weight
    gamma = clamp(gamma, *CURVE_GAMMA_RANGE)
    if gamma != 1.0:
        normalized = clamp(normalized) ** gamma
        details.append(f"calibrated curve exponent: {gamma:.2f}")
    # Missing data does not score zero, but it caps the credit available.
    points = maximum * min(normalized, 0.45 + 0.55 * coverage)
    return CategoryResult(points, maximum, coverage, details)


def score_categories(data: StockData, targets: dict[str, float], confidence: float,
                     integrity: float,
                     curve_gammas: Optional[dict[str, float]] = None) -> dict[str, CategoryResult]:
    m = data.metrics
    gammas = curve_gammas or {}
    g = lambda key: float(gammas.get(key, 1.0))
    margin = lambda v: interpolate(v, [(-0.20, 0), (0, 0.20), (0.05, 0.50), (0.15, 0.80), (0.30, 1.0)])
    gross = lambda v: interpolate(v, [(0.05, 0), (0.20, 0.30), (0.35, 0.60), (0.50, 0.85), (0.70, 1.0)])
    returns = lambda v: interpolate(v, [(-0.20, 0), (0, 0.20), (0.08, 0.55), (0.15, 0.80), (0.25, 1.0)])
    growth = lambda v: interpolate(v, [(-0.30, 0), (-0.05, 0.15), (0, 0.35), (0.05, 0.55),
                                       (0.10, 0.72), (0.20, 0.90), (0.35, 1.0)])
    steady = lambda v: interpolate(v, [(0, 0), (0.5, 0.5), (0.75, 0.8), (1.0, 1.0)])
    low_vol = lambda v: interpolate(v, [(0, 1), (0.25, 0.9), (0.75, 0.55), (1.5, 0.15), (3, 0)])
    liquidity = lambda v: interpolate(v, [(0, 0), (0.8, 0.2), (1.0, 0.45), (1.5, 0.8), (2.5, 1.0), (4.0, 0.85)])

    profitability = grouped_category(CATEGORY_MAXIMUMS["profitability"], gamma=g("profitability"), groups=[
        ("gross margin", 0.15, [metric_score(data, "gross_margin", gross)]),
        ("operating margins", 0.25, [metric_score(data, "operating_margin", margin),
                                     metric_score(data, "profit_margin", margin)]),
        ("return on capital", 0.40, [metric_score(data, "roic", returns)]),
        ("asset and equity returns", 0.10, [metric_score(data, "return_on_assets", returns),
                                            metric_score(data, "return_on_equity", returns)]),
        ("profit consistency", 0.10, [metric_score(data, "income_consistency", steady)]),
    ])
    growth_result = grouped_category(CATEGORY_MAXIMUMS["growth"], gamma=g("growth"), groups=[
        ("revenue growth", 0.30, [metric_score(data, "revenue_growth", growth),
                                  metric_score(data, "revenue_cagr3", growth),
                                  metric_score(data, "revenue_cagr5", growth)]),
        ("profit and cash growth", 0.28, [metric_score(data, "income_growth", growth),
                                          metric_score(data, "fcf_growth", growth),
                                          metric_score(data, "gross_profit_cagr3", growth)]),
        ("forward estimates", 0.18, [metric_score(data, "forward_revenue_growth", growth),
                                     metric_score(data, "forward_eps_growth", growth)]),
        ("growth stability", 0.14, [metric_score(data, "revenue_consistency", steady),
                                    metric_score(data, "revenue_volatility", low_vol)]),
        ("reinvestment runway", 0.10, [metric_score(data, "capex_to_depreciation",
                                                    lambda v: interpolate(v, [(0.3, 0.3), (1.0, 0.8), (1.5, 1.0), (3.5, 0.5)]))]),
    ])
    health = grouped_category(CATEGORY_MAXIMUMS["financial_health"], gamma=g("financial_health"), groups=[
        ("net cash position", 0.25, [metric_score(data, "cash_to_debt",
                                                  lambda v: interpolate(v, [(0, 0), (0.25, 0.25), (0.75, 0.6), (1, 0.8), (2, 1)]))]),
        ("leverage", 0.25, [metric_score(data, "debt_to_equity",
                                         lambda v: interpolate(v, [(-1, 0), (0, 1), (0.5, 0.9), (1, 0.65), (2, 0.3), (4, 0)])),
                            metric_score(data, "net_debt_to_ebitda",
                                         lambda v: interpolate(v, [(-2, 1), (0, 0.95), (1.5, 0.8), (3, 0.5), (5, 0.15), (8, 0)]))]),
        ("liquidity", 0.20, [metric_score(data, "current_ratio", liquidity),
                             metric_score(data, "quick_ratio", liquidity)]),
        ("interest coverage", 0.20, [metric_score(data, "interest_coverage",
                                                  lambda v: interpolate(v, [(-1, 0), (1, 0.15), (2, 0.4), (5, 0.75), (10, 1)]))]),
        ("working-capital cycle", 0.10, [metric_score(data, "cash_conversion_cycle",
                                                      lambda v: interpolate(v, [(-60, 1), (0, 0.95), (45, 0.75), (90, 0.45), (180, 0.1)]))]),
    ])
    valuation = grouped_category(CATEGORY_MAXIMUMS["valuation"], gamma=g("valuation"), groups=[
        ("earnings multiples", 0.30, [metric_score(data, "trailing_pe", lambda v: score_relative_multiple(v, targets["pe"])),
                                      metric_score(data, "forward_pe", lambda v: score_relative_multiple(v, targets["forward_pe"]))]),
        ("cash-flow multiples", 0.28, [metric_score(data, "price_to_fcf", lambda v: score_relative_multiple(v, targets["pfcf"]))]),
        ("enterprise multiples", 0.24, [metric_score(data, "ev_to_ebitda", lambda v: score_relative_multiple(v, targets["ev_ebitda"])),
                                        metric_score(data, "ev_to_ebit", lambda v: score_relative_multiple(v, targets["ev_ebit"]))]),
        ("sales multiple", 0.10, [metric_score(data, "price_to_sales", lambda v: score_relative_multiple(v, targets["ps"]))]),
        ("owner yield", 0.08, [metric_score(data, "fcf_yield",
                                            lambda v: interpolate(v, [(-0.05, 0), (0, 0.15), (0.03, 0.45), (0.06, 0.75), (0.10, 1.0)]))]),
    ])
    accounting = grouped_category(CATEGORY_MAXIMUMS["cash_accounting"], gamma=g("cash_accounting"), groups=[
        ("cash generation", 0.22, [metric_score(data, "fcf_margin", margin),
                                   metric_score(data, "fcf_consistency", steady)]),
        ("earnings conversion", 0.20, [metric_score(data, "ocf_to_net_income",
                                                    lambda v: interpolate(v, [(-1, 0), (0, 0.1), (0.5, 0.35), (0.8, 0.7),
                                                                              (1.0, 0.9), (1.3, 1.0), (2.5, 0.85)]))]),
        ("accruals", 0.18, [metric_score(data, "accrual_ratio",
                                         lambda v: interpolate(v, [(-0.25, 0.9), (-0.05, 1.0), (0.05, 0.8), (0.15, 0.35), (0.30, 0)]))]),
        ("dilution and SBC", 0.16, [metric_score(data, "shares_growth",
                                                 lambda v: interpolate(v, [(-0.10, 1), (0, 0.9), (0.03, 0.6), (0.10, 0.15), (0.25, 0)])),
                                    metric_score(data, "sbc_to_revenue",
                                                 lambda v: interpolate(v, [(0, 1), (0.03, 0.9), (0.08, 0.6), (0.15, 0.25), (0.30, 0)]))]),
        ("receivables and inventory", 0.12, [metric_score(data, "days_sales_outstanding",
                                                          lambda v: interpolate(v, [(0, 1), (45, 0.9), (75, 0.7), (120, 0.35), (200, 0)])),
                                             metric_score(data, "days_inventory",
                                                          lambda v: interpolate(v, [(0, 1), (60, 0.85), (120, 0.6), (240, 0.2), (400, 0)]))]),
        ("one-offs and balance sheet", 0.12, [metric_score(data, "restructuring_to_revenue",
                                                           lambda v: interpolate(v, [(0, 1), (0.02, 0.85), (0.05, 0.55), (0.15, 0)])),
                                              metric_score(data, "goodwill_to_assets",
                                                           lambda v: interpolate(v, [(0, 1), (0.15, 0.85), (0.35, 0.5), (0.60, 0.15), (1, 0)]))]),
    ])
    risk = grouped_category(CATEGORY_MAXIMUMS["risk_data"], gamma=g("risk_data"), groups=[
        ("data confidence", 0.25, [clamp(confidence / 100)]),
        ("statement integrity", 0.25, [clamp(integrity / 100)]),
        ("price volatility", 0.18, [metric_score(data, "annual_volatility",
                                                 lambda v: interpolate(v, [(0, 1), (0.20, 0.9), (0.35, 0.7), (0.60, 0.35), (1.0, 0)]))]),
        ("drawdown", 0.16, [metric_score(data, "max_drawdown",
                                         lambda v: interpolate(v, [(-1, 0), (-0.60, 0.25), (-0.40, 0.55), (-0.20, 0.85), (0, 1)]))]),
        ("estimate revisions", 0.16, [metric_score(data, "revision_balance",
                                                   lambda v: interpolate(v, [(-1, 0), (-0.25, 0.3), (0, 0.6), (0.25, 0.85), (1, 1)]))]),
    ])

    # Value-trap adjustment: applied only to the valuation category, and shown.
    traps = sum(bool(x) for x in (
        m.get("revenue_growth") is not None and m["revenue_growth"] < -0.02,
        m.get("operating_margin") is not None and m["operating_margin"] < 0,
        m.get("fcfe") is not None and m["fcfe"] <= 0,
        m.get("accrual_ratio") is not None and m["accrual_ratio"] > 0.15,
        data.forensics.altman_zone == "distress zone",
        data.forensics.beneish_flag == "elevated manipulation risk",
    ))
    if traps:
        valuation.points *= max(0.50, 1 - 0.10 * traps)
        valuation.details.append(f"value-trap adjustment: {traps} signal(s)")

    return {
        "profitability": profitability, "growth": growth_result, "financial_health": health,
        "valuation": valuation, "cash_accounting": accounting, "risk_data": risk,
    }


def freshness_factor(meta: MetricMeta, retrieved_at: datetime) -> float:
    if meta.as_of is None:
        return 1.0 if "market" in meta.source else 0.85
    as_of = pd.Timestamp(meta.as_of).to_pydatetime()
    if as_of.tzinfo is not None:
        as_of = as_of.replace(tzinfo=None)
    days = max(0, (retrieved_at.replace(tzinfo=None) - as_of).days)
    if meta.period in {"FY", "FY-end", "latest quarter", "TTM"}:
        return interpolate(days, [(0, 1), (120, 0.95), (240, 0.80), (400, 0.55), (700, 0.20)])
    return 1.0


def confidence_score(data: StockData) -> float:
    total = float(sum(CONFIDENCE_INPUT_WEIGHTS.values()))
    earned = 0.0
    for key, weight in CONFIDENCE_INPUT_WEIGHTS.items():
        if data.metrics.get(key) is None or key not in data.meta:
            continue
        meta = data.meta[key]
        earned += weight * meta.reliability * freshness_factor(meta, data.retrieved_at)
    history_bonus = 0.0
    for key in ("revenue", "net_income", "ocf", "capex", "assets", "diluted_shares"):
        observations = len(data.annual.get(key, pd.Series(dtype=float)).dropna())
        history_bonus += min(observations, 4) / 4
    return clamp((earned / total) * 0.90 + (history_bonus / 6) * 0.10) * 100


def data_quality_components(data: StockData) -> tuple[float, float]:
    total = float(sum(CONFIDENCE_INPUT_WEIGHTS.values()))
    available = 0.0
    freshness_earned = 0.0
    for key, weight in CONFIDENCE_INPUT_WEIGHTS.items():
        if data.metrics.get(key) is None or key not in data.meta:
            continue
        available += weight
        freshness_earned += weight * freshness_factor(data.meta[key], data.retrieved_at)
    coverage = 100 * available / total
    freshness = 100 * freshness_earned / available if available else 0.0
    return coverage, freshness


def model_fit_score(data: StockData) -> float:
    """How well a general FCFF/WACC corporate model suits this business."""
    if data.special_model:
        return 20.0
    sector, industry = data.sector.lower(), data.industry.lower()
    if "software" in industry or "internet" in industry:
        score = 82.0
    elif "semiconductor" in industry:
        score = 66.0
    elif "energy" in sector or "utilities" in sector:
        score = 56.0
    elif "industrial" in sector:
        score = 72.0
    elif any(term in industry for term in ("retail", "specialty", "department store")):
        score = 68.0
    elif "communication" in sector:
        score = 64.0
    elif "materials" in sector:
        score = 60.0
    else:
        score = 70.0
    if (data.metrics.get("capex_to_ocf") or 0) > 0.65:
        score -= 10.0
    if (data.metrics.get("revenue_volatility") or 0) > 0.45:
        score -= 8.0
    if not data.currency_compatible:
        score -= 30.0
    if len(data.annual.get("revenue", pd.Series(dtype=float)).dropna()) < 3:
        score -= 12.0
    return float(clamp(score, 0.0, 100.0))


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def generate_observations(data: StockData, categories: dict[str, CategoryResult],
                          confidence: float) -> tuple[list[str], list[str]]:
    m = data.metrics
    f = data.forensics
    strengths: list[str] = []
    concerns: list[str] = []

    if (m.get("roic") or 0) >= 0.15:
        strengths.append(f"Return on invested capital is {m['roic']:.1%}, comfortably above a typical cost of capital.")
    if (m.get("gross_margin") or 0) >= 0.45:
        strengths.append(f"Gross margin of {m['gross_margin']:.1%} indicates real pricing power.")
    if (m.get("operating_margin") or 0) >= 0.15:
        strengths.append(f"Operating margin is {m['operating_margin']:.1%}.")
    if (m.get("revenue_cagr3") or 0) >= 0.10:
        strengths.append(f"Three-year revenue CAGR is {m['revenue_cagr3']:.1%}.")
    if (m.get("fcfe") or 0) > 0 and (m.get("fcf_consistency") or 0) >= 0.75:
        strengths.append("Free cash flow has been positive in most of the available history.")
    if (m.get("cash_to_debt") or 0) >= 1:
        strengths.append("Cash and short-term investments equal or exceed total debt.")
    if f.piotroski is not None and f.piotroski >= 7:
        strengths.append(f"Piotroski F-Score of {f.piotroski}/9 indicates improving fundamental quality.")
    if f.altman_zone == "safe zone":
        strengths.append(f"Altman Z-Score of {f.altman_z:.2f} sits in the safe zone.")

    if (m.get("revenue_growth") or 0) < 0:
        concerns.append(f"Latest annual revenue growth was {m['revenue_growth']:.1%}.")
    if m.get("fcfe") is not None and m["fcfe"] <= 0:
        concerns.append("Free cash flow is negative on the latest basis.")
    if (m.get("accrual_ratio") or 0) > 0.10:
        concerns.append(f"An accrual ratio of {m['accrual_ratio']:.1%} weakens earnings quality.")
    if (m.get("sbc_to_revenue") or 0) > 0.10:
        concerns.append(f"Stock-based compensation runs at {m['sbc_to_revenue']:.1%} of revenue.")
    if (m.get("shares_growth") or 0) > 0.03:
        concerns.append(f"Diluted share count rose {m['shares_growth']:.1%} year over year.")
    if (m.get("net_debt_to_ebitda") or 0) > 3.5:
        concerns.append(f"Net debt is {m['net_debt_to_ebitda']:.1f}x EBITDA.")
    if f.beneish_flag == "elevated manipulation risk":
        concerns.append(f"Beneish M-Score of {f.beneish_m:.2f} falls in the elevated-risk range; scrutinise revenue recognition.")
    if f.altman_zone == "distress zone":
        concerns.append(f"Altman Z-Score of {f.altman_z:.2f} sits in the distress zone.")
    if f.piotroski is not None and f.piotroski <= 3:
        concerns.append(f"Piotroski F-Score of {f.piotroski}/9 points to deteriorating fundamentals.")
    if (m.get("max_drawdown") or 0) < -0.50:
        concerns.append(f"Maximum drawdown over the price history was {m['max_drawdown']:.1%}.")
    failed = [c.name for c in data.integrity if not c.passed]
    if failed:
        concerns.append("Statement integrity issues: " + ", ".join(failed) + ".")
    low_coverage = [name for name, result in categories.items()
                    if result.coverage < MIN_CATEGORY_COVERAGE and name != "risk_data"]
    if low_coverage:
        concerns.append("Thin input coverage in: " + ", ".join(low_coverage) + ".")
    if confidence < 75:
        concerns.append(f"Independent-input confidence is only {confidence:.0f}%.")
    return strengths[:7], concerns[:7]


def apply_category_weights(categories: dict[str, CategoryResult], weights: Optional[dict[str, float]]) -> None:
    if not weights:
        return
    for key, category in categories.items():
        ratio = divide(category.points, category.maximum) or 0.0
        category.maximum = weights[key]
        category.points = ratio * category.maximum


def load_category_weights(path: Optional[str]) -> Optional[dict[str, float]]:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != set(CATEGORY_MAXIMUMS):
        raise ValueError("The weights file must contain exactly: " + ", ".join(CATEGORY_MAXIMUMS))
    weights = {key: float(payload[key]) for key in CATEGORY_MAXIMUMS}
    if any(not math.isfinite(v) or v <= 0 for v in weights.values()):
        raise ValueError("Every category weight must be a positive finite number.")
    total = sum(weights.values())
    return {key: 100 * value / total for key, value in weights.items()}


def load_calibration(path: Optional[str]) -> Optional[Calibration]:
    """Load a calibration file written by --calibrate-out.

    Also accepts the legacy flat weights-only format for continuity, so an
    old --weights-json file can be passed to --calibration-json unchanged."""
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("The calibration file must be a JSON object.")
    if set(payload) == set(CATEGORY_MAXIMUMS):        # legacy weights-only file
        return Calibration(category_weights=load_category_weights(path), provenance="legacy weights file")
    weights = None
    if payload.get("category_weights"):
        raw = payload["category_weights"]
        if set(raw) != set(CATEGORY_MAXIMUMS):
            raise ValueError("category_weights must contain exactly: " + ", ".join(CATEGORY_MAXIMUMS))
        weights = {k: float(raw[k]) for k in CATEGORY_MAXIMUMS}
        if any(not math.isfinite(v) or v <= 0 for v in weights.values()):
            raise ValueError("Every category weight must be a positive finite number.")
        total = sum(weights.values())
        weights = {k: 100 * v / total for k, v in weights.items()}
    gammas = None
    if payload.get("curve_gamma"):
        gammas = {str(k): float(clamp(float(v), *CURVE_GAMMA_RANGE))
                  for k, v in payload["curve_gamma"].items()
                  if k in CATEGORY_MAXIMUMS and math.isfinite(float(v))}
    band = payload.get("dcf_band_scale")
    band = float(clamp(float(band), *DCF_BAND_SCALE_RANGE)) if band is not None and math.isfinite(float(band)) else None
    return Calibration(category_weights=weights, curve_gamma=gammas, dcf_band_scale=band,
                       provenance=str(payload.get("meta", {}).get("generated", "calibration file")))


def analyze(data: StockData, peer_symbols: list[str],
            category_weights: Optional[dict[str, float]] = None,
            mos_override: Optional[float] = None,
            calibration: Optional[Calibration] = None) -> AnalysisResult:
    calibration = calibration or Calibration()
    if category_weights is None:
        category_weights = calibration.category_weights

    peer_medians, peers_used = fetch_peer_medians(peer_symbols, data.warnings,
                                                  data.sector, data.industry)
    data.peers_used = peers_used
    data.peer_medians = peer_medians

    risk_free, rate_source = fetch_risk_free_rate(data.warnings)
    assumptions = build_assumptions(data, risk_free, rate_source)

    confidence = confidence_score(data)
    coverage, freshness = data_quality_components(data)
    integrity = integrity_score(data)
    live_pe = fetch_live_anchor_pe(data.sector)
    targets = valuation_targets(data, peer_medians, risk_free, live_pe)
    categories = score_categories(data, targets, confidence, integrity,
                                  curve_gammas=calibration.curve_gamma)
    apply_category_weights(categories, category_weights)
    overall = sum(result.points for result in categories.values())

    quality_keys = ("profitability", "growth", "financial_health", "cash_accounting")
    quality_max = sum(categories[k].maximum for k in quality_keys)
    quality = 100 * sum(categories[k].points for k in quality_keys) / quality_max if quality_max else 0.0
    model_fit = model_fit_score(data)

    fair_value = estimate_fair_value(
        data, targets, assumptions, confidence, coverage, freshness,
        quality, model_fit, integrity, mos_override,
        band_scale=calibration.dcf_band_scale if calibration.dcf_band_scale is not None else 1.0,
    )
    model_confidence = fair_value.confidence if fair_value.methods else min(
        40.0, 0.30 * coverage + 0.20 * freshness + 0.20 * model_fit)

    coverages_ok = all(categories[k].coverage >= MIN_CATEGORY_COVERAGE
                       for k in ("profitability", "growth", "financial_health", "valuation", "cash_accounting"))
    reliable = (confidence >= MIN_INPUT_CONFIDENCE and model_confidence >= MIN_INPUT_CONFIDENCE
                and coverages_ok and data.special_model is None
                and not any(not c.passed and c.severity == "critical" for c in data.integrity))

    uncertainty = 4.0 + (100 - confidence) * 0.15
    if data.metrics.get("annual_volatility") is not None:
        uncertainty += min(data.metrics["annual_volatility"] * 5, 4.0)
    score_low, score_high = max(0.0, overall - uncertainty), min(100.0, overall + uncertainty)

    strengths, concerns = generate_observations(data, categories, confidence)

    if data.special_model:
        conclusion = (f"A general corporate score is suppressed: {data.company_name} needs a "
                      f"{data.special_model} model built on different inputs.")
    elif fair_value.base is None:
        conclusion = (f"No fair value was issued. {fair_value.action}. "
                      "Use the coverage table and diagnostics as a data-gathering checklist rather than a verdict.")
    else:
        price = data.metrics.get("current_price")
        direction = ("upside" if (fair_value.upside_downside or 0) >= 0 else "downside")
        implied = ""
        if fair_value.dcf.implied_growth is not None:
            implied = (f" At the current price the market is pricing roughly "
                       f"{fair_value.dcf.implied_growth:.1%} annual cash-flow growth")
            if fair_value.dcf.implied_vs_actual is not None:
                realized = fair_value.dcf.implied_growth - fair_value.dcf.implied_vs_actual
                implied += f", against {realized:.1%} realized over the last three years"
            implied += "."
        conclusion = (
            f"Business quality scores {quality:.0f}/100. The economic fair value is "
            f"{price_text(fair_value.base, data.currency)} per share against a market price of "
            f"{price_text(price, data.currency)}, i.e. {abs(fair_value.upside_downside or 0):.1%} {direction}. "
            f"With a required margin of safety of {(fair_value.margin_of_safety or 0):.0%}, the buy-below price is "
            f"{price_text(fair_value.buy_below, data.currency)}.{implied} "
            f"Treat the {score_low:.0f}-{score_high:.0f} score band as a screening range, not a return forecast."
        )

    return AnalysisResult(
        categories=categories, overall=overall, score_low=score_low, score_high=score_high,
        business_quality=quality, confidence=confidence, data_coverage=coverage,
        data_freshness=freshness, model_confidence=model_confidence, model_fit=model_fit,
        integrity_score=integrity, fair_value=fair_value, reliable=reliable,
        strengths=strengths, concerns=concerns, conclusion=conclusion,
    )


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------

CURRENCY_SYMBOLS = {"USD": "$", "EUR": "EUR ", "GBP": "GBP ", "CAD": "C$", "AUD": "A$",
                    "JPY": "JPY ", "CHF": "CHF ", "INR": "INR ", "HKD": "HK$"}


def money(value: Optional[float], currency: str) -> str:
    if value is None:
        return "N/A"
    prefix = CURRENCY_SYMBOLS.get(currency.upper(), currency + " ")
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    for divisor, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if magnitude >= divisor:
            return f"{sign}{prefix}{magnitude / divisor:,.2f}{suffix}"
    return f"{sign}{prefix}{magnitude:,.2f}"


def price_text(value: Optional[float], currency: str) -> str:
    if value is None:
        return "N/A"
    return f"{CURRENCY_SYMBOLS.get(currency.upper(), currency + ' ')}{value:,.2f}"


def pct(value: Optional[float], digits: int = 1) -> str:
    return "N/A" if value is None else f"{value:.{digits}%}"


def multiple(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.2f}x"


def rating_label(score: float) -> str:
    if score >= 85:
        return "HIGH-QUALITY SCREEN"
    if score >= 72:
        return "ABOVE-AVERAGE SCREEN"
    if score >= 58:
        return "MIXED / WATCHLIST"
    if score >= 45:
        return "BELOW-AVERAGE SCREEN"
    return "HIGH-RISK SCREEN"


class TerminalStyle:
    CODES = {"title": "\033[96;1m", "bold": "\033[97;1m", "good": "\033[92m",
             "neutral": "\033[93m", "bad": "\033[91m", "value": "\033[95m",
             "info": "\033[94m", "dim": "\033[2;37m"}

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def apply(self, value: str, tone: Optional[str] = None) -> str:
        if not self.enabled or tone not in self.CODES:
            return value
        return self.CODES[tone] + value + "\033[0m"


class ReportPrinter:
    def __init__(self, use_color: bool, width: Optional[int] = None) -> None:
        self.width = max(76, min(width or shutil.get_terminal_size((96, 24)).columns, 100))
        self.style = TerminalStyle(use_color)
        encoding = (getattr(sys.stdout, "encoding", None) or "utf-8").lower()
        try:
            "\u2500\u2554\u2557\u255a\u2550\u2551\u2588\u2591".encode(encoding)
            self.unicode = True
        except (LookupError, UnicodeEncodeError):
            self.unicode = False

    def line(self, value: str = "", tone: Optional[str] = None) -> None:
        print(self.style.apply(value, tone))

    def rule(self) -> None:
        self.line(("\u2500" if self.unicode else "-") * self.width, "dim")

    def section(self, title: str) -> None:
        self.line()
        self.line(title.upper(), "title")
        self.rule()

    def wrapped(self, value: str, prefix: str = "", tone: Optional[str] = None) -> None:
        lines = textwrap.wrap(value, width=max(20, self.width - len(prefix)),
                              initial_indent=prefix, subsequent_indent=" " * len(prefix),
                              break_long_words=False)
        for item in lines or [prefix.rstrip()]:
            self.line(item, tone)

    def key_values(self, rows: Iterable[tuple[str, str, Optional[str]]]) -> None:
        for label, value, tone in rows:
            raw = label + ":"
            prefix = raw + " " if len(raw) >= 31 else f"{raw:<31}"
            if len(prefix) + len(value) <= self.width:
                self.line(prefix + value, tone)
            else:
                self.wrapped(value, prefix, tone)

    def box(self, rows: Iterable[tuple[str, Optional[str]]]) -> None:
        inner = self.width - 4
        corners = ("\u2554", "\u2550", "\u2557", "\u255a", "\u255d", "\u2551") if self.unicode else ("+", "-", "+", "+", "+", "|")
        tl, hz, tr, bl, br, side = corners
        self.line(tl + hz * (self.width - 2) + tr, "title")
        for value, tone in rows:
            for piece in textwrap.wrap(value, width=inner, break_long_words=False) or [""]:
                self.line(f"{side} {piece:<{inner}} {side}", tone)
        self.line(bl + hz * (self.width - 2) + br, "title")


def score_bar(value: float, maximum: float = 100.0, cells: int = 22, unicode_ok: bool = True) -> str:
    ratio = clamp(divide(value, maximum) or 0.0)
    filled = round(ratio * cells)
    full, empty = ("\u2588", "\u2591") if unicode_ok else ("#", ".")
    return "[" + full * filled + empty * (cells - filled) + "]"


def score_tone(value: float, maximum: float = 100.0) -> str:
    ratio = divide(value, maximum) or 0.0
    return "good" if ratio >= 0.80 else "neutral" if ratio >= 0.60 else "bad"


def action_tone(action: str) -> str:
    upper = action.upper()
    if "STRONG BUY" in upper or "BUY ZONE" in upper:
        return "good"
    if "INCONCLUSIVE" in upper or "INSUFFICIENT" in upper or "VERIFY" in upper or "SPECIALIZED" in upper:
        return "neutral"
    if "ABOVE" in upper:
        return "bad"
    return "neutral"


def terminal_color_enabled(no_color: bool) -> bool:
    if no_color or os.environ.get("NO_COLOR") is not None or not sys.stdout.isatty():
        return False
    return os.environ.get("TERM", "").lower() != "dumb"


def print_report(data: StockData, result: AnalysisResult, detail: str = "standard",
                 use_color: bool = False) -> None:
    m, fv, f = data.metrics, result.fair_value, data.forensics
    p = ReportPrinter(use_color)

    relation = "not issued"
    if fv.discount_premium is not None:
        relation = f"{abs(fv.discount_premium):.1%} {'discount' if fv.discount_premium >= 0 else 'premium'} to fair value"

    p.line()
    p.box((
        (f"{data.company_name.upper()} ({data.symbol})  -  VALUATION MODEL V5", "title"),
        (f"Current price      {price_text(m.get('current_price'), data.currency)}", None),
        (f"Fair value         {price_text(fv.base, data.currency)}   ({relation})",
         "value" if fv.base is not None else "bad"),
        (f"BUY BELOW          {price_text(fv.buy_below, data.currency)}   "
         f"(requires a {pct(fv.margin_of_safety, 0)} margin of safety)",
         "good" if fv.buy_below is not None else "bad"),
        (f"Strong-buy below   {price_text(fv.strong_buy_below, data.currency)}", "good"),
        (f"Verdict            {fv.status}",
         "good" if "UNDER" in fv.status else "bad" if "OVER" in fv.status else "neutral"),
        (f"Action             {fv.action}", action_tone(fv.action)),
        (f"Quality {result.business_quality:.0f}/100   Score {result.overall:.0f}/100   "
         f"Model confidence {result.model_confidence:.0f}%", score_tone(result.model_confidence)),
    ))

    if fv.base is not None:
        p.section("Price zones")
        p.key_values((
            ("Strong buy (max safety)", f"below {price_text(fv.strong_buy_below, data.currency)}", "good"),
            ("Buy zone", f"{price_text(fv.strong_buy_below, data.currency)} - {price_text(fv.buy_below, data.currency)}", "good"),
            ("Below fair, thin margin", f"{price_text(fv.buy_below, data.currency)} - {price_text(fv.base, data.currency)}", "neutral"),
            ("Fair to optimistic", f"{price_text(fv.base, data.currency)} - {price_text(fv.high, data.currency)}", "neutral"),
            ("Above the estimated range", f"above {price_text(fv.high, data.currency)}", "bad"),
            ("Bear / base / bull", f"{price_text(fv.low, data.currency)} / {price_text(fv.base, data.currency)} / {price_text(fv.high, data.currency)}", "value"),
            ("Upside to fair value", f"{fv.upside_downside:+.1%}" if fv.upside_downside is not None else "N/A",
             "good" if (fv.upside_downside or 0) >= 0 else "bad"),
            ("Analyst target (reference)", price_text(fv.analyst_reference, data.currency), "dim"),
        ))
        p.wrapped("Zones are valuation references. They are not technical levels and not instructions to trade.", tone="dim")

    p.section("What the market is pricing in (reverse DCF)")
    if fv.dcf.implied_growth is not None:
        realized3 = m.get("revenue_cagr3")
        realized5 = m.get("revenue_cagr5")
        gap = fv.dcf.implied_vs_actual
        verdict = "no view"
        if gap is not None:
            verdict = ("the market demands faster growth than recently delivered" if gap > 0.02
                       else "the market demands less growth than recently delivered" if gap < -0.02
                       else "the market is roughly extrapolating recent growth")
        p.key_values((
            ("Implied 5Y cash-flow growth", pct(fv.dcf.implied_growth), "value"),
            ("Realized revenue CAGR 3Y / 5Y", f"{pct(realized3)} / {pct(realized5)}", None),
            ("Implied minus realized", pct(gap) if gap is not None else "N/A",
             "bad" if (gap or 0) > 0.05 else "good"),
            ("Reading", verdict, "neutral"),
        ))
        p.wrapped("This is the most falsifiable figure in the report: it is the growth rate that makes "
                  "today's price exactly fair under the stated WACC and terminal growth.", tone="dim")
    else:
        p.wrapped("A reverse DCF could not be solved (non-positive or unavailable owner cash flow).", tone="dim")

    p.section("Valuation methods")
    p.line(f"{'Method':<30}{'Family':<12}{'Value':>11}{'Rel.':>7}{'Weight':>8}  Note")
    p.rule()
    for method in fv.methods:
        tone = "neutral" if "outlier" in method.status else "good"
        p.line(f"{method.name[:29]:<30}{method.family:<12}{price_text(method.value, data.currency):>11}"
               f"{method.reliability:>7.0%}{method.effective_weight:>8.0%}  {method.note or method.status}", tone)
    if fv.family_values:
        p.rule()
        for family, value in sorted(fv.family_values.items()):
            p.line(f"{'family: ' + family:<30}{'':<12}{price_text(value, data.currency):>11}", "dim")
    p.rule()
    p.key_values((
        ("Cross-family agreement", pct((fv.family_agreement or 0), 0), score_tone(100 * (fv.family_agreement or 0))),
        ("High / low family ratio", f"{fv.family_ratio:.2f}x" if fv.family_ratio else "N/A", "neutral"),
        ("Decision basis", fv.decision_basis or fv.action, "neutral"),
    ))

    if detail != "compact":
        p.section("Cost of capital and DCF")
        a = fv.assumptions_used
        p.key_values((
            ("Risk-free rate", f"{a.risk_free:.2%}  ({a.source})", "info"),
            ("Beta / equity risk premium", f"{a.beta:.2f} / {a.equity_risk_premium:.2%}", "info"),
            ("Cost of equity", f"{a.cost_of_equity:.2%}", "info"),
            ("After-tax cost of debt", f"{a.cost_of_debt * (1 - a.tax_rate):.2%}", "info"),
            ("Capital weights (E / D)", f"{a.equity_weight:.0%} / {a.debt_weight:.0%}", "info"),
            ("WACC", f"{a.wacc:.2%}", "value"),
            ("Terminal growth", f"{a.terminal_growth:.2%}", "info"),
            ("Effective tax rate", f"{a.tax_rate:.1%}", "info"),
        ))
        d = fv.dcf
        p.key_values((
            ("Starting FCFF (unlevered)", money(d.starting_fcff, data.currency), None),
            ("FCFE cross-check (levered)", money(d.fcfe_crosscheck, data.currency), None),
            ("Cross-check gap", pct(d.reconciliation_gap) if d.reconciliation_gap is not None else "N/A",
             "bad" if (d.reconciliation_gap or 0) > 0.35 else "good"),
            ("Normalized SBC deducted", money(fv.normalized_sbc, data.currency), None),
            ("Base-case growth used", pct(d.growth), None),
            ("PV of explicit years 1-10", money(d.explicit_pv, data.currency), None),
            ("PV of terminal value", money(d.terminal_pv, data.currency), None),
            ("Terminal share of value", pct(d.terminal_share),
             "bad" if (d.terminal_share or 0) > MAX_TERMINAL_SHARE else "neutral"),
            ("Enterprise value", money(d.enterprise_value, data.currency), None),
            ("Less net debt", money(d.net_debt, data.currency), None),
            ("Equity value", money(d.equity_value, data.currency), None),
            ("DCF value per share", price_text(d.per_share, data.currency), "value"),
        ))

        p.section("Forensic accounting screens")
        p.key_values((
            ("Piotroski F-Score", f"{f.piotroski}/9" if f.piotroski is not None else "unavailable",
             "good" if (f.piotroski or 0) >= 7 else "bad" if (f.piotroski is not None and f.piotroski <= 3) else "neutral"),
            ("Altman Z-Score", f"{f.altman_z:.2f}  ({f.altman_zone})" if f.altman_z is not None else f.altman_zone,
             "good" if f.altman_zone == "safe zone" else "bad" if f.altman_zone == "distress zone" else "neutral"),
            ("Beneish M-Score", f"{f.beneish_m:.2f}  ({f.beneish_flag})" if f.beneish_m is not None else f.beneish_flag,
             "bad" if f.beneish_flag == "elevated manipulation risk" else "good" if f.beneish_m is not None else "neutral"),
        ))
        if detail == "detailed" and f.piotroski_detail:
            for line in f.piotroski_detail:
                p.line(line, "dim")

        p.section("Business fundamentals")
        p.key_values((
            ("Revenue / 3Y / 5Y CAGR", f"{money(m.get('revenue'), data.currency)} / {pct(m.get('revenue_cagr3'))} / {pct(m.get('revenue_cagr5'))}", None),
            ("Gross / operating / net margin", f"{pct(m.get('gross_margin'))} / {pct(m.get('operating_margin'))} / {pct(m.get('profit_margin'))}", None),
            ("ROIC / ROE / ROA", f"{pct(m.get('roic'))} / {pct(m.get('return_on_equity'))} / {pct(m.get('return_on_assets'))}", None),
            ("R&D / SG&A intensity", f"{pct(m.get('rnd_intensity'))} / {pct(m.get('sga_intensity'))}", None),
            ("Cash / debt / net debt", f"{money(m.get('cash'), data.currency)} / {money(m.get('debt'), data.currency)} / {money(m.get('net_debt'), data.currency)}", None),
            ("Net debt / EBITDA", multiple(m.get("net_debt_to_ebitda")), None),
            ("Current / quick ratio", f"{multiple(m.get('current_ratio'))} / {multiple(m.get('quick_ratio'))}", None),
            ("DSO / DIO / DPO / CCC", f"{m.get('days_sales_outstanding'):.0f} / {m.get('days_inventory'):.0f} / "
             f"{m.get('days_payable'):.0f} / {m.get('cash_conversion_cycle'):.0f} days"
             if None not in (m.get('days_sales_outstanding'), m.get('days_inventory'),
                             m.get('days_payable'), m.get('cash_conversion_cycle')) else "N/A", None),
            ("Trailing / forward P/E", f"{multiple(m.get('trailing_pe'))} / {multiple(m.get('forward_pe'))}", None),
            ("P/FCF / P/S / P/B", f"{multiple(m.get('price_to_fcf'))} / {multiple(m.get('price_to_sales'))} / {multiple(m.get('price_to_book'))}", None),
            ("EV/EBITDA / EV/EBIT", f"{multiple(m.get('ev_to_ebitda'))} / {multiple(m.get('ev_to_ebit'))}", None),
            ("FCF yield / earnings yield", f"{pct(m.get('fcf_yield'))} / {pct(m.get('earnings_yield'))}", None),
        ))

    p.section("Scorecard")
    labels = {"profitability": "Profitability", "growth": "Growth", "financial_health": "Financial health",
              "valuation": "Market valuation", "cash_accounting": "Cash & accounting", "risk_data": "Risk & data"}
    for key, category in result.categories.items():
        p.line(f"{labels[key]:<22}{category.points:>5.1f}/{category.maximum:<5.0f} coverage {category.coverage:>4.0%}",
               score_tone(category.points, category.maximum))
    p.rule()
    for label, value in (("Business quality", result.business_quality),
                         ("Overall score", result.overall),
                         ("Model confidence", result.model_confidence),
                         ("Statement integrity", result.integrity_score)):
        p.line(f"{label:<20}{score_bar(value, 100, unicode_ok=p.unicode)} {value:5.1f}/100", score_tone(value))
    p.line(f"Score band: {result.score_low:.0f}-{result.score_high:.0f}/100   "
           f"({rating_label(result.overall)})", "dim")

    if detail != "compact":
        p.section("Model reliability")
        p.key_values((
            ("Data coverage", f"{result.data_coverage:.0f}%", score_tone(result.data_coverage)),
            ("Data freshness", f"{result.data_freshness:.0f}%", score_tone(result.data_freshness)),
            ("Independent input confidence", f"{result.confidence:.0f}%", score_tone(result.confidence)),
            ("Statement integrity", f"{result.integrity_score:.0f}%", score_tone(result.integrity_score)),
            ("Model fit for this business", f"{result.model_fit:.0f}%", score_tone(result.model_fit)),
            ("Peers used", ", ".join(data.peers_used) if data.peers_used else "none supplied", "info"),
            ("Overall rating reliable", "yes" if result.reliable else "no - treat as a data checklist",
             "good" if result.reliable else "bad"),
        ))

        p.section("Integrity checks")
        for check in data.integrity or []:
            p.wrapped(f"{check.name}: {check.detail}", prefix="PASS  " if check.passed else "FAIL  ",
                      tone="good" if check.passed else "bad")
        if not data.integrity:
            p.wrapped("No integrity check could be run with the available statements.", tone="dim")

        p.section("Cross-source validation")
        if data.validation:
            for check in data.validation:
                prefix = {"confirmed": "OK    ", "review": "?     ", "conflict": "FAIL  "}[check.verdict]
                tone = {"confirmed": "good", "review": "neutral", "conflict": "bad"}[check.verdict]
                p.wrapped(f"{check.key}: {price_text(check.primary, data.currency) if check.key == 'current_price' else money(check.primary, data.currency)} "
                          f"vs {check.source} - {check.detail}", prefix=prefix, tone=tone)
        else:
            p.wrapped("No secondary source was available for this symbol (non-US filer, "
                      "non-USD statements, network failure, or --no-secondary).", tone="dim")

        p.section("Valuation diagnostics")
        for item in dict.fromkeys(fv.diagnostics) or ["No valuation diagnostic was triggered."]:
            p.wrapped(item, prefix="! ", tone="bad" if fv.base is None else "neutral")

        p.section("Strengths")
        for item in result.strengths or ["No quantitative strength was confirmed from the available inputs."]:
            p.wrapped(item, prefix="+ ", tone="good")
        p.section("Concerns")
        for item in result.concerns or ["No quantitative concern triggered; qualitative risks still need review."]:
            p.wrapped(item, prefix="! ", tone="bad")

    if detail == "detailed":
        p.section("Assumptions")
        for item in fv.assumptions:
            p.wrapped(item, prefix="- ", tone="dim")
        p.section("Metric sources and periods")
        for key in sorted(data.meta):
            meta = data.meta[key]
            p.wrapped(f"{key}: {meta.source}; {meta.period}; reliability {meta.reliability:.0%}; "
                      f"as of {meta.as_of.date() if meta.as_of is not None else 'N/A'}", tone="dim")
        if data.warnings:
            p.section("Raw data warnings")
            for warning in dict.fromkeys(data.warnings):
                p.wrapped(warning, prefix="! ", tone="bad")

    p.section("Bottom line")
    p.wrapped(result.conclusion)
    p.line()
    p.key_values((
        ("Fair value", price_text(fv.base, data.currency), "value"),
        ("BUY BELOW", price_text(fv.buy_below, data.currency), "good"),
        ("Current price", price_text(m.get("current_price"), data.currency), None),
        ("Verdict", fv.status, "good" if "UNDER" in fv.status else "bad" if "OVER" in fv.status else "neutral"),
        ("Action", fv.action, action_tone(fv.action)),
    ))
    p.line()
    p.wrapped("Educational quantitative screen only. Not investment advice. Verify the filings and the "
              "assumptions above before acting on anything here.", tone="dim")
    p.rule()


# --------------------------------------------------------------------------
# JSON, snapshots, backtesting
# --------------------------------------------------------------------------

def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.ndarray, pd.Series)):
        return value.tolist()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return str(value)


def result_to_dict(data: StockData, result: AnalysisResult) -> dict[str, Any]:
    fv = result.fair_value
    f = data.forensics
    return {
        "symbol": data.symbol,
        "company_name": data.company_name,
        "sector": data.sector,
        "industry": data.industry,
        "currency": data.currency,
        "retrieved_at": data.retrieved_at,
        "price": data.metrics.get("current_price"),
        "special_model": data.special_model,
        "peers_used": data.peers_used,
        "decision": {
            "fair_value": fv.base,
            "buy_below": fv.buy_below,
            "strong_buy_below": fv.strong_buy_below,
            "margin_of_safety_required": fv.margin_of_safety,
            "bear": fv.low,
            "bull": fv.high,
            "upside_downside": fv.upside_downside,
            "discount_premium": fv.discount_premium,
            "status": fv.status,
            "action": fv.action,
            "decision_basis": fv.decision_basis,
        },
        "reverse_dcf": {
            "implied_growth": fv.dcf.implied_growth,
            "realized_revenue_cagr3": data.metrics.get("revenue_cagr3"),
            "realized_revenue_cagr5": data.metrics.get("revenue_cagr5"),
            "implied_minus_realized": fv.dcf.implied_vs_actual,
        },
        "cost_of_capital": asdict(fv.assumptions_used),
        "dcf": asdict(fv.dcf),
        "scores": {
            "overall": round(result.overall, 2),
            "score_low": round(result.score_low, 2),
            "score_high": round(result.score_high, 2),
            "business_quality": round(result.business_quality, 2),
            "rating_label": rating_label(result.overall),
        },
        "forensics": {
            "piotroski_f": f.piotroski,
            "altman_z": f.altman_z,
            "altman_zone": f.altman_zone,
            "beneish_m": f.beneish_m,
            "beneish_flag": f.beneish_flag,
        },
        "reliability": {
            "input_confidence": round(result.confidence, 1),
            "data_coverage": round(result.data_coverage, 1),
            "data_freshness": round(result.data_freshness, 1),
            "statement_integrity": round(result.integrity_score, 1),
            "model_fit": round(result.model_fit, 1),
            "model_confidence": round(result.model_confidence, 1),
            "cross_family_agreement": fv.family_agreement,
            "family_ratio": fv.family_ratio,
            "reliable": result.reliable,
        },
        "categories": {k: {"points": round(c.points, 2), "maximum": round(c.maximum, 2),
                           "coverage": round(c.coverage, 3)} for k, c in result.categories.items()},
        "methods": [{"name": mt.name, "family": mt.family, "value": mt.value, "bear": mt.bear,
                     "bull": mt.bull, "reliability": round(mt.reliability, 3),
                     "effective_weight": round(mt.effective_weight, 3), "status": mt.status}
                    for mt in fv.methods],
        "family_values": fv.family_values,
        "integrity_checks": [asdict(c) for c in data.integrity],
        "cross_source_validation": [asdict(c) for c in data.validation],
        "strengths": result.strengths,
        "concerns": result.concerns,
        "diagnostics": list(dict.fromkeys(fv.diagnostics)),
        "conclusion": result.conclusion,
        "warnings": list(dict.fromkeys(data.warnings)),
    }


SNAPSHOT_FIELDS = [
    "date", "ticker", "price", "score", "score_low", "score_high", "business_quality",
    "confidence", "data_coverage", "data_freshness", "statement_integrity", "model_fit",
    "model_confidence", "cross_family_agreement", "fair_value", "buy_below",
    "margin_of_safety", "upside_downside", "discount_premium", "valuation_status",
    "action", "implied_growth", "realized_cagr3", "piotroski_f", "altman_z", "beneish_m",
    "bear", "bull",
    *[f"cat_{key}" for key in CATEGORY_MAXIMUMS],
    "future_return", "benchmark_return",
]


def append_snapshot(path: str, data: StockData, result: AnalysisResult) -> None:
    """Point-in-time record. Downloading today's fundamentals for an old date
    would be look-ahead bias, so the only valid backtest is one accumulated
    forward, one run at a time.

    If the file already exists with an older column set, its header wins and
    new columns are silently dropped for that file - never corrupt an
    accumulating history by shifting columns mid-file."""
    destination = Path(path)
    exists = destination.exists()
    fieldnames = SNAPSHOT_FIELDS
    if exists:
        try:
            with destination.open("r", newline="", encoding="utf-8") as handle:
                first = handle.readline().strip()
            existing = [c.strip() for c in first.split(",")] if first else []
            if existing and "date" in existing:
                fieldnames = existing
        except OSError:
            pass
    fv, f = result.fair_value, data.forensics
    row = {
        "date": data.retrieved_at.date().isoformat(), "ticker": data.symbol,
        "price": data.metrics.get("current_price"),
        "score": round(result.overall, 4), "score_low": round(result.score_low, 4),
        "score_high": round(result.score_high, 4), "business_quality": round(result.business_quality, 4),
        "confidence": round(result.confidence, 4), "data_coverage": round(result.data_coverage, 4),
        "data_freshness": round(result.data_freshness, 4),
        "statement_integrity": round(result.integrity_score, 4),
        "model_fit": round(result.model_fit, 4), "model_confidence": round(result.model_confidence, 4),
        "cross_family_agreement": fv.family_agreement, "fair_value": fv.base,
        "buy_below": fv.buy_below, "margin_of_safety": fv.margin_of_safety,
        "upside_downside": fv.upside_downside, "discount_premium": fv.discount_premium,
        "valuation_status": fv.status, "action": fv.action,
        "implied_growth": fv.dcf.implied_growth, "realized_cagr3": data.metrics.get("revenue_cagr3"),
        "piotroski_f": f.piotroski, "altman_z": f.altman_z, "beneish_m": f.beneish_m,
        "bear": fv.low, "bull": fv.high,
        "future_return": "", "benchmark_return": "",
    }
    for key, category in result.categories.items():
        row[f"cat_{key}"] = round(100 * category.points / category.maximum, 4) if category.maximum else ""
    with destination.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", restval="")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"\nPoint-in-time snapshot appended to {destination}.")


def block_bootstrap_ci(frame: pd.DataFrame, column: str, iterations: int = 2000,
                       seed: int = 7) -> tuple[float, float]:
    """Confidence interval for the score/return rank correlation, resampling by
    DATE block rather than by row. Observations sharing a date are not
    independent, and pretending otherwise inflates significance badly."""
    rng = np.random.default_rng(seed)
    if "date" in frame.columns:
        blocks = [group for _, group in frame.groupby("date")]
    else:
        blocks = [frame.iloc[[i]] for i in range(len(frame))]
    if len(blocks) < 3:
        return float("nan"), float("nan")
    estimates: list[float] = []
    for _ in range(iterations):
        picked = rng.integers(0, len(blocks), size=len(blocks))
        sample = pd.concat([blocks[i] for i in picked], ignore_index=True)
        if sample[column].nunique() < 5 or sample["excess_return"].nunique() < 5:
            continue
        value = sample[[column, "excess_return"]].corr(method="spearman").iloc[0, 1]
        if math.isfinite(value):
            estimates.append(float(value))
    if len(estimates) < 100:
        return float("nan"), float("nan")
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def run_backtest(path: str, calibrate_out: Optional[str] = None) -> int:
    frame = pd.read_csv(path)
    required = {"date", "ticker", "score", "future_return"}
    missing = required - set(frame.columns)
    if missing:
        print("The backtest file is missing: " + ", ".join(sorted(missing)))
        return 1
    for column in ("score", "future_return"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["score", "future_return"])
    if len(frame) < 20:
        print("At least 20 completed point-in-time observations are needed for even a rough check.")
        return 1

    frame["excess_return"] = frame["future_return"]
    if "benchmark_return" in frame:
        benchmark = pd.to_numeric(frame["benchmark_return"], errors="coerce")
        frame["excess_return"] = frame["future_return"] - benchmark.fillna(0)

    dates = frame["date"].nunique() if "date" in frame.columns else len(frame)
    tickers = frame["ticker"].nunique() if "ticker" in frame.columns else len(frame)
    spearman = frame[["score", "excess_return"]].corr(method="spearman").iloc[0, 1]
    pearson = frame[["score", "excess_return"]].corr(method="pearson").iloc[0, 1]
    lower, upper = block_bootstrap_ci(frame, "score")

    frame["quintile"] = pd.qcut(frame["score"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    grouped = frame.groupby("quintile", observed=True)["excess_return"].agg(["count", "mean", "median"])
    top = frame[frame["quintile"] == 5]["excess_return"]
    bottom = frame[frame["quintile"] == 1]["excess_return"]
    spread = top.mean() - bottom.mean()

    print("\nPOINT-IN-TIME CALIBRATION REPORT")
    print(f"Observations:              {len(frame)}")
    print(f"Distinct dates / tickers:  {dates} / {tickers}")
    print(f"Spearman rank correlation: {spearman: .3f}")
    if math.isfinite(lower):
        print(f"  date-block bootstrap 95% CI: [{lower: .3f}, {upper: .3f}]")
        if lower <= 0 <= upper:
            print("  The interval spans zero: this sample does NOT establish predictive skill.")
    else:
        print("  Too few distinct dates for a meaningful bootstrap interval.")
    print(f"Pearson correlation:       {pearson: .3f}")
    print(f"Top-minus-bottom spread:   {spread: .2%}")
    print(f"Top-quintile hit rate:     {float((top > 0).mean()): .2%}")
    if dates < 12:
        print("\nWarning: fewer than 12 distinct observation dates. Results are dominated by a few "
              "market environments and should not be read as evidence of skill.")
    print("\nExcess return by score quintile:")
    print(grouped.to_string(formatters={"mean": lambda x: f"{x:.2%}", "median": lambda x: f"{x:.2%}"}))

    # Additional signals worth tracking separately from the composite.
    for column, label in (("implied_growth", "reverse-DCF implied growth"),
                          ("discount_premium", "discount to fair value"),
                          ("piotroski_f", "Piotroski F-Score"),
                          ("business_quality", "business quality")):
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = pd.DataFrame({"x": values, "excess_return": frame["excess_return"]}).dropna()
        if len(valid) >= 20:
            corr = float(valid.corr(method="spearman").iloc[0, 1])
            print(f"{label:<32}rank correlation {corr: .3f}  (n={len(valid)})")

    correlations: dict[str, float] = {}
    category_columns = [f"cat_{key}" for key in CATEGORY_MAXIMUMS if f"cat_{key}" in frame.columns]
    if category_columns:
        print("\nCategory rank correlations:")
        for column in category_columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            valid = pd.DataFrame({"x": values, "excess_return": frame["excess_return"]}).dropna()
            corr = float(valid.corr(method="spearman").iloc[0, 1]) if len(valid) >= 15 else float("nan")
            key = column.removeprefix("cat_")
            correlations[key] = corr
            print(f"  {key:<22}{corr: .3f}" if math.isfinite(corr) else f"  {key:<22} insufficient data")

    # Realized-error check on the published bear/bull bands: compare the
    # dispersion of (realized return - predicted upside) against the half-width
    # of the bands the model published on those same dates. Robust spread
    # (MAD-based) by date block, so one crash date cannot set the scale.
    band_scale: Optional[float] = None
    band_columns = {"bear", "bull", "fair_value", "upside_downside"}
    if band_columns.issubset(frame.columns):
        numeric = {c: pd.to_numeric(frame[c], errors="coerce") for c in band_columns}
        half_band = ((numeric["bull"] - numeric["bear"]) / (2 * numeric["fair_value"])).replace(
            [np.inf, -np.inf], np.nan)
        residual = (frame["future_return"] - numeric["upside_downside"]).replace(
            [np.inf, -np.inf], np.nan)
        valid = pd.DataFrame({"half_band": half_band, "residual": residual,
                              "date": frame.get("date")}).dropna(subset=["half_band", "residual"])
        valid = valid[valid["half_band"] > 0.01]
        if len(valid) >= 30 and valid["date"].nunique() >= 8:
            mad = float((valid["residual"] - valid["residual"].median()).abs().median())
            robust_spread = 1.4826 * mad
            median_band = float(valid["half_band"].median())
            if median_band > 0 and math.isfinite(robust_spread):
                band_scale = float(clamp(robust_spread / median_band, *DCF_BAND_SCALE_RANGE))
                verdict = ("too narrow - widen" if band_scale > 1.15
                           else "too wide - narrow" if band_scale < 0.85 else "about right")
                print(f"\nSensitivity-band check: realized robust error {robust_spread:.1%} vs "
                      f"published half-band {median_band:.1%} -> band scale {band_scale:.2f} ({verdict}).")
        else:
            print("\nSensitivity-band check: not enough bear/bull observations yet "
                  "(needs 30+ rows across 8+ dates with the bear/bull columns filled).")

    # Curve-shape signal per category: the rank correlation earned above maps
    # to a curve exponent. Strong ordering steepens the curve; absent ordering
    # flattens it. This is intentionally the ONLY shape parameter fitted -
    # fitting individual breakpoints to a sample this small would be pure
    # curve-fitting.
    curve_gamma: dict[str, float] = {}
    for key, corr in correlations.items():
        if math.isfinite(corr):
            curve_gamma[key] = round(float(clamp(1.0 + 2.5 * (corr - 0.05), *CURVE_GAMMA_RANGE)), 4)

    if calibrate_out:
        if dates < 20 or len(frame) < 200:
            print("\nRefusing to emit a calibration file: at least 200 observations across 20 distinct "
                  "dates are required. Fitting parameters on less than that is curve-fitting, not calibration.")
            return 1
        if set(correlations) != set(CATEGORY_MAXIMUMS) or not any(math.isfinite(v) for v in correlations.values()):
            print("\nCould not build a calibration file: every category column needs enough observations.")
            return 1
        signals = {k: max(correlations[k], 0.02) if math.isfinite(correlations[k]) else 0.02
                   for k in CATEGORY_MAXIMUMS}
        total = sum(signals.values())
        candidate = {k: clamp(100 * v / total, 5.0, 30.0) for k, v in signals.items()}
        renorm = sum(candidate.values())
        candidate = {k: round(100 * v / renorm, 4) for k, v in candidate.items()}
        payload = {
            "meta": {
                "generated": datetime.now().astimezone().isoformat(),
                "observations": int(len(frame)),
                "distinct_dates": int(dates),
                "distinct_tickers": int(tickers),
                "spearman": round(float(spearman), 4) if math.isfinite(spearman) else None,
                "note": "Validate on a holdout of separate dates AND tickers before use.",
            },
            "category_weights": candidate,
            "curve_gamma": {k: curve_gamma.get(k, 1.0) for k in CATEGORY_MAXIMUMS},
            "dcf_band_scale": band_scale if band_scale is not None else 1.0,
        }
        Path(calibrate_out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nCalibration written to {calibrate_out} (weights + curve exponents + band scale). "
              "Validate it on a separate holdout of dates AND tickers before passing --calibration-json.")

    if not math.isfinite(spearman) or spearman < 0.10 or spread <= 0:
        print("\nCalibration warning: the score ordering has weak or negative out-of-sample evidence here. "
              "Do not raise your confidence in the rating.")
    else:
        print("\nThe calibration signal is positive, but confirm it on separate dates and securities "
              "before changing any weights.")
    return 0


# --------------------------------------------------------------------------
# Self-test (no network required)
# --------------------------------------------------------------------------

def _synthetic_company() -> StockData:
    """A deterministic fixture used to exercise the valuation path offline."""
    dates = [pd.Timestamp(f"20{year}-12-31") for year in (24, 23, 22, 21, 20)]
    data = StockData(
        symbol="TEST", company_name="Test Manufacturing", sector="Industrials",
        industry="Specialty Industrial Machinery", quote_type="EQUITY",
        currency="USD", financial_currency="USD", currency_compatible=True,
        retrieved_at=datetime(2025, 3, 1).astimezone(), price_timestamp=None,
    )
    series = lambda values: pd.Series(dict(zip(dates, values)), dtype=float).sort_index(ascending=False)
    data.annual = {
        "revenue": series([1000, 920, 850, 780, 700]),
        "cost_of_revenue": series([600, 560, 525, 490, 445]),
        "gross_profit": series([400, 360, 325, 290, 255]),
        "operating_income": series([180, 160, 143, 125, 108]),
        "ebit": series([180, 160, 143, 125, 108]),
        "net_income": series([130, 115, 102, 88, 75]),
        "pretax_income": series([170, 150, 133, 115, 98]),
        "tax_expense": series([40, 35, 31, 27, 23]),
        "ocf": series([190, 172, 155, 138, 120]),
        "capex": series([55, 50, 48, 45, 42]),
        "depreciation": series([45, 42, 40, 38, 36]),
        "working_capital_change": series([-12, -10, -9, -8, -7]),
        "sbc": series([18, 16, 14, 12, 10]),
        "assets": series([1400, 1300, 1210, 1120, 1040]),
        "liabilities": series([700, 670, 640, 610, 580]),
        "equity": series([700, 630, 570, 510, 460]),
        "retained_earnings": series([520, 460, 405, 355, 310]),
        "cash": series([220, 190, 165, 140, 120]),
        "debt": series([300, 300, 300, 300, 300]),
        "long_term_debt": series([260, 265, 270, 275, 280]),
        "current_assets": series([560, 520, 480, 445, 410]),
        "current_liabilities": series([320, 305, 290, 275, 262]),
        "receivables": series([150, 138, 128, 118, 106]),
        "payables": series([110, 102, 96, 90, 84]),
        "inventory": series([130, 122, 115, 108, 100]),
        "net_ppe": series([420, 400, 385, 370, 355]),
        "short_term_investments": series([60, 50, 45, 40, 35]),
        "sga": series([160, 148, 138, 128, 117]),
        "diluted_shares": series([100, 101, 102, 103, 104]),
        "shares": series([100, 101, 102, 103, 104]),
        "diluted_eps": series([1.30, 1.14, 1.00, 0.85, 0.72]),
    }
    for key, value, reliability in (
        ("revenue", 1000.0, 0.95), ("cost_of_revenue", 600.0, 0.95), ("gross_profit", 400.0, 0.95),
        ("operating_income", 180.0, 0.95), ("ebit", 180.0, 0.95), ("net_income", 130.0, 0.95),
        ("pretax_income", 170.0, 0.95), ("tax_expense", 40.0, 0.95), ("ebitda", 225.0, 0.90),
        ("ocf", 190.0, 0.95), ("capex", 55.0, 0.95), ("depreciation", 45.0, 0.95),
        ("working_capital_change", -12.0, 0.90), ("sbc", 18.0, 0.90),
        ("assets", 1400.0, 0.95), ("liabilities", 700.0, 0.95), ("equity", 700.0, 0.95),
        ("retained_earnings", 520.0, 0.95), ("cash", 220.0, 0.95), ("debt", 300.0, 0.95),
        ("long_term_debt", 260.0, 0.95), ("current_assets", 560.0, 0.95),
        ("current_liabilities", 320.0, 0.95), ("receivables", 150.0, 0.95),
        ("payables", 110.0, 0.95), ("inventory", 130.0, 0.95), ("net_ppe", 420.0, 0.95),
        ("short_term_investments", 60.0, 0.95), ("sga", 160.0, 0.95),
        ("interest_expense", 14.0, 0.90), ("shares", 100.0, 0.90), ("diluted_shares", 100.0, 0.92),
        ("current_price", 22.0, 0.98), ("reported_market_cap", 2200.0, 0.85), ("beta", 1.05, 0.60),
        ("forward_eps", 1.45, 0.76), ("analyst_count", 12.0, 0.80),
        ("forward_eps_growth", 0.10, 0.72), ("forward_revenue_growth", 0.08, 0.72),
        ("net_change_cash", 30.0, 0.9), ("investing_cash_flow", -70.0, 0.9),
        ("financing_cash_flow", -90.0, 0.9),
    ):
        put_metric(data, key, value, "quarterly statements", "TTM", dates[0], reliability)
    run_integrity_checks(data)
    derive_metrics(data)
    data.forensics = compute_forensics(data)
    return data


def self_test() -> int:
    failures: list[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        if condition:
            print(f"  PASS  {name}")
        else:
            failures.append(name)
            print(f"  FAIL  {name} {detail}")

    print("Pure-function checks")
    check("interpolate endpoints", interpolate(-5, [(0, 1), (10, 0)]) == 1 and interpolate(15, [(0, 1), (10, 0)]) == 0)
    check("interpolate midpoint", abs(interpolate(5, [(0, 0), (10, 1)]) - 0.5) < 1e-9)
    check("divide guards zero", divide(1.0, 0.0) is None)
    check("safe_number rejects text", safe_number("abc") is None and safe_number(True) is None)
    check("sign-aware turnaround", sign_aware_change(5, -5) == (None, "turnaround"))
    check("weighted median", abs(weighted_median([(1, 1), (2, 1), (10, 5)]) - 10) < 1e-9)
    check("ticker validation", validate_ticker(" aapl ") == "AAPL")
    try:
        validate_ticker("not a ticker!")
        check("ticker rejection", False)
    except ValueError:
        check("ticker rejection", True)

    print("\nDCF mechanics")
    ev, parts = dcf_enterprise_value(100.0, 0.05, 0.09, 0.025)
    check("DCF returns a positive enterprise value", ev is not None and ev > 0)
    check("terminal share is a fraction", 0 < parts["terminal_share"] < 1)
    ev_high, _ = dcf_enterprise_value(100.0, 0.10, 0.09, 0.025)
    check("faster growth is worth more", ev_high > ev)
    ev_cheap, _ = dcf_enterprise_value(100.0, 0.05, 0.12, 0.025)
    check("a higher discount rate is worth less", ev_cheap < ev)
    check("g >= WACC is refused", dcf_enterprise_value(100.0, 0.05, 0.02, 0.025)[0] is None)

    print("\nSynthetic company")
    data = _synthetic_company()
    check("balance sheet articulates", all(c.passed for c in data.integrity if c.name == "Balance-sheet identity"))
    check("gross margin derived", abs((data.metrics.get("gross_margin") or 0) - 0.40) < 1e-6)
    check("FCFE derived", abs((data.metrics.get("fcfe") or 0) - 135.0) < 1e-6)
    check("ROIC positive", (data.metrics.get("roic") or 0) > 0.10)
    check("Piotroski computed", data.forensics.piotroski is not None,
          f"(got {data.forensics.piotroski})")
    check("Altman computed", data.forensics.altman_z is not None)
    check("Beneish computed", data.forensics.beneish_m is not None)

    assumptions = build_assumptions(data, 0.042, "self-test constant")
    check("WACC is between the bounds", MIN_WACC <= assumptions.wacc <= MAX_WACC)
    check("terminal growth is below WACC", assumptions.terminal_growth < assumptions.wacc)

    result = analyze(data, [], None, None)
    fv = result.fair_value
    check("fair value issued", fv.base is not None, f"(action {fv.action})")
    if fv.base is not None:
        check("buy-below is under fair value", fv.buy_below < fv.base)
        check("strong-buy is under buy-below", fv.strong_buy_below <= fv.buy_below)
        check("bear <= base <= bull", fv.low <= fv.base <= fv.high)
        check("margin of safety in range", 0.10 <= (fv.margin_of_safety or 0) <= 0.50)
        check("effective weights sum to one",
              abs(sum(mt.effective_weight for mt in fv.methods) - 1.0) < 1e-6)
    check("reverse DCF solved", fv.dcf.implied_growth is not None)
    if fv.dcf.implied_growth is not None:
        _, parts = dcf_per_share(data, fv.dcf.starting_fcff, fv.dcf.implied_growth, assumptions)
        implied_price = divide(parts.get("equity_value"), data.metrics["effective_shares"])
        check("reverse DCF round-trips to the market price",
              implied_price is not None and abs(implied_price - data.metrics["current_price"]) < 0.25,
              f"(got {implied_price})")
    check("score in range", 0 <= result.overall <= 100)
    check("JSON serialises", isinstance(json.dumps(result_to_dict(data, result), default=_json_default), str))

    print("\nOvervaluation direction")
    expensive = _synthetic_company()
    put_metric(expensive, "current_price", 200.0, "market quote", "current", None, 0.98)
    derive_metrics(expensive)
    expensive_result = analyze(expensive, [], None, None)
    check("a 10x higher price reads as overvalued",
          "OVER" in expensive_result.fair_value.status or "ABOVE" in expensive_result.fair_value.action,
          f"(got {expensive_result.fair_value.status} / {expensive_result.fair_value.action})")

    print("\nRate-sensitive anchors")
    base_anchors = SECTOR_PROFILES["technology"].copy()
    at_base = rate_adjusted_anchors(base_anchors, ANCHOR_BASE_RF)
    check("no adjustment at the baseline rate",
          all(abs(at_base[k] - base_anchors[k]) < 1e-9 for k in base_anchors))
    higher = rate_adjusted_anchors(base_anchors, ANCHOR_BASE_RF + 0.02)
    lower = rate_adjusted_anchors(base_anchors, ANCHOR_BASE_RF - 0.02)
    check("higher rates compress the P/E anchor", higher["pe"] < base_anchors["pe"])
    check("lower rates expand the P/E anchor", lower["pe"] > base_anchors["pe"])
    check("long-duration multiples move more than short",
          (base_anchors["pe"] - higher["pe"]) / base_anchors["pe"]
          > (base_anchors["ev_ebitda"] - higher["ev_ebitda"]) / base_anchors["ev_ebitda"])
    check("P/S moves less than P/E",
          (base_anchors["ps"] - higher["ps"]) / base_anchors["ps"]
          < (base_anchors["pe"] - higher["pe"]) / base_anchors["pe"])
    extreme = rate_adjusted_anchors(base_anchors, 0.15)
    check("rate adjustment respects the clamp",
          all(extreme[k] >= base_anchors[k] * ANCHOR_RATE_CLAMP[0] - 1e-9 for k in base_anchors))

    print("\nCalibrated curve exponents")
    groups = [("only", 1.0, [0.6])]
    neutral = grouped_category(10.0, groups).points
    steep = grouped_category(10.0, groups, gamma=1.5).points
    flat = grouped_category(10.0, groups, gamma=0.7).points
    check("gamma of 1 is the identity", abs(neutral - 6.0) < 1e-9, f"(got {neutral})")
    check("gamma above 1 penalizes a mediocre score", steep < neutral)
    check("gamma below 1 lifts a mediocre score", flat > neutral)
    check("a perfect score survives any gamma",
          abs(grouped_category(10.0, [("only", 1.0, [1.0])], gamma=1.5).points - 10.0) < 1e-9)

    print("\nCalibrated DCF sensitivity bands")
    wide = analyze(data, [], None, None, Calibration(dcf_band_scale=1.8))
    narrow = analyze(data, [], None, None, Calibration(dcf_band_scale=0.7))
    if None not in (fv.base, wide.fair_value.base, narrow.fair_value.base):
        base_spread = fv.high - fv.low
        check("a band scale above 1 widens bear/bull",
              wide.fair_value.high - wide.fair_value.low > base_spread)
        check("a band scale below 1 narrows bear/bull",
              narrow.fair_value.high - narrow.fair_value.low < base_spread)
        check("the base fair value is unmoved by band scaling",
              abs(wide.fair_value.base - fv.base) / fv.base < 0.02,
              f"(got {wide.fair_value.base} vs {fv.base})")

    print("\nCross-source validation mechanics")
    validated = _synthetic_company()
    rel_before = validated.meta["revenue"].reliability
    _apply_validation(validated, "revenue", 1000.0, 1005.0, "self-test source")
    check("a confirming source raises reliability",
          validated.meta["revenue"].reliability > rel_before
          and validated.validation[-1].verdict == "confirmed")
    rel_before = validated.meta["net_income"].reliability
    _apply_validation(validated, "net_income", 130.0, 210.0, "self-test source")
    check("a conflicting source cuts reliability hard",
          validated.meta["net_income"].reliability < rel_before * 0.7
          and validated.validation[-1].verdict == "conflict")
    check("a conflict is surfaced as a warning",
          any("Cross-source conflict" in w for w in validated.warnings))
    _apply_validation(validated, "ocf", 190.0, 170.0, "self-test source")
    check("a middling gap is flagged for review, not punished",
          validated.validation[-1].verdict == "review")

    print("\nCalibration round-trip and backtest emission")
    import contextlib
    import io
    with tempfile.TemporaryDirectory() as workdir:
        legacy_path = Path(workdir) / "legacy.json"
        legacy_path.write_text(json.dumps({k: v for k, v in CATEGORY_MAXIMUMS.items()}), encoding="utf-8")
        legacy = load_calibration(str(legacy_path))
        check("legacy weights file loads as a calibration",
              legacy is not None and legacy.category_weights is not None
              and legacy.curve_gamma is None)

        rng = np.random.default_rng(11)
        rows = []
        for day in range(24):
            date = f"2026-{1 + day // 28:02d}-{1 + day % 28:02d}"
            for t in range(10):
                score = float(rng.uniform(5, 95))
                ret = 0.004 * (score - 50) + float(rng.normal(0, 0.08))
                row = {"date": date, "ticker": f"T{t}", "score": score,
                       "future_return": ret, "benchmark_return": 0.0,
                       "fair_value": 100.0, "bear": 82.0, "bull": 118.0,
                       "upside_downside": float(rng.normal(0.02, 0.05)),
                       "discount_premium": 0.0, "implied_growth": 0.05,
                       "piotroski_f": 5, "business_quality": score}
                for key in CATEGORY_MAXIMUMS:
                    row[f"cat_{key}"] = clamp(score + float(rng.normal(0, 18)), 0, 100)
                rows.append(row)
        history_path = Path(workdir) / "history.csv"
        pd.DataFrame(rows).to_csv(history_path, index=False)
        calib_path = Path(workdir) / "calib.json"
        with contextlib.redirect_stdout(io.StringIO()):
            status = run_backtest(str(history_path), str(calib_path))
        check("backtest emits a calibration file", status == 0 and calib_path.exists())
        if calib_path.exists():
            calibration = load_calibration(str(calib_path))
            check("calibration file carries weights, gammas, and a band scale",
                  calibration is not None
                  and calibration.category_weights is not None
                  and calibration.curve_gamma is not None
                  and calibration.dcf_band_scale is not None)
            if calibration and calibration.curve_gamma:
                check("emitted gammas respect the clamp",
                      all(CURVE_GAMMA_RANGE[0] <= v <= CURVE_GAMMA_RANGE[1]
                          for v in calibration.curve_gamma.values()))
            if calibration and calibration.dcf_band_scale is not None:
                check("emitted band scale respects the clamp",
                      DCF_BAND_SCALE_RANGE[0] <= calibration.dcf_band_scale <= DCF_BAND_SCALE_RANGE[1])

    print("\nPeer weighting mechanics")
    check("weighted median favors the full-weight side",
          weighted_median([(10.0, 1.0), (30.0, 0.5), (10.0, 1.0)]) == 10.0)

    print(f"\n{'ALL CHECKS PASSED' if not failures else str(len(failures)) + ' CHECK(S) FAILED: ' + ', '.join(failures)}")
    return 0 if not failures else 1


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def analyze_symbol(symbol: str, peers: list[str], snapshot_csv: Optional[str],
                   weights_path: Optional[str] = None, detail: str = "standard",
                   use_color: bool = False, as_json: bool = False,
                   mos_override: Optional[float] = None,
                   calibration_path: Optional[str] = None) -> int:
    try:
        cleaned = validate_ticker(symbol)
        validated = [validate_ticker(peer) for peer in peers]
        peers = sorted({peer for peer in validated if peer != cleaned})
        if not as_json:
            print(f"Retrieving point-in-time data for {cleaned}...")
        data = fetch_stock_data(cleaned)
        weights = load_category_weights(weights_path)
        calibration = load_calibration(calibration_path)
        result = analyze(data, peers, weights, mos_override, calibration)
        if as_json:
            print(json.dumps(result_to_dict(data, result), default=_json_default, indent=2))
        else:
            print_report(data, result, detail=detail, use_color=use_color)
        if snapshot_csv:
            append_snapshot(snapshot_csv, data, result)
        return 0
    except (ValueError, StockDataError) as exc:
        print(json.dumps({"error": str(exc)}) if as_json else f"Unable to analyze the ticker: {exc}")
    except KeyboardInterrupt:
        print("\nAnalysis cancelled.")
    except Exception as exc:
        if as_json:
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        else:
            print(f"Unexpected provider or model error: {type(exc).__name__}: {exc}")
            print("The provider may have changed a field name. No rating was fabricated.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fundamental equity valuation model V5: fair value, buy-below price, and verdict.")
    parser.add_argument("--ticker", help="Ticker to analyze, for example AAPL")
    parser.add_argument("--peers", default="", help="Comma-separated peer tickers for relative multiples")
    parser.add_argument("--mos", type=float, help="Override the required margin of safety, e.g. 0.30")
    parser.add_argument("--snapshot-csv", help="Append today's point-in-time record for later backtesting")
    parser.add_argument("--backtest-csv", help="Evaluate completed point-in-time observations")
    parser.add_argument("--calibrate-out", help="With --backtest-csv, write a full calibration file "
                                                "(category weights + curve exponents + DCF band scale)")
    parser.add_argument("--weights-json", help="Apply previously validated category weights (legacy format)")
    parser.add_argument("--calibration-json", help="Apply a validated calibration file from --calibrate-out")
    parser.add_argument("--no-secondary", action="store_true",
                        help="Skip Stooq/SEC EDGAR cross-source validation")
    parser.add_argument("--detail", choices=REPORT_DETAILS, default="standard", help="Report depth")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    parser.add_argument("--no-cache", action="store_true", help="Bypass the on-disk provider cache")
    parser.add_argument("--clear-cache", action="store_true", help="Delete cached provider data and exit")
    parser.add_argument("--self-test", action="store_true", help="Run offline checks and exit")
    args = parser.parse_args()

    global CACHE_ENABLED, SECONDARY_ENABLED
    if args.no_cache:
        CACHE_ENABLED = False
    if args.no_secondary:
        SECONDARY_ENABLED = False
    if args.clear_cache:
        print(f"Cleared {clear_cache()} cached file(s) from {CACHE_DIR}.")
        return 0
    if args.self_test:
        return self_test()
    if args.backtest_csv:
        return run_backtest(args.backtest_csv, args.calibrate_out)

    use_color = terminal_color_enabled(args.no_color)
    if args.ticker:
        peers = [item.strip().upper() for item in args.peers.split(",") if item.strip()]
        return analyze_symbol(args.ticker, peers, args.snapshot_csv, args.weights_json,
                              detail=args.detail, use_color=use_color, as_json=args.json,
                              mos_override=args.mos, calibration_path=args.calibration_json)
    try:
        ticker = input("Enter a stock ticker symbol: ").strip()
    except (EOFError, KeyboardInterrupt):
        return 0
    while ticker:
        analyze_symbol(ticker, [], args.snapshot_csv, args.weights_json,
                       detail=args.detail, use_color=use_color, mos_override=args.mos,
                       calibration_path=args.calibration_json)
        try:
            ticker = input("\nEnter another ticker, or press Enter to exit: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0
    print("Goodbye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

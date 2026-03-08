"""
APAC market monitor reporting.
==============================

Provides data containers and report generation utilities for a
quantitative APAC rates, FX swap, and cross-currency swap market monitor.
The report is designed around sourced market data points so that every
displayed number can carry source and timestamp metadata.
"""

from __future__ import annotations

import html
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

TRADING_DAYS_1Y = 252
TRADING_DAYS_5Y = 1260


def _ensure_non_empty(value: str, field_name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must be provided")


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_timestamp(value: str) -> str:
    _ensure_non_empty(value, "timestamp")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return value


def _percentile_rank(history: Sequence[float], current: float) -> float:
    if not history:
        return float("nan")
    arr = np.asarray(history, dtype=float)
    return 100.0 * float(np.mean(arr <= current))


def _format_number(value: Optional[float], decimals: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:.{decimals}f}"


def _format_bps(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:.1f}bp"


def _format_pct(value: Optional[float], decimals: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{100.0 * value:.{decimals}f}%"


def _format_signed(value: Optional[float], decimals: int = 1, suffix: str = "bp") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:+.{decimals}f}{suffix}"


def _series_change(values: Sequence[float], lag: int) -> float:
    if len(values) <= lag:
        return float("nan")
    return float(values[-1] - values[-1 - lag])


def _sorted_unique(values: Sequence[float]) -> List[float]:
    return sorted({float(v) for v in values})


def _closest_value(options: Sequence[float], target: float) -> float:
    return min(options, key=lambda item: abs(float(item) - target))


def _standardize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Standardize a matrix column-wise with zero-variance protection."""
    std = matrix.std(axis=0)
    safe_std = np.where(std == 0, 1.0, std)
    return (matrix - matrix.mean(axis=0)) / safe_std


@dataclass(frozen=True)
class TimeSeries:
    """A sourced time series."""

    name: str
    dates: Sequence[str]
    values: Sequence[float]
    source: str
    timestamp: str
    unit: str = ""

    def __post_init__(self) -> None:
        _ensure_non_empty(self.name, "name")
        _ensure_non_empty(self.source, "source")
        object.__setattr__(self, "timestamp", _safe_timestamp(self.timestamp))
        if len(self.dates) != len(self.values):
            raise ValueError("dates and values must have the same length")
        if len(self.values) == 0:
            raise ValueError("values must not be empty")

    @property
    def latest(self) -> float:
        return float(self.values[-1])

    @property
    def latest_date(self) -> str:
        return self.dates[-1]

    def change(self, lag: int = 1) -> float:
        return _series_change(self.values, lag)


@dataclass(frozen=True)
class CurveSeries:
    """A sourced tenor-specific rate history."""

    tenor: float
    dates: Sequence[str]
    values: Sequence[float]
    source: str
    timestamp: str
    instrument: str = "yield"
    unit: str = "rate"

    def __post_init__(self) -> None:
        _ensure_non_empty(self.source, "source")
        object.__setattr__(self, "timestamp", _safe_timestamp(self.timestamp))
        if len(self.dates) != len(self.values):
            raise ValueError("dates and values must have the same length")
        if len(self.values) == 0:
            raise ValueError("values must not be empty")
        if self.tenor <= 0:
            raise ValueError("tenor must be positive")

    @property
    def latest(self) -> float:
        return float(self.values[-1])

    def change(self, lag: int = 1) -> float:
        return _series_change(self.values, lag)


@dataclass(frozen=True)
class CurveDataset:
    """A set of sourced curve histories for one market."""

    market: str
    currency: str
    series: Sequence[CurveSeries]

    def __post_init__(self) -> None:
        _ensure_non_empty(self.market, "market")
        _ensure_non_empty(self.currency, "currency")
        if len(self.series) == 0:
            raise ValueError("series must not be empty")

    @property
    def tenors(self) -> List[float]:
        return sorted(float(item.tenor) for item in self.series)

    @property
    def latest_date(self) -> str:
        return self.series[0].dates[-1]

    @property
    def latest_timestamp(self) -> str:
        return max(item.timestamp for item in self.series)

    def by_tenor(self) -> Dict[float, CurveSeries]:
        return {float(item.tenor): item for item in self.series}


@dataclass(frozen=True)
class TenorQuote:
    """A sourced tenor quote with prior observations."""

    tenor: str
    current: float
    previous_day: float
    previous_week: float
    source: str
    timestamp: str
    unit: str = "bp"

    def __post_init__(self) -> None:
        _ensure_non_empty(self.tenor, "tenor")
        _ensure_non_empty(self.source, "source")
        object.__setattr__(self, "timestamp", _safe_timestamp(self.timestamp))

    @property
    def daily_change(self) -> float:
        return float(self.current - self.previous_day)

    @property
    def weekly_change(self) -> float:
        return float(self.current - self.previous_week)


@dataclass(frozen=True)
class TenorQuoteSet:
    """A set of sourced tenor quotes."""

    name: str
    quotes: Sequence[TenorQuote]
    description: str = ""

    def __post_init__(self) -> None:
        _ensure_non_empty(self.name, "name")
        if len(self.quotes) == 0:
            raise ValueError("quotes must not be empty")


@dataclass(frozen=True)
class PolicyEvent:
    """A sourced central-bank decision or guidance event."""

    bank: str
    event_date: str
    decision: str
    summary: str
    source: str
    timestamp: str
    rates_impact_bps: Mapping[str, float] = field(default_factory=dict)
    basis_impact_bps: Optional[float] = None
    ccs_impact_bps: Optional[float] = None

    def __post_init__(self) -> None:
        _ensure_non_empty(self.bank, "bank")
        _ensure_non_empty(self.decision, "decision")
        _ensure_non_empty(self.summary, "summary")
        _ensure_non_empty(self.source, "source")
        object.__setattr__(self, "timestamp", _safe_timestamp(self.timestamp))


@dataclass(frozen=True)
class EventStudy:
    """Historical event-study outcomes for similar decisions."""

    label: str
    bank: str
    event_type: str
    outcomes: Mapping[str, Sequence[float]]
    source: str
    timestamp: str

    def __post_init__(self) -> None:
        _ensure_non_empty(self.label, "label")
        _ensure_non_empty(self.bank, "bank")
        _ensure_non_empty(self.event_type, "event_type")
        _ensure_non_empty(self.source, "source")
        object.__setattr__(self, "timestamp", _safe_timestamp(self.timestamp))
        if len(self.outcomes) == 0:
            raise ValueError("outcomes must not be empty")


@dataclass(frozen=True)
class MacroRelease:
    """A sourced macro data release."""

    region: str
    indicator: str
    release_date: str
    actual: float
    expected: Optional[float]
    previous: Optional[float]
    impact_summary: str
    source: str
    timestamp: str
    unit: str = ""

    def __post_init__(self) -> None:
        _ensure_non_empty(self.region, "region")
        _ensure_non_empty(self.indicator, "indicator")
        _ensure_non_empty(self.impact_summary, "impact_summary")
        _ensure_non_empty(self.source, "source")
        object.__setattr__(self, "timestamp", _safe_timestamp(self.timestamp))

    @property
    def surprise(self) -> Optional[float]:
        if self.expected is None:
            return None
        return float(self.actual - self.expected)


@dataclass(frozen=True)
class DeskExposure:
    """Linear-rates desk exposure metrics."""

    trade: str
    dv01: float
    carry_roll: float
    rolldown: float
    swap_spread_beta: float
    commentary: str
    source: str
    timestamp: str

    def __post_init__(self) -> None:
        _ensure_non_empty(self.trade, "trade")
        _ensure_non_empty(self.commentary, "commentary")
        _ensure_non_empty(self.source, "source")
        object.__setattr__(self, "timestamp", _safe_timestamp(self.timestamp))


@dataclass(frozen=True)
class VolSurface:
    """A sourced volatility surface."""

    name: str
    expiries: Sequence[str]
    pillars: Sequence[str]
    current: Sequence[Sequence[float]]
    previous: Sequence[Sequence[float]]
    source: str
    timestamp: str
    unit: str = "vol"

    def __post_init__(self) -> None:
        _ensure_non_empty(self.name, "name")
        _ensure_non_empty(self.source, "source")
        object.__setattr__(self, "timestamp", _safe_timestamp(self.timestamp))
        if len(self.expiries) == 0 or len(self.pillars) == 0:
            raise ValueError("surface axes must not be empty")
        current = np.asarray(self.current, dtype=float)
        previous = np.asarray(self.previous, dtype=float)
        if current.shape != (len(self.expiries), len(self.pillars)):
            raise ValueError("current surface shape does not match axes")
        if previous.shape != current.shape:
            raise ValueError("previous surface shape must match current")


@dataclass(frozen=True)
class SourcedObservation:
    """A sourced scalar observation."""

    label: str
    value: float
    source: str
    timestamp: str
    unit: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        _ensure_non_empty(self.label, "label")
        _ensure_non_empty(self.source, "source")
        object.__setattr__(self, "timestamp", _safe_timestamp(self.timestamp))


@dataclass(frozen=True)
class APACMarketMonitorInput:
    """Input bundle for the APAC market monitor."""

    apac_curves: Sequence[CurveDataset] = field(default_factory=list)
    benchmark_curves: Sequence[CurveDataset] = field(default_factory=list)
    fx_swap_quotes: Sequence[TenorQuoteSet] = field(default_factory=list)
    implied_funding_series: Sequence[TimeSeries] = field(default_factory=list)
    ccs_quotes: Sequence[TenorQuoteSet] = field(default_factory=list)
    basis_series: Sequence[TimeSeries] = field(default_factory=list)
    liquidity_indicators: Sequence[SourcedObservation] = field(default_factory=list)
    flow_commentary: Sequence[str] = field(default_factory=list)
    policy_events: Sequence[PolicyEvent] = field(default_factory=list)
    event_studies: Sequence[EventStudy] = field(default_factory=list)
    macro_releases: Sequence[MacroRelease] = field(default_factory=list)
    forward_guidance: Sequence[str] = field(default_factory=list)
    linear_exposures: Sequence[DeskExposure] = field(default_factory=list)
    relative_value_opportunities: Sequence[str] = field(default_factory=list)
    cross_market_rv: Sequence[str] = field(default_factory=list)
    flow_insights: Sequence[str] = field(default_factory=list)
    options_surfaces: Sequence[VolSurface] = field(default_factory=list)
    vol_rate_series: Sequence[TimeSeries] = field(default_factory=list)
    option_scenarios: Sequence[str] = field(default_factory=list)


@dataclass
class APACMarketMonitorReport:
    """Structured report output."""

    title: str
    generated_at: str
    sections: Mapping[str, str]
    tables: Mapping[str, Sequence[Mapping[str, str]]]
    charts: Mapping[str, str]

    def to_dict(self) -> Dict[str, object]:
        """Convert the report to a serializable dictionary."""
        return {
            "title": self.title,
            "generated_at": self.generated_at,
            "sections": dict(self.sections),
            "tables": {key: list(value) for key, value in self.tables.items()},
            "charts": dict(self.charts),
        }

    def to_text(self) -> str:
        """Render the report as plain text."""
        lines = [self.title, f"Generated: {self.generated_at}"]
        for heading, content in self.sections.items():
            lines.extend(["", heading, "-" * len(heading), content])
        return "\n".join(lines)

    def to_html(self) -> str:
        """Render the report as HTML."""
        parts = [
            "<html><head><meta charset='utf-8'>",
            f"<title>{html.escape(self.title)}</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 24px; color: #18212b; }",
            "h1, h2, h3 { color: #0c3b5d; }",
            "table { border-collapse: collapse; width: 100%; margin: 12px 0 24px 0; }",
            "th, td { border: 1px solid #ccd6e0; padding: 6px 8px; font-size: 13px; }",
            "th { background: #eef5fb; text-align: left; }",
            ".chart { margin: 16px 0 24px 0; }",
            ".section { margin-bottom: 28px; }",
            ".small { color: #4d6073; font-size: 12px; }",
            "ul { margin-top: 8px; }",
            "</style></head><body>",
            f"<h1>{html.escape(self.title)}</h1>",
            f"<p class='small'>Generated: {html.escape(self.generated_at)}</p>",
        ]
        for heading, content in self.sections.items():
            parts.append(f"<div class='section'><h2>{html.escape(heading)}</h2>{content}</div>")
            if heading in self.tables:
                parts.append(self._render_table(self.tables[heading]))
            if heading in self.charts:
                parts.append(f"<div class='chart'>{self.charts[heading]}</div>")
        parts.append("</body></html>")
        return "".join(parts)

    def save_html(self, path: str) -> str:
        """Save the HTML representation to disk."""
        rendered = self.to_html()
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(rendered)
        return path

    @staticmethod
    def _render_table(rows: Sequence[Mapping[str, str]]) -> str:
        if not rows:
            return "<p>No tabular data available.</p>"
        headers = list(rows[0].keys())
        parts = ["<table><thead><tr>"]
        parts.extend(f"<th>{html.escape(header)}</th>" for header in headers)
        parts.append("</tr></thead><tbody>")
        for row in rows:
            parts.append("<tr>")
            parts.extend(f"<td>{html.escape(str(row.get(header, '')))}</td>" for header in headers)
            parts.append("</tr>")
        parts.append("</tbody></table>")
        return "".join(parts)


class APACMarketMonitor:
    """
    Generate a fixed-structure APAC rates, FX swap, and CCS market monitor.

    Parameters
    ----------
    title : str, optional
        Report title shown in the rendered output.
    """

    SECTION_ORDER: Tuple[str, ...] = (
        "APAC Rates Overview",
        "FX Swap & CCS Market Moves",
        "Central Bank Policy & Macro Events",
        "Implications for the Linear Rates Desk",
        "Implications for the Options Rates Desk",
        "Historical Curve-Shape Analysis",
        "Data Appendix (with Charts & Tables)",
    )

    def __init__(self, title: str = "APAC Rates, FX Swap, and Cross-Currency Swap Market Monitor") -> None:
        self.title = title

    def build(self, data: APACMarketMonitorInput) -> APACMarketMonitorReport:
        """
        Build the report from sourced market data.

        Parameters
        ----------
        data : APACMarketMonitorInput
            Structured market data and commentary inputs.

        Returns
        -------
        APACMarketMonitorReport
            Render-ready report object containing sections, charts, and tables.
        """
        sections: "OrderedDict[str, str]" = OrderedDict()
        tables: Dict[str, List[Mapping[str, str]]] = {}
        charts: Dict[str, str] = {}

        sections["APAC Rates Overview"], tables["APAC Rates Overview"], charts["APAC Rates Overview"] = (
            self._build_rates_overview(data)
        )
        sections["FX Swap & CCS Market Moves"], tables["FX Swap & CCS Market Moves"], charts["FX Swap & CCS Market Moves"] = (
            self._build_fx_ccs(data)
        )
        sections["Central Bank Policy & Macro Events"], tables["Central Bank Policy & Macro Events"], charts["Central Bank Policy & Macro Events"] = (
            self._build_policy_macro(data)
        )
        sections["Implications for the Linear Rates Desk"], tables["Implications for the Linear Rates Desk"], charts["Implications for the Linear Rates Desk"] = (
            self._build_linear_desk(data)
        )
        sections["Implications for the Options Rates Desk"], tables["Implications for the Options Rates Desk"], charts["Implications for the Options Rates Desk"] = (
            self._build_options_desk(data)
        )
        sections["Historical Curve-Shape Analysis"], tables["Historical Curve-Shape Analysis"], charts["Historical Curve-Shape Analysis"] = (
            self._build_historical_shape(data)
        )
        sections["Data Appendix (with Charts & Tables)"], tables["Data Appendix (with Charts & Tables)"], charts["Data Appendix (with Charts & Tables)"] = (
            self._build_appendix(data)
        )

        return APACMarketMonitorReport(
            title=self.title,
            generated_at=_iso_now(),
            sections=sections,
            tables=tables,
            charts=charts,
        )

    def _build_rates_overview(
        self,
        data: APACMarketMonitorInput,
    ) -> Tuple[str, List[Mapping[str, str]], str]:
        if not data.apac_curves:
            return "<p>No APAC curve data provided.</p>", [], ""

        rows: List[Mapping[str, str]] = []
        narratives: List[str] = []
        comparison_chart_lines: List[Tuple[str, Sequence[float], Sequence[float]]] = []
        benchmark_map = {curve.currency: curve for curve in data.benchmark_curves}

        for curve in data.apac_curves:
            metrics = self._curve_metrics(curve)
            rows.extend(metrics["rows"])
            narratives.append(
                "<p><strong>{market}</strong>: 2s10s at {s210}, 5s10s at {s510}, 10s30s at {s1030}; "
                "PCA flags {dominant}. Historical percentiles place the 10Y yield at {p10y} (1y) / {p10y5} (5y). "
                "Derived from {sources} with latest timestamp {timestamp}.</p>".format(
                    market=html.escape(curve.market),
                    s210=_format_bps(metrics["spreads"].get("2s10s")),
                    s510=_format_bps(metrics["spreads"].get("5s10s")),
                    s1030=_format_bps(metrics["spreads"].get("10s30s")),
                    dominant=html.escape(metrics["dominant_factor"]),
                    p10y=_format_number(metrics["percentiles_1y"].get("10Y"), 1) + "th pct",
                    p10y5=_format_number(metrics["percentiles_5y"].get("10Y"), 1) + "th pct",
                    sources=html.escape(metrics["source_summary"]),
                    timestamp=html.escape(curve.latest_timestamp),
                )
            )
            comparison_chart_lines.append(
                (
                    curve.market,
                    curve.tenors,
                    [curve.by_tenor()[tenor].latest * 100.0 for tenor in curve.tenors],
                )
            )

            usd_curve = benchmark_map.get("USD")
            eur_curve = benchmark_map.get("EUR")
            if usd_curve is not None:
                narratives.append(self._cross_market_paragraph(curve, usd_curve, "USD"))
            if eur_curve is not None:
                narratives.append(self._cross_market_paragraph(curve, eur_curve, "EUR"))

        chart = self._line_chart_svg(
            title="Current APAC yield curves",
            x_values=[tenor for tenor in comparison_chart_lines[0][1]],
            series=[(name, values) for name, _, values in comparison_chart_lines],
            x_label="Tenor (Y)",
            y_label="Yield (%)",
        )
        return "".join(narratives), rows, chart

    def _build_fx_ccs(
        self,
        data: APACMarketMonitorInput,
    ) -> Tuple[str, List[Mapping[str, str]], str]:
        narratives: List[str] = []
        rows: List[Mapping[str, str]] = []

        for quote_set in list(data.fx_swap_quotes) + list(data.ccs_quotes):
            for quote in quote_set.quotes:
                rows.append(
                    {
                        "Market": quote_set.name,
                        "Tenor": quote.tenor,
                        "Current": _format_number(quote.current, 2),
                        "Daily Change": _format_signed(quote.daily_change, 2, ""),
                        "Weekly Change": _format_signed(quote.weekly_change, 2, ""),
                        "Source": quote.source,
                        "Timestamp": quote.timestamp,
                    }
                )
            descriptions = f" ({quote_set.description})" if quote_set.description else ""
            belly_index = len(quote_set.quotes) // 2
            narratives.append(
                "<p><strong>{name}</strong>{desc}: front-end move {front}, belly {belly}, 10Y {ten}. "
                "Daily changes point to {direction}. Source timestamps run through {timestamp}.</p>".format(
                    name=html.escape(quote_set.name),
                    desc=html.escape(descriptions),
                    front=_format_number(quote_set.quotes[0].current, 2),
                    belly=_format_number(quote_set.quotes[belly_index].current, 2),
                    ten=_format_number(quote_set.quotes[-1].current, 2),
                    direction=html.escape(self._move_description(quote_set.quotes)),
                    timestamp=html.escape(max(item.timestamp for item in quote_set.quotes)),
                )
            )

        if data.liquidity_indicators:
            liquidity_bits = [
                f"{html.escape(item.label)} {html.escape(_format_number(item.value, 2))}{html.escape(item.unit)} "
                f"({html.escape(item.source)}, {html.escape(item.timestamp)})"
                for item in data.liquidity_indicators
            ]
            narratives.append(
                "<p><strong>Liquidity:</strong> " + "; ".join(liquidity_bits) + ".</p>"
            )
        if data.flow_commentary:
            narratives.append(
                "<ul>" + "".join(f"<li>{html.escape(line)}</li>" for line in data.flow_commentary) + "</ul>"
            )

        corr_rows = self._correlation_rows(data.basis_series, data.implied_funding_series, data.apac_curves)
        rows.extend(corr_rows)
        if corr_rows:
            narratives.append(
                "<p>Correlation analysis links basis and funding pressure back to cross-market rate differentials. "
                "Positive/negative signs are computed on aligned historical daily observations using sourced series timestamps.</p>"
            )

        chart_series: List[Tuple[str, Sequence[float]]] = []
        chart_x: List[str] = []
        if data.basis_series:
            chart_x = list(data.basis_series[0].dates)
            chart_series.extend((series.name, series.values) for series in data.basis_series[:3])
        chart = self._time_series_chart_svg(
            title="Basis and funding time series",
            dates=chart_x,
            series=chart_series,
            y_label="Level",
        ) if chart_series else ""
        if not narratives and not rows:
            return "<p>No FX swap or CCS data provided.</p>", [], chart
        return "".join(narratives), rows, chart

    def _build_policy_macro(
        self,
        data: APACMarketMonitorInput,
    ) -> Tuple[str, List[Mapping[str, str]], str]:
        rows: List[Mapping[str, str]] = []
        narratives: List[str] = []

        if data.policy_events:
            for event in data.policy_events:
                rows.append(
                    {
                        "Type": "Policy",
                        "Entity": event.bank,
                        "Date": event.event_date,
                        "Headline": event.decision,
                        "Impact": self._event_impact_summary(event),
                        "Source": event.source,
                        "Timestamp": event.timestamp,
                    }
                )
            grouped = OrderedDict()
            for event in data.policy_events:
                grouped.setdefault(event.bank, []).append(event)
            for bank, events in grouped.items():
                latest = events[-1]
                narratives.append(
                    "<p><strong>{bank}</strong>: latest action was <em>{decision}</em> on {date}. "
                    "{summary} Derived from {source} ({timestamp}).</p>".format(
                        bank=html.escape(bank),
                        decision=html.escape(latest.decision),
                        date=html.escape(latest.event_date),
                        summary=html.escape(latest.summary),
                        source=html.escape(latest.source),
                        timestamp=html.escape(latest.timestamp),
                    )
                )

        if data.event_studies:
            narratives.append("<h3>Event-study analogs</h3>")
            narratives.append("<ul>")
            for study in data.event_studies:
                summary = self._event_study_summary(study)
                narratives.append(
                    f"<li><strong>{html.escape(study.bank)} / {html.escape(study.event_type)}</strong>: "
                    f"{html.escape(summary)} Source: {html.escape(study.source)} ({html.escape(study.timestamp)}).</li>"
                )
            narratives.append("</ul>")

        if data.macro_releases:
            narratives.append("<h3>Macro releases</h3>")
            narratives.append("<ul>")
            for release in data.macro_releases:
                surprise = release.surprise
                surprise_text = "n/a" if surprise is None else _format_number(surprise, 2)
                narratives.append(
                    "<li><strong>{region} {indicator}</strong>: actual {actual}{unit}, expected {expected}{unit}, "
                    "surprise {surprise}; {impact}. Source: {source} ({timestamp}).</li>".format(
                        region=html.escape(release.region),
                        indicator=html.escape(release.indicator),
                        actual=_format_number(release.actual, 2),
                        expected=_format_number(release.expected, 2),
                        unit=html.escape(release.unit),
                        surprise=surprise_text,
                        impact=html.escape(release.impact_summary),
                        source=html.escape(release.source),
                        timestamp=html.escape(release.timestamp),
                    )
                )
                rows.append(
                    {
                        "Type": "Macro",
                        "Entity": release.region,
                        "Date": release.release_date,
                        "Headline": release.indicator,
                        "Impact": release.impact_summary,
                        "Source": release.source,
                        "Timestamp": release.timestamp,
                    }
                )
            narratives.append("</ul>")

        if data.forward_guidance:
            narratives.append(
                "<p><strong>Forward guidance implications:</strong></p><ul>"
                + "".join(f"<li>{html.escape(item)}</li>" for item in data.forward_guidance)
                + "</ul>"
            )

        event_chart = self._bar_chart_svg(
            title="Average event-study impact (bp)",
            values=[
                (study.bank + " " + study.event_type, self._event_study_primary_move(study))
                for study in data.event_studies
            ],
            y_label="bp",
        ) if data.event_studies else ""

        if not narratives and not rows:
            return "<p>No central-bank or macro-event data provided.</p>", [], event_chart
        return "".join(narratives), rows, event_chart

    def _build_linear_desk(
        self,
        data: APACMarketMonitorInput,
    ) -> Tuple[str, List[Mapping[str, str]], str]:
        rows: List[Mapping[str, str]] = []
        narratives: List[str] = []
        if data.linear_exposures:
            total_dv01 = sum(item.dv01 for item in data.linear_exposures)
            total_carry = sum(item.carry_roll for item in data.linear_exposures)
            total_rolldown = sum(item.rolldown for item in data.linear_exposures)
            narratives.append(
                "<p>Desk DV01 totals {dv01}, with carry/roll of {carry} and rolldown of {rolldown}. "
                "Swap-spread sensitivity is concentrated in the positions listed below.</p>".format(
                    dv01=_format_bps(total_dv01),
                    carry=_format_bps(total_carry),
                    rolldown=_format_bps(total_rolldown),
                )
            )
            for item in data.linear_exposures:
                rows.append(
                    {
                        "Trade": item.trade,
                        "DV01": _format_bps(item.dv01),
                        "Carry/Roll": _format_bps(item.carry_roll),
                        "Rolldown": _format_bps(item.rolldown),
                        "Swap Spread Beta": _format_number(item.swap_spread_beta, 2),
                        "Commentary": item.commentary,
                        "Source": item.source,
                        "Timestamp": item.timestamp,
                    }
                )
        if data.relative_value_opportunities:
            narratives.append(
                "<p><strong>Relative-value opportunities:</strong></p><ul>"
                + "".join(f"<li>{html.escape(item)}</li>" for item in data.relative_value_opportunities)
                + "</ul>"
            )
        if data.cross_market_rv:
            narratives.append(
                "<p><strong>Cross-market RV:</strong></p><ul>"
                + "".join(f"<li>{html.escape(item)}</li>" for item in data.cross_market_rv)
                + "</ul>"
            )
        if data.flow_insights:
            narratives.append(
                "<p><strong>Flow-based insights:</strong></p><ul>"
                + "".join(f"<li>{html.escape(item)}</li>" for item in data.flow_insights)
                + "</ul>"
            )

        chart = self._bar_chart_svg(
            title="Desk DV01 by trade",
            values=[(item.trade, item.dv01) for item in data.linear_exposures],
            y_label="DV01 (bp)",
        ) if data.linear_exposures else ""
        if not narratives and not rows:
            return "<p>No linear-desk exposure data provided.</p>", [], chart
        return "".join(narratives), rows, chart

    def _build_options_desk(
        self,
        data: APACMarketMonitorInput,
    ) -> Tuple[str, List[Mapping[str, str]], str]:
        rows: List[Mapping[str, str]] = []
        narratives: List[str] = []
        charts: List[str] = []

        for surface in data.options_surfaces:
            current = np.asarray(surface.current, dtype=float)
            previous = np.asarray(surface.previous, dtype=float)
            change = current - previous
            atm_idx = self._atm_index(surface.pillars)
            atm_change = float(np.mean(change[:, atm_idx]))
            skew_change = float(change[0, -1] - change[0, 0]) if current.shape[1] >= 2 else float("nan")
            narratives.append(
                "<p><strong>{name}</strong>: average ATM change {atm}, front-end skew change {skew}. "
                "Source {source} ({timestamp}).</p>".format(
                    name=html.escape(surface.name),
                    atm=_format_signed(atm_change, 2, "vol"),
                    skew=_format_signed(skew_change, 2, "vol"),
                    source=html.escape(surface.source),
                    timestamp=html.escape(surface.timestamp),
                )
            )
            rows.append(
                {
                    "Surface": surface.name,
                    "Average ATM Change": _format_signed(atm_change, 2, "vol"),
                    "Skew Change": _format_signed(skew_change, 2, "vol"),
                    "Source": surface.source,
                    "Timestamp": surface.timestamp,
                }
            )
            charts.append(
                self._heatmap_svg(
                    title=f"{surface.name} surface change",
                    rows=surface.expiries,
                    columns=surface.pillars,
                    matrix=change.tolist(),
                )
            )

        corr_rows = self._vol_rate_correlation_rows(data.vol_rate_series)
        rows.extend(corr_rows)
        if corr_rows:
            narratives.append(
                "<p>Vol/rate correlations are computed on aligned daily changes to highlight gamma and vega feedback loops "
                "around policy events and directional sell-offs.</p>"
            )
        if data.option_scenarios:
            narratives.append(
                "<p><strong>Scenario analysis:</strong></p><ul>"
                + "".join(f"<li>{html.escape(item)}</li>" for item in data.option_scenarios)
                + "</ul>"
            )

        chart = "".join(charts)
        if not narratives and not rows:
            return "<p>No options data provided.</p>", [], chart
        return "".join(narratives), rows, chart

    def _build_historical_shape(
        self,
        data: APACMarketMonitorInput,
    ) -> Tuple[str, List[Mapping[str, str]], str]:
        if not data.apac_curves:
            return "<p>No historical curve data provided.</p>", [], ""

        rows: List[Mapping[str, str]] = []
        narratives: List[str] = []
        analog_lines: List[Tuple[str, Sequence[float]]] = []
        dates: Sequence[str] = []

        for curve in data.apac_curves:
            analog = self._closest_historical_analog(curve)
            narratives.append(
                "<p><strong>{market}</strong>: the closest historical analog is {analog_date}, with a standardized "
                "distance of {distance}. Current slope/curvature regime sits in the {regime} bucket.</p>".format(
                    market=html.escape(curve.market),
                    analog_date=html.escape(analog["date"]),
                    distance=_format_number(analog["distance"], 2),
                    regime=html.escape(analog["regime"]),
                )
            )
            rows.append(
                {
                    "Market": curve.market,
                    "Analog Date": analog["date"],
                    "Distance": _format_number(analog["distance"], 2),
                    "Regime": analog["regime"],
                    "Source": analog["source"],
                    "Timestamp": analog["timestamp"],
                }
            )
            history = self._curve_factor_history(curve)
            dates = history["dates"]
            analog_lines.append((curve.market + " slope", history["slope"]))
            analog_lines.append((curve.market + " curvature", history["curvature"]))

        chart = self._time_series_chart_svg(
            title="Historical slope and curvature",
            dates=dates,
            series=analog_lines,
            y_label="bp",
        )
        return "".join(narratives), rows, chart

    def _build_appendix(
        self,
        data: APACMarketMonitorInput,
    ) -> Tuple[str, List[Mapping[str, str]], str]:
        rows: List[Mapping[str, str]] = []
        charts: List[str] = []
        narratives = [
            "<p>The appendix consolidates yield-curve snapshots, factor diagnostics, FX swap/basis visuals, "
            "volatility heatmaps, and supporting tables with source/timestamp fields.</p>"
        ]

        for curve in data.apac_curves:
            by_tenor = curve.by_tenor()
            tenors = curve.tenors
            current = [by_tenor[tenor].latest * 100.0 for tenor in tenors]
            week = [
                (by_tenor[tenor].values[-6] if len(by_tenor[tenor].values) > 5 else by_tenor[tenor].values[0]) * 100.0
                for tenor in tenors
            ]
            month = [
                (by_tenor[tenor].values[-21] if len(by_tenor[tenor].values) > 20 else by_tenor[tenor].values[0]) * 100.0
                for tenor in tenors
            ]
            charts.append(
                self._line_chart_svg(
                    title=f"{curve.market} current vs 1w vs 1m",
                    x_values=tenors,
                    series=[
                        ("Current", current),
                        ("1w ago", week),
                        ("1m ago", month),
                    ],
                    x_label="Tenor (Y)",
                    y_label="Yield (%)",
                )
            )
            factor_history = self._curve_factor_history(curve)
            charts.append(
                self._time_series_chart_svg(
                    title=f"{curve.market} PCA proxy factors",
                    dates=factor_history["dates"],
                    series=[
                        ("Level", factor_history["level"]),
                        ("Slope", factor_history["slope"]),
                        ("Curvature", factor_history["curvature"]),
                    ],
                    y_label="bp",
                )
            )
            rows.append(
                {
                    "Dataset": curve.market,
                    "Latest Date": curve.latest_date,
                    "Tenors": ", ".join(f"{tenor:g}Y" for tenor in tenors),
                    "Source": ", ".join(sorted({item.source for item in curve.series})),
                    "Timestamp": curve.latest_timestamp,
                }
            )

        for quote_set in list(data.fx_swap_quotes) + list(data.ccs_quotes):
            charts.append(
                self._bar_chart_svg(
                    title=f"{quote_set.name} current levels",
                    values=[(item.tenor, item.current) for item in quote_set.quotes],
                    y_label=quote_set.quotes[0].unit,
                )
            )

        for surface in data.options_surfaces:
            charts.append(
                self._heatmap_svg(
                    title=f"{surface.name} current surface",
                    rows=surface.expiries,
                    columns=surface.pillars,
                    matrix=np.asarray(surface.current, dtype=float).tolist(),
                )
            )

        return "".join(narratives), rows, "".join(charts)

    def _curve_metrics(self, curve: CurveDataset) -> Dict[str, object]:
        by_tenor = curve.by_tenor()
        target_tenors = {
            "2Y": _closest_value(curve.tenors, 2.0),
            "5Y": _closest_value(curve.tenors, 5.0),
            "10Y": _closest_value(curve.tenors, 10.0),
            "30Y": _closest_value(curve.tenors, 30.0),
        }
        spreads = {
            "2s10s": 10000.0 * (by_tenor[target_tenors["10Y"]].latest - by_tenor[target_tenors["2Y"]].latest),
            "5s10s": 10000.0 * (by_tenor[target_tenors["10Y"]].latest - by_tenor[target_tenors["5Y"]].latest),
            "10s30s": 10000.0 * (by_tenor[target_tenors["30Y"]].latest - by_tenor[target_tenors["10Y"]].latest),
        }
        pca = self._pca_factor_decomposition(curve)
        rows = []
        for tenor in curve.tenors:
            series = by_tenor[tenor]
            history = list(series.values)
            label = f"{int(tenor) if tenor.is_integer() else tenor:g}Y"
            rows.append(
                {
                    "Market": curve.market,
                    "Tenor": label,
                    "Level": _format_pct(series.latest),
                    "Daily Change": _format_signed(series.change(1) * 10000.0, 1),
                    "Weekly Change": _format_signed(series.change(5) * 10000.0, 1),
                    "1Y Percentile": _format_number(_percentile_rank(history[-TRADING_DAYS_1Y:], series.latest), 1),
                    "5Y Percentile": _format_number(_percentile_rank(history[-TRADING_DAYS_5Y:], series.latest), 1),
                    "Source": series.source,
                    "Timestamp": series.timestamp,
                }
            )
        source_summary = ", ".join(sorted({item.source for item in curve.series}))
        return {
            "rows": rows,
            "spreads": spreads,
            "pca": pca,
            "dominant_factor": pca["dominant_factor"],
            "percentiles_1y": {
                "10Y": _percentile_rank(
                    list(by_tenor[target_tenors["10Y"]].values)[-TRADING_DAYS_1Y:],
                    by_tenor[target_tenors["10Y"]].latest,
                )
            },
            "percentiles_5y": {
                "10Y": _percentile_rank(
                    list(by_tenor[target_tenors["10Y"]].values)[-TRADING_DAYS_5Y:],
                    by_tenor[target_tenors["10Y"]].latest,
                )
            },
            "source_summary": source_summary,
        }

    def _cross_market_paragraph(self, curve: CurveDataset, benchmark: CurveDataset, benchmark_name: str) -> str:
        own = curve.by_tenor()
        other = benchmark.by_tenor()
        common_tenors = sorted(set(own).intersection(other))
        if not common_tenors:
            return ""
        spread_bits = []
        for target in (2.0, 5.0, 10.0, 30.0):
            tenor = _closest_value(common_tenors, target)
            spread = 10000.0 * (own[tenor].latest - other[tenor].latest)
            spread_bits.append(f"{tenor:g}Y {spread:+.1f}bp")
        return (
            f"<p><strong>{html.escape(curve.market)} vs {html.escape(benchmark_name)}</strong>: "
            + ", ".join(spread_bits)
            + f". Benchmark source set includes {html.escape(', '.join(sorted({item.source for item in benchmark.series})))}.</p>"
        )

    def _pca_factor_decomposition(self, curve: CurveDataset) -> Dict[str, object]:
        tenors = curve.tenors
        by_tenor = curve.by_tenor()
        matrix = np.asarray([by_tenor[tenor].values for tenor in tenors], dtype=float).T
        if matrix.shape[0] < 3 or matrix.shape[1] < 3:
            return {
                "eigenvalues": [float("nan")] * 3,
                "loadings": {},
                "scores": {"level": float("nan"), "slope": float("nan"), "curvature": float("nan")},
                "dominant_factor": "insufficient history",
            }
        demeaned = matrix - matrix.mean(axis=0)
        covariance = np.cov(demeaned, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        latest_curve = demeaned[-1, :]
        scores = eigenvectors[:, :3].T @ latest_curve
        factor_names = ("level", "slope", "curvature")
        loadings = {
            factor_names[idx]: {
                f"{tenor:g}Y": float(eigenvectors[pos, idx]) for pos, tenor in enumerate(tenors)
            }
            for idx in range(min(3, eigenvectors.shape[1]))
        }
        dominant_idx = int(np.argmax(np.abs(scores[:3])))
        return {
            "eigenvalues": eigenvalues[:3].tolist(),
            "loadings": loadings,
            "scores": {
                factor_names[idx]: float(scores[idx]) for idx in range(min(3, len(scores)))
            },
            "dominant_factor": factor_names[dominant_idx],
        }

    def _correlation_rows(
        self,
        basis_series: Sequence[TimeSeries],
        funding_series: Sequence[TimeSeries],
        apac_curves: Sequence[CurveDataset],
    ) -> List[Mapping[str, str]]:
        rows: List[Mapping[str, str]] = []
        if not basis_series:
            return rows

        curve_slope_series: List[TimeSeries] = []
        for curve in apac_curves:
            factor_history = self._curve_factor_history(curve)
            curve_slope_series.append(
                TimeSeries(
                    name=f"{curve.market} slope",
                    dates=factor_history["dates"],
                    values=factor_history["slope"],
                    source=", ".join(sorted({item.source for item in curve.series})),
                    timestamp=curve.latest_timestamp,
                    unit="bp",
                )
            )

        all_series = list(basis_series) + list(funding_series) + curve_slope_series
        for idx in range(len(all_series)):
            for jdx in range(idx + 1, len(all_series)):
                corr = self._aligned_correlation(all_series[idx], all_series[jdx])
                if np.isfinite(corr):
                    rows.append(
                        {
                            "Market": "Correlation",
                            "Tenor": f"{all_series[idx].name} vs {all_series[jdx].name}",
                            "Level": _format_number(corr, 2),
                            "Daily Change": "",
                            "Weekly Change": "",
                            "1Y Percentile": "",
                            "5Y Percentile": "",
                            "Source": f"{all_series[idx].source} | {all_series[jdx].source}",
                            "Timestamp": max(all_series[idx].timestamp, all_series[jdx].timestamp),
                        }
                    )
        return rows

    def _vol_rate_correlation_rows(self, series_list: Sequence[TimeSeries]) -> List[Mapping[str, str]]:
        rows: List[Mapping[str, str]] = []
        for idx in range(len(series_list)):
            for jdx in range(idx + 1, len(series_list)):
                corr = self._aligned_correlation(series_list[idx], series_list[jdx])
                if np.isfinite(corr):
                    rows.append(
                        {
                            "Surface": "Vol-rate correlation",
                            "Average ATM Change": _format_number(corr, 2),
                            "Skew Change": "",
                            "Source": f"{series_list[idx].source} | {series_list[jdx].source}",
                            "Timestamp": max(series_list[idx].timestamp, series_list[jdx].timestamp),
                        }
                    )
        return rows

    def _event_impact_summary(self, event: PolicyEvent) -> str:
        parts = []
        if event.rates_impact_bps:
            parts.append(
                "curve impact "
                + ", ".join(f"{key} {_format_signed(value, 1)}" for key, value in event.rates_impact_bps.items())
            )
        if event.basis_impact_bps is not None:
            parts.append(f"basis {_format_signed(event.basis_impact_bps, 1)}")
        if event.ccs_impact_bps is not None:
            parts.append(f"CCS {_format_signed(event.ccs_impact_bps, 1)}")
        return "; ".join(parts) if parts else event.summary

    def _event_study_summary(self, study: EventStudy) -> str:
        bits = []
        for key, values in study.outcomes.items():
            arr = np.asarray(values, dtype=float)
            bits.append(f"{key} avg {_format_number(float(arr.mean()), 1)}bp")
        return ", ".join(bits)

    def _event_study_primary_move(self, study: EventStudy) -> float:
        first_key = next(iter(study.outcomes))
        arr = np.asarray(study.outcomes[first_key], dtype=float)
        return float(arr.mean())

    def _curve_factor_history(self, curve: CurveDataset) -> Dict[str, List[float]]:
        by_tenor = curve.by_tenor()
        dates = list(curve.series[0].dates)
        target_2 = _closest_value(curve.tenors, 2.0)
        target_5 = _closest_value(curve.tenors, 5.0)
        target_10 = _closest_value(curve.tenors, 10.0)
        target_30 = _closest_value(curve.tenors, 30.0)
        level = []
        slope = []
        curvature = []
        for idx in range(len(dates)):
            y2 = by_tenor[target_2].values[idx]
            y5 = by_tenor[target_5].values[idx]
            y10 = by_tenor[target_10].values[idx]
            y30 = by_tenor[target_30].values[idx]
            level.append(10000.0 * np.mean([y2, y5, y10, y30]))
            slope.append(10000.0 * (y10 - y2))
            curvature.append(10000.0 * (2.0 * y10 - y5 - y30))
        return {"dates": dates, "level": level, "slope": slope, "curvature": curvature}

    def _closest_historical_analog(self, curve: CurveDataset) -> Dict[str, object]:
        by_tenor = curve.by_tenor()
        tenors = curve.tenors
        matrix = np.asarray([by_tenor[tenor].values for tenor in tenors], dtype=float).T
        z_matrix = _standardize_matrix(matrix)
        current = z_matrix[-1]
        if z_matrix.shape[0] <= 1:
            return {
                "date": curve.latest_date,
                "distance": 0.0,
                "regime": "single-observation",
                "source": ", ".join(sorted({item.source for item in curve.series})),
                "timestamp": curve.latest_timestamp,
            }
        distances = np.linalg.norm(z_matrix[:-1] - current, axis=1)
        index = int(np.argmin(distances))
        factor_history = self._curve_factor_history(curve)
        slope_pct = _percentile_rank(factor_history["slope"], factor_history["slope"][-1])
        curvature_pct = _percentile_rank(factor_history["curvature"], factor_history["curvature"][-1])
        if slope_pct > 66 and curvature_pct > 66:
            regime = "bear-steep / high-curvature"
        elif slope_pct < 33 and curvature_pct < 33:
            regime = "flat / low-curvature"
        else:
            regime = "mixed"
        return {
            "date": curve.series[0].dates[index],
            "distance": float(distances[index]),
            "regime": regime,
            "source": ", ".join(sorted({item.source for item in curve.series})),
            "timestamp": curve.latest_timestamp,
        }

    @staticmethod
    def _atm_index(pillars: Sequence[str]) -> int:
        """Return the ATM pillar index, falling back to the middle pillar."""
        for idx, pillar in enumerate(pillars):
            if pillar.strip().upper() == "ATM":
                return idx
        return len(pillars) // 2

    @staticmethod
    def _aligned_correlation(left: TimeSeries, right: TimeSeries) -> float:
        left_map = {date: value for date, value in zip(left.dates, left.values)}
        shared_dates = [date for date in right.dates if date in left_map]
        if len(shared_dates) < 3:
            return float("nan")
        left_values = np.asarray([left_map[date] for date in shared_dates], dtype=float)
        right_map = {date: value for date, value in zip(right.dates, right.values)}
        right_values = np.asarray([right_map[date] for date in shared_dates], dtype=float)
        left_diff = np.diff(left_values)
        right_diff = np.diff(right_values)
        if len(left_diff) < 2:
            return float("nan")
        return float(np.corrcoef(left_diff, right_diff)[0, 1])

    @staticmethod
    def _move_description(quotes: Sequence[TenorQuote]) -> str:
        avg_move = float(np.mean([item.daily_change for item in quotes]))
        if avg_move > 0:
            return "widening / richer pricing"
        if avg_move < 0:
            return "tightening / cheaper pricing"
        return "stable pricing"

    def _line_chart_svg(
        self,
        title: str,
        x_values: Sequence[float],
        series: Sequence[Tuple[str, Sequence[float]]],
        x_label: str,
        y_label: str,
    ) -> str:
        if not series or not x_values:
            return ""
        width = 640
        height = 280
        margin = 50
        colors = ["#0c3b5d", "#0091ad", "#9c6644", "#5a189a", "#e76f51"]
        y_values = [value for _, values in series for value in values]
        y_min = min(y_values)
        y_max = max(y_values)
        if math.isclose(y_min, y_max):
            y_min -= 1.0
            y_max += 1.0
        x_min = min(x_values)
        x_max = max(x_values)
        if math.isclose(x_min, x_max):
            x_min -= 1.0
            x_max += 1.0

        def x_pos(value: float) -> float:
            return margin + (width - 2 * margin) * (value - x_min) / (x_max - x_min)

        def y_pos(value: float) -> float:
            return height - margin - (height - 2 * margin) * (value - y_min) / (y_max - y_min)

        parts = [
            f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg'>",
            f"<text x='{width / 2:.0f}' y='24' text-anchor='middle' font-size='16' fill='#0c3b5d'>{html.escape(title)}</text>",
            f"<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' stroke='#7f8fa4'/>",
            f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height - margin}' stroke='#7f8fa4'/>",
            f"<text x='{width / 2:.0f}' y='{height - 10}' text-anchor='middle' font-size='12'>{html.escape(x_label)}</text>",
            f"<text x='16' y='{height / 2:.0f}' text-anchor='middle' font-size='12' transform='rotate(-90, 16, {height / 2:.0f})'>{html.escape(y_label)}</text>",
        ]
        for idx, (name, values) in enumerate(series):
            points = " ".join(f"{x_pos(float(x)):.1f},{y_pos(float(y)):.1f}" for x, y in zip(x_values, values))
            color = colors[idx % len(colors)]
            parts.append(f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{points}'/>")
            for x, y in zip(x_values, values):
                parts.append(f"<circle cx='{x_pos(float(x)):.1f}' cy='{y_pos(float(y)):.1f}' r='2.5' fill='{color}'/>")
            parts.append(
                f"<text x='{width - margin + 8}' y='{margin + 16 * idx:.0f}' font-size='11' fill='{color}'>{html.escape(name)}</text>"
            )
        parts.append("</svg>")
        return "".join(parts)

    def _time_series_chart_svg(
        self,
        title: str,
        dates: Sequence[str],
        series: Sequence[Tuple[str, Sequence[float]]],
        y_label: str,
    ) -> str:
        if not dates or not series:
            return ""
        x_values = list(range(len(dates)))
        return self._line_chart_svg(title, x_values, series, "Observation", y_label)

    def _bar_chart_svg(self, title: str, values: Sequence[Tuple[str, float]], y_label: str) -> str:
        if not values:
            return ""
        width = 640
        height = 260
        margin = 50
        numeric = [float(value) for _, value in values]
        max_abs = max(abs(min(numeric)), abs(max(numeric)), 1.0)
        zero_y = height / 2 if min(numeric) < 0 else height - margin
        parts = [
            f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg'>",
            f"<text x='{width / 2:.0f}' y='24' text-anchor='middle' font-size='16' fill='#0c3b5d'>{html.escape(title)}</text>",
            f"<line x1='{margin}' y1='{zero_y:.1f}' x2='{width - margin}' y2='{zero_y:.1f}' stroke='#7f8fa4'/>",
            f"<text x='16' y='{height / 2:.0f}' text-anchor='middle' font-size='12' transform='rotate(-90, 16, {height / 2:.0f})'>{html.escape(y_label)}</text>",
        ]
        slot = (width - 2 * margin) / len(values)
        for idx, (label, value) in enumerate(values):
            bar_height = (height - 2 * margin) * (abs(float(value)) / max_abs) / 2
            x = margin + idx * slot + slot * 0.15
            y = zero_y - bar_height if value >= 0 else zero_y
            color = "#0c3b5d" if value >= 0 else "#e76f51"
            parts.append(
                f"<rect x='{x:.1f}' y='{y:.1f}' width='{slot * 0.7:.1f}' height='{bar_height:.1f}' fill='{color}'/>"
            )
            parts.append(
                f"<text x='{x + slot * 0.35:.1f}' y='{height - margin + 14:.1f}' text-anchor='middle' font-size='10'>{html.escape(label)}</text>"
            )
        parts.append("</svg>")
        return "".join(parts)

    def _heatmap_svg(
        self,
        title: str,
        rows: Sequence[str],
        columns: Sequence[str],
        matrix: Sequence[Sequence[float]],
    ) -> str:
        if not rows or not columns:
            return ""
        matrix_np = np.asarray(matrix, dtype=float)
        width = 640
        height = 300
        left = 100
        top = 50
        cell_w = (width - left - 20) / len(columns)
        cell_h = (height - top - 30) / len(rows)
        min_value = float(np.min(matrix_np))
        max_value = float(np.max(matrix_np))
        span = max(max_value - min_value, 1e-9)

        def color(value: float) -> str:
            normalized = (value - min_value) / span
            red = int(255 * normalized)
            blue = int(255 * (1.0 - normalized))
            green = 80
            return f"rgb({red},{green},{blue})"

        parts = [
            f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg'>",
            f"<text x='{width / 2:.0f}' y='24' text-anchor='middle' font-size='16' fill='#0c3b5d'>{html.escape(title)}</text>",
        ]
        for col_idx, column in enumerate(columns):
            parts.append(
                f"<text x='{left + col_idx * cell_w + cell_w / 2:.1f}' y='{top - 12:.1f}' text-anchor='middle' font-size='11'>{html.escape(column)}</text>"
            )
        for row_idx, row in enumerate(rows):
            parts.append(
                f"<text x='{left - 8:.1f}' y='{top + row_idx * cell_h + cell_h / 2 + 4:.1f}' text-anchor='end' font-size='11'>{html.escape(row)}</text>"
            )
            for col_idx, value in enumerate(matrix_np[row_idx]):
                x = left + col_idx * cell_w
                y = top + row_idx * cell_h
                parts.append(
                    f"<rect x='{x:.1f}' y='{y:.1f}' width='{cell_w:.1f}' height='{cell_h:.1f}' fill='{color(float(value))}' stroke='#ffffff'/>"
                )
                parts.append(
                    f"<text x='{x + cell_w / 2:.1f}' y='{y + cell_h / 2 + 4:.1f}' text-anchor='middle' font-size='10' fill='#ffffff'>{float(value):.2f}</text>"
                )
        parts.append("</svg>")
        return "".join(parts)

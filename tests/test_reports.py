"""Tests for reports module."""

from __future__ import annotations

from pathlib import Path

import pytest

from quantpy.reports import (
    APACMarketMonitor,
    APACMarketMonitorInput,
    CurveDataset,
    CurveSeries,
    DeskExposure,
    EventStudy,
    MacroRelease,
    PolicyEvent,
    SourcedObservation,
    TenorQuote,
    TenorQuoteSet,
    TimeSeries,
    VolSurface,
)


def _make_curve(market: str, currency: str, base: float, slope: float) -> CurveDataset:
    dates = [f"2026-01-{day:02d}" for day in range(1, 31)]
    tenors = [2.0, 5.0, 10.0, 30.0]
    series = []
    for tenor in tenors:
        values = [
            base + slope * tenor + 0.00005 * idx + 0.00003 * ((idx + int(tenor)) % 3)
            for idx in range(len(dates))
        ]
        series.append(
            CurveSeries(
                tenor=tenor,
                dates=dates,
                values=values,
                source=f"{market} official data",
                timestamp="2026-03-08T04:42:37+00:00",
            )
        )
    return CurveDataset(market=market, currency=currency, series=series)


@pytest.fixture
def monitor_input() -> APACMarketMonitorInput:
    dates = [f"2026-02-{day:02d}" for day in range(1, 21)]
    basis_values = [-25.0 + 0.4 * idx for idx in range(len(dates))]
    funding_values = [-18.0 + 0.3 * idx for idx in range(len(dates))]
    vol_rate_values = [0.45 + 0.01 * idx for idx in range(len(dates))]
    rate_move_values = [5.0 + 0.4 * idx for idx in range(len(dates))]

    return APACMarketMonitorInput(
        apac_curves=[
            _make_curve("Australia", "AUD", 0.0350, 0.00018),
            _make_curve("Japan", "JPY", 0.0080, 0.00009),
        ],
        benchmark_curves=[
            _make_curve("United States", "USD", 0.0400, 0.00016),
            _make_curve("Euro Area", "EUR", 0.0220, 0.00012),
        ],
        fx_swap_quotes=[
            TenorQuoteSet(
                name="USD/JPY FX swap points",
                description="Tokyo close indicative points",
                quotes=[
                    TenorQuote("1Y", 520.0, 510.0, 500.0, "Broker composite", "2026-03-08T04:42:37+00:00"),
                    TenorQuote("2Y", 1025.0, 1008.0, 995.0, "Broker composite", "2026-03-08T04:42:37+00:00"),
                    TenorQuote("5Y", 2480.0, 2450.0, 2410.0, "Broker composite", "2026-03-08T04:42:37+00:00"),
                    TenorQuote("10Y", 4700.0, 4655.0, 4605.0, "Broker composite", "2026-03-08T04:42:37+00:00"),
                ],
            )
        ],
        implied_funding_series=[
            TimeSeries(
                name="JPY implied funding spread",
                dates=dates,
                values=funding_values,
                source="Treasury/cross-currency desk snapshot",
                timestamp="2026-03-08T04:42:37+00:00",
                unit="bp",
            )
        ],
        ccs_quotes=[
            TenorQuoteSet(
                name="USD/JPY CCS basis",
                quotes=[
                    TenorQuote("1Y", -31.0, -29.0, -28.0, "Interdealer runs", "2026-03-08T04:42:37+00:00"),
                    TenorQuote("2Y", -34.0, -33.0, -31.5, "Interdealer runs", "2026-03-08T04:42:37+00:00"),
                    TenorQuote("5Y", -41.0, -39.5, -38.0, "Interdealer runs", "2026-03-08T04:42:37+00:00"),
                    TenorQuote("10Y", -47.0, -45.0, -44.0, "Interdealer runs", "2026-03-08T04:42:37+00:00"),
                ],
            )
        ],
        basis_series=[
            TimeSeries(
                name="JPY/USD 5Y basis",
                dates=dates,
                values=basis_values,
                source="Interdealer runs",
                timestamp="2026-03-08T04:42:37+00:00",
                unit="bp",
            )
        ],
        liquidity_indicators=[
            SourcedObservation(
                label="3M basis bid/ask",
                value=1.5,
                source="Dealer runs",
                timestamp="2026-03-08T04:42:37+00:00",
                unit="bp",
            ),
            SourcedObservation(
                label="Tokyo turn premium",
                value=2.2,
                source="Funding desk",
                timestamp="2026-03-08T04:42:37+00:00",
                unit="bp",
            ),
        ],
        flow_commentary=[
            "Month-end reserve-manager receiving kept the 5Y basis from widening more aggressively.",
            "Fast-money paid the 10Y CCS sector after the local rates sell-off steepened the curve.",
        ],
        policy_events=[
            PolicyEvent("BoJ", "2026-03-07", "Policy unchanged, JGB purchases trimmed", "BoJ kept the policy rate steady while signalling a slower balance-sheet runoff.", "BoJ statement", "2026-03-08T04:42:37+00:00", {"2Y": 2.0, "10Y": 6.0}, -1.0, -0.5),
            PolicyEvent("RBA", "2026-03-05", "25bp hike with hawkish hold bias", "The Board highlighted sticky services inflation and resilient labour demand.", "RBA statement", "2026-03-08T04:42:37+00:00", {"2Y": 8.0, "10Y": 4.0}),
            PolicyEvent("RBNZ", "2026-02-26", "Hold with easing optionality", "Forward guidance retained a mild easing bias conditional on disinflation.", "RBNZ statement", "2026-03-08T04:42:37+00:00"),
            PolicyEvent("BoK", "2026-02-25", "Hold amid FX stability concerns", "The committee balanced soft growth against KRW stability considerations.", "BoK statement", "2026-03-08T04:42:37+00:00"),
            PolicyEvent("MAS", "2026-01-29", "Slightly flatter S$NEER slope", "MAS reduced the slope of the policy band while keeping width and centre unchanged.", "MAS statement", "2026-03-08T04:42:37+00:00"),
            PolicyEvent("PBoC", "2026-02-20", "LPR cut and liquidity injection", "The PBoC complemented a benchmark rate cut with targeted liquidity support.", "PBoC statement", "2026-03-08T04:42:37+00:00"),
        ],
        event_studies=[
            EventStudy(
                label="BoJ taper analogs",
                bank="BoJ",
                event_type="taper",
                outcomes={"2Y": [1.2, 2.1, 1.7], "10Y": [4.5, 6.0, 5.2], "2s10s": [3.3, 3.9, 3.5]},
                source="Historical event study",
                timestamp="2026-03-08T04:42:37+00:00",
            ),
            EventStudy(
                label="RBA hike analogs",
                bank="RBA",
                event_type="hike",
                outcomes={"2Y": [6.0, 8.0, 7.0], "10Y": [2.0, 3.0, 4.0], "2s10s": [-4.0, -5.0, -3.0]},
                source="Historical event study",
                timestamp="2026-03-08T04:42:37+00:00",
            ),
        ],
        macro_releases=[
            MacroRelease("Japan", "CPI ex fresh food", "2026-03-01", 2.4, 2.2, 2.1, "Upside inflation surprise lifted terminal-rate pricing and cheapened front-end JPY basis.", "Statistics Bureau of Japan", "2026-03-08T04:42:37+00:00", unit="%"),
            MacroRelease("Australia", "Employment change", "2026-03-04", 41.0, 22.0, 18.0, "The labour-market beat re-priced the RBA path and bear-flattened ACGBs.", "ABS labour force", "2026-03-08T04:42:37+00:00", unit="k"),
            MacroRelease("China", "Official manufacturing PMI", "2026-03-01", 50.8, 50.2, 49.8, "The PMI rebound supported CNH funding and narrowed the front-end basis.", "NBS China", "2026-03-08T04:42:37+00:00"),
        ],
        forward_guidance=[
            "BoJ balance-sheet normalization points to gradual JPY curve steepening with only modest front-end repricing.",
            "RBA guidance keeps AUD front-end carry attractive, but richer levels warrant selective flattener expressions.",
        ],
        linear_exposures=[
            DeskExposure("AUD 2s10s flattener", 12.5, 1.8, 0.6, -0.3, "Carry remains positive while policy risk skews toward further front-end richening.", "Linear desk", "2026-03-08T04:42:37+00:00"),
            DeskExposure("JPY 10s30s steepener", -8.0, 0.7, 0.5, 0.1, "Taper headlines historically steepen the long end faster than the belly.", "Linear desk", "2026-03-08T04:42:37+00:00"),
        ],
        relative_value_opportunities=[
            "Maintain AUD 2s10s flatteners versus neutral JPY as the RBA retains the stronger inflation reaction function.",
            "5s10s30s butterflies in JPY screen rich versus the closest taper analogs.",
        ],
        cross_market_rv=[
            "JGBs remain 18-22bp rich to matched-maturity USTs on the 10Y point after hedging costs.",
            "ACGBs look optically cheap versus USTs in 5Y once adjusted for carry and expected roll.",
        ],
        flow_insights=[
            "Japanese lifers continued to add unhedged ACGB exposure, dampening AUD long-end outright cheapening.",
            "Corporate issuance swaps paid USD/JPY basis in 3Y-5Y, consistent with the recent funding spread widening.",
        ],
        options_surfaces=[
            VolSurface(
                name="JPY swaption ATM/skew",
                expiries=["1M", "3M", "6M"],
                pillars=["25P", "ATM", "25C"],
                current=[[40.0, 42.0, 44.0], [43.0, 45.0, 47.5], [46.0, 48.0, 50.0]],
                previous=[[39.0, 41.5, 43.2], [42.5, 44.1, 46.0], [45.2, 47.0, 48.8]],
                source="Broker vol sheet",
                timestamp="2026-03-08T04:42:37+00:00",
            )
        ],
        vol_rate_series=[
            TimeSeries("JPY vol changes", dates, vol_rate_values, "Broker vol sheet", "2026-03-08T04:42:37+00:00"),
            TimeSeries("JPY rate changes", dates, rate_move_values, "BoJ official data", "2026-03-08T04:42:37+00:00"),
        ],
        option_scenarios=[
            "A 25bp bull-flattening would likely cheapen short-expiry payer tails while lifting receiver gamma in 1Yx10Y.",
            "A 15bp bear-steepening shock should reprice upper-right implieds higher given positive vol-rate beta.",
        ],
    )


def test_apac_market_monitor_generates_required_sections(monitor_input):
    """The report should render the fixed requested structure with cited content."""
    monitor = APACMarketMonitor()
    report = monitor.build(monitor_input)

    assert list(report.sections.keys()) == [
        "APAC Rates Overview",
        "FX Swap & CCS Market Moves",
        "Central Bank Policy & Macro Events",
        "Implications for the Linear Rates Desk",
        "Implications for the Options Rates Desk",
        "Historical Curve-Shape Analysis",
        "Data Appendix (with Charts & Tables)",
    ]
    html_output = report.to_html()
    assert "APAC Rates Overview" in html_output
    assert "FX Swap &amp; CCS Market Moves" in html_output
    assert "BoJ" in html_output
    assert "RBA" in html_output
    assert "Source" in html_output
    assert "Timestamp" in html_output
    assert "<svg" in html_output


def test_apac_market_monitor_tables_and_charts_are_populated(monitor_input):
    """Key analytics tables and charts should be produced for rates, FX/CCS, and options."""
    report = APACMarketMonitor().build(monitor_input)

    rates_rows = report.tables["APAC Rates Overview"]
    fx_rows = report.tables["FX Swap & CCS Market Moves"]
    options_rows = report.tables["Implications for the Options Rates Desk"]

    assert any(row["Market"] == "Australia" and row["Tenor"] == "10Y" for row in rates_rows)
    assert any("Correlation" == row["Market"] for row in fx_rows)
    assert any(row["Surface"] == "JPY swaption ATM/skew" for row in options_rows)
    assert "Current APAC yield curves" in report.charts["APAC Rates Overview"]
    assert "Basis and funding time series" in report.charts["FX Swap & CCS Market Moves"]
    assert "surface change" in report.charts["Implications for the Options Rates Desk"]


def test_apac_market_monitor_can_save_html_report(monitor_input, tmp_path: Path):
    """The report HTML should be writable for downstream publishing workflows."""
    report = APACMarketMonitor().build(monitor_input)
    output_path = tmp_path / "apac_monitor.html"
    saved_path = report.save_html(str(output_path))

    assert saved_path == str(output_path)
    assert output_path.exists()
    assert "Cross-Currency Swap Market Monitor" in output_path.read_text(encoding="utf-8")

"""Reviewed geopolitical exposure links and explicit residual stress assumptions.

Issuer filings establish an economic channel, NOT a stock-return coefficient.
The finite stress ranges below are analyst assumptions, not fitted loss bounds.
One shared severity draw drives the supply-chain add-ons, including the same
issuer held directly and through funds. An original fund's beta is counted once.
Gold is an alternative whole-position channel and REPLACES its market beta; its
measured episodes identify no sign, so it is drawn independently of that severity.
No provider calls, clock access, forecast or calibration claim is made here.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import math

from .portfolio import ExposureReport, Portfolio, WEIGHT_TOLERANCE, _finite_number


SOURCES = {
    "AAPL": {
        "title": "Apple Inc. 2025 Form 10-K",
        "url": "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm",
        "published_on": "2025-10-31", "reviewed_on": "2026-10-23",
        "locator": "Item 1A, Business Risks, printed pages 7-8; Note 12",
        "supports": "Manufacturing and component partners include Taiwan and other Asian locations; single-source components and disruption of outsourced manufacturing can harm operations.",
        "does_not_support": "A numerical Taiwan share of Apple's production or a stock-price loss conditional on a blockade.",
    },
    "AMD": {
        "title": "Advanced Micro Devices, Inc. 2025 Form 10-K",
        "url": "https://www.sec.gov/Archives/edgar/data/2488/000000248826000018/amd-20251227.htm",
        "published_on": "2026-02-04", "reviewed_on": "2026-10-23",
        "locator": "Item 1A, third-party manufacturing; export restrictions; Note 4",
        "supports": "TSMC produces AMD microprocessor/GPU wafers at nodes of 7 nm or smaller. MI308 was subject to the April 2025 US export license requirement; inventory charges are accounting effects, not an equity return.",
        "does_not_support": "A causal stock return, a proportional mapping from inventory charges to market capitalization, or automatic application to every future restriction.",
    },
    "TSM": {
        "title": "Taiwan Semiconductor Manufacturing Company 2025 Form 20-F",
        "url": "https://investor.tsmc.com/sites/ir/sec-filings/2025_20F%20Report.pdf",
        "published_on": "2026-04-16", "reviewed_on": "2026-10-23",
        "locator": "Item 3, risk factors, printed pages 5 and 11 (PDF pages 10 and 16)",
        "supports": "Principal production facilities and most operating revenue are concentrated in Taiwan; military disruption can affect operations and investment value.",
        "does_not_support": "An estimated invasion/blockade loss or guaranteed recovery horizon.",
    },
    "GLD": {
        "title": "SPDR Gold Shares issuer objective and World Gold Council original analysis",
        "url": "https://www.spdrgoldshares.com/usa/gld/",
        "research_url": "https://www.gold.org/goldhub/gold-focus/2023/10/you-asked-we-answered-whats-impact-of-geopolitics-on-gold",
        "published_on": "2026-10-23", "research_published_on": "2023-10-13", "reviewed_on": "2026-10-23",
        "locator": "Issuer Key Features; WGC What is gold's geopolitical premium?, October 13, 2023",
        "supports": "GLD tracks bullion less expenses. WGC's conditional regression reports about +2.5% gold return per 100-unit GPR increase; other macro drivers can offset that channel.",
        "does_not_support": "A Taiwan-to-GPR mapping, unconditional safe-haven guarantee, or a calibrated prediction interval for GLD.",
        "measured_episodes": "reports/r10_validation/geopolitical_episodes.json",
    },
    "MU": {
        "title": "Micron Technology, Inc. 2025 Form 10-K",
        "url": "https://www.sec.gov/Archives/edgar/data/723125/000072312525000028/mu-20250828.htm",
        "published_on": "2025-10-03", "reviewed_on": "2026-10-24",
        "locator": "Item 1A, We face geopolitical and other risks associated with our international operations",
        "supports": "\"A majority of our DRAM production output in 2025 was from our fabrication facilities in Taiwan\"; any loss of output could have a material adverse effect.",
        "does_not_support": "A Taiwan share of total revenue, an estimated blockade loss or a stock-price sensitivity.",
    },
}
# Gold over the LPL-table episodes inside GLD's price history (20 sessions from onset), measured
# by scripts/r10_measure_geopolitical_episodes.py: whole-position return, the channel GLD uses here.
GOLD_EPISODES = {"episodes": 11, "positive": 6, "median": .0084, "min": -.1427, "max": .0889,
                 "report": "reports/r10_validation/geopolitical_episodes.json"}


def _stress(point, low, high, label, *, mode="RESIDUAL_ADDON", rationale, basis="DECLARED_STRESS_NOT_ESTIMATED_EVENT_EFFECT"):
    return {"point": point, "range": [low, high], "unit": "fractional return over the scenario window",
            "label": label, "mode": mode, "basis": basis,
            "anchors": rationale, "calibration_status": "NOT_EMPIRICALLY_CALIBRATED"}


CHANNELS = {
    "TAIWAN_MAJOR_MILITARY_ACTION": {
        "AAPL": _stress(-.075, -.15, 0., "Apple component and manufacturing disruption", rationale="Declared extra Apple loss from zero to 15%, beyond the common market move. Filings establish the link; they do not calibrate these endpoints or the -7.5% midpoint."),
        "AMD": _stress(-.10, -.20, 0., "AMD foundry and packaging disruption", rationale="Declared extra AMD loss from zero to 20%. The same stress range as NVIDIA is a scenario choice, not equal estimated vulnerability."),
        "TSM": _stress(-.15, -.30, 0., "TSMC concentrated production disruption", rationale="Declared extra TSM loss from zero to 30%, reflecting a chosen direct-production stress. These are finite scenario endpoints, not maximum possible losses."),
        "MU": _stress(-.10, -.20, 0., "Micron Taiwan DRAM fabrication disruption", rationale="Declared extra Micron loss from zero to 20%: most of its DRAM output comes from its own Taiwan fabs. The same stress range as the foundry-dependent issuers is a scenario choice, not an estimated vulnerability."),
        "GLD": {**_stress(0., GOLD_EPISODES["min"], .10, "Gold demand, liquidity and opportunity-cost response", mode="REPLACES_MARKET_BETA",
                       basis="MEASURED_EPISODES_SIGN_NOT_IDENTIFIED",
                       rationale="Over the 11 LPL-table episodes in GLD's history gold rose in 6 and fell in 5 within 20 sessions "
                                 "(median +0.8%, from -14.3% to +8.9%; the largest fall came with a broad sell-off). No safe-haven "
                                 "sign is identified, so the reference is zero; the range keeps a +10% upper end for a larger war."),
                "dependence": "INDEPENDENT_OF_SUPPLY_SEVERITY"},
    },
    "TAIWAN_LIMITED_USE_OF_FORCE": {
        "AAPL": _stress(0., -.03, 0., "Apple limited disruption sensitivity", rationale="An encounter need not interrupt production: zero reference, declared downside sensitivity to -3%."),
        "AMD": _stress(0., -.05, 0., "AMD limited disruption sensitivity", rationale="Zero reference; declared residual downside to -5%, not a measured encounter effect."),
        "TSM": _stress(0., -.075, 0., "TSMC limited disruption sensitivity", rationale="Zero reference; declared residual downside to -7.5%, not a measured encounter effect."),
        "MU": _stress(0., -.05, 0., "Micron limited disruption sensitivity", rationale="Zero reference; declared residual downside to -5%, not a measured encounter effect."),
        "GLD": {**_stress(0., GOLD_EPISODES["min"], GOLD_EPISODES["max"], "Gold sign-uncertain response", mode="REPLACES_MARKET_BETA",
                       basis="MEASURED_EPISODES_SIGN_NOT_IDENTIFIED",
                       rationale="No directional reference: over the 11 LPL-table episodes gold rose in 6 and fell in 5 within 20 "
                                 "sessions. The range spans those episodes, -14.3% to +8.9%."),
                "dependence": "INDEPENDENT_OF_SUPPLY_SEVERITY"},
    },
    "US_AI_CHIP_EXPORT_CURB": {
        "AMD": _stress(-.02, -.06, 0., "AMD advanced-chip export access", rationale="The MI308 filing establishes exposure to licensing, not a price sensitivity. Extra stock loss 0% to 6%, with a -2% reference, is an explicit stress for a rule that reaches AMD products; no inventory-charge-to-equity conversion is made."),
    },
}


def channel_specs(profile_name, *, as_of, nvda_spec=None, overrides=None):
    """Return dated source-backed channels; optional changes stay labeled assumed.

An issuer is eligible only when its cited filing predates the analysis. The
    gold channel's availability is dated to the current issuer-page review;
    its older research date is retained separately.
"""
    stamp = datetime.fromisoformat(str(as_of).replace("Z", "+00:00"))
    if stamp.utcoffset() is None:
        raise ValueError("Channel analysis timestamp must be timezone-aware")
    specs = deepcopy(CHANNELS.get(profile_name, {}))
    excluded = []
    for symbol in list(specs):
        source = SOURCES[symbol]
        # These links were reviewed for R10. Do not backfill a historical run
        # with a source package that was unavailable to that implementation.
        if stamp.date().isoformat() < source["published_on"]:
            excluded.append({"symbol": symbol, "reason": "SOURCE_NOT_YET_PUBLISHED_AT_ANALYSIS"})
            del specs[symbol]
            continue
        specs[symbol]["source"] = deepcopy(source)
    if nvda_spec is not None:
        specs["NVDA"] = {**deepcopy(nvda_spec), "label": "NVIDIA supply/export residual",
                         "mode": "RESIDUAL_ADDON", "calibration_status": "NOT_EMPIRICALLY_CALIBRATED"}
    for symbol, override in (overrides or {}).items():
        if symbol not in specs or not isinstance(override, dict) or set(override) != {"point", "range"}:
            raise ValueError("Overrides require a reviewed channel and exactly point/range")
        point = _finite_number(override["point"], "Channel point")
        bounds = override["range"]
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError("Channel range requires two endpoints")
        lo, hi = (_finite_number(x, "Channel bound") for x in bounds)
        if not -1 <= lo <= point <= hi <= 1:
            raise ValueError("Channel overrides must satisfy -1 <= low <= point <= high <= 1")
        specs[symbol].update(point=point, range=[lo, hi], basis="OPERATOR_DECLARED_STRESS_OVERRIDE",
                             calibration_status="NOT_EMPIRICALLY_CALIBRATED")
    return specs, excluded


def issuer_exposure_weights(portfolio, exposure_report, symbols):
    """Exact original-position attribution; no inferred or normalized holdings.

Only stock identities marked direct/look_through enter issuer residuals. Gold
is intentionally handled as the original GLD position, not invented holdings.
"""
    if not isinstance(portfolio, Portfolio) or not isinstance(exposure_report, ExposureReport):
        raise ValueError("Validated portfolio and exposure ledger are required")
    positions = {p.symbol: p for p in portfolio.positions}
    masses = {s: [] for s in positions}
    weights = {s: {} for s in positions}
    ledger = []
    for entry in exposure_report.entries:
        if entry.original_symbol not in positions:
            raise ValueError("Exposure ledger belongs to another portfolio")
        w = _finite_number(entry.portfolio_weight, "Exposure weight")
        if w < 0:
            raise ValueError("Negative issuer exposure is unsupported")
        if entry.status == "direct" and (entry.symbol != entry.original_symbol or entry.asset_type != positions[entry.original_symbol].asset_type):
            raise ValueError("Direct exposure identity must match the original position")
        masses[entry.original_symbol].append(w)
        if entry.symbol in symbols and entry.asset_type == "stock" and entry.status in {"direct", "look_through"}:
            own = weights[entry.original_symbol]
            own[entry.symbol] = own.get(entry.symbol, 0.) + w
            ledger.append({"issuer": entry.symbol, "original_symbol": entry.original_symbol,
                           "portfolio_weight": w, "status": entry.status, "as_of": entry.as_of,
                           "retrieved_at": entry.retrieved_at, "source_url": entry.source_url})
    for symbol, position in positions.items():
        if not math.isclose(math.fsum(masses[symbol]), position.weight, abs_tol=WEIGHT_TOLERANCE, rel_tol=0):
            raise ValueError("Exposure ledger does not preserve each original position's mass")
        if math.fsum(weights[symbol].values()) > position.weight + WEIGHT_TOLERANCE:
            raise ValueError("Issuer exposures exceed the original position")
    unresolved = math.fsum(e.portfolio_weight for e in exposure_report.entries if e.status == "unresolved_etf")
    if not math.isclose(unresolved, exposure_report.unresolved_weight, abs_tol=WEIGHT_TOLERANCE, rel_tol=0):
        raise ValueError("Unresolved exposure summary disagrees with its ledger")
    return weights, ledger


def build_multiissuer_sensitivity(portfolio, exposure_report, *, profile_name, nvda_spec):
    """Broad-risk sensitivity without assigning returns to unreviewed holdings."""
    specs, excluded = channel_specs(profile_name, as_of=exposure_report.evaluated_at, nvda_spec=nvda_spec)
    weights, ledger = issuer_exposure_weights(portfolio, exposure_report, specs)
    rows = []
    for issuer, spec in specs.items():
        weight = math.fsum(by_issuer.get(issuer, 0.) for by_issuer in weights.values())
        if spec["mode"] == "REPLACES_MARKET_BETA":
            weight = math.fsum(p.weight for p in portfolio.positions if p.symbol == issuer and p.asset_type == "etf")
        if weight <= 0:
            continue
        amount = portfolio.notional * weight
        rows.append({"issuer": issuer, "portfolio_weight": weight,
                     "dollars_per_one_percent_return": amount * .01,
                     "declared_component_pnl_usd": amount * spec["point"],
                     "declared_component_range_usd": [amount * x for x in spec["range"]],
                     **deepcopy(spec)})
    return {"schema": "castle.multiissuer_geopolitical_sensitivity.v1", "channels": rows,
            "identified_channel_weight": math.fsum(r["portfolio_weight"] for r in rows),
            "unresolved_holdings_weight": exposure_report.unresolved_weight,
            "excluded_sources": excluded, "exposure_ledger": ledger,
            "scope": "IDENTIFIED_CHANNEL_COMPONENTS_ONLY_NOT_FULL_PORTFOLIO_PNL",
            "declared_component_pnl_usd": math.fsum(r["declared_component_pnl_usd"] for r in rows),
            "declared_component_range_usd": [math.fsum(r["declared_component_range_usd"][i] for r in rows) for i in (0, 1)],
            "unreviewed_exposures_imputed": False, "probability_assigned": False,
            "interpretation": "Documented economic links with assumed shock magnitudes. No price, probability or causal event effect is inferred from issuer filings."}

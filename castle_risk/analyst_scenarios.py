"""Reproducible, editable economic *assumptions* for Castle Option 1.

The source links establish definitions/mechanisms, NOT the numeric price moves.
Returns below are explicitly authored stress premises, not observations, fitted
event effects, forecasts or prices derived from prediction-market contracts.

An event may admit an exact-source A estimate, a separate B analyst scenario,
both, or neither. This module never changes primary event outcomes/admission.
"""
from __future__ import annotations

import math
import re

from .scenario_lab import ScenarioInputError, evaluate_scenario
from .r107_evidence import payroll_context_for_positions


SCHEMA = "castle.analyst_assumption.v1"

# A dated, independently published *instrument parameter*, not an event
# response estimate. The 10 bp curve move below is still an analyst premise.
TLT_DURATION = {
    "effective_years": 14.90,
    "as_of": "2026-10-29",
    "source_url": "https://www.ishares.com/us/products/239454/ishares-20-year-treasury-bond-etf",
    "source_name": "iShares TLT portfolio characteristics: effective duration",
}
LONG_AUCTION_YIELD_SHOCK_BP = 10.0

# Shocks are percent event-YES minus event-NO *value gaps*, not moves from the
# current share price. The two-state pricing weight below is a displayed,
# deliberately UNIDENTIFIED sensitivity assumption. 50% is not a market quote.
# Fixed parameters make scenario arithmetic reproducible for any input NAV.
PROFILES = {
    "shutdown": {
        "condition": "A federal funding lapse persists for three weeks, delaying activity and increasing risk aversion.",
        "why": "CBO documented delayed spending and temporarily lower output during the 2018–19 shutdown. The assumed market-risk and flight-to-quality moves are illustrative, not CBO's asset-return estimates.",
        "source": ("Congressional Budget Office: effects of the 2018–19 shutdown", "https://www.cbo.gov/publication/54937"),
        "gap": {"equity": -.7, "tech": -.7, "bank": -.9, "industrial": -.9,
                "health": -.6, "energy": -.8, "utilities": -.4, "treasury_long": .6,
                "treasury_intermediate": .3, "aggregate_bond": .2, "gold": .4,
                "emerging": -.7, "usd": .1},
    },
    "china_tariff": {
        "condition": "A new broad China import-tariff package raises effective duties by about 10 percentage points.",
        "why": "Higher import costs and possible retaliation motivate an illustrative equity-risk shock; a Treasury/gold safe-haven response is a separate explicit assumption. The tariff magnitude does not mechanically identify asset returns.",
        "source": ("U.S. Trade Representative: Section 301 China tariff actions", "https://ustr.gov/issue-areas/enforcement/section-301-investigations/tariff-actions"),
        "gap": {"equity": -1.2, "tech": -1.7, "bank": -1.0, "industrial": -1.6,
                "health": -.8, "energy": -1.0, "utilities": -.6, "treasury_long": .4,
                "treasury_intermediate": .2, "aggregate_bond": .1, "gold": .8,
                "emerging": -2.0, "usd": .5},
    },
    "tariff": {
        "condition": "An import-tariff action causes a moderate risk-off repricing; product coverage and severity are held hypothetical.",
        "why": "Product-specific trade restrictions can affect margins and supply chains unevenly. The broad-market stress is a first-pass assumption, not an estimated effect on the named issuer.",
        "source": ("U.S. Trade Representative: trade enforcement", "https://ustr.gov/issue-areas/enforcement"),
        "gap": {"equity": -.8, "tech": -1.1, "bank": -.7, "industrial": -1.3,
                "health": -.5, "energy": -.8, "utilities": -.5, "treasury_long": .3,
                "treasury_intermediate": .15, "aggregate_bond": .1, "gold": .6,
                "emerging": -1.3, "usd": .4},
    },
    "long_auction": {
        "condition": "The Treasury signals larger 20-year auction supply and yields relevant to TLT rise by an assumed 10 bp.",
        "why": "Supply does not imply a 10 bp yield rise. That rise is assumed. For TLT alone, a dated iShares effective duration of 14.90 years maps 10 bp to approximately -1.49%; convexity, carry and curve-shape risk remain outside the first-order calculation. Other holdings retain separate analyst assumptions.",
        "source": ("U.S. Treasury: quarterly refunding and 20-year auction sizes", "https://home.treasury.gov/news/press-releases/sb0590"),
        "gap": {"equity": -.2, "tech": -.3, "bank": -.1, "industrial": -.2,
                "health": -.2, "energy": -.2, "utilities": -.4,
                "treasury_intermediate": -.5, "aggregate_bond": -.7, "gold": -.4,
                "emerging": -.3, "usd": .1},
        "alternative": "A larger auction does not guarantee higher yields; absent the stated 10 bp move, the TLT result does not apply.",
    },
    "payroll": {
        "condition": "The payroll report surprises roughly 100,000 jobs below consensus; the release itself is not the adverse outcome.",
        "why": "The Bureau of Labor Statistics publishes the scheduled release; the miss relative to consensus is an additional hypothetical condition. Equity risk-off and falling nominal yields are assumed jointly, not estimated from the release date.",
        "source": ("Bureau of Labor Statistics: Employment Situation release calendar", "https://www.bls.gov/schedule/news_release/empsit.htm"),
        "gap": {"equity": -.8, "tech": -.8, "bank": -1.0, "industrial": -1.1,
                "health": -.6, "energy": -1.1, "utilities": -.3, "treasury_long": .8,
                "treasury_intermediate": .35, "aggregate_bond": .3, "gold": .5,
                "emerging": -.8, "usd": -.1},
        "alternative": "A weaker jobs report could instead lift some equities if traders emphasize easier future Fed policy. The negative equity gaps are a chosen risk-off regime, not a universal event response.",
    },
    "basel": {
        "condition": "A final capital rule is materially stricter for large banks than investors had anticipated.",
        "why": "Capital requirements affect risk-weighted assets, payout capacity and business economics. The bank-price gap is a declared rule-severity assumption, not a valuation of the still-uncertain final regulation.",
        "source": ("Federal Reserve: 2026 proposed Basel III and capital-rule changes", "https://www.federalreserve.gov/publications/2026-june-supervision-and-regulation-report-regulatory-developments.htm"),
        "gap": {"equity": 0, "tech": 0, "bank": -1.5, "industrial": 0,
                "health": 0, "energy": 0, "utilities": 0, "treasury_long": 0,
                "treasury_intermediate": 0, "aggregate_bond": 0, "gold": 0,
                "emerging": 0, "usd": 0},
    },
    "cyber": {
        "condition": "The named issuer reports a material cyber incident on SEC Form 8-K Item 1.05; its equity-value gap is assumed to be -3%.",
        "why": "An Item 1.05 filing means the issuer determined the incident material. It does not establish loss severity or a -3% price effect. Other stocks are held flat as an explicit first-order assumption.",
        "source": ("SEC: material cybersecurity incidents and Item 1.05", "https://www.sec.gov/resources-small-businesses/small-business-compliance-guides/cybersecurity-risk-management-strategy-governance-incident-disclosure"),
        "issuer_gap": -3.0,
    },
    "antitrust": {
        "condition": "A federal antitrust complaint is formally filed against the named issuer; its equity-value gap is assumed to be -2%.",
        "why": "Filing is not an adverse judgment or an estimated remedy. The -2% stock-price gap is a declared moderate legal-risk stress; other equities are held flat.",
        "source": ("U.S. Department of Justice: Antitrust Division", "https://www.justice.gov/atr"),
        "issuer_gap": -2.0,
    },
    "credit_rating": {
        "condition": "The named issuer suffers a one-notch senior-credit rating downgrade and a -1.5% illustrative equity gap.",
        "why": "Financing-cost and risk-premium effects motivate a company-specific stress. Rating criteria and actual issuer bond spreads must be assessed separately.",
        "source": ("Federal Reserve: financial stability and credit markets", "https://www.federalreserve.gov/financial-stability.htm"),
        "issuer_gap": -1.5,
    },
    "rail_strike": {
        "condition": "A material Class I railroad work stoppage persists for five business days; the named rail stock gaps -2%.",
        "why": "Disrupted transportation volume is the economic channel. Five-day duration and -2% are hypothetical; this is not a measured strike response.",
        "source": ("Surface Transportation Board: rail industry oversight", "https://www.stb.gov/"),
        "issuer_gap": -2.0,
    },
    "hormuz_disruption": {
        "condition": "Commercial tanker traffic through Hormuz is severely disrupted for seven days; energy equities gain under an assumed oil-scarcity regime while other equities absorb higher costs.",
        "why": "EIA documents Hormuz as a major oil transit chokepoint and recorded both higher and more volatile crude prices amid 2026 disruptions. A 3% energy-equity gain and the other asset gaps are illustrative stress inputs, not EIA estimates or contract-implied moves.",
        "source": ("EIA: petroleum markets and Hormuz disruptions, July 2026", "https://www.eia.gov/Todayinenergy/detail.php?id=67865"),
        "gap": {"equity": -1.5, "tech": -2.0, "bank": -1.5, "industrial": -1.5,
                "health": -1.0, "energy": 3.0, "utilities": -1.2, "treasury_long": -.6,
                "treasury_intermediate": -.3, "aggregate_bond": -.2, "gold": 1.5,
                "emerging": -2.0, "usd": .3},
        "alternative": "A strong flight to Treasury safety could reverse the assumed negative bond return. A weak oil-demand regime could also erase the energy-equity gain.",
    },
    "opec_output_increase": {
        "condition": "OPEC+ formally raises aggregate oil output targets and the market prices a sustained supply increase; energy equities fall in a lower-oil-price stress.",
        "why": "OPEC publishes decisions adjusting required production. The assumed 3% energy-equity loss and lower-inflation responses are analyst premises; a target announcement alone does not measure barrels delivered or any share-price move.",
        "source": ("OPEC: official production-adjustment announcements, October 2026", "https://opec.org/pr-detail/616-4-october-2026.html"),
        "gap": {"equity": .3, "tech": .5, "bank": .2, "industrial": .5,
                "health": .2, "energy": -3.0, "utilities": .2, "treasury_long": .35,
                "treasury_intermediate": .2, "aggregate_bond": .15, "gold": -.3,
                "emerging": .4, "usd": -.1},
        "alternative": "If the extra production was already priced or offset by demand, oil and energy equities could barely move or even rise.",
    },
    "taiwan_shipping_blockade": {
        "condition": "A two-week blockade or enforced maritime quarantine seriously interrupts Taiwan-bound semiconductor logistics; TSM is assigned a -12% value gap in this severe illustration.",
        "why": "TSMC's Form 20-F describes concentrated Taiwan facilities, supply disruptions and geopolitical risks. It does not estimate a blockade's probability or a -12% equity response. The position gaps are scenario parameters only.",
        "source": ("TSMC: 2025 Form 20-F geopolitical and supply-chain risk disclosures", "https://investor.tsmc.com/sites/ir/sec-filings/2025_20F%20Report.pdf"),
        "gap": {"equity": -2.5, "tech": -3.5, "bank": -2.0, "industrial": -1.0,
                "health": -1.0, "energy": -1.5, "utilities": -.8, "treasury_long": 1.0,
                "treasury_intermediate": .6, "aggregate_bond": .4, "gold": 2.0,
                "emerging": -3.5, "usd": .5},
        "stock_overrides": {"TSM": -12.0},
        "alternative": "Defense suppliers may benefit from additional procurement; bond yields could instead rise with inflation or fiscal stress. The direct TSM shock is especially uncertain.",
    },
    "chip_export_controls": {
        "condition": "A new U.S. rule materially restricts advanced-chip or foundry-service exports to China beyond the current baseline.",
        "why": "Commerce BIS publishes advanced-node semiconductor end-use and export restrictions. The selected negative technology gaps assume lost demand and compliance costs; reshuffled customer demand could produce the opposite effect for some firms.",
        "source": ("Commerce BIS: EAR Part 744 advanced computing and semiconductor controls", "https://www.bis.gov/regulations/ear/744"),
        "gap": {"equity": -.6, "tech": -1.2, "bank": -.3, "industrial": -.3,
                "health": -.2, "energy": 0.0, "utilities": 0.0, "treasury_long": .3,
                "treasury_intermediate": .2, "aggregate_bond": .1, "gold": .2,
                "emerging": -1.5, "usd": .1},
        "stock_overrides": {"TSM": -3.0},
        "alternative": "Substitution toward unrestricted chips or non-China customers can offset or reverse the modeled issuer effect.",
    },
    "defense_funding_approval": {
        "condition": "A full-year U.S. Defense appropriations act is enacted, reducing funding uncertainty; named major defense contractors receive an illustrative +1.5% equity-value gap.",
        "why": "GAO documents how temporary appropriations disrupt selected Defense programs. Removing that uncertainty has a plausible contractor cash-flow channel, but the +1.5% response is an analyst stress, not a GAO asset-price estimate.",
        "source": ("GAO: effects of continuing resolutions on Defense programs, 2026", "https://www.gao.gov/assets/gao-26-107065-highlights.pdf"),
        "gap": {"equity": 0.0, "tech": 0.0, "bank": 0.0, "industrial": 0.0,
                "health": 0.0, "energy": 0.0, "utilities": 0.0, "treasury_long": 0.0,
                "treasury_intermediate": 0.0, "aggregate_bond": 0.0, "gold": 0.0,
                "emerging": 0.0, "usd": 0.0},
        "stock_overrides": {"LMT": 1.5, "RTX": 1.5, "NOC": 1.5, "GD": 1.5, "LHX": 1.5},
        "alternative": "If funding was anticipated or procurement allocations favor other suppliers, a named contractor may not reprice positively.",
    },
    "us_sovereign_downgrade": {
        "condition": "A major agency downgrades the U.S. long-term sovereign rating and a short-lived risk-premium shock lifts Treasury yields despite the downgrade's separate safe-haven effects.",
        "why": "Creditworthiness and financial stability affect risk premia, but a rating downgrade does not establish an increase in Treasury yields. All equity and bond price gaps here are hypothetical, especially the sign of the Treasury response.",
        "source": ("Federal Reserve: Financial Stability Reports", "https://www.federalreserve.gov/financial-stability.htm"),
        "gap": {"equity": -1.2, "tech": -1.5, "bank": -1.5, "industrial": -.8,
                "health": -.8, "energy": -.8, "utilities": -.6, "treasury_long": -1.0,
                "treasury_intermediate": -.4, "aggregate_bond": -.6, "gold": .8,
                "emerging": -1.5, "usd": .3},
        "alternative": "Treasuries may rally as investors seek liquidity even after a downgrade. The duration channel could therefore change sign.",
    },
}


STOCK_ROLES = {
    "tech": {"MSFT", "NVDA", "AAPL", "AMZN", "GOOG", "GOOGL", "META", "AMD", "MU", "TSLA", "CRM", "ORCL", "ADBE", "INTC", "TSM"},
    "bank": {"JPM", "BAC", "C", "WFC", "GS", "MS", "USB", "PNC", "BK", "SCHW"},
    "industrial": {"CAT", "UNP", "DE", "BA", "HON", "GE", "LMT", "RTX", "FDX"},
    "health": {"UNH", "HUM", "CVS", "CI", "ELV", "PFE", "MRK", "JNJ", "LLY"},
    "energy": {"XOM", "CVX", "COP", "OXY", "SLB"},
    "utilities": {"NEE", "DUK", "SO", "AEP", "EXC"},
}
FUND_ROLES = {
    "tech": {"QQQ", "QQQM", "XLK", "VGT", "SMH", "SOXX"},
    "bank": {"XLF", "KBE", "KRE", "VFH"},
    "equity": {"SPY", "VOO", "IVV", "VTI", "DIA", "IWM", "SCHB", "ITOT"},
    "treasury_long": {"TLT", "VGLT", "EDV", "GOVZ", "ZROZ"},
    "treasury_intermediate": {"IEF", "VGIT", "IEI", "SHY"},
    # LQD and HYG require separate credit-spread sensitivities. Treating them
    # as Treasury-heavy aggregate bond ETFs would invent exposure coverage.
    "aggregate_bond": {"BND", "AGG", "SCHZ"},
    "gold": {"GLD", "GLDM", "IAU", "SGOL"},
    "emerging": {"VWO", "EEM", "FXI", "KWEB"},
    "energy": {"XLE", "XOP", "USO", "OIH"},
    "usd": {"UUP"},
}
NON_ISSUER_FUNDS = (FUND_ROLES["treasury_long"] | FUND_ROLES["treasury_intermediate"] |
                    FUND_ROLES["aggregate_bond"] | FUND_ROLES["gold"] | FUND_ROLES["usd"])


def _profile(risk):
    # The analyst profile applies to the *settlement outcome* displayed in the
    # title. A broad original research question can mention several competing
    # events (e.g. shutdown vs. full-year Defense funding) and must never
    # silently choose the opposite economic premise.
    description = " ".join(str(risk.get(k) or "") for k in ("title", "id")).lower()
    approval_negated = bool(re.search(
        r"\b(?:not|never|without)\b(?:\s+[a-z]+){0,3}\s+"
        r"(?:signed|enacted|approved|passed)\b", description))
    if (re.search(r"\b(?:defense|pentagon|department of defense|dod)\b", description)
            and re.search(r"\b(?:appropriations? act|full.year.*appropriations?|funding bill)\b", description)
            and re.search(r"\b(?:signed|enacted|approved|passed)\b", description)
            and not approval_negated):
        return "defense_funding_approval"
    if re.search(r"\bhormuz\b", description) and re.search(
            r"\b(?:tanker|transit|shipping|suspend|interrupt|disrupt|clos|block)\w*\b", description):
        return "hormuz_disruption"
    if (re.search(r"\bopec\b", description)
            and re.search(r"\b(?:increas|rais|expand|boost)\w*\b", description)
            and re.search(r"\b(?:crude|oil|production|output|supply)\b", description)):
        return "opec_output_increase"
    if (re.search(r"\btaiwan\b", description)
            and re.search(r"\b(?:blockade|quarantine|maritime inspection|shipping disruption)\b", description)
            and not re.search(r"\b(?:no|not|without|never)\b(?:\s+[a-z]+){0,4}\s+"
                              r"(?:blockade|quarantine|shipping disruption)\b", description)):
        return "taiwan_shipping_blockade"
    if (re.search(r"\b(?:export controls?|export restriction|bis.*rule)\b", description)
            and re.search(r"\b(?:chips?|semiconductors?|advanced.node|foundry|ai chips?)\b", description)
            and re.search(r"\bchina\b", description)):
        return "chip_export_controls"
    if (re.search(r"\b(?:sovereign|united states|u\.?s\.?)\b", description)
            and re.search(r"\bdowngrad\w*\b", description)
            and re.search(r"\b(?:rating|credit|sovereign)\b", description)):
        return "us_sovereign_downgrade"
    if "item 1.05" in description or "cyber" in description:
        return "cyber"
    if "antitrust" in description:
        return "antitrust"
    if "downgrad" in description and "rating" in description:
        return "credit_rating"
    if "rail" in description and ("strike" in description or "stoppage" in description):
        return "rail_strike"
    if "shutdown" in description:
        return "shutdown"
    if "auction" in description and ("20-year" in description or "20y" in description or "twenty-year" in description):
        return "long_auction"
    if "basel" in description or "bank capital rule" in description:
        return "basel"
    if "tariff" in description or "trade-remedy" in description:
        return "china_tariff" if "china" in description or "chinese" in description else "tariff"
    if "payroll" in description or "jobs report" in description or "employment situation" in description:
        return "payroll"
    return None


def _role(position):
    symbol = str(position["symbol"]).upper()
    asset_type = position.get("type") or position.get("asset_type")
    if asset_type == "cash":
        return "cash"
    groups = FUND_ROLES if asset_type == "etf" else STOCK_ROLES
    for name, members in groups.items():
        if symbol in members:
            return name
    return "equity" if asset_type == "stock" else None


def _issuer(risk, positions, lookthrough):
    tokens = set(re.findall(r"[A-Z][A-Z0-9.]{1,10}",
                            " ".join(str(risk.get(k) or "") for k in ("id", "title", "original_question")).upper()))
    candidates = {str(p["symbol"]).upper() for p in positions if (p.get("type") or p.get("asset_type")) == "stock"}
    candidates |= {str(r.get("symbol") or "").upper() for r in lookthrough if isinstance(r, dict)}
    return next(iter(sorted((tokens & candidates) - {"US", "USD", "ETF", "SEC"})), None)


def _issuer_etf_shock(case, etf_symbol, etf_weight, issuer, issuer_gap):
    # A verified look-through is expressed in shares of TOTAL portfolio NAV.
    # Dividing by the parent ETF NAV recovers the issuer fraction *inside* it.
    # Never sum that child capital on top of the original ETF investment.
    for row in case.get("lookthrough") or []:
        if not isinstance(row, dict) or row.get("symbol") != issuer or row.get("unresolved") is True:
            continue
        via = (row.get("via") or {}).get(etf_symbol)
        if isinstance(via, (int, float)) and not isinstance(via, bool) and math.isfinite(via) and 0 <= via <= etf_weight + 1e-9:
            return issuer_gap * via / etf_weight, "VERIFIED_ISSUER_LOOKTHROUGH_ASSUMPTION"
    return None, "ETF_ISSUER_HOLDINGS_UNRESOLVED"


def propose_analyst_scenario(case, risk):
    """Separate B scenario for a defined target; never promote it into A."""
    if not isinstance(case, dict) or not isinstance(risk, dict) or risk.get("source_defined_scenario") is True:
        return None
    if (case.get("instrument_identity") or {}).get("blocked_symbols"):
        return None
    if (risk.get("quality") or {}).get("definition_review", {}).get("status") not in (None, "DEFINED_FOR_REVIEW"):
        return None
    key = _profile(risk)
    if key is None:
        return None
    template = PROFILES[key]
    positions = case.get("positions") or []
    issuer = _issuer(risk, positions, case.get("lookthrough") or []) if "issuer_gap" in template else None
    if "issuer_gap" in template and not issuer:
        return None
    shocks, channels, unknown, parameter_basis = {}, {}, [], {}
    for p in positions:
        if not isinstance(p, dict):
            return None
        symbol = p.get("symbol")
        role = _role(p)
        if role == "cash":
            shock, channel = 0., "NOMINAL_USD_CASH_FIXED"
        elif key == "long_auction" and role == "treasury_long":
            if symbol == "TLT":
                # Duration (years) times a yield change (bp); convert to a
                # percentage value change without treating a Treasury auction
                # announcement itself as a measured price response.
                shock = -TLT_DURATION["effective_years"] * LONG_AUCTION_YIELD_SHOCK_BP / 100.
                channel = "DATED_EFFECTIVE_DURATION_TIMES_ASSUMED_CURVE_SHOCK"
            else:
                shock, channel = None, "NO_VERIFIED_DURATION_FOR_THIS_LONG_BOND_ETF"
        elif "issuer_gap" not in template:
            overrides = template.get("stock_overrides") or {}
            if ((p.get("type") or p.get("asset_type")) == "stock" and symbol in overrides):
                shock, channel = overrides[symbol], "DECLARED_NAMED_STOCK_STRESS"
            else:
                shock = template["gap"].get(role)
                channel = "DECLARED_" + str(role or "UNKNOWN").upper() + "_MARKET_STRESS"
        elif symbol == issuer and (p.get("type") or p.get("asset_type")) == "stock":
            shock, channel = template["issuer_gap"], "ASSUMED_DIRECT_ISSUER_REPRICING"
        elif (p.get("type") or p.get("asset_type")) == "stock" or symbol in NON_ISSUER_FUNDS:
            shock, channel = 0., "ASSUMED_NO_FIRST_ORDER_ISSUER_SPILLOVER"
        elif (p.get("type") or p.get("asset_type")) == "etf":
            shock, channel = _issuer_etf_shock(case, symbol, p["weight"], issuer, template["issuer_gap"])
        else:
            shock, channel = None, "INSTRUMENT_CHANNEL_UNAVAILABLE"
        if shock is None:
            unknown.append(symbol)
        else:
            shocks[symbol] = shock
        channels[symbol] = channel
        if channel == "DATED_EFFECTIVE_DURATION_TIMES_ASSUMED_CURVE_SHOCK":
            parameter_basis[symbol] = {"kind": "PUBLISHED_DURATION_WITH_ASSUMED_YIELD_MOVE",
                                       "effective_duration_years": TLT_DURATION["effective_years"],
                                       "parameter_as_of": TLT_DURATION["as_of"],
                                       "assumed_yield_change_bp": LONG_AUCTION_YIELD_SHOCK_BP,
                                       "source_url": TLT_DURATION["source_url"],
                                       "source_label": TLT_DURATION["source_name"],
                                       "linear_approximation_only": True}
        elif shock is None:
            parameter_basis[symbol] = {"kind": "UNMODELED_NO_JUSTIFIED_INSTRUMENT_CHANNEL"}
        elif channel == "NOMINAL_USD_CASH_FIXED":
            parameter_basis[symbol] = {"kind": "DECLARED_NOMINAL_CASH_ASSUMPTION"}
        elif channel == "ASSUMED_NO_FIRST_ORDER_ISSUER_SPILLOVER":
            parameter_basis[symbol] = {"kind": "ASSUMED_ZERO_FIRST_ORDER_SPILLOVER"}
        elif channel == "VERIFIED_ISSUER_LOOKTHROUGH_ASSUMPTION":
            parameter_basis[symbol] = {"kind": "HOLDINGS_LOOKTHROUGH_TIMES_DECLARED_ISSUER_SHOCK"}
        else:
            parameter_basis[symbol] = {"kind": "DECLARED_ANALYST_RETURN_GAP_NOT_CALIBRATED"}
    if not any(v != 0 for v in shocks.values()):
        return None
    request = {"hypothesis": template["condition"], "mode": "binary_gap",
               "position_shocks": shocks, "pricing_weight": .5,
               "pricing_weight_range": [.25, .75]}
    try:
        result = evaluate_scenario(case, request)
    except (ScenarioInputError, TypeError, KeyError, ZeroDivisionError):
        return None
    # All monetary interval endpoints scale from original NAV. No false
    # confidence intervals are constructed from the severity multipliers.
    center = result["yes_pnl_usd"]
    severity_range = sorted([.5 * center, 1.5 * center]) if center is not None else None
    return {"schema": SCHEMA, "key": key, "status": "COMPLETE" if result["complete"] else "PARTIAL",
            "scenario_condition": template["condition"], "economic_reasoning": template["why"],
            "source": {"label": template["source"][0], "url": template["source"][1],
                       "supports": "EVENT_DEFINITION_OR_MECHANISM_NOT_SHOCK_MAGNITUDE"},
            "horizon": "ILLUSTRATIVE_SHORT_EVENT_WINDOW_NOT_A_PREDICTION",
            "position_return_gap_pct": shocks, "channels": channels,
            "position_parameter_basis": parameter_basis,
            "empirical_context": payroll_context_for_positions(positions) if key == "payroll" else None,
            "alternative_economic_regime": template.get("alternative"),
            "unmodeled_symbols": unknown, "result": result,
            "severity_sensitivity_yes_usd": severity_range,
            "severity_multipliers": [.5, 1.5],
            "equity_pricing_q_origin": "ILLUSTRATIVE_50_PERCENT_NOT_AN_EMPIRICAL_ESTIMATE",
            "event_probability": {"value": None, "status": "NOT_IDENTIFIED_BY_THIS_ANALYST_SCENARIO"},
            "pricing_q_limitation": "The same assumed two-state q for stocks, bonds and gold is a scenario convention, not an inferred asset-price expectation or a market quote.",
            "numeric_price_moves_are": "EXPLICIT_ANALYST_ASSUMPTIONS_NOT_SOURCE_ESTIMATES",
            "prediction_market_probability_used": False,
            "changes_existing_primary_outcomes": False}


def attach_analyst_scenarios(case):
    """Annotate public risks after strict source admission, without changing A."""
    for risk in case.get("risks") or []:
        proposal = propose_analyst_scenario(case, risk)
        if proposal is not None:
            risk["analyst_assumption"] = proposal
    return case

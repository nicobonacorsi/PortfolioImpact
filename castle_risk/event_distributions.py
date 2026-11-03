"""Source-conditioned FOMC directions and disjoint CPI settlement-value bins.

This presentation adapter preserves the existing semantic and probability engines.
It does not turn a meeting into a binary event, equate a research hypothesis with
a contract, repair an ambiguous strike, normalize inconsistent quotes, or infer
physical probabilities. Complete partitions are conditional on the displayed
common-settlement assumptions. The original bid/ask feasible set is exported.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, ROUND_FLOOR
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qs, urlsplit
from zoneinfo import ZoneInfo

import numpy as np
from scipy.optimize import linprog

from .adapters import Market, normalize_market, normalize_quote, parse_time
from .agentic_bridge import _receipt, AgenticEvidenceError
from .contract_questions import source_defined_yes_question
from .fed_outcome_pricing import evaluate_fed_outcome_quotes, FedOutcomePricingError
from .semantics import FED_PRIMARY_RE, CPI_TERMS_SHA256
from .strike_lattice import evaluate_strike_lattice, _parse_source, StrikeLatticeError


class EventDistributionError(ValueError):
    """No valid common source-conditioned distribution is available."""


_CPI_SECONDARY_RE = re.compile(
    r"The market will always close at 8:29 AM ET on the scheduled day of the data release "
    r"\((?P<release>[A-Z][a-z]+ \d{1,2}, 20\d{2})\)\. Please note: the Expiration Value is the "
    r"single-decimal value published at the Source Agency\. In the case of a delay in data caused "
    r"by a federal government shutdown impacting the reliability of the Source Agency, the market’s "
    r"latest Expiration Date will be extended to the sooner of the release of the Underlying or "
    r"six months after the end of the government shutdown\."
)
# The same terms without the close sentence (Kalshi's December 2026 CPI event). The release
# date is then not stated in the rules; it is read from a close at exactly 8:29 AM ET.
_CPI_SECONDARY_NO_CLOSE_SENTENCE_RE = re.compile(
    r"Please note: the Expiration Value is the single-decimal value published at the Source Agency\. "
    r"In the case of a delay in data caused by a federal government shutdown impacting the reliability "
    r"of the Source Agency, the market’s latest Expiration Date will be extended to the sooner of the "
    r"release of the Underlying or six months after the end of the government shutdown\."
)


def coobserved_event_markets(markets: Sequence[Market]) -> list[Market]:
    """Recover the full same-event cohort in an already-verified response body.

    The discovery UI may retain only eight rows from a twelve-strike response.
    This reads the original response once; it makes no provider request and does
    not combine different observation times. Every returned row is independently
    checked through the unchanged original source receipt verifier.
    """
    if not isinstance(markets, (list, tuple)) or not 1 <= len(markets) <= 40:
        raise EventDistributionError("Provide one to forty normalized markets for one event.")
    if any(not isinstance(market, Market) for market in markets):
        raise EventDistributionError("Original normalized Market objects are required.")
    if len({market.ticker for market in markets}) != len(markets):
        raise EventDistributionError("Duplicate input market identity.")
    scopes = []
    for market in markets:
        try:
            receipt = _receipt(market)
        except (AgenticEvidenceError, OSError, ValueError) as exc:
            raise EventDistributionError("Original source receipt failed: " + str(exc)) from None
        scopes.append((market.event_ticker, receipt["source_response_sha256"],
                       receipt["request_url"], receipt["retrieved_at_utc"]))
    if any(scope != scopes[0] for scope in scopes[1:]):
        raise EventDistributionError("MIXED_SOURCE_SNAPSHOT: event, response and receipt time must agree.")
    first = markets[0]
    body = json.loads(Path(first.raw_path).read_bytes())
    rows = body.get("markets", [body["market"]] if "market" in body else [])
    rows = [row for row in rows if isinstance(row, dict) and row.get("event_ticker") == first.event_ticker]
    if not 2 <= len(rows) <= 40:
        raise EventDistributionError("A bounded cohort of at least two same-event source rows is required.")
    if len({row.get("ticker") for row in rows}) != len(rows):
        raise EventDistributionError("Duplicate market identity in the original response.")
    result = [normalize_market(row, first.retrieved_at, first.raw_path) for row in rows]
    for market in result:
        _receipt(market)
    return sorted(result, key=lambda market: market.ticker)


def _current_quotes(markets: Sequence[Market], as_of: str | datetime,
                    max_age_seconds: int) -> tuple[datetime, list[dict[str, Any]]]:
    if (isinstance(max_age_seconds, bool) or not isinstance(max_age_seconds, int)
            or not 0 <= max_age_seconds <= 300):
        raise EventDistributionError("Receipt freshness must be an integer from zero to the original 300-second ceiling.")
    try:
        now = parse_time(as_of) if isinstance(as_of, str) else as_of
        if not isinstance(now, datetime) or now.utcoffset() is None:
            raise ValueError("Timezone required")
        now = now.astimezone(timezone.utc)
    except (ValueError, TypeError):
        raise EventDistributionError("An explicit timezone-aware evaluation time is required.") from None
    result = []
    for market in markets:
        try:
            if now < parse_time(market.quote_timestamp):
                raise EventDistributionError("Quote receipt is later than the declared evaluation time.")
            if now >= parse_time(market.close_time):
                raise EventDistributionError("Market is already closed at the declared evaluation time.")
        except (ValueError, TypeError):
            raise EventDistributionError("Invalid or closed market timing.") from None
        quote = normalize_quote(market, now=now, max_age_seconds=max_age_seconds)
        if not quote.usable:
            raise EventDistributionError("Unusable quotation for " + market.ticker + ": " + quote.exclusion_reason)
        result.append({"ticker": market.ticker, "lower": quote.lower, "upper": quote.upper,
                       "quote_notes": quote.notes})
    return now, result


def _lp_kwargs(constraints: dict[str, Any]) -> dict[str, Any]:
    return {"A_ub": constraints["A_ub"] or None, "b_ub": constraints["b_ub"] or None,
            "A_eq": constraints["A_eq"], "b_eq": constraints["b_eq"],
            "bounds": constraints["bounds"], "method": "highs"}


def _checked_witness(witness: Sequence[float], constraints: dict[str, Any]) -> list[float]:
    point = [float(value) for value in witness]
    if any(not math.isfinite(value) for value in point):
        raise EventDistributionError("Nonfinite common-Q LP witness.")
    for value, (lower, upper) in zip(point, constraints["bounds"]):
        if not lower - 1e-8 <= value <= upper + 1e-8:
            raise EventDistributionError("LP witness violates an atom probability bound.")
    for row, rhs in zip(constraints["A_ub"], constraints["b_ub"]):
        if math.fsum(a * b for a, b in zip(row, point)) > rhs + 1e-8:
            raise EventDistributionError("LP witness violates a quote constraint.")
    for row, rhs in zip(constraints["A_eq"], constraints["b_eq"]):
        if abs(math.fsum(a * b for a, b in zip(row, point)) - rhs) > 1e-8:
            raise EventDistributionError("LP witness violates a partition equality.")
    return point


def distribution_linear_bounds(constraints: dict[str, Any], values: Sequence[float]) -> dict[str, Any]:
    """Sharp linear expectation bounds and attaining witnesses on this finite set."""
    if (len(values) != len(constraints["atom_ids"])
            or any(isinstance(value, bool) or not math.isfinite(float(value)) for value in values)):
        raise EventDistributionError("A finite value for every probability atom is required.")
    low = linprog(list(values), **_lp_kwargs(constraints))
    high = linprog([-float(value) for value in values], **_lp_kwargs(constraints))
    if not (low.success and high.success):
        raise EventDistributionError("INCOHERENT_COMMON_Q_QUOTE_BASKET: no simplex distribution satisfies all original intervals.")
    return {"interval": [float(low.fun), float(-high.fun)],
            "lower_witness": _checked_witness(low.x, constraints),
            "upper_witness": _checked_witness(high.x, constraints)}


# Largest Kalshi taker fee per one-dollar contract under the fee schedule pinned
# in event_distribution_fees (0.07 x p x (1 - p), maximal at p = 0.5). A quote
# basket that is incoherent by less than this is not an executable arbitrage,
# so its smallest uniform relaxation is measured and disclosed, not rejected.
MAX_QUOTE_RELAXATION = 0.0175
QUOTE_RELAXATION_LP_SLACK = 1e-6  # numerical room for the LP witnesses on top of epsilon*


def minimal_quote_relaxation(constraints: dict[str, Any]) -> float:
    """Smallest epsilon making every quote bound and inequality feasible on the simplex.

    Each quoted atom bound [l, u] becomes [max(0, l - eps), min(1, u + eps)] and
    each quote inequality row a.q <= b becomes a.q <= b + eps. Simplex equality
    is never relaxed and no quote is normalized or dropped.
    """
    n = len(constraints["atom_ids"])
    bounds = constraints["bounds"]
    # Variables: q_1..q_n, eps. Encode relaxed bounds as inequalities.
    a_ub, b_ub = [], []
    for i, (lower, upper) in enumerate(bounds):
        row = [0.0] * (n + 1)
        if lower > 0:
            row[i], row[n] = -1.0, -1.0
            a_ub.append(row); b_ub.append(-lower)
        if upper < 1:
            row = [0.0] * (n + 1)
            row[i], row[n] = 1.0, -1.0
            a_ub.append(row); b_ub.append(upper)
    for matrix_row, rhs in zip(constraints["A_ub"], constraints["b_ub"]):
        a_ub.append(list(matrix_row) + [-1.0]); b_ub.append(rhs)
    a_eq = [list(row) + [0.0] for row in constraints["A_eq"]]
    result = linprog([0.0] * n + [1.0], A_ub=a_ub or None, b_ub=b_ub or None,
                     A_eq=a_eq, b_eq=constraints["b_eq"],
                     bounds=[(0.0, 1.0)] * n + [(0.0, 1.0)], method="highs")
    if not result.success:
        raise EventDistributionError("INCOHERENT_COMMON_Q_QUOTE_BASKET: no relaxation restores a simplex distribution.")
    return float(result.x[n])


def _relax_if_incoherent(constraints: dict[str, Any], *, allow_fee_adjusted_quotes: bool = False) -> dict[str, Any]:
    """Measure basket incoherence; relax by epsilon* only within the fee tolerance."""
    probe = linprog([0.0] * len(constraints["atom_ids"]), **_lp_kwargs(constraints))
    if probe.success:
        return {"status": "COHERENT_RAW_QUOTES", "epsilon": 0.0, "tolerance": MAX_QUOTE_RELAXATION}
    epsilon = minimal_quote_relaxation(constraints)
    if epsilon > MAX_QUOTE_RELAXATION and not allow_fee_adjusted_quotes:
        raise EventDistributionError(
            f"INCOHERENT_COMMON_Q_QUOTE_BASKET: minimal uniform quote relaxation {epsilon:.4f} exceeds "
            f"the {MAX_QUOTE_RELAXATION:.4f} fee tolerance; no distribution is reported.")
    if allow_fee_adjusted_quotes:
        # The fee adapter supplies its own documented cost envelope.
        raise EventDistributionError("INCOHERENT_COMMON_Q_QUOTE_BASKET: fee-adjusted bounds remain infeasible.")
    slack = epsilon + QUOTE_RELAXATION_LP_SLACK  # disclosed epsilon is unrounded
    constraints["raw_bounds"] = [list(b) for b in constraints["bounds"]]
    constraints["bounds"] = [[max(0.0, lower - slack) if lower > 0 else lower,
                              min(1.0, upper + slack) if upper < 1 else upper]
                             for lower, upper in constraints["bounds"]]
    constraints["b_ub"] = [rhs + slack for rhs in constraints["b_ub"]]
    return {"status": "MINIMAL_UNIFORM_QUOTE_RELAXATION_APPLIED", "epsilon": epsilon,
            "tolerance": MAX_QUOTE_RELAXATION,
            "rule": "Every quote bound widened by epsilon*, the smallest value restoring one simplex distribution; "
                    "accepted only when epsilon* is below the largest per-contract taker fee.",
            "raw_quotes_normalized_or_dropped": False}


def _attach_bounds(atoms: list[dict[str, Any]], constraints: dict[str, Any]) -> list[float]:
    witnesses = []
    for i, atom in enumerate(atoms):
        bound = distribution_linear_bounds(constraints, np.eye(len(atoms))[i])
        atom["q_interval"] = [max(0.0, bound["interval"][0]), max(0.0, min(1.0, bound["interval"][1]))]
        atom["lower_attaining_joint_witness"] = bound["lower_witness"]
        atom["upper_attaining_joint_witness"] = bound["upper_witness"]
        witnesses.extend([bound["lower_witness"], bound["upper_witness"]])
    # An average of feasible points is feasible. This reference is a transparent
    # display/model choice, not normalized midquotes or a statistically estimated P.
    reference = [math.fsum(point[i] for point in witnesses) / len(witnesses) for i in range(len(atoms))]
    _checked_witness(reference, constraints)
    for atom, weight in zip(atoms, reference):
        atom["q_reference"] = weight
    return reference


def _retained_quote_evidence(markets: list[Market], body: dict[str, Any],
                             source: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    """Retain source labels, not new predicates or additional market admission.

    Callers have already verified this same-response cohort. Read labels from
    its matched provider rows, never mutable presentation fields or invented
    disjoint-bin labels. Keep explicit test provenance without forwarding the
    receipt's arbitrary/private fields, credentials, headers or local paths.
    """
    rows = body.get('markets', [body['market']] if 'market' in body else [])
    by_id = {row.get('ticker'): row for row in rows if isinstance(row, dict)}
    retained = {}
    for market in markets:
        raw = by_id.get(market.ticker)
        if not isinstance(raw, dict):
            raise EventDistributionError('Verified source body omits this exact contract.')
        def text(name):
            value = raw.get(name)
            return ' '.join(value.split()) if isinstance(value, str) else ''
        title, yes_label, subtitle = text('title'), text('yes_sub_title'), text('subtitle')
        if yes_label:
            question = (title + ' — ' if title else '') + 'YES: ' + yes_label
            basis = 'SOURCE_TITLE_WITH_EXPLICIT_SOURCE_YES_LABEL'
        elif subtitle:
            question = (title + ' — ' if title else '') + subtitle
            basis = 'SOURCE_TITLE_WITH_SOURCE_SUBTITLE'
        elif text('rules_primary'):
            question, basis = text('rules_primary'), 'SOURCE_PRIMARY_RULE_TEXT'
        else:
            question, basis = title, 'SOURCE_TITLE_ONLY'
        primary, secondary = raw.get('rules_primary'), raw.get('rules_secondary')
        rule_text = (primary if isinstance(primary, str) else '') + '\n' + (secondary if isinstance(secondary, str) else '')
        record = {name: raw.get(name, '') for name in
                  ('title', 'subtitle', 'yes_sub_title', 'no_sub_title', 'rules_primary', 'rules_secondary')}
        record.update(contract_id=market.ticker, event_ticker=market.event_ticker,
            venue='Kalshi', question=question, question_construction=basis,
            quote_scope='EXACT_SOURCE_CONTRACT_YES_QUOTE_NOT_PHYSICAL_PROBABILITY',
            quote_received_at=source['retrieved_at_utc'], quote_timestamp_kind='retrieval_receipt',
            source_url=market.source_url, source=dict(source), fixture=receipt.get('fixture'),
            source_receipt_status='LOCAL_HTTP_RECEIPT_BYTES_VERIFIED_NOT_PROVIDER_SIGNATURE',
            exchange_quote_update_time_verified=False,
            rules_sha256=sha256(rule_text.encode('utf-8')).hexdigest(),
            rules_hash_basis='UTF8_PRIMARY_OR_EMPTY_PLUS_NEWLINE_PLUS_SECONDARY_OR_EMPTY',
            numeric_predicate_inferred_from_display_label=False)
        for name in ('test_data', 'synthetic', 'synthetic_mutation', 'provenance_kind'):
            if name in receipt and isinstance(receipt[name], (str, bool)):
                record[name] = receipt[name]
        retained[market.ticker] = record
    return retained


def _base(markets: list[Market], now: datetime) -> dict[str, Any]:
    source = _receipt(markets[0])
    body = json.loads(Path(markets[0].raw_path).read_bytes())
    receipt = json.loads(Path(markets[0].raw_path).with_suffix('.meta.json').read_text(encoding='utf-8'))
    query = parse_qs(urlsplit(source["request_url"]).query)
    return {
        "schema": "castle.option1.event_distribution.v1",
        "status": "CONDITIONAL_SOURCE_DISTRIBUTION",
        "event_ticker": markets[0].event_ticker,
        "event_date": parse_time(markets[0].close_time).date().isoformat(),
        "release_date": parse_time(markets[0].close_time).date().isoformat(),
        "evaluated_at": now.isoformat(), "source": source,
        "source_market_close_at_utc": min(parse_time(market.close_time) for market in markets).astimezone(timezone.utc).isoformat(),
        "source_close_basis": "EARLIEST_VERIFIED_CONSTITUENT_MARKET_CLOSE",
        "source_contract_ids": [market.ticker for market in markets],
        "source_quote_evidence": _retained_quote_evidence(markets, body, source, receipt),
        "source_market_catalog_complete": (query.get("event_ticker") == [markets[0].event_ticker]
                                           and body.get("cursor") == ""),
        "underlying_outcome_catalog_exhaustive_claimed": False,
        "partition_complete_under_declared_common_settlement_model": True,
        "q_reference_basis": "ARITHMETIC_BARYCENTER_OF_ALL_COORDINATE_EXTREMUM_LP_WITNESSES_NOT_NORMALIZED_MIDQUOTES",
        "physical_probability_P": None,
        "broad_research_event_equivalence": "NOT_INFERRED_FROM_FAMILY_MEMBERSHIP",
        "assumptions": [
            "Common unit-dollar payout pricing measure, unit discount convention, transaction fees and execution depth not modeled.",
            "Probabilities are quote-compatible Q weights, not physical probabilities or statistical confidence intervals.",
            "Original receipt time is verified locally; exchange quote-update times are not independently verified.",
            "The displayed reference distribution is one feasible modeling choice; all original constraints and endpoint witnesses remain available.",
        ],
    }


def _fed(markets: list[Market], now: datetime, quotes: list[dict[str, Any]], *,
         allow_fee_adjusted_quotes: bool = False) -> dict[str, Any]:
    try:
        original = evaluate_fed_outcome_quotes(markets, as_of=now)
    except (FedOutcomePricingError, ValueError) as exc:
        # The optional fee adapter preserves the original failure after the
        # unchanged engine has completed every source/semantic check. No other
        # failure can be bypassed, and the ordinary public builder never opts in.
        # The infeasibility message is raised only after every source, receipt
        # and exclusivity check has passed. The basket's incoherence is then
        # measured (minimal quote relaxation) instead of discarding the event.
        if str(exc) == "Infeasible single-Q outcome simplex for source bid/ask intervals":
            original = {"status": "INCOHERENT_RAW_COMMON_Q_QUOTE_BASKET", "reason": str(exc)}
        else:
            raise EventDistributionError(str(exc)) from None
    atoms = []
    for market in markets:
        match = FED_PRIMARY_RE.fullmatch(market.rules_primary)
        if match is None:
            raise EventDistributionError("Unsupported FOMC source rule.")
        action, _, amount, _ = match.groups()
        question = source_defined_yes_question(market)
        predicate = question.get("typed_predicate")
        exact = (isinstance(predicate, dict) and predicate.get("operator") == "eq"
                 and predicate.get("unit") == "basis_points")
        value = float(predicate["threshold"]) if exact else None
        direction = "hold" if value == 0 else ("cut" if action == "Cut" else "hike")
        if not exact and float(amount) <= 0:
            raise EventDistributionError("Ambiguous zero policy action cannot define a direction.")
        support = [value, value] if exact else ([None, 0.0] if direction == "cut" else [0.0, None])
        label = ("Hold / canceled meeting" if direction == "hold" else
                 f"{direction.capitalize()} {abs(value):g} bp" if exact else
                 f"{direction.capitalize()} source bucket; magnitude ambiguous")
        atoms.append({"outcome_id": market.ticker, "label": label, "direction": direction,
                      "unit": "basis_points", "value": value, "value_interval": support,
                      "lower_inclusive": exact, "upper_inclusive": exact,
                      "numeric_mapping_status": "SOURCE_EXACT_NUMERIC_PREDICATE" if exact else "SOURCE_DIRECTION_ONLY_MAGNITUDE_UNVERIFIED",
                      "source_contract_ids": [market.ticker],
                      "rules_sha256": question["verified_rules_sha256"],
                      "original_numeric_rule_status": question["numeric_rule_status"]})
    # The exact hold contract covers zero change and cancellation. All remaining
    # policy actions are negative or positive under the explicitly stated model.
    # If that contract is absent, zero is retained as a separate unobserved atom.
    residual_directions = ["cut", "hike"] if any(a["direction"] == "hold" for a in atoms) else ["cut", "hold", "hike"]
    for direction in residual_directions:
        atoms.append({"outcome_id": "UNOBSERVED_" + direction.upper(),
                      "label": "Other / unquoted " + direction + " outcomes", "direction": direction,
                      "unit": "basis_points", "value": 0.0 if direction == "hold" else None,
                      "value_interval": [0.0, 0.0] if direction == "hold" else ([None, 0.0] if direction == "cut" else [0.0, None]),
                      "lower_inclusive": direction == "hold", "upper_inclusive": direction == "hold",
                      "numeric_mapping_status": "UNOBSERVED_DIRECTION_RESIDUAL_MAGNITUDE_UNIDENTIFIED",
                      "source_contract_ids": []})
    n = len(atoms)
    constraints = {"atom_ids": [atom["outcome_id"] for atom in atoms], "A_ub": [], "b_ub": [],
                   "A_eq": [[1.0] * n], "b_eq": [1.0],
                   "bounds": [[quote["lower"], quote["upper"]] for quote in quotes] + [[0.0, 1.0]] * len(residual_directions)}
    relaxation = _relax_if_incoherent(constraints, allow_fee_adjusted_quotes=allow_fee_adjusted_quotes)
    reference = _attach_bounds(atoms, constraints)
    groups = []
    for direction in ("cut", "hold", "hike"):
        indicator = [1.0 if atom["direction"] == direction else 0.0 for atom in atoms]
        bound = distribution_linear_bounds(constraints, indicator)
        groups.append({"outcome_id": direction.upper(), "label": "Hold / canceled meeting" if direction == "hold" else direction.capitalize(),
                       "q_interval": [max(0.0, bound["interval"][0]), max(0.0, min(1.0, bound["interval"][1]))],
                       "q_reference": math.fsum(x * y for x, y in zip(indicator, reference)),
                       "member_atom_ids": [atom["outcome_id"] for atom in atoms if atom["direction"] == direction],
                       "lower_attaining_joint_witness": bound["lower_witness"], "upper_attaining_joint_witness": bound["upper_witness"]})
    result = _base(markets, now)
    result.update(family="FED_POLICY", metric_identity="SCHEDULED_FED_TARGET_RATE_CHANGE_BP_WITH_CANCELLATION_AS_ZERO",
                  atoms=atoms, outcomes=atoms, outcome_groups=groups, constraints=constraints,
                  q_reference=reference, source_quote_constraints=quotes,
                  original_source_outcome_pricing_status=original["status"],
                  quote_relaxation=relaxation,
                  probability_question="Distribution of negative, zero/canceled, and positive policy changes at this scheduled meeting.",
                  event_level_probability=None, release_date_status="SOURCE_PRIMARY_MEETING_DATE_CHECKED_AGAINST_CLOSE")
    result["assumptions"].extend([
        "Source-declared mutually exclusive YES settlements share one policy-change variable; cancellation is the source-defined zero/hold outcome.",
        "Unquoted outcomes are retained and allocated only across possible directions; all source cut/hike words identify direction, not the garbled magnitude of C26/H26.",
        "No discretionary nonbinary settlement outside that common policy-change model is asserted to obey this partition.",
        "Unobserved or ambiguous policy-change magnitudes have no finite data-derived expectation bound; economic representatives require separate assumptions.",
    ])
    return result


def _cpi(markets: list[Market], now: datetime, quotes: list[dict[str, Any]]) -> dict[str, Any]:
    parsed = []
    for market in markets:
        try:
            strike, scope = _parse_source(market)
        except (StrikeLatticeError, ValueError) as exc:
            raise EventDistributionError(str(exc)) from None
        raw = market.metadata["raw"]
        if (raw.get("strike_type") != "greater" or isinstance(raw.get("floor_strike"), bool)
                or raw.get("floor_strike") != strike or raw.get("cap_strike") is not None
                or raw.get("custom_strike") not in (None, {}) or raw.get("functional_strike") not in (None, "")):
            raise EventDistributionError("CPI structured strike contradicts the strict-threshold source rule.")
        expected_label = "Above " + market.ticker.split("-T", 1)[1] + "%"
        if raw.get("yes_sub_title") != expected_label:
            raise EventDistributionError("CPI YES label contradicts the source strike.")
        secondary = _CPI_SECONDARY_RE.fullmatch(market.rules_secondary)
        close = parse_time(market.close_time).astimezone(ZoneInfo("America/New_York"))
        reference = datetime.strptime(scope["reference_month"] + " " + scope["reference_year"], "%B %Y")
        if secondary is not None:
            release = datetime.strptime(secondary["release"], "%B %d, %Y").date()
            release_date_status = "FULL_SOURCE_SECONDARY_RELEASE_DATE_CHECKED_AGAINST_0829_ET_CLOSE"
        elif (scope["primary_wording"] == "AS_REPORTED_TO_ONE_DECIMAL_PLACE"
              and _CPI_SECONDARY_NO_CLOSE_SENTENCE_RE.fullmatch(market.rules_secondary)):
            # The rules state no release date: the 8:29 AM ET close is the release day, and it
            # must fall in the month after the reference month, when BLS publishes that CPI.
            release = close.date()
            following = (reference.year + reference.month // 12, reference.month % 12 + 1)
            if (release.year, release.month) != following:
                raise EventDistributionError("CPI close is not in the month after the reference month.")
            release_date_status = "SOURCE_CLOSE_AT_0829_ET_IN_THE_FOLLOWING_MONTH_RELEASE_DATE_NOT_STATED_IN_RULES"
        else:
            raise EventDistributionError("CPI secondary provisions are outside the reviewed source grammar.")
        if close.date() != release or (close.hour, close.minute, close.second) != (8, 29, 0):
            raise EventDistributionError("CPI source release date and market close contradict.")
        if market.event_ticker != "KXCPI-" + reference.strftime("%y%b").upper():
            raise EventDistributionError("CPI reference month contradicts the source event identifier.")
        parsed.append((strike, market))
    try:
        lattice = evaluate_strike_lattice(markets, observed_at=now)
    except (StrikeLatticeError, ValueError) as exc:
        raise EventDistributionError(str(exc)) from None
    parsed.sort(key=lambda pair: pair[0])
    strikes = [strike for strike, _ in parsed]
    n = len(strikes) + 1
    atoms = []
    for index in range(n):
        lower = None if index == 0 else strikes[index - 1]
        upper = None if index == n - 1 else strikes[index]
        lower_grid = None if lower is None else float((Decimal(str(lower)) * 10).to_integral_value(rounding=ROUND_FLOOR) / 10 + Decimal("0.1"))
        upper_grid = None if upper is None else float((Decimal(str(upper)) * 10).to_integral_value(rounding=ROUND_FLOOR) / 10)
        exact = lower_grid is not None and upper_grid is not None and abs(lower_grid - upper_grid) < 1e-10
        label = (f"CPI <= {upper:g}%" if lower is None else f"CPI > {lower:g}%" if upper is None else
                 f"CPI = {upper_grid:g}%" if exact else f"{lower:g}% < CPI <= {upper:g}%")
        atoms.append({"outcome_id": "CPI_BIN_" + str(index), "label": label, "direction": None,
                      "unit": "percentage_points", "value": upper_grid if exact else None,
                      "value_interval": [lower_grid, upper_grid],
                      "strict_threshold_bin": {"lower": lower, "upper": upper, "lower_inclusive": False, "upper_inclusive": True},
                      "lower_inclusive": lower_grid is not None, "upper_inclusive": upper_grid is not None,
                      "numeric_mapping_status": "SOURCE_SINGLE_DECIMAL_BIN_EXACT_VALUE" if exact else "SOURCE_SINGLE_DECIMAL_INTERVAL_WITH_OPEN_TAIL" if lower is None or upper is None else "SOURCE_SINGLE_DECIMAL_MULTIVALUE_BIN",
                      "source_contract_ids": [market.ticker for _, market in parsed[max(0, index-1):min(index+1, len(parsed))]]})
    constraints = {"atom_ids": [atom["outcome_id"] for atom in atoms], "A_ub": [], "b_ub": [],
                   "A_eq": [[1.0] * n], "b_eq": [1.0], "bounds": [[0.0, 1.0]] * n}
    quoted = {quote["ticker"]: quote for quote in quotes}
    for index, (_, market) in enumerate(parsed):
        row = [0.0] * (index+1) + [1.0] * (n-index-1)
        constraints["A_ub"].extend([row, [-value for value in row]])
        constraints["b_ub"].extend([quoted[market.ticker]["upper"], -quoted[market.ticker]["lower"]])
    relaxation = _relax_if_incoherent(constraints)
    reference_weights = _attach_bounds(atoms, constraints)
    result = _base(markets, now)
    result.update(family="US_INFLATION", metric_identity="HEADLINE_CPI_U_SEASONALLY_ADJUSTED_MONTH_OVER_MONTH_SINGLE_DECIMAL_SETTLEMENT_PERCENT",
                  source_terms_reference={"url": "https://assets.kalshi.com/contract_terms/CPI.pdf", "reviewed_sha256": CPI_TERMS_SHA256,
                                          "current_terms_bytes_reverified_by_this_function": False,
                                          "exceptional_terms_precedence": "UNRESOLVED_API_SIX_MONTH_VS_PDF_THREE_MONTH_AND_FALLBACK"},
                  atoms=atoms, outcomes=atoms, outcome_groups=[], constraints=constraints,
                  q_reference=reference_weights, source_quote_constraints=quotes,
                  original_strike_lattice_status=lattice["status"], quote_relaxation=relaxation,
                  probability_question="Distribution of the common source CPI settlement value across disjoint bins, including both tails.",
                  event_level_probability=None,
                  reference_period=reference.strftime("%Y-%m"), reference_month=reference.strftime("%Y-%m"),
                  release_date_status=release_date_status,
                  expectation_without_tail_assumptions="UNBOUNDED_OR_UNIDENTIFIED_FROM_FINITE_STRIKE_QUOTES")
    result["assumptions"].extend([
        "All quoted thresholds refer to one common signed, one-decimal headline CPI-U seasonally-adjusted monthly settlement statistic, not core CPI or year-over-year inflation.",
        "The complete source secondary terms agree; strict-threshold payouts apply to the same settlement value without discretionary contract-specific overrides.",
        "API shutdown extensions and PDF exceptional-expiration/fallback provisions have unresolved precedence; the distribution is conditional on a common resolution convention, not guaranteed first-release BLS matching.",
        "Both unbounded tails are retained. Finite tail representatives or caps are economic modeling assumptions and are not implied by these quotes.",
    ])
    return result


def build_event_distribution(markets: Sequence[Market], *, as_of: str | datetime,
                             max_age_seconds: int = 300) -> dict[str, Any]:
    """Build a receipt-verified, same-snapshot distribution; fail closed otherwise.

    For archived replay pass the original observation time explicitly. Passing
    current time to old quotes is rejected. No event-family probability is set
    to one: a scheduled meeting/release is the index of an outcome distribution.
    """
    cohort = coobserved_event_markets(markets)
    now, quotes = _current_quotes(cohort, as_of, max_age_seconds)
    if any(market.series_ticker != cohort[0].series_ticker for market in cohort):
        raise EventDistributionError("Mixed source series cannot define one distribution.")
    if cohort[0].series_ticker == "KXFEDDECISION":
        return _fed(cohort, now, quotes)
    if cohort[0].series_ticker == "KXCPI":
        return _cpi(cohort, now, quotes)
    raise EventDistributionError("No reviewed finite-outcome adapter for this source family.")


def build_distributions_from_markets(market_objects: Mapping[str, Market], *,
                                    as_of: str | datetime) -> dict[str, dict[str, Any]]:
    """One result per source event, retaining unavailable/incoherent results.

    When the discovery loop has several original snapshots for the same event,
    select the latest receipt by time and source hash before inspecting prices.
    Expand its original response cohort; never assemble quotes from different
    snapshots or fall back to an older more attractive/coherent distribution.
    """
    if not isinstance(market_objects, Mapping):
        raise EventDistributionError("Market objects must be an identity-keyed mapping.")
    grouped: dict[str, list[Market]] = {}
    failures: dict[str, list[str]] = {}
    for identifier, market in market_objects.items():
        if not isinstance(market, Market):
            raise EventDistributionError("The market mapping contains a non-Market value.")
        grouped.setdefault(market.event_ticker, []).append(market)
        if identifier != market.ticker:
            failures.setdefault(market.event_ticker, []).append("Mapping key differs from original contract identity.")
    output = {}
    for event_ticker, markets in sorted(grouped.items()):
        sources = []
        for market in markets:
            try:
                receipt = _receipt(market)
                sources.append((parse_time(receipt["retrieved_at_utc"]), receipt["source_response_sha256"], market))
            except (AgenticEvidenceError, OSError, ValueError) as exc:
                failures.setdefault(event_ticker, []).append(str(exc))
        if failures.get(event_ticker):
            output[event_ticker] = {"status": "SOURCE_INTEGRITY_FAILED", "event_ticker": event_ticker,
                                    "reasons": failures[event_ticker], "atoms": [], "outcomes": [],
                                    "physical_probability_P": None}
            continue
        selected = max(sources, key=lambda row: (row[0], row[1], row[2].ticker))[2]
        try:
            result = build_event_distribution([selected], as_of=as_of)
        except (EventDistributionError, ValueError, TypeError) as exc:
            reason = str(exc)
            status = "INCOHERENT_COMMON_Q_QUOTE_BASKET" if "infeasible" in reason.lower() or "incoherent" in reason.lower() or "no common-q" in reason.lower() else "DISTRIBUTION_UNAVAILABLE"
            try:
                source_close_date = parse_time(selected.close_time).date().isoformat()
            except (TypeError, ValueError):
                source_close_date = None
            result = {"status": status, "event_ticker": event_ticker, "reason": reason,
                      "family": {"KXFEDDECISION": "FED_POLICY", "KXCPI": "US_INFLATION"}.get(selected.series_ticker),
                      "event_date": source_close_date, "release_date": source_close_date,
                      "release_date_status": "SOURCE_CLOSE_DATE_ONLY_DISTRIBUTION_VALIDATION_FAILED",
                      "source": _receipt(selected), "atoms": [], "outcomes": [],
                      "physical_probability_P": None, "quotes_normalized_or_dropped": False}
        result["snapshot_selection"] = {
            "rule": "LATEST_RECEIPT_THEN_SOURCE_HASH_BEFORE_PRICE_INSPECTION_NO_OLDER_COHERENT_FALLBACK",
            "available_distinct_response_count": len({(source[0], source[1]) for source in sources}),
            "selected_response_sha256": _receipt(selected)["source_response_sha256"],
        }
        output[event_ticker] = result
    return output

"""Narrow Fed/TLT sensitivity bridge with archived issuer evidence.

This is a documented bond-price sensitivity conditional on user-specified
long-yield shocks. It is not an estimated Fed-event effect, a forecast, or a
causally identified coefficient. Kalshi probabilities never enter this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from .portfolio import Portfolio, _finite_number
from .semantics import validate_predicate

ISSUER_URL = "https://www.ishares.com/us/products/239454/ishares-20%20-year-treasury-bond-etf"
MAX_DURATION_AGE_DAYS = 30
MAX_EVENT_HORIZON_DAYS = 90
MAX_NET_YIELD_SHOCK_BPS = 50.0


def _timestamp(value: str | datetime) -> datetime:
    stamp = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("A timezone-aware timestamp is required.")
    return stamp.astimezone(timezone.utc)


@dataclass(frozen=True)
class DurationEvidence:
    symbol: str
    cusip: str
    effective_duration_years: float
    data_as_of: str
    retrieved_at: str
    source_url: str
    raw_file: str
    raw_sha256: str
    extraction_locator: str
    evidence_basis: str = "archived_official_issuer_characteristic"

    def __post_init__(self) -> None:
        duration = _finite_number(self.effective_duration_years, "Effective duration")
        if not 0 < duration <= 50:
            raise ValueError("Effective duration must be positive and no greater than 50 years.")
        object.__setattr__(self, "effective_duration_years", duration)
        if self.symbol != "TLT" or self.cusip != "464287432":
            raise ValueError("The reviewed duration adapter only supports TLT / CUSIP 464287432.")
        observed = date.fromisoformat(self.data_as_of)
        received = _timestamp(self.retrieved_at)
        if observed > received.date():
            raise ValueError("Issuer characteristic date cannot follow its retrieval date.")
        if self.source_url != ISSUER_URL:
            raise ValueError("The reviewed duration adapter requires the official issuer URL.")
        if not re.fullmatch(r"[a-f0-9]{64}", self.raw_sha256):
            raise ValueError("Duration evidence requires a SHA-256 archive digest.")
        if not self.raw_file or not self.extraction_locator:
            raise ValueError("Duration evidence requires archive and extraction provenance.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_issuer_duration(raw_html: str, metadata: Mapping[str, Any]) -> DurationEvidence:
    """Parse the actual issuer's embedded modelOad object, failing closed.

    Read the public two-decimal display (14.85), not extra hidden precision.
    All repeated displayed duration/date fields must agree. This narrow parser
    is deliberately not a general webpage or other-fund inference mechanism.
    """
    if metadata.get("http_status") != 200 or metadata.get("source_url") != ISSUER_URL:
        raise ValueError("Only a successful official issuer retrieval is admitted.")
    digest = hashlib.sha256(raw_html.encode("utf-8")).hexdigest()
    if digest != metadata.get("raw_sha256"):
        raise ValueError("Issuer HTML SHA-256 mismatch.")
    document = html.unescape(raw_html)
    if not re.search(r'"ticker"\s*:\s*"TLT"', document):
        raise ValueError("Issuer ticker does not identify TLT.")
    decoder = json.JSONDecoder()
    cusips = []
    for match in re.finditer(r'"cusip"\s*:\s*', document):
        try:
            obj, _ = decoder.raw_decode(document[match.end():])
        except json.JSONDecodeError as exc:
            raise ValueError("Malformed issuer CUSIP object.") from exc
        if isinstance(obj, dict) and obj.get("formattedValue"):
            cusips.append(obj["formattedValue"])
    if not cusips or set(cusips) != {"464287432"}:
        raise ValueError("Issuer CUSIP is absent, ambiguous, or does not identify TLT.")
    found = set()
    for match in re.finditer(r'"modelOad"\s*:\s*', document):
        try:
            obj, _ = decoder.raw_decode(document[match.end():])
        except json.JSONDecodeError as exc:
            raise ValueError("Malformed issuer effective-duration object.") from exc
        if not isinstance(obj, dict) or obj.get("label") != "Effective Duration":
            raise ValueError("Effective-duration field label changed.")
        value_match = re.fullmatch(r"([0-9]+\.[0-9]{2}) yrs", str(obj.get("formattedValue", "")))
        if not value_match:
            raise ValueError("Effective-duration display has an unsupported numeric format or unit.")
        duration = float(value_match[1])
        observed = datetime.strptime(str(obj.get("formattedAsOfDate", "")), "%b %d, %Y").date()
        if obj.get("asOfDate") is not None and str(obj["asOfDate"]) != observed.strftime("%Y%m%d"):
            raise ValueError("Issuer duration date fields disagree.")
        if obj.get("value") is not None and round(_finite_number(obj["value"], "Underlying duration"), 2) != duration:
            raise ValueError("Issuer underlying and displayed duration disagree.")
        found.add((duration, observed.isoformat()))
    if len(found) != 1:
        raise ValueError("No unique dated issuer effective duration was found.")
    duration, observed = found.pop()
    return DurationEvidence(
        "TLT", "464287432", duration, observed, str(metadata["retrieved_at"]), ISSUER_URL,
        str(metadata["raw_file"]), digest,
        'HTML-unescaped JSON objects named modelOad: label, formattedValue, formattedAsOfDate; ticker and cusip cross-check',
    )


def load_duration_evidence(repo_root: Path | str | None = None) -> DurationEvidence:
    """Load and verify the frozen real GET response; never perform network I/O."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    folder = root / "data" / "evidence" / "impact"
    metadata = json.loads((folder / "tlt_issuer_20261016.metadata.json").read_text(encoding="utf-8"))
    if metadata.get("raw_file") != "tlt_issuer_20261016.html":
        raise ValueError("Unexpected duration archive filename.")
    raw = (folder / metadata["raw_file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != metadata.get("raw_sha256"):
        raise ValueError("Issuer raw-byte SHA-256 mismatch.")
    return parse_issuer_duration(raw.decode("utf-8"), metadata)


def analyze_fed_duration_scenario(
    portfolio: Portfolio,
    target: Mapping[str, Any],
    *,
    evidence: DurationEvidence,
    as_of: str | datetime,
    yes_yield_shift_bps: float,
    no_yield_shift_bps: float,
    reference_yield_shift_bps: float = 0.0,
    yes_yield_shift_range_bps: tuple[float, float] | None = None,
    no_yield_shift_range_bps: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Return YES/NO conditional TLT repricing components or explicit UNAVAILABLE.

    Branch shifts and the reference shift are separate user assumptions. The
    event does not identify any of them. In particular, a 25bp Fed decision is
    not treated as a 25bp move in long Treasury yields. NO is supplied separately,
    not inferred to be an opposite-sign surprise. No probability input exists.

    The reference shift can explore an assumed priced-in adjustment, but does
    not estimate it. Zero reference means no adjustment in this scenario, not
    evidence that the event is unpriced. Source availability gates are strict:
    a replay before this archive's retrieval is not a point-in-time experiment.
    """
    now = _timestamp(as_of)
    if not isinstance(evidence, DurationEvidence):
        raise ValueError("A validated DurationEvidence is required.")
    yes = _finite_number(yes_yield_shift_bps, "YES assumed long-yield shift")
    no = _finite_number(no_yield_shift_bps, "NO assumed long-yield shift")
    reference = _finite_number(reference_yield_shift_bps, "Assumed reference yield shift")
    if max(abs(yes), abs(no), abs(reference)) > 100:
        raise ValueError("Absolute branch and reference shift assumptions must be within 100 basis points.")

    def check_range(value: tuple[float, float] | None, point: float, label: str) -> tuple[float, float]:
        if value is None:
            return (point, point)
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{label} range requires two endpoints.")
        lower, upper = (_finite_number(v, f"{label} yield endpoint") for v in value)
        if not -100 <= lower <= point <= upper <= 100:
            raise ValueError(f"{label} yield range must be ordered, contain the branch assumption and stay within +/-100bp.")
        return lower, upper

    yes_range = check_range(yes_yield_shift_range_bps, yes, "YES")
    no_range = check_range(no_yield_shift_range_bps, no, "NO")
    max_net_shock = max(abs(x - reference) for x in (*yes_range, *no_range))
    if max_net_shock > MAX_NET_YIELD_SHOCK_BPS:
        raise ValueError("Net parallel-yield shifts exceed the reviewed first-order +/-50bp scenario domain.")
    basis = "documented_sensitivity_with_assumed_event_shocks"
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "UNAVAILABLE",
        "basis": basis,
        "sensitivity_basis": "documented_sensitivity",
        "event_to_yield_basis": "user_assumption",
        "causal_identification": "NOT_ESTABLISHED",
        "empirical_event_effect_estimation": "NOT_RUN",
        "analysis_as_of": now.isoformat(),
        "target_event_id": target.get("event_id"),
        "target_definition": dict(target),
        "evidence": evidence.to_dict(),
        "source_age_days": (now.date() - date.fromisoformat(evidence.data_as_of)).days,
        "formula": "return_component = -effective_duration_years * (branch_yield_shift_bps - reference_yield_shift_bps) / 10000",
        "horizon": "Instantaneous event repricing component with current duration frozen; not a holding-period total return to the meeting.",
        "price_probability_bridge": "NONE; no Kalshi quotation, midpoint, probability or equity pricing weight is an input.",
        "priced_in_status": "NOT_IDENTIFIED; the reference yield shift is an independent user scenario assumption.",
        "reference_yield_shift_bps": reference,
        "branch_shift_assumptions_bps": {"YES": yes, "NO": no},
        "branch_yield_ranges_bps": {"YES": list(yes_range), "NO": list(no_range)},
        "interval_interpretation": "User-selected yield sensitivity ranges only; not confidence, prediction or probability intervals.",
        "branches": [],
        "assumptions": [
            "The independently defined Fed target only labels YES and NO; it does not imply a Treasury-yield shock or a stock return.",
            "YES and NO long-yield shifts are separately supplied scenario assumptions, not an empirical event-to-yield estimate.",
            "Each branch can contain several Fed outcomes (including cancellation in NO for a positive threshold); one common branch yield shift is an additional approximation, not a logical consequence.",
            "A parallel change in the relevant long Treasury yield curve is applied to the issuer's displayed effective duration.",
            "Duration is held fixed from its source date to the event; positions and portfolio weights are held fixed.",
            "Only the first-order price component is calculated. Convexity, curve twists, term-premium changes beyond the assumed parallel shock, carry, distributions, roll-down, fees and NAV/market-price basis are omitted.",
            "The issuer reports convexity, but its unit convention is not validated here, so no second-order correction is asserted.",
            "The reference yield shift is an optional hypothetical priced-in adjustment. Zero does not establish that the event is unpriced.",
            "No Kalshi probability is used to weight returns or equated to a stock or bond market pricing weight.",
            "Unmodeled non-TLT positions remain unknown. A known contribution is not labeled total portfolio P&L.",
            "The 30-day source-age, 90-day event-horizon and 50bp net-shock gates are conservative engineering scope limits, not calibrated error guarantees.",
        ],
    }

    def unavailable(reason: str) -> dict[str, Any]:
        result["reason"] = reason
        return result

    if target.get("definition_origin") not in {"USER_DEFINED", "REVIEWED_REGISTRY"}:
        return unavailable("A defined physical Fed event is required; exact-contract settlement identity is not silently substituted.")
    try:
        predicate = validate_predicate(target["predicate"])
        meeting = date.fromisoformat(predicate["period"])
    except (KeyError, TypeError, ValueError) as exc:
        return unavailable(f"The target is missing a valid dated Fed predicate: {exc}")
    context = {
        "metric": "fed_target_rate_change", "source": "Federal Reserve", "vintage": "scheduled_meeting_decision",
        "unit": "basis_points", "cancellation_policy": "no_change",
    }
    if any(predicate.get(k) != v for k, v in context.items()):
        return unavailable("Unsupported event context; only an explicitly dated scheduled Fed rate decision is supported.")
    if target.get("event_date", meeting.isoformat()) != meeting.isoformat():
        return unavailable("Target event date and predicate period disagree.")
    result["event_date"] = meeting.isoformat()
    result["days_to_event"] = (meeting - now.date()).days
    if not 1 <= result["days_to_event"] <= MAX_EVENT_HORIZON_DAYS:
        return unavailable("The defined meeting must be future and at most 90 calendar days away; no long-horizon frozen-duration extrapolation.")
    if _timestamp(evidence.retrieved_at) > now or date.fromisoformat(evidence.data_as_of) > now.date():
        return unavailable("Duration source was not yet available at the requested analysis timestamp; no look-ahead replay is permitted.")
    if result["source_age_days"] > MAX_DURATION_AGE_DAYS:
        return unavailable("Archived duration is stale for this analysis timestamp (over 30 days). Refresh and review issuer evidence.")

    rows = []
    for p in portfolio.positions:
        supported = p.symbol == evidence.symbol and p.asset_type == "etf"
        rows.append({"symbol": p.symbol, "asset_type": p.asset_type, "weight": p.weight,
                     "position_value": p.weight * portfolio.notional,
                     "status": "documented_sensitivity" if supported else "zero_weight" if p.weight == 0 else "unmodeled",
                     "effective_duration_years": evidence.effective_duration_years if supported else None})
    modeled_weight = math.fsum(r["weight"] for r in rows if r["status"] == "documented_sensitivity")
    unmodeled_weight = math.fsum(r["weight"] for r in rows if r["status"] == "unmodeled")
    result.update(portfolio_notional=portfolio.notional, modeled_weight=modeled_weight, unmodeled_weight=unmodeled_weight,
                  coverage_status="COMPLETE_SENSITIVITY_COMPONENT" if unmodeled_weight == 0 else "PARTIAL_KNOWN_CONTRIBUTION")
    if modeled_weight == 0:
        return unavailable("No positive-weight TLT ETF position is supported by the archived sensitivity; no numeric portfolio impact is asserted.")

    def branch(label: str, shift: float, shift_range: tuple[float, float]) -> dict[str, Any]:
        duration = evidence.effective_duration_years
        component = -duration * (shift - reference) / 10_000
        return_range = (-duration * (shift_range[1] - reference) / 10_000,
                        -duration * (shift_range[0] - reference) / 10_000)
        contributions = []
        for row in rows:
            supported = row["status"] == "documented_sensitivity"
            zero = row["weight"] == 0
            pnl = portfolio.notional * row["weight"] * component if supported else 0.0 if zero else None
            contributions.append({**row, "return_component": component if supported else 0.0 if zero else None,
                                  "pnl_component": pnl,
                                  "pnl_range": [portfolio.notional * row["weight"] * v for v in return_range] if supported else [0.0, 0.0] if zero else None})
        known_pnl = portfolio.notional * modeled_weight * component
        known_range = [portfolio.notional * modeled_weight * v for v in return_range]
        return {"state": label, "assumed_yield_shift_bps": shift, "net_yield_shock_bps": shift - reference,
                "tlt_return_component": component, "tlt_return_component_range": list(return_range),
                "known_pnl_component": known_pnl, "known_pnl_range": known_range,
                "total_pnl_component": known_pnl if unmodeled_weight == 0 else None,
                "total_pnl_range": known_range if unmodeled_weight == 0 else None,
                "contributions": contributions}

    result["branches"] = [branch("YES", yes, yes_range), branch("NO", no, no_range)]
    result["zero_shock_baseline"] = branch("ZERO_NET_YIELD_SHOCK", reference, (reference, reference))
    result["status"] = "AVAILABLE_CONDITIONAL_SENSITIVITY"
    result["reason"] = "Documented issuer duration plus explicit conditional shock assumptions; event effect is not empirically estimated."
    return result

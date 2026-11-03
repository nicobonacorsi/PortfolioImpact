"""Auditable composition of adapters, semantics, probability and portfolio stress."""
from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
from datetime import datetime, timezone
from itertools import combinations
import hashlib
import json
from pathlib import Path
from typing import Any

from .bounds import Relation, QuoteInterval, coherent_bounds
from .portfolio import parse_portfolio, recover_exposures
from .impact import analyze_scenario
from .semantics import (verify_relationship, parse_market_predicate, deterministic_relation,
                        make_fed_target, make_contract_target, make_cpi_settlement_target,
                        CPI_TERMS_SHA256)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PORTFOLIO = [
    {"symbol": "NVDA", "weight": 0.35, "asset_type": "stock"},
    {"symbol": "QQQ", "weight": 0.45, "asset_type": "etf"},
    {"symbol": "TLT", "weight": 0.20, "asset_type": "etf"},
]
DEFAULT_STRESSES = {"NVDA": -0.12, "QQQ": -0.08, "TLT": -0.03}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def dt(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("A timezone-aware observation timestamp is required.")
    return result


def _read_verified(path: Path) -> tuple[dict, dict]:
    raw = path.read_bytes()
    metadata = json.loads(path.with_suffix(".meta.json").read_text())
    if hashlib.sha256(raw).hexdigest() != metadata["sha256"]:
        raise ValueError(f"Archived source hash mismatch: {path.name}")
    if metadata.get("fixture") is not False or metadata.get("status") != 200:
        raise ValueError("This replay loader only admits archived successful real retrievals.")
    return json.loads(raw), metadata


def load_real_snapshot(family: str = "oct26") -> dict:
    """Read a frozen real API response; never label it live or fabricated fixture."""
    from .adapters import market_from_raw
    if family == "cpi_oct26":
        return _load_cpi_snapshot()
    folder = ROOT / "data" / "evidence"
    prefix = {"oct26": "fed_oct", "jan28": "fed_jan28"}[family]
    raw, meta = _read_verified(folder / f"{prefix}_event.json")
    markets, books, receipts = [], {}, [meta["retrieved_at"]]
    for record in raw["markets"]:
        market = market_from_raw(record, retrieved_at=meta["retrieved_at"], raw_path=str(folder / f"{prefix}_event.json"))
        markets.append(market)
        suffix = record["ticker"].rsplit("-", 1)[-1]
        book, bm = _read_verified(folder / f"{prefix}_{suffix}_book.json")
        book["_provenance"] = bm
        books[record["ticker"]] = book
        receipts.append(bm["retrieved_at"])
    return {"markets": markets, "orderbooks": books, "as_of": max(receipts, key=dt),
            "mode": "ARCHIVED_REAL_REPLAY", "event_ticker": raw["event"]["event_ticker"],
            "retrieved_at": meta["retrieved_at"], "source": meta["url"]}


def _load_cpi_snapshot() -> dict:
    """Reconstruct reviewed source context from hash-checked archived responses.

    The case file selects records; its duplicate context is never certification
    evidence. PDF bytes are checked independently from JSON responses.
    """
    from .adapters import market_from_raw
    folder = ROOT / "data" / "evidence" / "nonidentity"
    spec = json.loads((folder / "case.json").read_text(encoding="utf-8"))
    def source_path(relative: str) -> Path:
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(folder.resolve()):
            raise ValueError("CPI evidence path escapes the reviewed archive folder.")
        return path
    event_path = source_path(spec["source_artifacts"]["event"])
    series_path = source_path(spec["source_artifacts"]["series"])
    terms_path = source_path(spec["source_artifacts"]["terms"])
    event, event_meta = _read_verified(event_path)
    series, series_meta = _read_verified(series_path)
    terms_meta = json.loads(terms_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    actual_terms_hash = hashlib.sha256(terms_path.read_bytes()).hexdigest()
    if (actual_terms_hash != terms_meta.get("sha256") or actual_terms_hash != CPI_TERMS_SHA256
            or terms_meta.get("status") != 200 or terms_meta.get("fixture") is not False):
        raise ValueError("CPI terms are not the successful reviewed real source version.")
    if (event_meta.get("url") != "https://external-api.kalshi.com/trade-api/v2/events/KXCPI-26OCT"
            or series_meta.get("url") != "https://external-api.kalshi.com/trade-api/v2/series/KXCPI"
            or terms_meta.get("url") != "https://assets.kalshi.com/contract_terms/CPI.pdf"
            or event.get("event", {}).get("event_ticker") != "KXCPI-26OCT"):
        raise ValueError("CPI archived source endpoint or event identity does not match the reviewed case.")
    context = {"event": event["event"], "series": series["series"], "terms_sha256": actual_terms_hash}
    selected = spec["market_ids"]
    if not isinstance(selected, list) or not 1 <= len(selected) <= 10 or len(set(selected)) != len(selected):
        raise ValueError("CPI case must select one to ten distinct archived markets.")
    records = event.get("markets", [])
    if any(sum(r.get("ticker") == ticker for r in records) != 1 for ticker in selected):
        raise ValueError("Selected CPI market is missing or duplicated in the source response.")
    books, markets, receipts = {}, [], [event_meta["retrieved_at"]]
    for ticker in selected:
        raw = deepcopy(next(r for r in records if r.get("ticker") == ticker))
        raw["_reviewed_context"] = deepcopy(context)
        markets.append(market_from_raw(raw, retrieved_at=event_meta["retrieved_at"], raw_path=str(event_path)))
        book_path = source_path(spec["orderbook_artifacts"][ticker])
        book, metadata = _read_verified(book_path)
        if metadata.get("url") != f"https://external-api.kalshi.com/trade-api/v2/markets/{ticker}/orderbook":
            raise ValueError("CPI orderbook source is not the selected market.")
        book["_provenance"] = metadata
        books[ticker] = book
        receipts.append(metadata["retrieved_at"])
    target = make_cpi_settlement_target()
    natural_target = deepcopy(target)
    natural_target.update(
        event_id="cpi-bls-first-release-2026-10-ge-0.35",
        title="First BLS October 2026 headline CPI-U seasonally adjusted MoM release is at least 0.35%",
        description="Use the first published October 2026 BLS headline CPI-U seasonally adjusted monthly percent change. No settlement fallback or later revision is substituted.",
        definition_scope="UNRESTRICTED_BLS_FIRST_RELEASE",
    )
    natural_target["predicate"].update(metric="headline_cpi_u_sa_mom", source="Bureau of Labor Statistics",
                                      vintage="first_published_release", cancellation_policy="no_release_means_event_undefined")
    natural_target.pop("accepted_model_assumptions")
    natural_target.pop("required_model_assumptions")
    return {
        "markets": markets, "orderbooks": books, "as_of": max(receipts, key=dt),
        "mode": "ARCHIVED_REAL_REPLAY", "event_ticker": event["event"]["event_ticker"],
        "retrieved_at": event_meta["retrieved_at"], "source": event_meta["url"],
        "default_target": target, "natural_macro_target": natural_target,
        "case_label": "CPI Oct 2026 — nonidentity, conditional settlement-value model",
        "source_limitations": spec.get("source_limitations", []),
        "source_provenance": {"event": event_meta, "series": series_meta, "terms": terms_meta},
        "retrieval_accounting": spec.get("retrieval_accounting", {}),
        "selected_market_ids": selected,
        "excluded_event_market_ids": [r["ticker"] for r in records if r.get("ticker") not in selected],
    }


def _conditional_model_adopted(parsed: dict, target: dict) -> bool:
    """An LLM cannot adopt a model, including via market-to-market crosslinks."""
    if parsed.get("status") != "CONDITIONAL_VERIFIED":
        return parsed.get("status") == "CERTIFIED"
    accepted, required = target.get("accepted_model_assumptions"), parsed.get("required_model_assumptions")
    return (target.get("definition_origin") in {"USER_DEFINED", "REVIEWED_REGISTRY", "CONTRACT_IDENTITY"}
            and isinstance(accepted, list) and isinstance(required, list) and bool(required)
            and all(condition in accepted for condition in required))


def market_record(market: Any) -> dict:
    if isinstance(market, dict):
        return market.get("metadata", {}).get("raw", market.get("raw", market))
    return market.metadata.get("raw", asdict(market))


def analyze(
    portfolio_rows: list[dict], stresses: dict[str, float], snapshot: dict, *,
    target: dict | None = None, notional: float = 100_000,
    disabled_markets: list[str] | None = None, holdings: dict | None = None,
    proposals: list[dict] | None = None, max_quote_age_seconds: int = 300,
) -> dict:
    """Recompute from explicit inputs; no LLM call or hidden network effects."""
    from .adapters import normalize_quote
    now = dt(snapshot["as_of"]) if snapshot["mode"] == "ARCHIVED_REAL_REPLAY" else utcnow()
    target = target or default_target(snapshot)
    portfolio = parse_portfolio(portfolio_rows, notional)
    exposure = recover_exposures(portfolio, holdings or {}, now=utcnow())
    scenario = analyze_scenario(portfolio, stresses, name="Conditional event stress — user assumptions",
                                assumption_note="Scenario horizon: the selected event's resolution window. These returns are editable assumptions from today's marks.")
    disabled = set(disabled_markets or [])
    proposal_by_id = {p["market_id"]: p for p in (proposals or [])}
    relations, quotes, matches, quote_rows, parsed = [], [], [], [], {}
    parsed_diagnostics, conditional_constraints, required_assumptions = {}, [], []
    markets = snapshot["markets"]
    # The slice admits one venue/event/expiration group. No aggregation across
    # unrelated venue measures or different settlement dates is silently allowed.
    scope_event = snapshot.get("event_ticker")
    expirations = {market_record(m).get("expiration_time") for m in markets
                   if market_record(m).get("event_ticker") == scope_event}
    common_expiration = next(iter(expirations)) if len(expirations) == 1 and None not in expirations else None
    measure = f"kalshi:USD:{scope_event}:{common_expiration}:D=1"
    for market in markets:
        raw = market_record(market)
        ticker = raw["ticker"]
        match = verify_relationship(target, raw, proposal_by_id.get(ticker))
        match["challenged"] = ticker in disabled
        match["source_url"] = getattr(market, "source_url", "")
        match["rules_primary"] = raw.get("rules_primary", "")
        match["rules_secondary"] = raw.get("rules_secondary", "")
        in_scope = raw.get("event_ticker") == scope_event and common_expiration is not None
        match_admitted = (match["status"] in {"CERTIFIED", "CONDITIONAL_VERIFIED"}
                          and _conditional_model_adopted(match, target)
                          and match["relationship"] != "NONE" and ticker not in disabled and in_scope)
        match["constraint_admitted"] = match_admitted
        if match_admitted:
            relations.append(Relation(ticker, match["relationship"], target["event_id"],
                                      evidence="\n".join(match["evidence"]), verified=True))
            if match["status"] == "CONDITIONAL_VERIFIED":
                conditional_constraints.append({"left": ticker, "right": target["event_id"], "relationship": match["relationship"]})
        matches.append(match)
        p = parse_market_predicate(raw)
        parsed_diagnostics[ticker] = p
        for assumption in p.get("required_model_assumptions", []):
            if assumption not in required_assumptions:
                required_assumptions.append(assumption)
        if _conditional_model_adopted(p, target) and ticker not in disabled and in_scope:
            parsed[ticker] = p
        quote = normalize_quote(market, snapshot.get("orderbooks", {}).get(ticker),
                                now=now, max_age_seconds=max_quote_age_seconds)
        qr = asdict(quote)
        qr.update(market_id=ticker, source_url=getattr(market, "source_url", ""))
        reason = quote.exclusion_reason
        usable = quote.usable and in_scope and ticker not in disabled
        if ticker in disabled:
            reason = "Excluded by user challenge; all associated constraints removed."
        elif not in_scope:
            reason = "No common event/expiration group; a shared pricing measure is not asserted."
        qr.update(price_eligible=usable, admission_reason=reason)
        quote_rows.append(qr)
        quotes.append(QuoteInterval(ticker, quote.lower if quote.lower is not None else 0.0,
                                    quote.upper if quote.upper is not None else 1.0,
                                    measure, ticker, usable, reason))
    # Independently parsed pairwise constraints tighten bounds beyond a star.
    for left, right in combinations(sorted(parsed), 2):
        relation = deterministic_relation(parsed[right]["predicate"], parsed[left]["predicate"])
        if relation != "NONE":
            conditional = any(parsed[t]["status"] == "CONDITIONAL_VERIFIED" for t in (left, right))
            relations.append(Relation(left, relation, right,
                                      evidence="Complete reviewed source templates; exact scalar-set relation."
                                               + (" Conditional on the explicitly adopted common settlement-value assumptions." if conditional else ""),
                                      verified=True))
            if conditional:
                conditional_constraints.append({"left": left, "right": right, "relationship": relation})
    bound = coherent_bounds(target["event_id"], relations, quotes)
    active_events = set(bound.diagnostics.get("events", []))
    conditional_constraints = [r for r in conditional_constraints
                               if r["left"] in active_events and r["right"] in active_events]
    admitted_ids = {q["event_id"] for q in bound.diagnostics.get("admitted_quotes", [])}
    for row in quote_rows:
        row["admitted_to_lp"] = row["market_id"] in admitted_ids
        row["lp_exclusion_reason"] = ("" if row["admitted_to_lp"] else row["admission_reason"] or
                                      "No admitted semantic path connects this quote to the selected target.")
    semantic_status = ("CONDITIONALLY_VERIFIED_RELATIONSHIPS" if conditional_constraints else
                       "VERIFIED_RELATIONSHIPS" if any(m["constraint_admitted"] for m in matches) else
                       "UNVERIFIED_RELATIONSHIPS" if any(m["status"] not in {"CERTIFIED", "CONDITIONAL_VERIFIED"} for m in matches) else "VERIFIED_NONMATCH")
    result = {
        "schema_version": 1, "mode": snapshot["mode"], "calculated_at": utcnow().isoformat(),
        "evaluation_time": now.isoformat(), "source": snapshot.get("source"), "target": target,
        "portfolio": portfolio.to_dict(), "exposures": exposure.to_dict(),
        "semantic_status": semantic_status, "relationships": matches,
        "admitted_constraints": bound.diagnostics.get("admitted_relations", []),
        "market_predicates": parsed_diagnostics,
        "semantic_model": {
            "status": "CONDITIONAL_ASSUMPTIONS_ADOPTED" if conditional_constraints else
                      "CONDITIONAL_ASSUMPTIONS_NOT_USED" if required_assumptions else "NO_ADDITIONAL_CONDITIONAL_MODEL",
            "required_assumptions": required_assumptions,
            "accepted_assumptions": target.get("accepted_model_assumptions", []),
            "conditional_constraint_count": len(conditional_constraints),
            "conditional_constraints": conditional_constraints,
            "natural_macro_target": snapshot.get("natural_macro_target"),
            "source_limitations": snapshot.get("source_limitations", []),
        },
        "source_provenance": snapshot.get("source_provenance", {}),
        "quotes": quote_rows, "bounds": bound.to_dict(), "scenario": scenario.to_dict(),
        "agent_execution": "NOT_RUN" if proposals is None else "PROPOSALS_SUPPLIED_AND_REVERIFIED",
        "assumptions": [
            "One Kalshi event, USD unit payout; common settlement/numeraire approximation and discount factor D=1.",
            "Bounds assume a coherent shadow-price measure lies inside the observed fee-free bid/ask inequalities.",
            "Identical expiration metadata does not prove identical actual cash settlement timing.",
            "Order-book snapshots are sequential; receipt freshness is known, exchange quote age is not.",
            "Displayed normalized prices exclude fees, depth-dependent execution, financing and settlement risk.",
            "These are conditional pricing-measure bounds, not physical probabilities or statistical confidence intervals.",
            "A feasible LP does not validate the semantics. An infeasible LP does not prove executable arbitrage.",
        ] + (required_assumptions if conditional_constraints else []),
    }
    digest_input = {"portfolio": result["portfolio"], "target": target, "quotes": quote_rows,
                    "disabled": sorted(disabled), "stresses": stresses, "relationships": matches}
    result["input_sha256"] = hashlib.sha256(json.dumps(digest_input, sort_keys=True, default=str).encode()).hexdigest()
    return result


def replay(**kwargs: Any) -> dict:
    return analyze(kwargs.pop("portfolio_rows", DEFAULT_PORTFOLIO), kwargs.pop("stresses", DEFAULT_STRESSES),
                   load_real_snapshot(kwargs.pop("family", "oct26")), **kwargs)


def fetch_event_snapshot(event_ticker: str, data_dir: Path | str = "runtime/kalshi") -> dict:
    """Read one public event and at most ten books; preserve every raw response."""
    from .adapters import KalshiClient, normalize_market
    client = KalshiClient(data_dir, cache_ttl=0)
    response = client.get_event(event_ticker)
    meta = response["_provenance"]
    records = response.get("markets", [])
    if not records:
        raise ValueError("This event returned no markets.")
    if len(records) > 10:
        raise ValueError("This first slice supports at most ten contracts per event; choose a smaller event.")
    markets = [normalize_market(m, meta["retrieved_at"], meta["raw_path"]) for m in records]
    books = {m.ticker: client.get_orderbook(m.ticker) for m in markets}
    return {"markets": markets, "orderbooks": books, "as_of": utcnow().isoformat(),
            "mode": "LIVE_API_OBSERVATION", "event_ticker": event_ticker,
            "retrieved_at": meta["retrieved_at"], "source": meta["url"]}


def default_target(snapshot: dict) -> dict:
    """A usable reviewed definition without silently repairing ambiguous rules."""
    if "default_target" in snapshot:
        return deepcopy(snapshot["default_target"])
    if snapshot["event_ticker"] == "KXFEDDECISION-28JAN":
        return make_fed_target("2028-01-26", operator="ge", threshold=25)
    markets = snapshot["markets"]
    chosen = next((m for m in markets if market_record(m)["ticker"].endswith("-H25")), markets[0])
    return make_contract_target(market_record(chosen))

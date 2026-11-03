"""R125 point-2 evidence stage: per-event targeted research and two-venue market review.

Runs only inside the authenticated live analysis, after the original event
slate exists and before the structural forecaster. It owns a separate, bounded
Anthropic client and separate public Kalshi/Polymarket read budgets, so the
discovery and point-3 stages keep their R124 budgets untouched. Every per-event
failure is recorded and never raised; the forecaster then sees exactly the
evidence that was actually retrieved.

Nothing here produces a probability or a portfolio number.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .agent import AgentError, AnthropicClient
from .event_evidence_research import public_event_evidence, research_event_evidence
from .event_market_search import review_event_markets
from .live_catalog import BoundedKalshiCatalog
from .provider_access import ValidatedKalshiClient

STAGE_SCHEMA = "castle.point2_evidence_stage.v1"
POINT2_KALSHI_REQUESTS = 300
POINT2_GAMMA_REQUESTS = 120
POINT2_MAX_SERIES_PER_QUERY = 4
POINT2_OUTPUT_TOKENS = 16_384


def _definition_detail(event_id: str, discovery_events: list, default_events: list) -> str | None:
    for event in discovery_events or []:
        if isinstance(event, dict) and event.get("event_id") == event_id:
            for key in ("question_we_care_about", "description"):
                value = event.get(key)
                if isinstance(value, str) and value.strip():
                    return " ".join(value.split())[:1500]
    for event in default_events or []:
        if isinstance(event, dict) and event.get("event_id") == event_id:
            value = event.get("original_question")
            if isinstance(value, str) and value.strip():
                return " ".join(value.split())[:1500]
    return None


def _proposed_queries(event_id: str, discovery_events: list) -> list[str]:
    for event in discovery_events or []:
        if isinstance(event, dict) and event.get("event_id") == event_id:
            queries = event.get("search_queries")
            if isinstance(queries, list):
                return [q for q in queries if isinstance(q, str)][:3]
    return []


def _seed_candidates(event_id: str, geopolitical_search: object) -> list[dict]:
    """Already captured family-search cards for this event become review seeds."""
    seeds = []
    if not isinstance(geopolitical_search, dict):
        return seeds
    for target in geopolitical_search.get("targets") or []:
        if not isinstance(target, dict) or target.get("event_id") != event_id:
            continue
        for card in target.get("candidates") or []:
            if not isinstance(card, dict) or not isinstance(card.get("rules_text"), str):
                continue
            reference = card.get("contract_quote_reference") or {}
            interval = reference.get("interval") if isinstance(reference, dict) else None
            seeds.append({
                "venue": card.get("venue"), "contract_id": str(card.get("contract_id") or ""),
                "question": str(card.get("market_question") or "")[:300],
                "rules_text": card["rules_text"], "rules_sha256": card.get("rules_sha256"),
                "close_time": str(card.get("contract_end_time") or ""),
                "market_open_at_evaluation": bool(card.get("market_open_at_evaluation")),
                "quote_interval": list(interval) if isinstance(interval, list) and len(interval) == 2 else None,
                "quote_scope": (reference.get("scope") if isinstance(reference, dict) else None)
                or "FAMILY_SEARCH_CAPTURE_REFERENCE",
                "quote_reasons": list(reference.get("reasons") or []) if isinstance(reference, dict) else [],
                "source_url": card.get("source_url"),
                "retrieved_at": (card.get("source") or {}).get("retrieved_at"),
                "source_response_sha256": (card.get("source") or {}).get("source_response_sha256"),
                "search_query": "FAMILY_SEARCH_SEED",
            })
    return [row for row in seeds if row["contract_id"] and row["venue"] in ("Kalshi", "Polymarket")]


def run_point2_evidence_stage(candidates: list[dict], *, run_dir: Path, anthropic_key: str | None,
                              model: str | None, as_of: datetime, discovery_events: list | None = None,
                              default_events: list | None = None, geopolitical_search: object = None,
                              progress: Callable | None = None, client: Any = None,
                              kalshi_client: Any = None, gamma_client: Any = None,
                              resume_archives: bool = False, archive_reuse: dict | None = None,
                              portfolio_search: object = None) -> dict:
    """Targeted evidence and market review for each eligible original event."""
    notify = progress or (lambda message: None)
    run_dir = Path(run_dir)
    from .restart_recovery import prime_verified_http_cache
    reuse_counts = archive_reuse if archive_reuse is not None else {}
    started = datetime.now(timezone.utc)
    evidence: dict[str, dict] = {}
    public_evidence: dict[str, dict] = {}
    reviews: dict[str, dict] = {}
    details: dict[str, str] = {}
    failures: list[dict] = []
    stage = {"schema": STAGE_SCHEMA, "status": "NOT_RUN", "event_count": len(candidates),
             "started_at_utc": started.isoformat()}
    if not candidates:
        stage["status"] = "NO_ELIGIBLE_ORIGINAL_EVENT"
        return {"evidence": evidence, "public_evidence": public_evidence, "market_reviews": reviews,
                "definition_details": details, "stage": stage, "ledger": []}
    try:
        point2_client = client or AnthropicClient(
            api_key=anthropic_key, **({"model": model} if model else {}),
            archive_dir=run_dir / "claude_point2",
            reuse_verified_archives=bool(resume_archives),
            max_requests=min(50, 4 * len(candidates) + 4), max_output_tokens=POINT2_OUTPUT_TOKENS,
            max_total_tokens=min(12_000_000, (4 * len(candidates) + 4) * (100_000 + POINT2_OUTPUT_TOKENS + 2048)),
            timeout_seconds=240, min_interval_seconds=1, max_request_bytes=100_000,
            max_attempts_per_request=2)
    except AgentError as exc:
        stage.update(status="FAILED_CLIENT_CONSTRUCTION", reason=str(exc)[:160])
        return {"evidence": evidence, "public_evidence": public_evidence, "market_reviews": reviews,
                "definition_details": details, "stage": stage, "ledger": []}
    try:
        kalshi = kalshi_client or ValidatedKalshiClient(run_dir / "point2_public" / "kalshi",
                                                         max_requests=POINT2_KALSHI_REQUESTS,
                                                         max_retries=2, cache_ttl=0)
        if resume_archives and kalshi_client is None:
            reuse_counts["KalshiPoint2"] = prime_verified_http_cache(
                kalshi, run_dir / "point2_public" / "kalshi")
        catalog = BoundedKalshiCatalog(kalshi, as_of=as_of.date(), expanded=True,
                                       max_series_per_query=POINT2_MAX_SERIES_PER_QUERY)
    except (ValueError, TypeError, OSError) as exc:
        catalog = None
        failures.append({"scope": "KALSHI_CLIENT", "reason": type(exc).__name__})
    try:
        if gamma_client is None:
            from .geopolitical_market_search import GammaSearchClient
            gamma_client = GammaSearchClient(run_dir / "point2_public" / "polymarket",
                                             max_requests=POINT2_GAMMA_REQUESTS, max_retries=2, cache_ttl=0)
            if resume_archives:
                reuse_counts["PolymarketPoint2"] = prime_verified_http_cache(
                    gamma_client, run_dir / "point2_public" / "polymarket")
    except (ValueError, TypeError, OSError) as exc:
        gamma_client = None
        failures.append({"scope": "GAMMA_CLIENT", "reason": type(exc).__name__})
    for index, context in enumerate(candidates, start=1):
        event_id = context.get("event_id")
        if not isinstance(event_id, str):
            continue
        detail = _definition_detail(event_id, discovery_events or [], default_events or [])
        if detail:
            details[event_id] = detail
        notify(f"Event {index}/{len(candidates)}: targeted evidence search for the exact original event.")
        try:
            row = research_event_evidence(point2_client, context, definition_detail=detail,
                                          portfolio_search=portfolio_search)
        except Exception as exc:  # explicit per-event failure, never a stage crash
            row = None
            failures.append({"scope": "TARGETED_RESEARCH", "event_id": event_id, "reason": type(exc).__name__})
        if row is not None:
            evidence[event_id] = row
            if str(row.get("status", "")).endswith(("FAILED_NO_SOURCES", "SEARCH_ERROR")):
                failures.append({"scope": "TARGETED_RESEARCH", "event_id": event_id, "reason": row["status"]})
            public = public_event_evidence(row)
            if public is not None:
                public_evidence[event_id] = public
        notify(f"Event {index}/{len(candidates)}: Kalshi and Polymarket contract search and relation review.")
        try:
            review = review_event_markets(
                point2_client, context, kalshi_catalog=catalog, gamma_client=gamma_client,
                proposed_queries=_proposed_queries(event_id, discovery_events or []),
                definition_detail=detail, evaluated_at=datetime.now(timezone.utc).isoformat(),
                seed_candidates=_seed_candidates(event_id, geopolitical_search))
        except Exception as exc:  # explicit per-event failure, never a stage crash
            review = None
            failures.append({"scope": "MARKET_REVIEW", "event_id": event_id, "reason": type(exc).__name__})
        if review is not None:
            reviews[event_id] = review
    ledger = [dict(entry, stage="POINT2_EVIDENCE") for entry in getattr(point2_client, "request_ledger", [])]
    counts = {"exact": 0, "entailment": 0, "related": 0, "rejected": 0, "not_reviewed": 0}
    for review in reviews.values():
        for key, value in (review.get("counts") or {}).items():
            counts[key] = counts.get(key, 0) + int(value)
    stage.update({
        "status": ("COMPLETED_WITH_FAILURES" if failures else "COMPLETED"),
        "targeted_research_events": len(evidence),
        "targeted_leads_total": sum(len(r.get("leads") or []) for r in evidence.values()),
        "url_verified_findings_total": sum(len(r.get("findings") or []) for r in evidence.values()),
        "model_reviewed_source_links_total": sum(len(r.get("source_relevance_links") or []) for r in evidence.values()),
        "web_search_requests_reported": sum(int(r.get("web_search_requests_reported") or 0) for r in evidence.values()),
        "market_reviews": len(reviews),
        "market_contract_counts": counts,
        "events_with_exact_model_reviewed_contract": sum(
            1 for r in reviews.values() if (r.get("counts") or {}).get("exact")),
        "anthropic_requests": len(ledger),
        "kalshi_requests": getattr(getattr(catalog, "client", None), "request_count", 0) if catalog else 0,
        "polymarket_requests": getattr(gamma_client, "request_count", 0) if gamma_client else 0,
        "failures": failures,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "TARGETED_LEADS_DIGESTS_AND_REVIEWED_CONTRACTS_NOT_FACT_CERTIFICATION",
        "probability_produced_here": False,
    })
    return {"evidence": evidence, "public_evidence": public_evidence, "market_reviews": reviews,
            "definition_details": details, "stage": stage, "ledger": ledger,
            "catalog_search_ledger": list(getattr(catalog, "search_ledger", [])) if catalog else []}

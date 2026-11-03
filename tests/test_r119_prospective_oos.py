"""Synthetic R119 prospective score regression tests (NOT resolved real events)."""
from __future__ import annotations

from copy import deepcopy
import json
from math import log

import pytest

from castle_risk.prospective_oos_score import (
    ProspectiveScoreError, evaluate_prospective_lock,
)


def _event(i, p=.8, family=None, exact=True):
    return {
        "event_id": f"EVENT{i}",
        "target_condition": f"Precisely defined synthetic release event {i} on future date",
        "event_family_id": family or f"FAMILY{i}",
        "model_forecast": {
            "status": "MODEL_FORECAST_UNCALIBRATED",
            "model_version": "frozen-test-model-v1",
            "provider_message_id": f"provider-msg-{i}",
            "provider_response_received_at_utc": "2026-11-02T00:00:00Z",
            "last_evidence_retrieved_at_utc": "2026-11-01T23:59:00Z",
            "probability_center": p,
            "credal_interval": [max(0,p-.1), min(1,p+.1)],
        },
        "market_quote": {
            "ticker": f"EXACT{i}",
            "captured_at_utc": "2026-11-01T23:57:00Z",
            "yes_bid": .48, "yes_ask": .52,
            "match_status": ("EXACT_CONTRACT_RULES_AUDITED" if exact
                             else "PROVISIONAL_MATCH_NOT_SCORED"),
        }
    }


def _lock(events):
    return {"schema": "castle.prospective_event_lock.v1",
            "locked_at_utc": "2026-11-02T00:01:00Z", "events": events}


def _resolved(e, y):
    return {
        "event_id": e["event_id"],
        "target_condition": e["target_condition"],
        "outcome": y,
        "resolved_at_utc": "2027-02-01T00:00:00Z",
        "resolution_type": "VENUE_SETTLEMENT",
        "resolution_source_url": "https://external-api.kalshi.com/trade-api/v2/markets/exact",
    }


def _score(lock, outcomes):
    return evaluate_prospective_lock(lock, outcomes, audit_time_utc="2027-02-02T00:00:00Z",
                                     min_independent_families=2)


def test_pending_is_not_misreported_as_successful_holdout():
    e = _event(1)
    row = _score(_lock([e]), [])
    assert row["status"] == "NO_RESOLVED_HOLDOUT_EVENTS_YET"
    assert row["model_brier"] is None
    assert row["paired_exact_market_count"] == 0
    assert row["forecast_empirically_calibrated"] is False


def test_exact_matched_quote_and_model_score_same_events():
    a, b = _event(1, .8), _event(2, .2)
    output = _score(_lock([a,b]), [_resolved(a, 1), _resolved(b, 0)])
    assert output["status"] == "DESCRIPTIVE_PROSPECTIVE_SCORES_REQUIRING_EXTERNAL_AUDIT"
    assert output["model_brier"] == pytest.approx(.04)
    assert output["market_midpoint_brier_on_paired_subset"] == pytest.approx(.25)
    assert output["market_minus_model_brier_by_family"] == pytest.approx(.21)
    assert output["model_log_loss"] == pytest.approx(-log(.8))
    assert output["independent_scored_families"] == 2
    assert output["cluster_bootstrap_95_percent_difference_interval"] is None
    assert output["external_outcome_authenticity_verified_by_this_module"] is False


def test_abstentions_and_provisional_quote_not_scored_selectively():
    a, b = _event(1, .6, exact=False), _event(2, .4)
    b["model_forecast"].update(status="ABSTAIN", probability_center=None,
                                credal_interval=None)
    out = _score(_lock([a,b]), [_resolved(a, 1), _resolved(b, 0)])
    assert out["scored_model_count"] == 1
    assert out["resolved_abstention_count"] == 1
    assert out["paired_exact_market_count"] == 0
    assert out["model_score_coverage_of_resolved"] == .5


def test_same_calendar_release_strikes_are_not_independent_families():
    a = _event(1, .8, family="ONE_CPI_RELEASE")
    b = _event(2, .8, family="ONE_CPI_RELEASE")
    out = _score(_lock([a,b]), [_resolved(a, 1), _resolved(b, 1)])
    assert out["independent_scored_families"] == 1
    assert out["independent_paired_market_families"] == 1
    assert out["status"] == "INSUFFICIENT_INDEPENDENT_FAMILIES"


@pytest.mark.parametrize("bad", [
    lambda lock, rows: lock["events"][0]["model_forecast"].update(
        provider_response_received_at_utc="2027-02-02T00:00:00Z"),
    lambda lock, rows: lock["events"][0]["model_forecast"].update(
        probability_center=float("nan")),
    lambda lock, rows: lock["events"][0]["market_quote"].update(
        yes_bid=.9,yes_ask=.1),
    lambda lock, rows: lock["events"][0]["market_quote"].update(
        captured_at_utc="2027-02-01T00:00:00Z"),
    lambda lock, rows: lock["events"][0]["market_quote"].update(
        captured_at_utc="2026-11-02T00:00:30Z"),
    lambda lock, rows: rows[0].update(target_condition="Different event entirely"),
    lambda lock, rows: rows[0].update(outcome=True),
    lambda lock, rows: rows[0].update(resolved_at_utc="2026-11-01T00:00:00Z"),
    lambda lock, rows: rows[0].update(resolved_at_utc="2027-02-03T00:00:00Z"),
])
def test_chronology_and_predicate_mutations_rejected(bad):
    e = _event(1)
    lock, rows = _lock([e]), [_resolved(e, 1)]
    bad(lock, rows)
    with pytest.raises(ProspectiveScoreError):
        _score(lock, rows)


def test_mixed_model_versions_cannot_be_misreported_as_one_skill_estimate():
    a, b = _event(1), _event(2)
    b["model_forecast"]["model_version"] = "different-after-results-model"
    with pytest.raises(ProspectiveScoreError, match="Mixed"):
        _score(_lock([a,b]), [])


def test_infinite_loss_is_reported_without_nonfinite_mean():
    a, b = _event(1, p=0.0), _event(2, p=1.0)
    out = _score(_lock([a,b]), [_resolved(a, 1), _resolved(b, 0)])
    assert out["model_brier"] == 1.0
    assert out["model_log_loss"] is None
    assert out["model_log_loss_infinite_due_to_confident_error"] is True
    assert all(row["model_log_loss"] is None for row in out["score_rows"])
    json.dumps(out, allow_nan=False)  # no invalid JSON Infinity


def test_duplicate_locked_ids_and_resolution_rejected():
    a = _event(1)
    with pytest.raises(ProspectiveScoreError):
        _score(_lock([a,deepcopy(a)]), [_resolved(a, 1)])
    with pytest.raises(ProspectiveScoreError):
        _score(_lock([a]), [_resolved(a, 1), _resolved(a, 1)])


def test_bootstrap_only_after_ten_independent_paired_families_and_deterministic():
    events = [_event(i,p=.1 if i%2 else .9) for i in range(12)]
    adjudications = [_resolved(e, int(i%3==0)) for i,e in enumerate(events)]
    out = _score(_lock(events), adjudications)
    ci = out["cluster_bootstrap_95_percent_difference_interval"]
    assert len(ci) == 2 and ci[0] <= ci[1]
    assert ci == _score(_lock(events), adjudications)[
        "cluster_bootstrap_95_percent_difference_interval"]

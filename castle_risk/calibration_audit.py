"""Archived historical-slope stresses, never causal coefficients or live data.

The source event-study receipts belong to the R11/R13 retrospective evidence.
They have weaker identification than a high-frequency surprise instrument and
are NEVER promoted to a probability distribution or confidence interval.
"""
from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import json
import math
from pathlib import Path

from .tariff_scenarios import compare_anchor_calibrations


ARCHIVES = {
    "cpi_fomc": "reports/r11_validation/event_study_evidence.json",
    "payroll": "reports/r13_validation/payroll_evidence.json",
}
ARCHIVE_SAFE_AS_OF = datetime.fromisoformat("2026-10-25T00:00:00+00:00")


def load_archived_calibration_sources(root):
    """Read packaged audit receipts only; report missing files instead of refitting."""
    data, receipts = {}, {}
    for name, relative in ARCHIVES.items():
        path = Path(root) / relative
        if not path.is_file():
            receipts[name] = {"status": "ARCHIVED_SOURCE_UNAVAILABLE", "path": relative}
            continue
        raw = path.read_bytes()
        try:
            data[name] = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            receipts[name] = {"status": "ARCHIVED_SOURCE_MALFORMED", "path": relative}
            continue
        receipts[name] = {"status": "PACKAGED_ARCHIVED_RECEIPT_NOT_NEW_PROVIDER_VALIDATION",
                          "path": relative, "sha256": sha256(raw).hexdigest()}
    return data, receipts


def _book_name(report, recorded):
    portfolio = (report.get("portfolio") or {}).get("positions") or []
    given = {p["symbol"]: float(p["weight"]) for p in portfolio}
    if not given or not math.isclose(sum(given.values()), 1, rel_tol=0, abs_tol=1e-7):
        return None
    for label, book in sorted(recorded.items()):
        if given.keys() == book.keys() and all(math.isclose(given[s], w, rel_tol=0, abs_tol=1e-8)
                                               for s, w in book.items()):
            return label
    return None


def _pnl_for_cpi_slope(table, slope):
    positions = table["positions"]
    equity_loading = math.fsum(p["weight"] * p["loading"] for p in positions if p.get("channel") == "EQUITY")
    rate_loading = math.fsum(p["weight"] * p["loading"] for p in positions if p.get("channel") == "RATES")
    rate_point = table["factors"]["rates"]["point"]
    unit = equity_loading * slope + rate_loading * rate_point / 10000
    notional = table["notional_usd"]
    mean = table["reference_expectation"]
    return [notional * unit * (row["state"] - mean) for row in table["outcomes"]]


def _pnl_for_payroll_slope(table, slope):
    notional = table["notional_usd"]
    mean = table["reference_expectation"]
    return [notional * slope * (row["state"] - mean) / 100000 for row in table["outcomes"]]


def historical_stress_audit(report, sources):
    """Additional P&L vectors at frozen Q for *verified* exact-statistic matches.

    Does not choose whichever retrospective slope makes a preferred result.
    All available dated specifications remain in the robust ranking as a union.
    """
    alt, diagnostics = {}, []
    recorded = sources.get("payroll", {}).get("recorded_portfolios") or {}
    book = _book_name(report, recorded)
    evaluated = report.get("evaluated_at_utc") or report.get("evaluated_at")
    try:
        report_time = datetime.fromisoformat(str(evaluated).replace("Z", "+00:00"))
    except ValueError:
        report_time = None
    if report_time is None or report_time.utcoffset() is None or report_time < ARCHIVE_SAFE_AS_OF:
        return {"schema": "castle.option1.calibration_audit.v1", "alternative_vectors_by_event": {},
                "diagnostics": [{"status": "ARCHIVED_CALIBRATION_NOT_CHRONOLOGY_SAFE",
                                 "reason": "The packaged October 2026 empirical receipts cannot be applied to an earlier or undated assessment."}],
                "portfolio_exact_payroll_reference_match": None,
                "source_evidence": "NO_TIME_COMPATIBLE_ARCHIVED_DATA_USED",
                "causal_identification_established": False, "future_interval_coverage_established": False}
    for event in (report.get("default_analysis") or {}).get("events", []):
        eid, family = event["event_id"], event.get("family")
        table = event.get("outcome_table") or {}
        if not str(table.get("status", "")).startswith("COMPLETE_") or not table.get("outcomes"):
            continue
        if family == "TRADE_TARIFFS":
            try:
                anchors = compare_anchor_calibrations(table)
            except (KeyError, ValueError, TypeError) as exc:
                diagnostics.append({"event_id": eid, "status": "TARIFF_ANCHOR_AUDIT_UNAVAILABLE",
                                    "reason": str(exc)})
                continue
            alt[eid] = [{"name": "2025-04-03_reciprocal_tariffs_confounded",
                         "pnl_if_outcome_usd": [row["pnl_april_usd"] for row in anchors["outcomes"]]}]
            diagnostics.append({"event_id": eid, "status": "TWO_ANNOUNCEMENT_DAY_SPECIFICATIONS_NOT_CAUSAL",
                                "source": "R15_ARCHIVED_TARIFF_ANCHORS",
                                "direction_reversal_count": anchors["direction_reversal_count"],
                                "assumption": "The historical tariff episodes transfer linearly to this future rate change."})
        elif family == "US_INFLATION":
            history = ((sources.get("cpi_fomc") or {}).get("cpi") or {}).get("period_slopes", {}).get("SPY") or []
            if table.get("positions") is None or table.get("factors") is None or len(history) < 2:
                diagnostics.append({"event_id": eid, "status": "NO_FULL_CPI_REGIME_COMPARISON"})
                continue
            try:
                alt[eid] = [{"name": "SPY_CPI_" + str(p["period"]) + "_retrospective",
                             "pnl_if_outcome_usd": _pnl_for_cpi_slope(table, float(p["return_per_unit_surprise"]))}
                            for p in history if p.get("n", 0) >= 5]
            except (KeyError, TypeError, ValueError) as exc:
                alt.pop(eid, None)
                diagnostics.append({"event_id": eid, "status": "CPI_REGIME_ARITHMETIC_REJECTED", "reason": str(exc)})
                continue
            diagnostics.append({"event_id": eid, "status": "HISTORICAL_SPY_CPI_REGIME_SENSITIVITY_ONLY",
                                "n_by_period": {p["period"]: p["n"] for p in history},
                                "slope_by_period": {p["period"]: p["return_per_unit_surprise"] for p in history},
                                "does_not_identify": "Next-release causal CPI response or future predictive coverage"})
        elif family == "US_PAYROLLS":
            fit = ((sources.get("payroll") or {}).get("fits") or {}).get("all", {}).get(book or "")
            last = fit.get("last") if isinstance(fit, dict) else None
            if (not isinstance(fit, dict) or fit.get("status") != "ESTIMATED"
                    or not last or not evaluated or last >= str(evaluated)[:10]):
                diagnostics.append({"event_id": eid, "status": "NO_CHRONOLOGY_SAFE_EXACT_BOOK_PAYROLL_SPLIT",
                                    "portfolio_match": book})
                continue
            if any(fit.get(k) is None for k in ("slope_first_two_thirds", "slope_last_third")):
                continue
            alt[eid] = [{"name": "payroll_first_two_thirds_retrospective",
                         "pnl_if_outcome_usd": _pnl_for_payroll_slope(table, fit["slope_first_two_thirds"])},
                        {"name": "payroll_last_third_retrospective",
                         "pnl_if_outcome_usd": _pnl_for_payroll_slope(table, fit["slope_last_third"])}]
            diagnostics.append({"event_id": eid, "status": "EXACT_PORTFOLIO_PAYROLL_REGIME_SPLIT_NOT_CAUSAL",
                                "portfolio_match": book, "last_observation": last, "training_n": fit.get("n"),
                                "slope_first_two_thirds": fit["slope_first_two_thirds"],
                                "slope_last_third": fit["slope_last_third"],
                                "does_not_identify": "Future slope or information about wages and revisions"})
        elif family == "FED_POLICY":
            fomc = (sources.get("cpi_fomc") or {}).get("fomc") or {}
            diagnostics.append({"event_id": eid, "status": "POLICY_DECISION_ONLY_COMMUNICATION_UNIDENTIFIED",
                                "historical_decision_surprise_n": fomc.get("n"),
                                "historical_median_abs_decision_surprise_bp": fomc.get("median_abs_surprise"),
                                "unmodeled": ["Statement/path guidance", "Press conference", "Concurrent news"],
                                "no_unjustified_communication_pnl": True})
    return {"schema": "castle.option1.calibration_audit.v1", "alternative_vectors_by_event": alt,
            "diagnostics": diagnostics, "portfolio_exact_payroll_reference_match": book,
            "source_evidence": "REPLAYED_ARCHIVED_RETROSPECTIVE_NOT_NEW_PROVIDER_OBSERVATIONS",
            "causal_identification_established": False,
            "future_interval_coverage_established": False}

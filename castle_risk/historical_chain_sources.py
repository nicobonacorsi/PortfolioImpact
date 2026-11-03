"""Auditable, offline admission of historical reference probabilities.

This module never substitutes current prices, invents a missing quote, or uses
asset outcomes to choose a reference probability. All exclusions are retained.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re


def moment(value):
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.utcoffset() is None:
        raise ValueError("Timezone-aware timestamp required")
    return result


def new_york_daytime_utc(day, hour, minute):
    """US post-2007 DST rule for unambiguous daytime clocks, without tzdata.

    Only years represented by this protocol and times after 04:00 are admitted;
    this is not a replacement for a general timezone implementation.
    """
    local_day = date.fromisoformat(str(day)[:10])
    if not 2007 <= local_day.year <= 2037 or not 4 <= hour <= 23:
        raise ValueError("Unsupported civil-time conversion")
    march = date(local_day.year, 3, 1)
    start = march + timedelta(days=(6-march.weekday()) % 7 + 7)
    november = date(local_day.year, 11, 1)
    end = november + timedelta(days=(6-november.weekday()) % 7)
    offset = 4 if start <= local_day < end else 5
    return datetime(local_day.year, local_day.month, local_day.day, hour, minute,
                    tzinfo=timezone(timedelta(hours=-offset))).astimezone(timezone.utc)


def verify_source(folder, filename, expected):
    """Verify only paths contained in the explicit acquired-source directory."""
    folder = Path(folder).resolve()
    path = (folder / filename).resolve()
    if not path.is_relative_to(folder):
        raise ValueError("Source path escapes acquisition directory")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError("Source SHA-256 mismatch: " + str(filename))
    return actual


def project_box_simplex(midpoints, lower, upper, *, tolerance=1e-11):
    """Unique Euclidean projection onto a probability simplex with hard bounds."""
    if not (len(midpoints) == len(lower) == len(upper)) or not lower:
        raise ValueError("Nonempty equal-length vectors required")
    if any(not math.isfinite(x) for row in (midpoints, lower, upper) for x in row):
        raise ValueError("Nonfinite probability")
    if any(not 0 <= lo <= hi <= 1 for lo, hi in zip(lower, upper)):
        raise ValueError("Invalid probability interval")
    if math.fsum(lower) > 1 + tolerance or math.fsum(upper) < 1 - tolerance:
        raise ValueError("INFEASIBLE_HARD_QUOTE_POLYTOPE")
    lo_lambda = min(m - hi for m, hi in zip(midpoints, upper)) - 1
    hi_lambda = max(m - lo for m, lo in zip(midpoints, lower)) + 1
    for _ in range(100):
        shift = (lo_lambda + hi_lambda) / 2
        q = [min(hi, max(lo, m - shift)) for m, lo, hi in zip(midpoints, lower, upper)]
        if math.fsum(q) > 1:
            lo_lambda = shift
        else:
            hi_lambda = shift
    # Correct floating point summation only, within the original interval.
    delta = 1 - math.fsum(q)
    for i in range(len(q)):
        adjustment = min(upper[i] - q[i], max(lower[i] - q[i], delta))
        q[i] += adjustment
        delta -= adjustment
    if abs(math.fsum(q) - 1) > tolerance:
        raise ValueError("Simplex projection failed numerical validation")
    return q


def project_nested_thresholds(midpoints, lower, upper, *, tolerance=1e-11):
    """Bounded pool-adjacent-violators projection of decreasing tail masses.

Each block minimizes its sum of squared distances on the intersection of its
hard intervals. An impossible merged interval certifies infeasibility; no
interval is widened. Source thresholds must be strictly increasing.
"""
    if not (len(midpoints) == len(lower) == len(upper)) or not lower:
        raise ValueError("Nonempty equal-length vectors required")
    if any(not math.isfinite(x) for row in (midpoints, lower, upper) for x in row):
        raise ValueError("Nonfinite probability")
    if any(not 0 <= lo <= hi <= 1 for lo, hi in zip(lower, upper)):
        raise ValueError("Invalid probability interval")
    # A lower bound on a later tail must not exceed an earlier upper bound.
    running_hi = 1.0
    for lo, hi in zip(lower, upper):
        running_hi = min(running_hi, hi)
        if lo > running_hi + tolerance:
            raise ValueError("INFEASIBLE_HARD_QUOTE_POLYTOPE")
    blocks = []
    for i, (target, lo, hi) in enumerate(zip(midpoints, lower, upper)):
        blocks.append({"start": i, "end": i + 1, "sum": target, "n": 1,
                       "lo": lo, "hi": hi, "value": min(hi, max(lo, target))})
        while len(blocks) >= 2 and blocks[-2]["value"] < blocks[-1]["value"] - tolerance:
            right, left = blocks.pop(), blocks.pop()
            lo, hi = max(left["lo"], right["lo"]), min(left["hi"], right["hi"])
            if lo > hi + tolerance:
                raise ValueError("INFEASIBLE_HARD_QUOTE_POLYTOPE")
            total, n = left["sum"] + right["sum"], left["n"] + right["n"]
            blocks.append({"start": left["start"], "end": right["end"], "sum": total,
                           "n": n, "lo": lo, "hi": hi,
                           "value": min(hi, max(lo, total / n))})
    result = [0.0] * len(lower)
    for block in blocks:
        result[block["start"]:block["end"]] = [block["value"]] * block["n"]
    if (any(x < lo - tolerance or x > hi + tolerance for x, lo, hi in zip(result, lower, upper))
            or any(a < b - tolerance for a, b in zip(result, result[1:]))):
        raise ValueError("Nested projection failed numerical validation")
    return result


def quote_constraints(contract, cutoff, *, max_age_seconds=0):
    bid, ask = contract.get("bid"), contract.get("ask")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
           for x in (bid, ask)):
        raise ValueError("MISSING_HISTORICAL_BID_ASK")
    if not 0 <= bid <= ask <= 1:
        raise ValueError("INVALID_HISTORICAL_BID_ASK")
    observed = moment(contract.get("candle_end_time_utc"))
    age = (moment(cutoff) - observed).total_seconds()
    if age < 0:
        raise ValueError("FUTURE_QUOTE")
    if age > max_age_seconds:
        raise ValueError("NO_QUOTE_IN_FROZEN_FRESHNESS_WINDOW")
    if contract.get("created_time") and moment(contract["created_time"]) > observed:
        raise ValueError("QUOTE_BEFORE_CONTRACT_CREATION")
    return float(bid), float(ask), age


def verify_cpi_contract_record(contract, source_root, cutoff):
    """Bind derived metadata, strike, and selected candle to authenticated bytes."""
    for prefix in ("source", "candle_source"):
        filename, expected = contract.get(prefix + "_file"), contract.get(prefix + "_sha256")
        if not filename or not expected:
            raise ValueError("MISSING_RAW_SOURCE_HASH_OR_FILENAME")
        verify_source(source_root, filename, expected)
    metadata = json.loads((Path(source_root) / contract["source_file"]).read_text(encoding="utf-8"))
    markets = metadata.get("markets", [])
    matched = [m for m in markets if m.get("ticker") == contract["ticker"]]
    if len(matched) != 1:
        raise ValueError("SOURCE_MARKET_NOT_UNIQUE")
    raw = matched[0]
    for field in ("rules_primary", "rules_secondary", "title", "result", "expiration_value", "created_time"):
        if raw.get(field) != contract.get(field):
            raise ValueError("NORMALIZED_MARKET_FIELD_DISAGREES_WITH_RAW: " + field)
    phrase = re.search(r"(?:more than|greater than)\s*(-?\d+(?:\.\d+)?)\s*%", raw.get("rules_primary", ""), re.I)
    if phrase is None:
        phrase = re.search(r"(?:more than|greater than)\s*(-?\d+(?:\.\d+)?)\s*%", raw.get("title", ""), re.I)
    if phrase is None or Decimal(phrase.group(1)) != Decimal(str(contract["threshold_value"])):
        raise ValueError("SOURCE_STRICT_THRESHOLD_NOT_REPRODUCED")
    if raw.get("floor_strike") is not None and Decimal(str(raw["floor_strike"])) != Decimal(phrase.group(1)):
        raise ValueError("SOURCE_STRIKE_RULE_DISAGREEMENT")
    candles = json.loads((Path(source_root) / contract["candle_source_file"]).read_text(encoding="utf-8"))
    if candles.get("ticker") != contract["ticker"]:
        raise ValueError("RAW_CANDLE_TICKER_MISMATCH")
    available = []
    for candle in candles.get("candlesticks", []):
        stamp = candle.get("end_period_ts")
        if not isinstance(stamp, int) or stamp > moment(cutoff).timestamp():
            continue
        pair = []
        for side in ("yes_bid", "yes_ask"):
            values = candle.get(side) or {}
            value = values.get("close_dollars", values.get("close"))
            pair.append(float(value) if value is not None else None)
        if all(x is not None and math.isfinite(x) for x in pair) and 0 <= pair[0] <= pair[1] <= 1:
            available.append((stamp, *pair))
    if not available:
        raise ValueError("NO_VALID_BID_ASK_IN_RAW_CANDLES")
    latest = max(available)
    if sum(row[0] == latest[0] for row in available) != 1:
        raise ValueError("DUPLICATE_RAW_CANDLE_TIMESTAMP")
    normalized = (moment(contract["candle_end_time_utc"]).timestamp(), float(contract["bid"]), float(contract["ask"]))
    if latest != normalized:
        raise ValueError("SELECTED_QUOTE_NOT_LATEST_AUTHENTICATED_PRE_CUTOFF_CANDLE")


def attach_returns(event, local):
    """Carry original-position returns and provenance without price substitution."""
    event["asset_returns"] = {}
    event["asset_return_evidence"] = {}
    if local is None:
        event["exclusion_reasons"].append("NO_MATCHED_ORIGINAL_PRICE_EVENT")
        return
    if moment(local["release_timestamp_utc"]) != moment(event["release_time_utc"]):
        event["exclusion_reasons"].append("PRICE_EVENT_RELEASE_TIME_MISMATCH")
        return
    event["asset_return_evidence"] = local["asset_returns"]
    for symbol, row in local["asset_returns"].items():
        value = row.get("simple_return")
        if row.get("status") == "OBSERVED_EXACT_MINUTE_WINDOW_RETURN" and isinstance(value, (int, float)) and math.isfinite(value):
            event["asset_returns"][symbol] = value
    event["available_time_utc"] = (moment(event["release_time_utc"]) + timedelta(minutes=15)).isoformat()
    event["return_convention"] = "EXACT_ELIGIBLE_MINUTE_CLOSE_PRICE_RETURN_EXCLUDING_CASH_DIVIDENDS"


def build_cpi_events(tape, asset_payload, source_root, *, state_model, max_age_seconds=0):
    """Frozen strict or separately declared lagged CPI reference cohort."""
    if max_age_seconds not in (0, 86400):
        raise ValueError("Only frozen strict and A5 exploratory cohorts allowed")
    lookup = {r["event_date"]: r for r in asset_payload["events"] if r["event_family"] == "CPI"}
    result = []
    for source in tape["events"]:
        event = {"event_id": source["event_ticker"], "family": "CPI",
                 "metric": "headline_CPI_SA_MoM_percentage_points", "state_unit": "percentage_points",
                 "release_time_utc": source.get("release_time_utc"),
                 "cutoff_time_utc": source.get("cutoff_time_utc"),
                 "reference_period": source.get("reference_period"),
                 "source_hashes": [], "states": [], "asset_returns": {}, "exclusion_reasons": [],
                 "source_vintage": source.get("source_vintage"),
                 "model_domain_id": "CPI_PRODUCTION_FINITE_TAILS_V1",
                 "cohort": "CPI_EXACT_0820_REFERENCE" if max_age_seconds == 0 else "CPI_LAGGED_REFERENCE_24H",
                 "horizon": "08:29 minute close to 08:44 minute close America/New_York",
                 "horizon_id": "CPI_0829_CLOSE_0844_CLOSE",
                 "quote_contracts": source.get("contracts", [])}
        local = lookup.get(source.get("release_date"))
        try:
            if not source.get("release_time_utc") or not source.get("cutoff_time_utc"):
                raise ValueError(source.get("exclusion_reason") or "NO_VERIFIED_RELEASE_CALENDAR_DATE")
            if (moment(source["release_time_utc"]) - moment(source["cutoff_time_utc"])).total_seconds() != 600:
                raise ValueError("WRONG_QUOTE_LEAD_TIME")
            if moment(source["release_time_utc"]) != new_york_daytime_utc(source["release_date"], 8, 30):
                raise ValueError("NONSTANDARD_RELEASE_CLOCK_OUTSIDE_PROTOCOL")
            attach_returns(event, local)
            if local is None or local.get("actual_first_release") is None:
                raise ValueError("NO_OFFICIAL_FIRST_RELEASE_SETTLEMENT")
            actual = Decimal(str(local["actual_first_release"]))
            event.update(settlement_value=str(actual), settlement_source_url=local.get("actual_source_url"),
                         settlement_source_sha256=local.get("actual_source_sha256"),
                         settlement_admission_tier=local.get("actual_admission_tier"),
                         settlement_reissue_notice=local.get("actual_reissue_notice"))
            if not source.get("metadata_pagination_complete"):
                raise ValueError("INCOMPLETE_CONTRACT_CATALOGUE")
            contracts = sorted(source["contracts"], key=lambda c: Decimal(str(c["threshold_value"])))
            thresholds = [Decimal(str(c["threshold_value"])) for c in contracts]
            if not contracts:
                raise ValueError("NO_SOURCE_THRESHOLDS")
            if len(set(thresholds)) != len(thresholds):
                raise ValueError("DUPLICATE_SOURCE_THRESHOLD")
            lower, upper, mid, ages = [], [], [], []
            for contract, threshold in zip(contracts, thresholds):
                if contract.get("threshold_operator") != ">":
                    raise ValueError("NONSTRICT_OR_UNVERIFIED_CPI_BOUNDARY")
                # Independent source result must match the archived official value.
                expected = "yes" if actual > threshold else "no"
                if contract.get("result") != expected:
                    raise ValueError("OFFICIAL_SETTLEMENT_VENUE_RESULT_DISAGREEMENT")
                expiry = contract.get("expiration_value")
                if expiry not in (None, "") and Decimal(str(expiry).strip().rstrip("%").strip()) != actual:
                    raise ValueError("OFFICIAL_SETTLEMENT_EXPIRATION_VALUE_DISAGREEMENT")
                bid, ask, age = quote_constraints(contract, source["cutoff_time_utc"], max_age_seconds=max_age_seconds)
                verify_cpi_contract_record(contract, source_root, source["cutoff_time_utc"])
                lower.append(bid); upper.append(ask); mid.append((bid + ask) / 2); ages.append(age)
                for prefix in ("source", "candle_source"):
                    filename, digest = contract.get(prefix + "_file"), contract.get(prefix + "_sha256")
                    if not filename or not digest:
                        raise ValueError("MISSING_RAW_SOURCE_HASH_OR_FILENAME")
                    verify_source(source_root, filename, digest)
                    event["source_hashes"].append(digest)
            tails = project_nested_thresholds(mid, lower, upper)
            masses = [1 - tails[0]] + [a - b for a, b in zip(tails, tails[1:])] + [tails[-1]]
            if any(mass < -1e-11 for mass in masses) or abs(math.fsum(masses) - 1) > 1e-11:
                raise ValueError("INVALID_PROJECTED_ATOM_MASSES")
            raw_atoms, bounds = [], []
            for i in range(len(masses)):
                lo = thresholds[i - 1] if i else None
                hi = thresholds[i] if i < len(thresholds) else None
                raw_atoms.append({"atom_id": event["event_id"] + ":" + str(i),
                                  "q_reference": max(0.0, masses[i]),
                                  "value_interval": [float(lo) if lo is not None else None,
                                                     float(hi) if hi is not None else None]})
                bounds.append({"lower": str(lo) if lo is not None else None,
                               "upper": str(hi) if hi is not None else None,
                               "lower_inclusive": False, "upper_inclusive": True})
            production = state_model(raw_atoms, "US_INFLATION")
            event["source_atoms"] = raw_atoms
            event["state_mapping_sha256"] = hashlib.sha256(
                Path(__file__).with_name("event_state_mapping.py").read_bytes()).hexdigest()
            event["quote_timestamps_utc"] = [c["candle_end_time_utc"] for c in contracts]
            settled = sum(actual > threshold for threshold in thresholds)
            for i, (model, mass, boundary) in enumerate(zip(production, masses, bounds)):
                event["states"].append({"id": model["outcome_id"], "value": model["value"],
                                        "lower": model["value_interval"][0], "upper": model["value_interval"][1],
                                        "q": max(0.0, mass), "settlement_bounds": boundary,
                                        "production_state": model, "source_atom": raw_atoms[i]})
            event["settled_state_id"] = event["states"][settled]["id"]
            actual_float = float(actual)
            state = event["states"][settled]
            event["realization_outside_finite_state_domain"] = not state["lower"] <= actual_float <= state["upper"]
            event["reference_q"] = {"method": "HARD_BOUNDED_DECREASING_ISOTONIC_MIDPOINT_PROJECTION",
                                    "thresholds": [str(t) for t in thresholds], "raw_midpoints": mid,
                                    "bid_lower": lower, "ask_upper": upper, "projected_tails": tails,
                                    "squared_projection_distance": math.fsum((x-y)**2 for x,y in zip(tails, mid)),
                                    "quote_age_seconds": ages, "all_quotes_exact_at_cutoff": all(a == 0 for a in ages),
                                    "max_admitted_age_seconds": max_age_seconds,
                                    "standing_quotes_at_cutoff_verified": False,
                                    "max_constraint_violation": max([0.0] + [lo-x for lo,x in zip(lower,tails)] +
                                        [x-hi for x,hi in zip(tails,upper)] + [b-a for a,b in zip(tails,tails[1:])])}
            event["source_hashes"] = sorted(set(event["source_hashes"] + [local["actual_source_sha256"]]))
        except (ValueError, TypeError, KeyError, OSError, ArithmeticError) as exc:
            event["exclusion_reasons"].append(str(exc))
        event["eligible"] = not event["exclusion_reasons"]
        result.append(event)
    return result

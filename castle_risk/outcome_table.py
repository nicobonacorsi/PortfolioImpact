"""One outcome table per scheduled macro event: Q, conditional P&L, priced-in residual.

Model (two factors, declared and testable):

    surprise      s_k = x_k - E_Q[x]                       (settlement state minus Q-reference mean)
    equity factor dSPY_k = gamma * s_k                     (published broad-equity response)
    rate factor   dy_k   = kappa * s_k  [yield bp]          (published long-yield response)
    position i    r_ik = B_i * dSPY_k          for stocks and equity ETFs
                  r_ik = -D_i * dy_k / 10^4   for bond ETFs with source-derived DV01
    portfolio     dV_k = NAV * sum_i w_i * r_ik

Because the model is linear in s_k and sum_k q_k s_k = 0, the Q-weighted P&L is zero:
dV_k is the part of the outcome NOT already reflected in prices under the declared
assumption that the contract Q is the weight embedded in asset prices.

Equity loadings combine the all-day and same-family announcement-day betas by
precision unless a pre-declared heterogeneity test (|z| > 1.96) separates them. The
two estimates share price history, so the loading's standard error is the
dependence-robust bound w*se_d + (1-w)*se_e (perfect positive correlation), never
the narrower independent-sample value. For each parameter/state draw the P&L is
optimized over every Q in the source polytope. The reported range is the 5th
percentile of these minima to the 95th percentile of these maxima. It contains
the corresponding empirical quantile range for every fixed feasible Q, but is
not a simultaneous bound on all parameter draws or a forecast coverage claim.
Parameter uncertainty is propagated by seeded Monte Carlo. Narrow ranges reflect
declared published coefficients, not more data.
No network, clock read or refit happens here.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import math

import numpy as np
from scipy.optimize import linprog
from scipy.stats import t as student_t

from .default_event_economics import (_number, _polytope, _range, _state_model,
                                      published_response_model, validate_event_day_report)
from .fund_composition import (RATE_SCOPE_NAV_TOLERANCE, review_rate_model_scope, review_rate_dollar_binding,
                               uncovered_bond_symbols)
from .portfolio import Portfolio
from .scheduled_event_timing import reviewed_macro_window

SCHEMA = "castle.option1.outcome_table.v1"
HETEROGENEITY_Z = 1.96
DEFAULT_DRAWS = 20000
DEFAULT_SEED = 20261023

KUTTNER_2001 = {
    "title": "Kuttner (2001), Monetary Policy Surprises and Interest Rates: Evidence from the Fed Funds Futures Market",
    "url": "https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr99.pdf",
    "locator": "Table 3, 30-year maturity, response to surprise change",
    "published_point": 19.4, "published_absolute_t": 2.3,
    "published_unit": "30-year Treasury yield basis points per 100 policy-surprise basis points",
}
BEECHEY_WRIGHT_2009 = {
    "title": "Beechey and Wright (2009), The High-Frequency Impact of News on Long-Term Yields and Forward Rates: Is it Real?",
    "url": "https://www.newyorkfed.org/medialibrary/media/research/conference/2009/inflation/Beechey_Wright.pdf",
    "locator": "Table 4 (daily yield changes), Core CPI row, 10-year nominal column: 1.57 (s.e. 0.78) per one-standard-deviation surprise; Table 1: core CPI surprise s.d. 0.09 percentage points",
    "published_point": 1.57, "published_standard_error": .78, "surprise_standard_deviation_pp": .09,
    "published_unit": "10-year nominal yield basis points per one-standard-deviation CORE CPI surprise (2004-2008)",
}
# 1.57 bp per 0.09 pp core surprise = 17.44 bp per pp. Headline surprises carry
# food and energy noise; the declared proxy therefore spans zero to the core response.
CPI_RATE_PROXY_UPPER = 1.57 / .09


def _t_crit(n_train):
    if isinstance(n_train, int) and n_train > 2:
        return float(student_t.ppf(.975, n_train - 2))
    return 1.96


def _standard_error(fit):
    beta = _number(fit["beta"], "Loading beta")
    lo, hi = _range(fit["beta_interval"], "Loading beta interval", point=beta)
    result = (hi - lo) / (2 * _t_crit(fit.get("n_train")))
    if not math.isfinite(result):
        raise ValueError("Loading uncertainty must be finite")
    return result


def combine_loadings(daily_fit, event_fit):
    """Precision-weighted equity loading with an explicit heterogeneity gate.

    Zero reported uncertainty uses the limiting precision weight only when
    the estimates are compatible. Contradictory exact fits are unidentified;
    disagreement with one finite-error fit uses their interval hull without
    choosing the specification by its holdout score. A reported zero SE does
    not establish parameter certainty outside the supplied model.
    """
    if not daily_fit or daily_fit.get("beta") is None or daily_fit.get("beta_interval") is None:
        return None
    beta_d, se_d = float(daily_fit["beta"]), _standard_error(daily_fit)
    record = {"daily_beta": beta_d, "daily_se": se_d, "event_day_beta": None, "event_day_se": None,
              "heterogeneity_z": None}
    estimated = (event_fit and str(event_fit.get("status", "")).startswith("ESTIMATED")
                 and event_fit.get("beta") is not None and event_fit.get("beta_interval") is not None)
    if not estimated:
        record.update(loading=beta_d, se=se_d, basis="DAILY_BETA_NO_ANNOUNCEMENT_DAY_ESTIMATE")
        return record
    beta_e, se_e = float(event_fit["beta"]), _standard_error(event_fit)
    scale = max(se_d, se_e)
    if scale == 0:
        record.update(event_day_beta=beta_e, event_day_se=se_e,
                      event_day_holdout=deepcopy(event_fit.get("transfer_check", {}).get("status")),
                      zero_standard_error_policy="REPORTED_ZERO_VARIANCE_IS_NOT_OUT_OF_SAMPLE_CERTAINTY")
        if beta_d != beta_e:
            record.update(loading=None, se=None, basis="CONFLICTING_ZERO_VARIANCE_ASSOCIATIONS_UNIDENTIFIED")
        else:
            record.update(loading=beta_d, se=0.0, heterogeneity_z=0.0,
                          basis="AGREEING_ZERO_VARIANCE_ASSOCIATIONS", se_if_independent=0.0,
                          se_rule="IDENTICAL_REPORTED_ZERO_VARIANCE_NO_PRECISION_DIVISION")
        return record
    z = (beta_e - beta_d) / math.hypot(se_d, se_e)
    # Algebraically identical normalized precision weights, without divisions
    # by zero or overflow when a reported standard error is tiny.
    scaled_d, scaled_e = se_d / scale, se_e / scale
    share = scaled_e ** 2 / (scaled_d ** 2 + scaled_e ** 2)
    pooled = share * beta_d + (1 - share) * beta_e
    smaller = min(se_d, se_e)
    pooled_se = smaller / math.sqrt(1 + (smaller / scale) ** 2)
    record.update(event_day_beta=beta_e, event_day_se=se_e, heterogeneity_z=z,
                  event_day_holdout=deepcopy(event_fit.get("transfer_check", {}).get("status")))
    # Both betas come from the same price history (announcement days can sit in
    # the all-day sample), so independence is not assumed: under any correlation
    # rho <= 1, sd(w b_d + (1-w) b_e) <= w se_d + (1-w) se_e.
    robust_se = share * se_d + (1 - share) * se_e
    record.update(event_day_beta=beta_e, event_day_se=se_e, heterogeneity_z=z,
                  event_day_holdout=deepcopy(event_fit.get("transfer_check", {}).get("status")),
                  se_if_independent=pooled_se,
                  se_rule="DEPENDENCE_ROBUST_BOUND_NOT_INDEPENDENT_POOLING")
    if min(se_d, se_e) == 0:
        record["zero_standard_error_policy"] = "LIMITING_PRECISION_WEIGHT_ONLY_WHEN_COMPATIBLE_OTHERWISE_INTERVAL_HULL"
        if abs(z) > HETEROGENEITY_Z:
            lo = min(daily_fit["beta_interval"][0], event_fit["beta_interval"][0])
            hi = max(daily_fit["beta_interval"][1], event_fit["beta_interval"][1])
            record.update(loading=(lo + hi) / 2, se=(hi - lo) / (2 * HETEROGENEITY_Z),
                          heterogeneity_interval_hull=[lo, hi],
                          basis="ZERO_VARIANCE_HETEROGENEITY_UNRESOLVED_INTERVAL_HULL",
                          se_rule="DECLARED_NORMAL_95PCT_SCALE_FROM_COMPONENT_INTERVAL_HULL")
            return record
    if abs(z) <= HETEROGENEITY_Z:
        record.update(loading=pooled, se=robust_se, basis="PRECISION_WEIGHTED_DAILY_AND_ANNOUNCEMENT_DAY")
    elif event_fit.get("transfer_check", {}).get("status") == "PASS":
        record.update(loading=beta_e, se=se_e, basis="ANNOUNCEMENT_DAY_HETEROGENEOUS_AND_HOLDOUT_PASS")
    else:
        record.update(loading=pooled, se=math.hypot(robust_se, (beta_e - beta_d) / 2),
                      basis="HETEROGENEOUS_UNRESOLVED_POOLED_POINT_WIDENED")
    return record


def _factor_model(family):
    equity = published_response_model(family if family == "FED_POLICY" else "US_INFLATION",
                                       "HEADLINE_CPI_MONTH_ON_MONTH" if family != "FED_POLICY" else None)
    if family == "FED_POLICY":
        specs = [(-4.68, 4.68 / 3.03), (-2.55, 2.55 / 2.79)]
        rate = {"point": .194, "draw": ("normal", .194, .194 / 2.3), "source": deepcopy(KUTTNER_2001),
                "status": "DECLARED_HISTORICAL_RESPONSE_TRANSFER",
                "unit": "long_yield_bp_per_policy_bp"}
        return {"equity_point": -4.68e-4,
                "equity_draw": ("mixture", [(m / 1e4, s / 1e4) for m, s in specs]),
                "equity_source": equity["source"], "equity_unit": "fractional_SPY_return_per_policy_bp",
                "equity_note": "Equal mixture of the published full-sample and no-outlier specifications.",
                "rate": rate, "state_unit": "policy_basis_points"}
    rate = {"point": CPI_RATE_PROXY_UPPER / 2, "draw": ("uniform", 0.0, CPI_RATE_PROXY_UPPER),
            "source": deepcopy(BEECHEY_WRIGHT_2009), "status": "DECLARED_PROXY_CORE_CPI_RESPONSE_BOUNDS_HEADLINE",
            "unit": "long_yield_bp_per_headline_cpi_pp"}
    return {"equity_point": -.0142, "equity_draw": ("normal", -.0142, .0058),
            "equity_source": equity["source"], "equity_unit": "fractional_SPY_return_per_headline_cpi_pp",
            "equity_note": "Normal approximation from the published robust standard error.",
            "rate": rate, "state_unit": "headline_monthly_cpi_percentage_points"}


def _draw(spec, rng, n):
    kind = spec[0]
    if kind == "normal":
        return rng.normal(spec[1], spec[2], n)
    if kind == "uniform":
        return rng.uniform(spec[1], spec[2], n)
    if kind == "mixture":
        parts = spec[1]
        choice = rng.integers(0, len(parts), n)
        means = np.array([p[0] for p in parts])[choice]
        sds = np.array([p[1] for p in parts])[choice]
        return rng.normal(means, sds)
    raise ValueError("Unknown parameter distribution")


def _rate_position_scope(row, instrument_source=None):
    """Recheck a full rate proxy from ledger facts, never a bare approval flag.

    Old saved DV01 records may be used in fresh computation only with an
    explicitly verified bond classification AND a complete fixed-income NAV
    record. Missing composition fields do not turn an arbitrary ETF into a
    bond. Source-known contrary composition always wins over a taxonomy.
    """
    dv01 = row.get("usd_per_bp_parallel_rise")
    if (isinstance(dv01, bool) or not isinstance(dv01, (float, int))
            or not math.isfinite(dv01) or dv01 > 0):
        return {"whole_fund_rate_proxy_allowed": False,
                "reason": "A non-short bond fund requires a finite nonpositive covered DV01"}
    if "fund_composition_scope_inputs" in row:
        ledger = row["fund_composition_scope_inputs"]
        scope = review_rate_model_scope(ledger)
        if scope.get("whole_fund_rate_proxy_allowed"):
            covered = row.get("covered_fund_nav_weight")
            uncovered = row.get("uncovered_fixed_income_fund_nav_weight")
            if ledger.get("fund_symbol") != row.get("symbol"):
                scope.update(whole_fund_rate_proxy_allowed=False,
                             reason="Rate position symbol differs from its retained source composition")
            elif (isinstance(covered, bool) or isinstance(uncovered, bool)
                    or not isinstance(covered, (float, int)) or not isinstance(uncovered, (float, int))
                    or not math.isfinite(covered) or not math.isfinite(uncovered)
                    or abs(covered - scope["covered_bond_nav_weight"]) > RATE_SCOPE_NAV_TOLERANCE
                    or abs(uncovered - scope["uncovered_fixed_income_nav_weight"]) > RATE_SCOPE_NAV_TOLERANCE):
                scope.update(whole_fund_rate_proxy_allowed=False,
                             reason="Rate position NAV differs from its retained source composition")
        return scope
    source = instrument_source or {}
    known = source.get("holdings_composition") or {}
    # A source-class summary is positive/negative evidence even when a caller
    # drops the newer scope fields. A mixed known ledger cannot regain authority.
    class_weights = known.get("reported_class_nav_weights") or {}
    if any(isinstance(class_weights.get(key), (int, float))
           and abs(class_weights[key]) > RATE_SCOPE_NAV_TOLERANCE
           for key in ("equity", "commodity", "unclassified")):
        return {"whole_fund_rate_proxy_allowed": False,
                "reason": "Known non-bond source composition requires an explicit multi-channel model"}
    covered = row.get("covered_fund_nav_weight")
    uncovered = row.get("uncovered_fixed_income_fund_nav_weight")
    verified_bond = (source.get("instrument_class") == "bond_etf"
                     and source.get("classification_status") == "PROVIDER_TYPED_ETF_ASSET_CLASS"
                     and source.get("instrument_identity_verified") is True)
    if (verified_bond and not isinstance(covered, bool) and not isinstance(uncovered, bool)
            and isinstance(covered, (int, float)) and isinstance(uncovered, (int, float))
            and math.isfinite(covered) and math.isfinite(uncovered)
            and covered > 0 and uncovered >= 0
            and abs(covered + uncovered - 1.) <= RATE_SCOPE_NAV_TOLERANCE):
        return {"status": "LEGACY_COMPLETE_FIXED_INCOME_WITH_VERIFIED_BOND_CLASS",
                "whole_fund_rate_proxy_allowed": True,
                "fixed_income_nav_weight": covered + uncovered,
                "covered_bond_nav_weight": covered,
                "uncovered_fixed_income_nav_weight": uncovered,
                "cash_like_nav_weight": 0., "legacy_scope": True,
                "reason": "Legacy complete fixed-income NAV and independently retained verified bond taxonomy; no ticker-based inference."}
    return {"whole_fund_rate_proxy_allowed": False,
            "reason": "A covered DV01 alone does not establish whole-fund fixed-income scope; retain only the covered bond component"}


def _bond_duration(position, notional, rate_sensitivity, instrument_source=None):
    if position.asset_type != "etf":
        return None
    for row in (rate_sensitivity or {}).get("positions", []):
        if row.get("symbol") != position.symbol:
            continue
        if not review_rate_dollar_binding(row, symbol=position.symbol, portfolio_notional_usd=notional,
                position_weight=position.weight, position_asset_type=position.asset_type)["binding_valid"]:
            return None
        dv01, covered = row.get("usd_per_bp_parallel_rise"), row.get("covered_fund_nav_weight")
        if (isinstance(dv01, bool) or isinstance(covered, bool)
                or not isinstance(dv01, (int, float)) or not isinstance(covered, (int, float))
                or not math.isfinite(dv01) or not math.isfinite(covered)
                or dv01 > 0 or covered <= 0 or position.weight <= 0):
            return None
        scope = _rate_position_scope(row, instrument_source)
        if not scope.get("whole_fund_rate_proxy_allowed"):
            return None
        covered_duration = -float(dv01) / (notional * position.weight) * 1e4
        fixed_income_nav = scope["fixed_income_nav_weight"]
        full = covered_duration / float(covered) * fixed_income_nav
        return {"duration": full, "duration_range": sorted([covered_duration, 2 * full - covered_duration]),
                "covered_fund_nav_weight": float(covered), "covered_duration": covered_duration,
                "fixed_income_fund_nav_weight": fixed_income_nav,
                "cash_like_fund_nav_weight": scope.get("cash_like_nav_weight"),
                "dollar_sensitivity_binding": deepcopy(row["dollar_sensitivity_binding"]),
                "whole_fund_rate_model_scope": scope,
                "basis": "SOURCE_HOLDINGS_DV01_EXTENDED_ONLY_TO_RECORDED_FIXED_INCOME_NAV",
                "assumption": "Uncovered FIXED-INCOME holdings have the covered bonds' duration on average; "
                              "the draw spans covered-only duration to an equally wide upper side. "
                              "Recorded cash-like NAV is assumed unchanged over the announcement window; "
                              "this includes source-identified money-market funds and is not a claim of zero "
                              "MMF duration or risk. Cash accrual and MMF repricing are not modeled; "
                              "equity or other non-rate NAV is never assigned a bond duration."}


def _partial_bond_component(position, rate_sensitivity, instrument_source=None, *, notional=None):
    """Retain a supported partial rate sensitivity without pricing the remainder."""
    for row in (rate_sensitivity or {}).get("positions", []):
        if row.get("symbol") != position.symbol:
            continue
        if not review_rate_dollar_binding(row, symbol=position.symbol, portfolio_notional_usd=notional,
                position_weight=position.weight, position_asset_type=position.asset_type)["binding_valid"]:
            return None
        dv01, covered = row.get("usd_per_bp_parallel_rise"), row.get("covered_fund_nav_weight")
        if (isinstance(dv01, bool) or isinstance(covered, bool)
                or not isinstance(dv01, (int, float)) or not isinstance(covered, (int, float))
                or not math.isfinite(dv01) or not math.isfinite(covered) or covered <= 0):
            return None
        component_scope = _rate_position_scope(row, instrument_source)
        if "fund_composition_scope_inputs" in row:
            ledger = row["fund_composition_scope_inputs"]
            if not isinstance(ledger, dict) or ledger.get("fund_symbol") != position.symbol:
                return None
            coverage_ledger = ledger.get("fixed_income_duration_coverage")
            if not isinstance(coverage_ledger, dict):
                return None
            source_covered = coverage_ledger.get("covered_nav_weight")
            if (isinstance(source_covered, bool) or not isinstance(source_covered, (int, float))
                    or not math.isfinite(source_covered)
                    or abs(source_covered - covered) > RATE_SCOPE_NAV_TOLERANCE):
                return None
            # A positive signed DV01 is possible for long/short bond baskets,
            # never for the supported long-only nominal bullet-bond channel.
            # The whole-fund gate rejects such signed baskets; a separately
            # retained partial contribution still needs actual short evidence.
            component_scope = review_rate_model_scope(ledger)
        if dv01 > 0 and component_scope.get("noncash_short_gross_nav_weight", 0.) <= RATE_SCOPE_NAV_TOLERANCE:
            return None
        return {"covered_usd_per_bp_parallel_rise": float(dv01),
                "dollar_sensitivity_binding": deepcopy(row["dollar_sensitivity_binding"]),
                "covered_fund_nav_weight": float(covered),
                "remaining_fund_nav_weight_not_priced": 1. - float(covered),
                "remaining_nav_weight_is_signed_accounting_not_risk_coverage": True,
                "whole_fund_return_unavailable": True,
                "whole_fund_rate_model_scope": _rate_position_scope(row, instrument_source),
                "scope": "COVERED_BOND_COMPONENT_ONLY_REMAINDER_NOT_ASSUMED_ZERO"}
    return None


def holdings_quality_gaps(inputs, compositions=None, reviews=None):
    """Per-ETF snapshot quality failures, with their measured size when known.

    A fund the class-aware holdings review accepted (``reviews`` by symbol) is not
    a gap: its declared tolerances and measured values are shown separately.
    """
    gaps = {}
    for symbol, holdings in sorted(((inputs or {}).get("holdings") or {}).items()):
        if not isinstance(holdings, dict):
            continue
        issues = list((holdings.get("metadata") or {}).get("quality_issues", []))
        if holdings.get("reliable") is not False and not issues:
            continue
        if ((reviews or {}).get(symbol) or {}).get("holdings_accepted_for_assessment") is True:
            continue
        composition = (compositions or {}).get(symbol) or {}
        gaps[symbol] = {
            "status": "HOLDINGS_SNAPSHOT_QUALITY_GAP", "reliable": holdings.get("reliable"), "issues": issues,
            "measured": {key: composition.get(key) for key in (
                "reported_weight_sum", "gross_short_weight", "unreported_or_residual_nav_weight",
                "repeated_ticker_groups_all_distinguished_by_identifier") if key in composition},
            "treatment": "Disclosed, not repaired. Durations use only rows whose DV01 is known and widen for the rest.",
        }
    return gaps


def instrument_model_gate(portfolio, *, instrument_exposures=None,
                          identity_blocked_symbols=None, rate_sensitivity=None, two_factor_fits=None):
    """Do not price a contradictory identity or an unidentified fund as equity.

    This guard consumes already-reviewed identity/composition evidence only;
    it neither guesses an asset class from a ticker nor acquires missing data.
    Explicitly unverified historical stock declarations remain distinguishable
    from provider-resolved identities. A direct return regression (payroll)
    needs the identity gate, but not a separate equity/rates routing decision.
    """
    from .issuer_fund_class import reviewed_original_fund_equity_class

    active = {p.symbol for p in portfolio.positions if p.weight}
    identity = set(identity_blocked_symbols or ()) & active
    asset_class = set()
    by_symbol = {r.get("symbol"): r for r in (instrument_exposures or {}).get("positions", [])}
    rate_rows = {r.get("symbol"): r for r in (rate_sensitivity or {}).get("positions", [])
                 if r.get("symbol")}
    binding_checks = {p.symbol: review_rate_dollar_binding(rate_rows[p.symbol], symbol=p.symbol,
                      portfolio_notional_usd=portfolio.notional, position_weight=p.weight,
                      position_asset_type=p.asset_type)
                      for p in portfolio.positions if p.symbol in rate_rows}
    covered_rates = {symbol for symbol, row in rate_rows.items()
                     if binding_checks.get(symbol, {}).get("binding_valid")
                     and _rate_position_scope(row, by_symbol.get(symbol)).get("whole_fund_rate_proxy_allowed")}
    partial_rates = set(rate_rows) - covered_rates
    known_bonds = uncovered_bond_symbols(rate_sensitivity) & active
    reviewed_equity_funds = {}
    for position in portfolio.positions:
        if not position.weight or position.asset_type == "cash":
            continue
        source = by_symbol.get(position.symbol) or {}
        classification = source.get("classification_status")
        provider_type = source.get("provider_instrument_type")
        if classification in {"UNSUPPORTED_PROVIDER_INSTRUMENT", "LEVERAGED_OR_INVERSE_FUND_UNSUPPORTED"}:
            identity.add(position.symbol)
        elif (classification == "DECLARATION_OR_INSTRUMENT_TYPE_UNSUPPORTED"
              and not (provider_type == "ETV" and position.asset_type == "etf")):
            identity.add(position.symbol)
        if position.symbol in covered_rates or position.symbol in identity:
            continue
        if position.symbol in partial_rates:
            asset_class.add(position.symbol)
            continue
        if position.symbol in known_bonds:
            asset_class.add(position.symbol)
            continue
        # Exchange-traded vehicles are not necessarily equity funds. In
        # particular a physically backed gold trust must not inherit the SPY
        # equity-factor model merely because a historical beta was fitted.
        # Separate issuer-grounded commodity stress is handled independently.
        if position.asset_type == "etf" and provider_type == "ETV":
            asset_class.add(position.symbol)
            continue
        # A current ETF identity does not establish its underlying asset class.
        # A complete, source-bound composition remains a separate, inspectable
        # basis for the declared original-fund proxy; absent even that evidence,
        # the equity branch must not be the default fallback.
        if (position.asset_type == "etf" and provider_type == "ETF"
                and source.get("instrument_class") == "unresolved_etf"):
            composition = source.get("holdings_composition") or {}
            if not (composition.get("eligible_for_research") is True
                    and composition.get("predominant_observed_class") in {"equity", "commodity"}):
                # A separately reviewed *primary issuer* document can classify
                # the original XLE equity fund even when constituent NAV does
                # not reconcile. Never infer class from the ticker itself.
                provenance = source.get("metadata_provenance") or {}
                issuer_review = (reviewed_original_fund_equity_class(position.symbol,
                    evaluated_at=(instrument_exposures or {}).get("evaluated_at_utc"))
                    if source.get("instrument_identity_verified") is True
                       and provenance.get("status") == "LOCAL_RESPONSE_BYTES_AND_FIELDS_VERIFIED"
                       and provenance.get("local_response_verified") is True
                       and composition.get("predominant_observed_class") not in {"fixed_income", "commodity"}
                    else None)
                if issuer_review:
                    reviewed_equity_funds[position.symbol] = issuer_review
                else:
                    asset_class.add(position.symbol)
    # A fund whose class or duration the holdings cannot establish is priced by
    # its own measured response to the model's two factors when that response
    # passed its holdout (two_factor_response). Identity blocks never qualify.
    measured = {symbol for symbol in asset_class
                if ((two_factor_fits or {}).get(symbol) or {}).get("admitted") is True}
    asset_class -= measured
    blocked = sorted(identity | asset_class)
    return {
        "two_factor_measured_symbols": sorted(measured),
        "status": "BLOCKED_INSTRUMENT_IDENTITY" if blocked else "IDENTITY_SCOPE_CHECKED",
        "blocked_symbols": blocked,
        "identity_blocked_symbols": sorted(identity),
        "asset_class_blocked_symbols": sorted(asset_class),
        "partial_bond_component_symbols": sorted(partial_rates & active),
        "dollar_binding_blocked_symbols": sorted(symbol for symbol in active & set(binding_checks)
                                                  if not binding_checks[symbol]["binding_valid"]),
        "dollar_sensitivity_binding_checks": binding_checks,
        "issuer_documented_original_equity_funds": reviewed_equity_funds,
        "reason": ("Instrument identity or underlying fund-class evidence is unresolved for "
                   + ", ".join(blocked) + ". Missing fund holdings or duration are not replaced by an equity proxy; "
                   "covered bonds inside a mixed or unresolved fund do not price its non-bond remainder."
                   if blocked else "No contradictory supported-instrument identity or unresolved source-typed fund routing was admitted."
                   + (" An independent primary-issuer equity mandate supports original-fund classification, "
                      "not constituent look-through or a validated event response."
                      if reviewed_equity_funds else "")),
        "no_class_inferred_from_ticker_or_name": True,
        "source_class_evidence_does_not_validate_return_prediction": True,
    }


def composite_model_reliability(positions, daily_fits=None, event_fits=None):
    """Keep component checks separate from validation of the combined model.

    A source-backed duration is not a tested future yield response. Likewise a
    beta's historical PASS does not validate its published-response composition,
    or a pool/selection of coefficients. This summary is deliberately not a
    new PASS gate and never overrides the retained daily/event diagnostics.
    """
    diagnostics, failed, untested = [], [], []
    for row in positions:
        symbol = row["symbol"]
        channel = row.get("channel")
        daily = deepcopy(((daily_fits or {}).get(symbol) or {}).get("transfer_check") or {})
        event = deepcopy(((event_fits or {}).get(symbol) or {}).get("transfer_check") or {})
        applicable = bool(row.get("weight")) and channel != "NONE"
        if applicable:
            untested.append(symbol)
        if channel == "EQUITY" and any(c.get("status") == "FAIL" for c in (daily, event)):
            failed.append(symbol)
        diagnostics.append({
            "symbol": symbol, "weight": row.get("weight"), "channel": channel,
            "basis": row.get("basis"),
            "combined_response_check": "NOT_TESTED" if applicable else "NOT_APPLICABLE",
            "daily_association_check": daily,
            "announcement_day_association_check": event,
            "association_checks_used_by_channel": channel == "EQUITY",
            "duration_coverage": deepcopy(row.get("duration")),
        })
    return {
        "status": "NOT_INDEPENDENTLY_VALIDATED" if untested else "NOT_APPLICABLE",
        "scenario_transfer_status": "NOT_TESTED" if untested else "NOT_APPLICABLE",
        "historical_diagnostic_failed_symbols": failed,
        "scenario_transfer_untested_symbols": untested,
        "positions": diagnostics,
        "forward_prediction_certified": False,
        "scope": "The asset-class-aware combined scenario model has no independent end-to-end holdout. Component association diagnostics remain separate; neither a PASS nor source-derived duration certifies future portfolio P&L.",
    }


def _expectation_ranges(lp_kwargs, values):
    """Sharp per-row expectation extrema on one validated probability polytope.

    Bounded simplexes use the continuous-knapsack solution (fill the smallest
    or largest coefficients first). For repeated coefficient orderings, an
    optional acceleration proves simultaneous attainment of all nested-tail
    extrema: E_Q[x] = x[0] + sum_j (x[j]-x[j-1]) Q(index >= j).
    If the extremal tails do not form feasible joint witnesses, each objective
    is solved directly. No unsupported constraint is dropped, and LP failure
    is fatal rather than silently collapsing uncertainty to the reference Q.

    The optimization is exact up to the LP solver's numerical tolerance.
    """
    values = np.asarray(values, dtype=float)
    bounds = np.asarray(lp_kwargs["bounds"], dtype=float)
    n = len(bounds)
    if values.ndim != 2 or values.shape[1] != n or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Finite state vectors with one column per probability atom are required")
    a_eq = np.asarray(lp_kwargs.get("A_eq", []), dtype=float)
    b_eq = np.asarray(lp_kwargs.get("b_eq", []), dtype=float)
    a_ub = np.asarray(lp_kwargs.get("A_ub") or [], dtype=float)
    b_ub = np.asarray(lp_kwargs.get("b_ub") or [], dtype=float)
    # Exact structural comparisons: small nonconstant constraints are still
    # constraints, not a reason to take the bounded-simplex shortcut.
    simplex_only = (len(a_eq) > 0 and not len(a_ub)
                    and any(np.all(row == 1.0) and rhs == 1.0 for row, rhs in zip(a_eq, b_eq))
                    and all(np.all(row == row[0]) and rhs == row[0] for row, rhs in zip(a_eq, b_eq)))
    lower, upper = np.empty(len(values)), np.empty(len(values))
    methods, lp_count = set(), 0

    def solve(coefficients):
        nonlocal lp_count
        result = linprog(coefficients, **lp_kwargs)
        lp_count += 1
        if not result.success or not np.isfinite(result.fun):
            raise ValueError("Per-draw probability optimization failed: " + str(result.message))
        return float(result.fun)

    if simplex_only:
        floors, capacities = bounds[:, 0], bounds[:, 1] - bounds[:, 0]
        remainder = 1.0 - float(floors.sum())
        if remainder < -1e-10 or remainder > float(capacities.sum()) + 1e-10:
            raise ValueError("Probability bounds do not admit a simplex distribution")
        remainder = max(0.0, min(remainder, float(capacities.sum())))
        base = values @ floors
        for reverse, target in ((False, lower), (True, upper)):
            order = np.argsort(-values if reverse else values, axis=1, kind="stable")
            capacity = capacities[order]
            prior = np.cumsum(capacity, axis=1) - capacity
            fill = np.minimum(capacity, np.maximum(0.0, remainder - prior))
            target[:] = base + np.sum(np.take_along_axis(values, order, axis=1) * fill, axis=1)
        methods.add("BOUNDED_SIMPLEX_GREEDY_EXACT")
    else:
        orders, membership = np.unique(np.argsort(values, axis=1, kind="stable"), axis=0, return_inverse=True)

        def feasible(point):
            tolerance = 1e-9
            return (np.all(point >= bounds[:, 0] - tolerance)
                    and np.all(point <= bounds[:, 1] + tolerance)
                    and (not len(a_eq) or np.all(np.abs(a_eq @ point - b_eq) <= tolerance))
                    and (not len(a_ub) or np.all(a_ub @ point <= b_ub + tolerance)))

        for group, order in enumerate(orders):
            indices = np.flatnonzero(membership == group)
            certified = False
            if len(indices) > n:  # certification only saves LPs for repeated orderings
                tail_lo, tail_hi = [], []
                for j in range(1, n):
                    indicator = np.zeros(n)
                    indicator[order[j:]] = 1.0
                    tail_lo.append(solve(indicator))
                    tail_hi.append(-solve(-indicator))
                witnesses = []
                for tails in (tail_lo, tail_hi):
                    point = np.empty(n)
                    point[order] = -np.diff(np.array([1.0, *tails, 0.0]))
                    witnesses.append(point)
                if all(feasible(point) for point in witnesses):
                    ordered = values[indices][:, order]
                    increments = np.diff(ordered, axis=1)
                    lower[indices] = ordered[:, 0] + increments @ np.asarray(tail_lo)
                    upper[indices] = ordered[:, 0] + increments @ np.asarray(tail_hi)
                    methods.add("CERTIFIED_NESTED_TAIL_EXTREMA")
                    certified = True
            if not certified:
                # Repeated identical vectors are solved only once. Different
                # objectives retain their own extrema even with the same order.
                unique, inverse = np.unique(values[indices], axis=0, return_inverse=True)
                minima = np.array([solve(row) for row in unique])
                maxima = np.array([-solve(-row) for row in unique])
                lower[indices], upper[indices] = minima[inverse], maxima[inverse]
                methods.add("DIRECT_PER_OBJECTIVE_LINEAR_PROGRAMS")
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)) or np.any(lower > upper + 1e-8):
        raise ValueError("Invalid per-draw probability expectation envelope")
    return lower, upper, {"methods": sorted(methods), "linear_programs": lp_count,
                          "scope": "SHARP_EXPECTATION_EXTREMA_FOR_EACH_STATE_DRAW_UP_TO_SOLVER_TOLERANCE"}


def _pnl_q_envelope(notional, unit_draw, x_draw, mean_lower, mean_upper):
    """Joint portfolio extrema: one common Q and state draw for every position."""
    first = notional * unit_draw[:, None] * (x_draw - mean_lower[:, None])
    second = notional * unit_draw[:, None] * (x_draw - mean_upper[:, None])
    return np.minimum(first, second), np.maximum(first, second)


def build_outcome_table(portfolio, distribution, daily_beta_report, *, release_day_report=None,
                        rate_sensitivity=None, holdings_quality=None, n_draws=DEFAULT_DRAWS, seed=DEFAULT_SEED,
                        empirical_response=None, instrument_exposures=None,
                        identity_blocked_symbols=None, two_factor_fits=None):
    """Complete per-outcome P&L and priced-in residual for one source outcome distribution."""
    if not isinstance(portfolio, Portfolio):
        raise ValueError("A validated original-position portfolio is required")
    identity_gate = instrument_model_gate(portfolio, instrument_exposures=instrument_exposures,
        identity_blocked_symbols=identity_blocked_symbols, rate_sensitivity=rate_sensitivity,
        two_factor_fits=two_factor_fits)
    two_factor_symbols = set(identity_gate["two_factor_measured_symbols"])
    if empirical_response is not None:
        # Explicit opt-in: the direct empirical response has a different horizon
        # and prediction distribution. Its adapter never falls back silently.
        from .empirical_chain_adapter import build_empirical_outcome_table
        _polytope(distribution, distribution.get("atoms") or [])
        if identity_gate["identity_blocked_symbols"]:
            return {"status": "BLOCKED_INSTRUMENT_IDENTITY", "outcomes": [],
                    "instrument_identity_gate": identity_gate, "legacy_fallback_allowed": False,
                    "reason": identity_gate["reason"], "summary": {"q_standard_deviation_usd": None}}
        return build_empirical_outcome_table(portfolio, distribution, empirical_response)
    if isinstance(n_draws, bool) or not isinstance(n_draws, int) or n_draws <= 0:
        raise ValueError("n_draws must be a positive integer")
    family = distribution.get("family")
    if family not in {"FED_POLICY", "US_INFLATION"}:
        raise ValueError("Outcome tables are defined for FOMC and CPI source distributions")
    atoms = distribution.get("atoms") or []
    if not atoms or not distribution.get("constraints"):
        raise ValueError("A coherent source outcome distribution is required")
    beta_time = datetime.fromisoformat(daily_beta_report["analysis_as_of"].replace("Z", "+00:00"))
    if beta_time.utcoffset() is None:
        raise ValueError("Beta analysis time must be timezone-aware")
    source_time = distribution.get("evaluated_at", distribution.get("evaluated_at_utc"))
    if source_time is not None:
        source_time = datetime.fromisoformat(source_time.replace("Z", "+00:00"))
        if source_time.utcoffset() is None or beta_time > source_time:
            raise ValueError("Sensitivity analysis must not follow the source distribution evaluation")
    cutoff = source_time or beta_time
    schedule_window = reviewed_macro_window(
        family, distribution["event_date"], cutoff,
        source_close_at=distribution.get("source_market_close_at_utc"))
    if not schedule_window["pending"]:
        raise ValueError("The event must be in the future at the evaluation time")
    announcement_day_admission = None
    if release_day_report is not None:
        announcement_day_admission = validate_event_day_report(release_day_report, family, not_after=cutoff)
    kwargs, q, identifiers = _polytope(distribution, atoms)
    states = _state_model(atoms, family)
    model = _factor_model(family)
    rng = np.random.default_rng(seed)
    notional = portfolio.notional
    gamma = _draw(model["equity_draw"], rng, n_draws)
    kappa = _draw(model["rate"]["draw"], rng, n_draws)
    positions, missing = [], []
    exposure_point = {"equity": 0.0, "rate": 0.0}
    exposure_draw = {"equity": np.zeros(n_draws), "rate": np.zeros(n_draws)}
    daily = daily_beta_report.get("instruments", {})
    instrument_sources = {row.get("symbol"): row for row in (instrument_exposures or {}).get("positions", [])}
    blocked_bonds = uncovered_bond_symbols(rate_sensitivity)
    identity_blocked = set(identity_gate["blocked_symbols"])
    event = (release_day_report or {}).get("instruments", {})
    for position in portfolio.positions:
        row = {"symbol": position.symbol, "weight": position.weight, "asset_type": position.asset_type}
        if position.symbol in (holdings_quality or {}):
            row["data_quality"] = deepcopy(holdings_quality[position.symbol])
        if position.weight and position.symbol in identity_blocked:
            basis = ("DOLLAR_DV01_PORTFOLIO_BINDING_MISMATCH_OR_MISSING"
                     if position.symbol in identity_gate["dollar_binding_blocked_symbols"]
                     and position.symbol not in identity_gate["identity_blocked_symbols"]
                     else "KNOWN_BOND_WITHOUT_DURATION_NO_EQUITY_PROXY"
                     if position.symbol in blocked_bonds and position.symbol not in identity_gate["identity_blocked_symbols"]
                     else "INSTRUMENT_IDENTITY_OR_FUND_CLASS_UNRESOLVED")
            row.update(channel="UNKNOWN", basis=basis,
                       loading=None, instrument_identity_gate=deepcopy(identity_gate))
            if position.symbol not in identity_gate["identity_blocked_symbols"]:
                partial = _partial_bond_component(position, rate_sensitivity,
                                                   instrument_sources.get(position.symbol), notional=notional)
                if partial:
                    row["covered_bond_component"] = partial
            missing.append(position.symbol)
            positions.append(row)
            continue
        if position.asset_type == "cash" or position.weight == 0:
            row.update(channel="NONE", basis="UNCHANGED_USD_CASH_OR_ZERO_WEIGHT", loading=0.0)
            positions.append(row)
            continue
        if position.symbol in two_factor_symbols:
            fit = two_factor_fits[position.symbol]
            draws = rng.multivariate_normal([fit["spy_loading"], fit["rate_loading_per_bp"]],
                                            fit["covariance"], n_draws)
            row.update(channel="TWO_FACTOR", basis="MEASURED_SPY_AND_10Y_YIELD_RESPONSE",
                       loading=fit["spy_loading"], rate_loading_per_bp=fit["rate_loading_per_bp"],
                       loading_unit="fractional_return_per_SPY_return_and_per_10Y_yield_bp",
                       two_factor_fit={k: deepcopy(fit.get(k)) for k in (
                           "status", "spy_loading", "spy_loading_se", "rate_loading_per_bp",
                           "rate_loading_se_per_bp", "n_train", "n_holdout", "training_span", "holdout_span",
                           "holdout_oos_r2", "holdout_event_days", "holdout_event_rmse",
                           "holdout_event_zero_rmse", "admission_rule", "scope")})
            exposure_point["equity"] += position.weight * fit["spy_loading"]
            exposure_point["rate"] += position.weight * fit["rate_loading_per_bp"] * 1e4
            exposure_draw["equity"] += position.weight * draws[:, 0]
            exposure_draw["rate"] += position.weight * draws[:, 1] * 1e4
            positions.append(row)
            continue
        bond = _bond_duration(position, notional, rate_sensitivity,
                              instrument_sources.get(position.symbol))
        if bond is not None:
            draws = rng.uniform(*bond["duration_range"], n_draws) if bond["duration_range"][1] > bond["duration_range"][0] else np.full(n_draws, bond["duration"])
            row.update(channel="RATES", loading=-bond["duration"], loading_unit="fractional_return_per_10000_yield_bp",
                       duration=bond, basis=bond["basis"])
            exposure_point["rate"] += position.weight * -bond["duration"]
            exposure_draw["rate"] += position.weight * -draws
            positions.append(row)
            continue
        if position.symbol in blocked_bonds:
            row.update(channel="UNKNOWN", basis="KNOWN_BOND_WITHOUT_DURATION_NO_EQUITY_PROXY", loading=None)
            missing.append(position.symbol)
            positions.append(row)
            continue
        combined = combine_loadings(daily.get(position.symbol), event.get(position.symbol))
        if combined is None or combined["loading"] is None:
            row.update(channel="UNKNOWN", **(combined or {"basis": "MISSING_INSTRUMENT_SENSITIVITY", "loading": None}))
            missing.append(position.symbol)
            positions.append(row)
            continue
        draws = rng.normal(combined["loading"], combined["se"], n_draws)
        row.update(channel="EQUITY", loading_unit="fractional_return_per_SPY_return", **combined)
        row["daily_holdout_status"] = (daily.get(position.symbol) or {}).get("transfer_check", {}).get("status", "NOT_TESTED")
        # Retain the original component parameter sets. The deterministic R32
        # sensitivity analysis must not reconstruct a smaller normal interval
        # from a small-sample Student-t fit or choose a component by its score.
        for prefix, fitted in (("daily", daily.get(position.symbol) or {}),
                               ("event_day", event.get(position.symbol) or {})):
            if fitted.get("beta_interval") is not None:
                row[prefix + "_beta_interval"] = deepcopy(fitted["beta_interval"])
            if fitted.get("n_train") is not None:
                row[prefix + "_n_train"] = fitted["n_train"]
        exposure_point["equity"] += position.weight * combined["loading"]
        exposure_draw["equity"] += position.weight * draws
        positions.append(row)
    x_point = np.array([s["value"] for s in states], dtype=float)
    lows = np.array([s["value_interval"][0] for s in states], dtype=float)
    highs = np.array([s["value_interval"][1] for s in states], dtype=float)
    x_draw = rng.uniform(lows, highs, (n_draws, len(states)))
    q_arr = np.array(q)
    mean_point = float(x_point @ q_arr)
    s_point = x_point - mean_point
    s_draw = x_draw - (x_draw @ q_arr)[:, None]
    # Magnitudes and Q vary together: optimize E_Q[x_draw] for EACH draw.
    # A constant shift computed at x_point is invalid for uncertain tail states.
    mean_lower, mean_upper, q_optimization = _expectation_ranges(kwargs, np.vstack([x_point, x_draw]))
    mean_range = [float(mean_lower[0]), float(mean_upper[0])]
    # Per unit of surprise: equity factor gamma, rate factor kappa (yield bp), bond return -D * dy / 1e4.
    unit_point = exposure_point["equity"] * model["equity_point"] + exposure_point["rate"] * model["rate"]["point"] / 1e4
    unit_draw = exposure_draw["equity"] * gamma + exposure_draw["rate"] * kappa / 1e4
    pnl_point = notional * unit_point * s_point
    pnl_draw = notional * unit_draw[:, None] * s_draw
    pnl_low_q, pnl_high_q = _pnl_q_envelope(notional, unit_draw, x_draw, mean_lower[1:], mean_upper[1:])
    point_low_q, point_high_q = _pnl_q_envelope(notional, np.array([unit_point]), x_point[None, :],
                                               mean_lower[:1], mean_upper[:1])
    q_std_draw = np.sqrt((pnl_draw ** 2) @ q_arr)
    rows = []
    for k, atom in enumerate(atoms):
        point = float(pnl_point[k])
        lo_ref, hi_ref = (float(v) for v in np.percentile(pnl_draw[:, k], [5, 95]))
        lo, hi = float(np.percentile(pnl_low_q[:, k], 5)), float(np.percentile(pnl_high_q[:, k], 95))
        sign = None if abs(point) < 1e-9 else float(np.mean(np.sign(pnl_draw[:, k]) == np.sign(point)))
        contributions = []
        partial_bond_contributions = []
        for row in positions:
            if row["channel"] == "EQUITY":
                value = notional * row["weight"] * row["loading"] * model["equity_point"] * s_point[k]
            elif row["channel"] == "RATES":
                value = notional * row["weight"] * row["loading"] * model["rate"]["point"] * s_point[k] / 1e4
            elif row["channel"] == "TWO_FACTOR":
                value = notional * row["weight"] * (row["loading"] * model["equity_point"]
                                                    + row["rate_loading_per_bp"] * model["rate"]["point"]) * s_point[k]
            elif row["channel"] == "NONE":
                value = 0.0
            else:
                value = None
            contributions.append({"symbol": row["symbol"], "channel": row["channel"], "pnl_usd": value})
            partial = row.get("covered_bond_component")
            if partial:
                partial_value = partial["covered_usd_per_bp_parallel_rise"] * model["rate"]["point"] * s_point[k]
                partial_bond_contributions.append({
                    "symbol": row["symbol"], "pnl_usd": float(partial_value),
                    "covered_fund_nav_weight": partial["covered_fund_nav_weight"],
                    "remaining_fund_nav_weight_not_priced": partial["remaining_fund_nav_weight_not_priced"],
                    "remaining_nav_weight_is_signed_accounting_not_risk_coverage": True,
                    "scope": partial["scope"],
                    "event_yield_move_is_declared_model_assumption": True})
        rows.append({
            "outcome_id": identifiers[k], "label": atom.get("label", identifiers[k]),
            "q_interval": deepcopy(atom.get("q_interval")), "q_reference": q[k],
            "state": states[k]["value"], "state_interval": states[k]["value_interval"],
            "state_basis": states[k]["basis"], "surprise": float(s_point[k]),
            "pnl_if_outcome_usd": point, "pnl_model_range_5_95_usd": [lo, hi],
            "pnl_model_range_5_95_at_reference_q_usd": [lo_ref, hi_ref],
            "pnl_at_point_parameters_range_over_q_usd": [float(point_low_q[0, k]), float(point_high_q[0, k])],
            "sign_determined_for_every_feasible_q": bool(point_low_q[0, k] > 0 or point_high_q[0, k] < 0),
            "sign_determined_scope": "POINT_PARAMETERS_AND_POINT_STATES_OVER_ALL_FEASIBLE_Q",
            "model_5_95_range_excludes_zero": bool(lo > 0 or hi < 0),
            "sign_probability_under_model": sign,
            "q_weighted_contribution_usd": q[k] * point,
            "position_contributions": contributions,
            "partial_covered_bond_contributions": partial_bond_contributions,
            "known_positions_plus_covered_bond_components_pnl_usd": (
                point + math.fsum(partial["pnl_usd"] for partial in partial_bond_contributions)
                if partial_bond_contributions else None),
            "source_contract_ids": deepcopy(atom.get("source_contract_ids", [])),
        })
    weighted = float(pnl_point @ q_arr)
    if abs(weighted) > 1e-6 * max(1.0, notional):
        raise ValueError("Q-centering identity failed")
    material = [r for r in rows if r["q_reference"] >= .01]
    worst = min(material or rows, key=lambda r: r["pnl_if_outcome_usd"])
    best = max(material or rows, key=lambda r: r["pnl_if_outcome_usd"])
    likely = max(rows, key=lambda r: r["q_reference"])
    missing_weight = math.fsum(p.weight for p in portfolio.positions if p.symbol in missing)
    q_std = float(math.sqrt(max(0.0, float((pnl_point ** 2) @ q_arr))))
    quality = [{"symbol": p["symbol"], **p["data_quality"]} for p in positions if p.get("data_quality")]
    if missing:
        for row in rows:
            row["known_positions_only_pnl_usd"] = row["pnl_if_outcome_usd"]
            row["pnl_if_outcome_usd"] = None
            row["q_weighted_contribution_usd"] = None
            row["pnl_model_range_5_95_usd"] = None
            row["pnl_model_range_5_95_at_reference_q_usd"] = None
            row["pnl_at_point_parameters_range_over_q_usd"] = None
            row["sign_determined_for_every_feasible_q"] = None
            row["model_5_95_range_excludes_zero"] = None
            row["sign_probability_under_model"] = None
    reliability = composite_model_reliability(positions, daily, event)
    return {
        "schema": SCHEMA,
        "status": "COMPLETE_DECLARED_MODEL" if not missing else "PARTIAL_KNOWN_POSITIONS_ONLY",
        "instrument_identity_gate": identity_gate,
        "legacy_fallback_allowed": not bool(identity_gate["blocked_symbols"]),
        "data_quality_gaps": quality,
        "family": family, "event_date": distribution.get("event_date"),
        "schedule_window": schedule_window,
        "announcement_day_admission": announcement_day_admission,
        "event_ticker": distribution.get("event_ticker"),
        "notional_usd": notional, "missing_symbols": missing, "missing_weight": missing_weight,
        "reference_expectation": mean_point, "reference_expectation_range_over_q_polytope": list(mean_range),
        "state_unit": model["state_unit"],
        "outcomes": rows,
        "summary": {
            "most_likely_outcome": {"label": likely["label"], "q_reference": likely["q_reference"],
                                    "pnl_if_outcome_usd": likely["pnl_if_outcome_usd"]},
            "largest_loss_outcome_reference_q_at_least_1pct": {"label": worst["label"], "q_reference": worst["q_reference"],
                                                    "pnl_if_outcome_usd": worst["pnl_if_outcome_usd"],
                                                    "pnl_model_range_5_95_usd": worst["pnl_model_range_5_95_usd"]},
            "largest_gain_outcome_reference_q_at_least_1pct": {"label": best["label"], "q_reference": best["q_reference"],
                                                    "pnl_if_outcome_usd": best["pnl_if_outcome_usd"],
                                                    "pnl_model_range_5_95_usd": best["pnl_model_range_5_95_usd"]},
            "q_standard_deviation_usd": None if missing else q_std,
            "known_positions_only_q_standard_deviation_usd": q_std if missing else None,
            "q_standard_deviation_model_range_5_95_usd": ([float(v) for v in np.percentile(q_std_draw, [5, 95])]
                                                           if not missing else None),
            "q_standard_deviation_scope": "REFERENCE_Q_ONLY_NOT_OPTIMIZED_OVER_Q",
            "q_weighted_pnl_usd": None if missing else weighted,
            "ranking_basis": "Q-standard deviation of the not-yet-priced announcement P&L: how far this event can move the portfolio beyond what the contract prices imply.",
        },
        "positions": positions,
        "model_reliability": reliability,
        "scenario_transfer_status": reliability["scenario_transfer_status"],
        "scenario_transfer_failed_symbols": [],
        "scenario_transfer_untested_symbols": reliability["scenario_transfer_untested_symbols"],
        "historical_diagnostic_failed_symbols": reliability["historical_diagnostic_failed_symbols"],
        "scenario_transfer_check_scope": reliability["scope"],
        "future_total_return_prediction_available": False,
        "factors": {
            "equity": {"point": model["equity_point"], "unit": model["equity_unit"],
                       "source": model["equity_source"], "uncertainty": model["equity_note"]},
            "rates": {"point": model["rate"]["point"], "unit": model["rate"]["unit"],
                      "status": model["rate"]["status"], "source": model["rate"]["source"],
                      "draw": list(model["rate"]["draw"])},
        },
        "uncertainty": {"method": "SEEDED_MONTE_CARLO_INDEPENDENT_PARAMETERS", "draws": n_draws, "seed": seed,
                        "q_optimization": q_optimization,
                        "includes": ["published factor responses", "instrument loadings (dependence-robust standard errors)",
                                     "bond duration coverage", "declared representatives of open or ambiguous outcomes",
                                     "every Q in the source polytope (lower edge = 5th percentile of the worst feasible Q, upper edge = 95th of the best)"],
                        "excludes": ["model misspecification", "other news on the announcement day",
                                     "uncertainty in the published coefficients beyond their reported standard errors"],
                        "interpretation": ("Model-conditional empirical quantile envelope: 5th percentile of per-draw Q-minima "
                                           "to 95th percentile of per-draw Q-maxima. It contains the empirical 5-95% interval "
                                           "for each fixed feasible Q, not every parameter draw; not forecast coverage. A narrow range comes from "
                                           "declared published coefficients and pooled betas, not from more evidence about this event."),
                        "sign_probability_scope": "Under the reference Q only."},
        "physical_probability_P": None,
        "assumptions": [
            "Contract Q is the outcome weight embedded in asset prices before the announcement, so the Q-weighted P&L is zero and each row is the not-yet-priced move if that outcome settles.",
            "Published historical factor responses transfer to this announcement; Q is not a physical probability.",
            "Stocks and equity ETFs move with SPY through their combined loading; bond ETFs move with long yields through their holdings-derived duration. Each original position is counted once.",
            "The model is linear in the surprise: information effects (for example a cut read as bad news) and convexity are not modeled.",
            *(["CPI rate channel: the core-CPI daily 10-year yield response (Beechey and Wright) bounds the headline response from above; the draw is uniform between zero and that bound. The 20+ year yield is assumed to move like the 10-year."]
              if family == "US_INFLATION" else []),
        ],
    }

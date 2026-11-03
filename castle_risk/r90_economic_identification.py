"""R90: source-grounded economic identification with explicit non-identification.

This module deliberately separates three objects that cannot be exchanged:

* High-frequency *contemporaneous* FOMC market responses, evaluated out of time.
* SEC-disclosed *issuer-level* earnings / fuel-cost scenario sensitivities.
* Sharp event/marginal coupling bounds under a *stipulated common measure*.

None of them alone identifies a future arbitrary-asset event-YES stock return,
an equity portfolio's risk-neutral event weight, or a new trade's expected alpha.
"""
from __future__ import annotations

from datetime import date
import math
from collections.abc import Mapping, Sequence

import numpy as np


SCHEMA = "castle.option1.r90_economic_identification.v1"
SF_FED_SOURCE = (
    "https://www.frbsf.org/research-and-insights/data-and-indicators/"
    "us-monetary-policy-event-study-database/"
)
SEC_EXXON_SOURCE = (
    "https://www.sec.gov/Archives/edgar/data/34088/"
    "000003408826000045/xom-20251231.htm"
)
SEC_DELTA_SOURCE = (
    "https://www.sec.gov/Archives/edgar/data/27904/"
    "000002790426000013/dal-20251231.htm"
)


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: finite real number required")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label}: finite real number required")
    return result


def issuer_cashflow_channel(
    symbol: str,
    *,
    driver: str,
    driver_change: float,
    fraction_of_reference_year: float = 1.0,
) -> dict:
    """Quantify the *company-wide accounting channel*, not its stock price.

    Exxon 2025 Form 10-K, filed February 18 2026: for 2026, +$1/bbl
    Brent corresponds to approximately +$700m annual after-tax Upstream
    consolidated plus equity-company earnings excluding derivatives. The
    disclosure embeds oil-linked LNG exposure. A pro-rata period fraction is
    a user-imposed constant-run-rate approximation, not company guidance.

    Delta 2025 10-K: 4,269m gallons consumed in 2025. A +$0.01/gallon jet
    fuel shock at unchanged gallon volume means $42.69m extra *gross* annual
    fuel cost, before refinery, hedges, fares, taxes and volume adaptation.
    """
    unit = _finite_number(driver_change, "driver_change")
    fraction = _finite_number(fraction_of_reference_year, "fraction_of_reference_year")
    if not 0 < fraction <= 1:
        raise ValueError("The scenario fraction must be strictly positive and at most one")
    symbol = symbol.strip().upper() if isinstance(symbol, str) else ""
    if (symbol, driver) == ("XOM", "BRENT_USD_PER_BARREL"):
        delta = 700_000_000.0 * unit * fraction
        return {
            "schema": SCHEMA, "status": "SEC_DISCLOSED_ISSUER_EARNINGS_SENSITIVITY",
            "symbol": symbol, "driver": driver, "driver_change": unit,
            "fraction_of_reference_year": fraction,
            "reference_year": 2026, "accounting_quantity": "after_tax_upstream_earnings_usd",
            "issuer_accounting_effect_usd": delta,
            "source": SEC_EXXON_SOURCE,
            "source_statement": "Approx. $700m annual 2026 after-tax Upstream earnings per $1/bbl Brent, excluding derivatives; includes oil-linked LNG.",
            "scenario_fiction": "Constant source sensitivity and fractional-year scaling are declared assumptions; they are not a future earnings forecast.",
            "portfolio_equity_pnl_usd": None,
            "equity_price_effect_identified": False,
            "already_priced_in_identified": False,
            "cannot_infer": ["equity_valuation_multiple", "event_to_brent_shock",
                             "risk_premium", "portfolio_stock_return"],
        }
    if (symbol, driver) == ("DAL", "JET_FUEL_USD_PER_GALLON"):
        delta = -4_269_000_000.0 * unit * fraction
        return {
            "schema": SCHEMA, "status": "SEC_VOLUME_HOLD_FIXED_GROSS_FUEL_COST",
            "symbol": symbol, "driver": driver, "driver_change": unit,
            "fraction_of_reference_year": fraction,
            "reference_year": 2025, "accounting_quantity": "gross_pre_offset_fuel_cost_effect_usd",
            "issuer_accounting_effect_usd": delta,
            "source": SEC_DELTA_SOURCE,
            "source_statement": "2025 gallons consumed: 4,269m; gross fuel expense run-rate at constant volume, prior to refinery, hedges, fares and taxes.",
            "scenario_fiction": "2025 volume carried forward and fractional-year scaling are hypothetical; not official DAL guidance.",
            "portfolio_equity_pnl_usd": None,
            "equity_price_effect_identified": False,
            "already_priced_in_identified": False,
            "cannot_infer": ["brent_to_jet_fuel_pass_through", "refinery_netting",
                             "fare_adjustment", "equity_price_response"],
        }
    raise ValueError("No exact SEC sensitivity for that symbol-driver pair; do not substitute a proxy")


def event_marginal_coupling_bounds(
    terminal_underlying_values: Sequence[float],
    common_measure_masses: Sequence[float],
    event_state_price_weight: float,
    *,
    probability_provenance: str,
) -> dict:
    """Compute *sharp* finite-support Fréchet bounds on an event conditional mean.

    If a common (stipulated) pricing measure Q carries the underlying marginal
    and event A with Q(A)=q, let z_i=Q(X=x_i,A). The only constraints are
    0 <= z_i <= p_i and sum_i z_i=q. The minimum / maximum of
    (sum_i x_i z_i)/q are obtained by allocating q from the lowest / highest
    underlying values. We return explicit constructive joint witnesses.

    WARNING: Kalshi/Polymarket probabilities and an oil option-implied
    marginal do *not* supply a common Q. Such inputs must abstain. Even an
    authorized hypothetical coupling of OIL FUTURES is NOT an equity-portfolio
    return; all model bounds are against the underlying's own Q-mean.
    """
    if (not isinstance(terminal_underlying_values, Sequence)
            or isinstance(terminal_underlying_values, (str, bytes))
            or not isinstance(common_measure_masses, Sequence)
            or isinstance(common_measure_masses, (str, bytes))
            or len(terminal_underlying_values) != len(common_measure_masses)
            or len(terminal_underlying_values) < 2):
        raise ValueError("At least two paired support values and masses are required")
    xs = [_finite_number(x, "terminal_underlying") for x in terminal_underlying_values]
    ps = [_finite_number(p, "common_measure_mass") for p in common_measure_masses]
    q = _finite_number(event_state_price_weight, "event_state_price_weight")
    if not 0 < q < 1 or any(p < 0 or p > 1 for p in ps) or abs(sum(ps) - 1) > 1e-10:
        raise ValueError("Nondegenerate event weight and normalized nonnegative masses required")
    if probability_provenance != "STIPULATED_SAME_MEASURE_HYPOTHESIS":
        return {
            "schema": SCHEMA, "status": "ABSTAIN_NO_COMMON_PRICING_MEASURE",
            "underlying_conditional_bound_identified": False,
            "equity_pricing_weight_identified": False,
            "portfolio_equity_pnl_identified": False,
            "yes_conditional_underlying": None,
            "no_conditional_underlying": None,
            "reason": "A prediction-market quote and an option-implied marginal on a different market cannot be assumed to share a state-pricing measure. An instrument-independent Q is not observed.",
        }

    def allocate(ascending: bool) -> list[float]:
        order = sorted(range(len(xs)), key=lambda i: (xs[i], i), reverse=not ascending)
        allocation = [0.0] * len(xs)
        residual = q
        for i in order:
            take = min(residual, ps[i])
            allocation[i] = take
            residual -= take
        if residual > 1e-10:
            raise ArithmeticError("Event mass cannot be allocated")
        return allocation

    low_joint = allocate(True)
    high_joint = allocate(False)
    marginal_mean = math.fsum(x * p for x, p in zip(xs, ps))

    def two_state_means(joint: list[float]) -> tuple[float, float]:
        event_mean = math.fsum(x*z for x, z in zip(xs, joint)) / q
        complement_mean = math.fsum(x*(p-z) for x, p, z in zip(xs, ps, joint)) / (1-q)
        return event_mean, complement_mean

    yl, nh = two_state_means(low_joint)
    yh, nl = two_state_means(high_joint)
    return {
        "schema": SCHEMA, "status": "SHARP_STIPULATED_MEASURE_BOUNDS_NOT_EQUITY_PNL",
        "underlying_conditional_bound_identified": True,
        "equity_pricing_weight_identified": False,
        "portfolio_equity_pnl_identified": False,
        "common_measure_is_observed": False,
        "provenance": probability_provenance,
        "state_price_weight_assumed": q, "underlying_unconditional_mean": marginal_mean,
        "yes_conditional_underlying": [yl, yh],
        "no_conditional_underlying": [nl, nh],
        "yes_minus_current_underlying": [yl - marginal_mean, yh - marginal_mean],
        "no_minus_current_underlying": [nl - marginal_mean, nh - marginal_mean],
        "low_yes_joint_witness": low_joint,
        "high_yes_joint_witness": high_joint,
        "assumptions": [
            "One hypothetical, common Q for the specified underlying and event indicator.",
            "A fully observed terminal-underlying Q-marginal, not sparse 10th/90th percentiles.",
            "No known dependence of event on underlying; rearrangement extrema are sharp.",
            "These are underlying conditional Q-expectations, not trading P&L or equity prices.",
        ],
    }


def _fomc_row_date(value: object) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError as exc:
            raise ValueError("Invalid FOMC date") from exc
    raise ValueError("FOMC date must be ISO date")


def evaluate_fomc_intraday_holdout(
    rows: Sequence[Mapping],
    *,
    split_date: str = "2020-01-01",
) -> dict:
    """Frozen 2011-2019 train / 2020+ test of *same-window* ED4 -> SP500.

    Rate surprise and S&P response are both realized inside the FOMC event
    window; the model is *not* available at pre-event pricing time. Dates
    before 2011, unscheduled meetings, or invalid/missing source values are
    excluded. One row per FOMC communication event. Neither the holdout
    slope nor its outcomes enter fitting. A clock-shift placebo falsifies
    broad sample co-movement when the event identities are disrupted.
    """
    split = date.fromisoformat(split_date)
    if split != date(2020, 1, 1):
        raise ValueError("The frozen R90 reproduction split is fixed at 2020-01-01")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValueError("Rows must be a sequence of source event observations")
    eligible: list[tuple[date, float, float]] = []
    seen = set()
    excluded_unscheduled = excluded_missing = excluded_early = 0
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Malformed FOMC source row")
        d = _fomc_row_date(row.get("Date"))
        if d in seen:
            raise ValueError("Duplicated FOMC event date")
        seen.add(d)
        if d < date(2011, 1, 1):
            excluded_early += 1
            continue
        unscheduled = row.get("Unscheduled")
        if isinstance(unscheduled, bool) or unscheduled not in (0, 0.0, 1, 1.0):
            raise ValueError("Source Unscheduled indicator must be 0 or 1")
        if unscheduled == 1:
            excluded_unscheduled += 1
            continue
        try:
            x, y = _finite_number(row.get("ED4"), "ED4"), _finite_number(row.get("SP500"), "SP500")
        except ValueError:
            excluded_missing += 1
            continue
        eligible.append((d, x, y))
    eligible.sort(key=lambda row: row[0])
    train = [r for r in eligible if r[0] < split]
    test = [r for r in eligible if r[0] >= split]
    if len(train) < 50 or len(test) < 30:
        raise ValueError("Not enough scheduled historical events for frozen chronological holdout")

    tx = np.array([r[1] for r in train], dtype=float)
    ty = np.array([r[2] for r in train], dtype=float)
    vx = np.array([r[1] for r in test], dtype=float)
    vy = np.array([r[2] for r in test], dtype=float)
    design = np.column_stack((np.ones(len(tx)), tx))
    if np.linalg.matrix_rank(design) != 2:
        raise ValueError("FOMC training surprise is rank deficient")
    fit, *_ = np.linalg.lstsq(design, ty, rcond=None)
    baseline = float(np.mean(ty))
    pred = fit[0] + fit[1] * vx
    rmse_base = float(np.sqrt(np.mean((vy - baseline)**2)))
    rmse = float(np.sqrt(np.mean((vy - pred)**2)))
    if rmse_base <= 0:
        raise ValueError("Degenerate FOMC holdout baseline")

    placebo = {}
    for lag in (1, 5, 11):
        # Surprises of other *meetings* cannot identify the event-time asset move.
        shifted_train = np.roll(tx, lag)
        shifted_test = np.roll(vx, lag)
        pfit, *_ = np.linalg.lstsq(np.column_stack((np.ones(len(tx)), shifted_train)), ty, rcond=None)
        prmse = float(np.sqrt(np.mean((vy - (pfit[0] + pfit[1]*shifted_test))**2)))
        placebo[str(lag)] = {
            "wrong_meeting_shift": lag, "oos_rmse_pct_points": prmse,
            "relative_improvement": 1 - prmse/rmse_base,
        }
    train_slope = float(fit[1])
    holdout_fit, *_ = np.linalg.lstsq(np.column_stack((np.ones(len(vx)), vx)), vy, rcond=None)
    segment_checks = {}
    for label, start, end in (
        ("2020_2022", date(2020, 1, 1), date(2023, 1, 1)),
        ("2023_2026", date(2023, 1, 1), date(2027, 1, 1)),
    ):
        ix = [i for i, r in enumerate(test) if start <= r[0] < end]
        if not ix:
            continue
        act = vy[ix]
        baseline_error = float(np.sqrt(np.mean((act - baseline)**2)))
        model_error = float(np.sqrt(np.mean((act - pred[ix])**2)))
        segment_checks[label] = {
            "n": len(ix), "baseline_rmse_pct_points": baseline_error,
            "model_rmse_pct_points": model_error,
            "relative_improvement": 1-model_error/baseline_error if baseline_error > 0 else None,
        }
    # Exploratory post-holdout robustness: *cluster* each resampling draw by
    # calendar year (rather than treating all ~53 event windows as IID).
    # This is not a distribution-free coverage guarantee or preregistered gate.
    # It deliberately leaves the 2011-19 fit fixed and never retrains on holdout.
    years = sorted({r[0].year for r in test})
    by_year = {year: np.array([i for i, r in enumerate(test) if r[0].year == year],
                               dtype=int) for year in years}
    year_leave_out = {}
    for year in years:
        kept = np.array([i for i, r in enumerate(test) if r[0].year != year], dtype=int)
        b = float(np.sqrt(np.mean((vy[kept]-baseline)**2)))
        m = float(np.sqrt(np.mean((vy[kept]-pred[kept])**2)))
        year_leave_out[str(year)] = 1-m/b if b > 0 else None
    year_rng = np.random.default_rng(9001)
    cluster_improvement = []
    for _ in range(5_000):
        sampled_years = year_rng.choice(years, size=len(years), replace=True)
        selected = np.concatenate([by_year[int(year)] for year in sampled_years])
        b = float(np.sqrt(np.mean((vy[selected]-baseline)**2)))
        m = float(np.sqrt(np.mean((vy[selected]-pred[selected])**2)))
        if b > 0:
            cluster_improvement.append(1-m/b)
    cluster_improvement.sort()
    lower, upper = (float(np.quantile(cluster_improvement, q)) for q in (.025, .975))
    return {
        "schema": SCHEMA, "status": "OOS_VALIDATED_CONTEMPORANEOUS_INDEX_RESPONSE_ONLY",
        "source": SF_FED_SOURCE, "source_sheet": "Monetary Events",
        "response_variable": "SP500 intraday event-window percent return",
        "surprise_variable": "ED4 event-window quarterly futures-implied rate change in percentage points",
        "scheduled_only": True, "train_end_exclusive": split.isoformat(),
        "retrospective_holdout_seen_during_prior_exploration": True,
        "prospective_uninspected_holdout": False,
        "training_start_inclusive": "2011-01-01",
        "train_count": len(train), "holdout_count": len(test),
        "first_train_event": train[0][0].isoformat(),
        "last_holdout_event": test[-1][0].isoformat(),
        "excluded_before_2011": excluded_early,
        "excluded_unscheduled_after_2011": excluded_unscheduled,
        "excluded_missing_after_2011": excluded_missing,
        "train_intercept_pct_points": float(fit[0]),
        "train_beta_sppct_per_ratepct": train_slope,
        "baseline_holdout_rmse_pct_points": rmse_base,
        "model_holdout_rmse_pct_points": rmse,
        "holdout_rmse_relative_improvement": 1-rmse/rmse_base,
        "holdout_slope_diagnostic_not_used_for_fitting": float(holdout_fit[1]),
        "holdout_slope_sign_stable": train_slope*float(holdout_fit[1]) > 0,
        "holdout_subperiods": segment_checks,
        "exploratory_year_cluster_resampling": {
            "year_clusters": years, "seed": 9001, "draws": len(cluster_improvement),
            "resampled_oos_rmse_gain_95pct_interval": [lower, upper],
            "leave_one_holdout_year_out_gain": year_leave_out,
            "all_leave_one_year_out_gains_positive": all(
                gain is not None and gain > 0 for gain in year_leave_out.values()
            ),
            "status": "POST_HOLDOUT_SENSITIVITY_NOT_INDEPENDENT_CONFIRMATORY_TEST",
        },
        "wrong_meeting_placebos": placebo,
        "index_contemporaneous_response_evaluated": True,
        "ex_ante_future_return_prediction_validated": False,
        "arbitrary_holding_portfolio_response_validated": False,
        "portfolio_conditional_yes_pnl_identified": False,
        "already_priced_in_fraction_identified": False,
        "prohibition": "Cannot apply post-announcement ED4 to pre-announcement trading, transfer S&P beta to XOM/DAL/TSM or infer the YES/NO portfolio state payoff.",
    }


def evaluate_opec_original_surprise_daily_holdout(
    rows: Sequence[Mapping],
    *,
    split_date: str = "2017-01-01",
) -> dict:
    """Secondary pre-fixed OPEC falsification from Degasperi's public daily file.

    Regress the dataset's S&P *observed event-day* move on the ORIGINAL
    ``PC_Kaenzig`` oil-futures surprise; never use ``sp500_neg`` or
    ``sp500_pos``, which were explicitly selected using contemporaneous
    S&P price directions. Train before 2017, test 2017–2025, check frozen
    5%-RMSE and train-versus-holdout-slope sign gates.

    This is a same-day correlation diagnostic on the source's supplied
    dSP500 field, not a future return forecast or a causal instrument-level
    conditional profit estimate. OPEC sample definitions may evolve.
    """
    if split_date != "2017-01-01":
        raise ValueError("R90 OPEC falsification split is frozen at 2017-01-01")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValueError("OPEC rows must be a sequence")
    seen: set[date] = set()
    eligible: list[tuple[date, float, float]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Malformed OPEC event row")
        d = _fomc_row_date(row.get("AnnouncementDate"))
        if d in seen:
            raise ValueError("Duplicate OPEC announcement date")
        seen.add(d)
        try:
            x = _finite_number(row.get("PC_Kaenzig"), "PC_Kaenzig")
            y = _finite_number(row.get("dSP500"), "dSP500")
        except ValueError:
            continue
        eligible.append((d, x, y))
    eligible.sort(key=lambda item: item[0])
    train = [item for item in eligible if item[0] < date(2017, 1, 1)]
    test = [item for item in eligible if item[0] >= date(2017, 1, 1)]
    if len(train) < 70 or len(test) < 15:
        raise ValueError("Insufficient independent historical OPEC event observations")
    tx, ty = np.array([r[1] for r in train]), np.array([r[2] for r in train])
    vx, vy = np.array([r[1] for r in test]), np.array([r[2] for r in test])
    a = np.column_stack((np.ones(len(train)), tx))
    if np.linalg.matrix_rank(a) != 2:
        raise ValueError("Rank-deficient original OPEC surprise")
    fitted, *_ = np.linalg.lstsq(a, ty, rcond=None)
    hfit, *_ = np.linalg.lstsq(np.column_stack((np.ones(len(test)), vx)), vy, rcond=None)
    mean_train = float(np.mean(ty))
    base_rmse = float(np.sqrt(np.mean((vy-mean_train)**2)))
    extended_rmse = float(np.sqrt(np.mean((vy-fitted[0]-fitted[1]*vx)**2)))
    if base_rmse <= 0:
        raise ValueError("Degenerate OPEC holdout benchmark")
    improvement = 1-extended_rmse/base_rmse
    stable = float(fitted[1]*hfit[1]) > 0
    passed = bool(stable and improvement >= .05)
    return {
        "schema": SCHEMA,
        "status": ("OPEC_SAME_DAY_DIAGNOSTIC_PASSED_NOT_CAUSAL" if passed
                   else "OPEC_SAME_DAY_DIAGNOSTIC_FAILED_NOT_PROMOTED"),
        "train_count": len(train), "holdout_count": len(test),
        "cutoff_test_start": split_date,
        "last_holdout_event": test[-1][0].isoformat(),
        "predictor": "PC_Kaenzig original oil-futures surprise, not the sign-selected stock-return instruments",
        "response": "Degasperi dSP500 event-date observed equity change; source units",
        "baseline_holdout_rmse_source_units": base_rmse,
        "model_holdout_rmse_source_units": extended_rmse,
        "relative_oos_rmse_improvement": improvement,
        "train_slope": float(fitted[1]),
        "holdout_slope_diagnostic_only": float(hfit[1]),
        "slope_sign_stable": stable,
        "passes_minimum_5pct_rmse_and_sign_gate": passed,
        "leakage_protection": "sp500_pos and sp500_neg are deliberately excluded because their construction uses the target equity sign.",
        "causal_portfolio_equity_effect_identified": False,
        "ex_ante_future_return_prediction_validated": False,
    }

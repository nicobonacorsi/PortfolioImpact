"""Chronologically checked fund-price proxy sensitivities for declared stresses.

Frozen rules: data/r103_proxy_extension_protocol.json. An oil-futures ETF return
is not spot oil; a high-yield ETF return is not a pure credit-spread change.
No event-to-proxy response or physical probability is estimated here.
"""
from __future__ import annotations

from datetime import date
import math
from collections.abc import Mapping

import numpy as np

from .two_factor_response import _changes

PROXY_SYMBOLS = ("USO", "DBC", "GLD", "HYG", "UUP", "VWO")
PROXY_SOURCES = {
    "USO": {"label": "USO oil futures fund price", "url": "https://www.uscfinvestments.com/uso",
            "scope": "WTI futures fund with rolls, collateral and expenses; not spot crude oil."},
    "DBC": {"label": "DBC commodity futures fund price", "url": "https://www.invesco.com/us/en/financial-products/etfs/invesco-db-commodity-index-tracking-fund.html",
            "scope": "Diversified futures basket, not spot commodities. Index methodology changed November 10, 2025."},
    "GLD": {"label": "GLD gold trust share price", "url": "https://www.ssga.com/us/en/individual/etfs/spdr-gold-shares-gld",
            "scope": "Gold bullion less expenses; share price can differ from net asset value."},
    "HYG": {"label": "HYG high yield bond fund price", "url": "https://www.ishares.com/us/literature/fact-sheet/hyg-ishares-iboxx-high-yield-corporate-bond-etf-fund-fact-sheet-en-us.pdf",
            "scope": "High yield corporate bonds, including interest-rate and credit risks; not a pure credit-spread change."},
    "UUP": {"label": "UUP dollar futures fund price", "url": "https://www.sec.gov/Archives/edgar/data/1383151/000119312526083557/uup-20251231.htm",
            "scope": "U.S. dollar futures basket against six currencies, with collateral and expenses; not a pure spot FX move."},
    "VWO": {"label": "VWO emerging market fund price", "url": "https://fund-docs.vanguard.com/F0964.pdf",
            "scope": "Emerging market equities with country and currency risks; not a pure geopolitical factor."},
}
FAMILY_PROXIES = {
    "HORMUZ_ENERGY": ("USO",), "OPEC_SUPPLY": ("USO",),
    "RUSSIA_ENERGY_SANCTIONS": ("USO",), "ENERGY_SUPPLY": ("USO",),
    "ENERGY_POLICY": ("USO",), "MIDDLE_EAST_ENERGY": ("USO",),
    "VENEZUELA_ENERGY": ("USO",), "CHINA_CRITICAL_MINERALS": ("DBC",),
    "BANK_CREDIT_STRESS": ("HYG",), "BANK_CAPITAL_RULES": ("HYG",),
    "US_SOVEREIGN_CREDIT": ("HYG",), "TREASURY_CREDIT": ("HYG",),
    "TRADE_TARIFFS": ("UUP", "VWO"), "US_CHINA_TARIFFS": ("UUP", "VWO"),
    "US_TARIFFS": ("UUP", "VWO"), "US_IMPORT_TARIFF_POLICY": ("UUP", "VWO"),
    "TAIWAN_SUPPLY": ("VWO",), "EXPORT_CONTROLS": ("VWO",),
    "EMERGING_MARKETS": ("VWO",), "DOLLAR": ("UUP",), "GOLD": ("GLD",),
}


def fit_proxy_response(prices, spy_prices, yields, proxy_prices, *, train_end,
                       release_dates=()):
    """OLS incremental loading with an untouched, fixed-comparator holdout."""
    out = {"status": "UNAVAILABLE", "admitted": False, "train_end": train_end,
           "basis": "HISTORICAL_THREE_FACTOR_ASSOCIATION_NOT_CAUSAL_EVENT_RESPONSE",
           "price_basis": "Split-adjusted closes; cash distributions are not added back."}
    try:
        date.fromisoformat(train_end)
        series = [_changes(rows, field, price=is_price) for rows, field, is_price in
                  ((prices, "close", True), (spy_prices, "close", True),
                   (yields, "yield_pct", False), (proxy_prices, "close", True))]
    except (ValueError, TypeError, KeyError) as exc:
        return {**out, "status": "INVALID_SOURCE_SERIES", "reason": str(exc)}
    common = sorted(set.intersection(*(set(s) for s in series)))
    days = [d for d in common if len({s[d][0] for s in series}) == 1
            and (date.fromisoformat(d) - date.fromisoformat(series[0][d][0])).days <= 4]
    train = np.array([d <= train_end for d in days], dtype=bool)
    ntrain, nhold = int(sum(train)), int(sum(~train))
    out.update(n_train=ntrain, n_holdout=nhold,
               excluded_misaligned_or_gapped_intervals=len(common)-len(days))
    if ntrain < 250 or nhold < 120:
        return {**out, "status": "INSUFFICIENT_HISTORY"}
    values = np.array([[s[d][1] for s in series] for d in days])
    if not np.isfinite(values).all():
        return {**out, "status": "INVALID_SOURCE_SERIES"}
    y, factors = values[:, 0], values[:, 1:]
    std = np.std(factors[train], axis=0)
    if np.any(std <= 0):
        return {**out, "status": "RANK_DEFICIENT_DESIGN"}
    mean = np.mean(factors[train], axis=0)
    z = (factors-mean)/std
    design = np.column_stack((np.ones(len(days)), z))
    if np.linalg.matrix_rank(design[train]) != 4 or np.linalg.cond(design[train]) > 1e10:
        return {**out, "status": "RANK_DEFICIENT_DESIGN"}
    coef = np.linalg.lstsq(design[train], y[train], rcond=None)[0]
    coef2 = np.linalg.lstsq(design[train, :3], y[train], rcond=None)[0]
    slopes = coef[1:]/std
    slopes2 = coef2[1:]/std[:2]
    # Same no-intercept incremental-price convention as the shipped two-factor
    # response. An estimated drift is not mistaken for a shock contribution.
    pred3 = factors @ slopes
    pred2 = factors[:, :2] @ slopes2
    mean_pred = np.full(len(days), float(y[train].mean()))
    rmse = lambda pred, select: float(np.sqrt(np.mean((y[select]-pred[select])**2)))
    r3, r2, r0 = rmse(pred3, ~train), rmse(pred2, ~train), rmse(np.zeros(len(days)), ~train)
    gain2 = 1-r3/r2 if r2 > 0 else None
    gain0 = 1-r3/r0 if r0 > 0 else None
    release_set = set(release_dates)
    event_mask = np.array([not train[i] and d in release_set for i, d in enumerate(days)])
    nevents = int(sum(event_mask))
    event3, event2 = (rmse(pred3, event_mask), rmse(pred2, event_mask)) if nevents else (None, None)
    admitted = (gain2 is not None and gain0 is not None and gain2 >= .02 and gain0 >= .02
                and (nevents < 8 or event3 <= event2))
    try:
        inverse = np.linalg.inv(design[train].T @ design[train])
    except np.linalg.LinAlgError:
        return {**out, "status": "RANK_DEFICIENT_DESIGN"}
    residuals = y[train] - design[train] @ coef
    leverage = np.einsum("ij,jk,ik->i", design[train], inverse, design[train])
    if np.any(leverage >= 1-1e-10):
        return {**out, "status": "UNSTABLE_COVARIANCE"}
    meat = (design[train]*(residuals/(1-leverage))[:, None]**2).T @ design[train]
    covariance = (inverse @ meat @ inverse)[1:, 1:] / std[:, None] / std[None, :]
    if not np.isfinite(covariance).all() or np.any(np.diag(covariance) < -1e-12):
        return {**out, "status": "UNSTABLE_COVARIANCE"}
    training_days = [d for d in days if d <= train_end]
    holdout_days = [d for d in days if d > train_end]
    return {**out, "status": "ADMITTED_PROXY_RESPONSE" if admitted else "PROXY_HOLDOUT_FAILED",
            "admitted": bool(admitted), "spy_loading": float(slopes[0]),
            "rate_loading_per_bp": float(slopes[1]), "proxy_loading": float(slopes[2]),
            "covariance": covariance.tolist(),
            "training_span": [training_days[0], training_days[-1]],
            "holdout_span": [holdout_days[0], holdout_days[-1]],
            "holdout_rmse": r3, "holdout_two_factor_rmse": r2, "holdout_zero_rmse": r0,
            "holdout_training_mean_rmse": rmse(mean_pred, ~train),
            "holdout_gain_vs_two_factor": gain2, "holdout_gain_vs_zero": gain0,
            "holdout_event_days": nevents, "holdout_event_rmse": event3,
            "holdout_event_two_factor_rmse": event2,
            "admission_rule": "At least 2% holdout RMSE improvement over both two-factor and zero; no worse on at least eight macro-release days.",
            "admission_period_is_not_an_independent_post_selection_test": True,
            "forecast_validated": False}


def build_proxy_models(portfolio, prices_by_symbol, proxy_prices, yields, *, train_end,
                       evaluated_at, release_dates=()):
    """One fixed extension per proxy, shared across all event-date repetitions."""
    through = evaluated_at.date().isoformat()
    result = {}
    merged = {**(proxy_prices or {}), **{k: v for k, v in (prices_by_symbol or {}).items() if v}}
    spy = merged.get("SPY") or []
    for symbol in PROXY_SYMBOLS:
        rows = merged.get(symbol) or []
        if not rows:
            result[symbol] = {"status": "PROXY_PRICES_UNAVAILABLE", "positions": {}}
            continue
        positions = {}
        for p in portfolio.positions:
            if p.asset_type == "cash" or p.symbol == "SPY":
                positions[p.symbol] = {"status": "EXACT_PRICE_IDENTITY", "admitted": True,
                    "proxy_loading": 0., "basis": "OWN_PRICE_IDENTITY_WITH_SPY_AND_OTHER_PRICES_FIXED"}
            elif p.symbol == symbol:
                positions[p.symbol] = {"status": "EXACT_PRICE_IDENTITY", "admitted": True,
                    "proxy_loading": 1., "basis": "SHOCKED_FUND_IS_THE_ORIGINAL_HOLDING"}
            elif not merged.get(p.symbol):
                positions[p.symbol] = {"status": "POSITION_PRICES_UNAVAILABLE", "admitted": False}
            else:
                # Defensive as-of clipping occurs before fitting and admission.
                before = lambda sequence: [r for r in sequence if not isinstance(r, Mapping)
                                            or str(r.get("date") or "") < through]
                positions[p.symbol] = fit_proxy_response(before(merged[p.symbol]), before(spy),
                    before(yields), before(rows), train_end=train_end, release_dates=release_dates)
        try:
            # Preserve malformed rows for the strict reader to reject; silently
            # deleting them could bridge an unknown price interval.
            increments = _changes([r for r in rows if not isinstance(r, Mapping)
                                   or str(r.get("date") or "") < through], "close", price=True)
            eligible = [(d, 100*value) for d, (prev, value) in increments.items() if d <= train_end
                        and (date.fromisoformat(d)-date.fromisoformat(prev)).days <= 4]
            changes = [value for _, value in eligible]
            scale = float(np.std(changes, ddof=1)) if len(changes) >= 250 else None
            scale = scale if scale and math.isfinite(scale) else None
        except (ValueError, TypeError, KeyError):
            scale, eligible = None, []
        result[symbol] = {"status": "COMPUTED", "positions": positions,
                          "scale_pct": scale, "training_end": train_end,
                          "training_span": [eligible[0][0], eligible[-1][0]] if eligible else None,
                          "scale_basis": "ONE_TRAINING_DAILY_STANDARD_DEVIATION_NOT_EVENT_SEVERITY"}
    return result

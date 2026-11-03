"""R43: independently test 7Y-vs-10Y Treasury *proxy* basis risk.

The output is a historical sensitivity diagnostic, never a deterministic
future IEF yield, prediction-market probability or total portfolio P&L.
"""
from __future__ import annotations

import csv
from datetime import datetime
from hashlib import sha256
from io import StringIO
from math import ceil, isfinite
from collections.abc import Mapping

from .conditional_factor_transmission import _bond_per_bp
from .direct_yield_threshold_bridge import (
    SCHEMA as R42_SCHEMA,EVENT as R42_EVENT,SOURCE_DATE as R42_DATE,
    STRIKES as R42_STRIKES,
)


SCHEMA = 'castle.treasury_curve_basis_audit.v1'
CALIBRATION = tuple(range(2017, 2022))
VALIDATION = tuple(range(2022, 2026))
ALL_YEARS = CALIBRATION + VALIDATION
BLOCK_SIZE = 46  # Two endpoints separated by 45 source observation sessions.


class CurveBasisError(ValueError):
    pass


def _parse_one_year(year, content, expected_sha):
    if not isinstance(year, int) or year not in ALL_YEARS:
        raise CurveBasisError('Unreviewed source year')
    if not isinstance(content, bytes) or not 100 < len(content) <= 1_000_000:
        raise CurveBasisError('Missing, oversized or altered source bytes')
    if not isinstance(expected_sha, str) or sha256(content).hexdigest() != expected_sha:
        raise CurveBasisError('Annual source SHA256 mismatch')
    try:
        source = csv.DictReader(StringIO(content.decode('utf-8-sig')))
        if (not source.fieldnames or source.fieldnames.count('Date') != 1
                or source.fieldnames.count('7 Yr') != 1
                or source.fieldnames.count('10 Yr') != 1):
            raise CurveBasisError('Missing official unique 7Y/10Y source columns')
        records = []
        for row in source:
            when = datetime.strptime(row['Date'], '%m/%d/%Y').date()
            if when.year != year:
                raise CurveBasisError('Year contamination in official annual source')
            a, b = float(row['7 Yr']), float(row['10 Yr'])
            if not (isfinite(a) and isfinite(b) and 0 <= a <= 30 and 0 <= b <= 30):
                raise CurveBasisError('Invalid Treasury yield units')
            records.append((when, a, b))
    except CurveBasisError:
        raise
    except (ValueError, KeyError, UnicodeError, TypeError, AttributeError) as exc:
        raise CurveBasisError('Malformed official annual source data') from exc
    if not 230 <= len(records) <= 270 or len({r[0] for r in records}) != len(records):
        raise CurveBasisError('Missing/duplicate dated official observations')
    records.sort(key=lambda r: r[0])
    if (records[0][0].month != 1 or records[0][0].day > 10
            or records[-1][0].month != 12 or records[-1][0].day < 10
            or any((b[0]-a[0]).days > 6 for a,b in zip(records,records[1:]))):
        raise CurveBasisError('Incomplete annual Treasury calendar coverage')
    return records


def _nonoverlapping_basis_changes(records):
    changes = []
    for i in range(0, len(records) - BLOCK_SIZE + 1, BLOCK_SIZE):
        first, last = records[i], records[i+BLOCK_SIZE-1]
        dy7 = 100*(last[1] - first[1])
        dy10 = 100*(last[2] - first[2])
        changes.append({'first_date': first[0].isoformat(),
                        'last_date': last[0].isoformat(),
                        'basis_change_bp': round(dy7 - dy10, 12),
                        'treasury_7y_change_bp': round(dy7, 12),
                        'treasury_10y_change_bp': round(dy10, 12)})
    return changes


def audit_curve_basis(*, annual_sources, annual_sha256, rate_sensitivity,
                      notional_usd):
    """Audit a frozen temporal split; never fetch or fit on validation data."""
    if not isinstance(annual_sources, Mapping) or not isinstance(annual_sha256, Mapping):
        raise CurveBasisError('Annual source mapping is required')
    if set(annual_sources) != set(ALL_YEARS) or set(annual_sha256) != set(ALL_YEARS):
        raise CurveBasisError('All nine unique source years are required')
    audited = {}
    for year in ALL_YEARS:
        records = _parse_one_year(year, annual_sources[year], annual_sha256[year])
        windows = _nonoverlapping_basis_changes(records)
        if len(windows) < 4:
            raise CurveBasisError('Insufficient independent approximate-quarter windows')
        audited[year] = {'official_observations':len(records),
                         'source_sha256':annual_sha256[year], 'windows':windows}
    train = [abs(row['basis_change_bp']) for year in CALIBRATION
             for row in audited[year]['windows']]
    n = len(train)
    rank = ceil(0.9*(n+1))
    if rank > n:
        raise CurveBasisError('Inapplicable finite-sample rank for calibration')
    tolerance_bp = sorted(train)[rank-1]
    val = [row for year in VALIDATION for row in audited[year]['windows']]
    covered = [abs(row['basis_change_bp']) <= tolerance_bp+1e-10 for row in val]
    zero_covered = [abs(row['basis_change_bp']) <= 1e-10 for row in val]
    coverage = sum(covered)/len(covered)
    by_year = {}
    for year in VALIDATION:
        samples = [abs(row['basis_change_bp']) for row in audited[year]['windows']]
        by_year[str(year)] = {
            'windows':len(samples),
            'covered_count':sum(x<=tolerance_bp+1e-10 for x in samples),
            'coverage':sum(x<=tolerance_bp+1e-10 for x in samples)/len(samples),
            'maximum_absolute_twist_bp':max(samples)}
    admitted = coverage >= 0.85-1e-12 and all(r['coverage'] >= .60-1e-12 for r in by_year.values())
    per_bp, fund_covered, fund_uncovered = _bond_per_bp(rate_sensitivity, notional_usd)
    return {'schema':SCHEMA,
            'status':('RETROSPECTIVE_PROXY_BASIS_DIAGNOSTIC_ADMISSION_PASSED' if admitted
                      else 'RETROSPECTIVE_PROXY_BASIS_DIAGNOSTIC_ADMISSION_FAILED'),
            'source_years':list(ALL_YEARS),
            'source_receipts_by_year':audited,
            'calibration_years':list(CALIBRATION),
            'validation_years':list(VALIDATION),
            'nonoverlapping_observations_per_window':BLOCK_SIZE,
            'calibration_samples':n,
            'absolute_rank_index_one_based':rank,
            'historical_90pct_absolute_proxy_basis_tolerance_bp':tolerance_bp,
            'heldout_samples':len(val),
            'heldout_proxy_basis_coverage':coverage,
            'heldout_zero_basis_coverage':sum(zero_covered)/len(val),
            'heldout_by_year':by_year,
            'heldout_minimum_signed_basis_change_bp':min(r['basis_change_bp'] for r in val),
            'heldout_maximum_signed_basis_change_bp':max(r['basis_change_bp'] for r in val),
            'historical_hypothetical_ief_dollar_tolerance':abs(per_bp)*tolerance_bp,
            'source_holdings_derived_ief_usd_per_bp':per_bp,
            'covered_ief_fund_nav_weight':fund_covered,
            'unresolved_ief_fund_nav_weight':fund_uncovered,
            'admission_passed':admitted,
            'future_deterministic_ief_yield_bound_identified':False,
            'event_conditional_total_portfolio_pnl_identified':False,
            'release_allowed':False,
            'limitations':['7Y CMT is only an imperfect proxy for bond-level IEF yield changes.',
                           'The 45-source-session windows only approximate the 2026-10-27 to 2026-12-31 calendar horizon.',
                           'Historical out-of-time diagnostics cannot guarantee future yield-curve-basis coverage.',
                           'Carry, convexity, changing ETF holdings, unresolved NAV and CAT/UNP/NEE responses are excluded.']}


def hypothetically_widen_r42_ief_proxy_bands(*, r42, r43):
    """Optional 7Y-proxy stress sensitivity; NEVER claim future guaranteed bounds.

    Reject incompatible cohorts and any 7Y scenario exceeding the original
    declared +-100bp first-order model domain. Historical 90% coverage is
    not a probability of physical outcomes under a Kalshi YES event.
    """
    if not isinstance(r42, Mapping) or r42.get('schema') != R42_SCHEMA:
        raise CurveBasisError('The exact R42 original-market bridge is required')
    if not isinstance(r43, Mapping) or r43.get('schema') != SCHEMA or not r43.get('admission_passed'):
        raise CurveBasisError('Unadmitted annual Treasury basis audit cannot widen a market stress')
    if (r42.get('event_id')!=R42_EVENT or r42.get('settlement_date')!=R42_DATE.isoformat()
            or r42.get('exact_original_market_threshold_count')!=len(R42_STRIKES)
            or r42.get('status')!='SOURCE_EXACT_YIELD_BAND_TO_COVERED_BOND_STRESS_NOT_EVENT_PNL'
            or r42.get('release_allowed') is not False
            or r42.get('event_yes_no_full_portfolio_pnl') is not None
            or r43.get('calibration_years')!=list(CALIBRATION)
            or r43.get('validation_years')!=list(VALIDATION)
            or r43.get('nonoverlapping_observations_per_window')!=BLOCK_SIZE
            or r43.get('release_allowed') is not False):
        raise CurveBasisError('The source event, portfolio or frozen curve cohort changed')
    per_bp = r43.get('source_holdings_derived_ief_usd_per_bp')
    source_bp = r42.get('source_price_implied_covered_ief_usd_per_parallel_bp')
    if (type(per_bp) not in (int,float) or type(source_bp) not in (int,float)
            or not isfinite(per_bp) or not isfinite(source_bp) or per_bp >= 0
            or abs(per_bp-source_bp) > 1e-8):
        raise CurveBasisError('R42/R43 original-position dollar sensitivity mismatch')
    tol = r43.get('historical_90pct_absolute_proxy_basis_tolerance_bp')
    if type(tol) not in (int,float) or not isfinite(tol) or not 0 <= tol <= 100:
        raise CurveBasisError('Invalid or unreviewed basis proxy tolerance')
    baseline = r42.get('treasury_baseline_yield_10y_pct')
    if (type(baseline) not in (int,float) or not isfinite(baseline)
            or r42.get('first_order_yield_scenario_bps_domain') != [-100.,100.]):
        raise CurveBasisError('Original R42 local 10Y baseline/domain changed')
    rows = []
    original_bands=r42.get('source_defined_yield_bands', [])
    if not isinstance(original_bands,list) or len(original_bands)!=len(R42_STRIKES)+1:
        raise CurveBasisError('Source-exact strike partition absent')
    for i,b in enumerate(original_bands):
        original=b.get('covered_ief_linear_parallel_stress_usd_interval')
        interval=b.get('source_treasury_10y_yield_interval_pct')
        canonical_lower=float(R42_STRIKES[i-1]) if i else None
        canonical_upper=float(R42_STRIKES[i]) if i<len(R42_STRIKES) else None
        if (not isinstance(interval,Mapping)
                or interval.get('lower_exclusive')!=canonical_lower
                or interval.get('upper_inclusive')!=canonical_upper):
            raise CurveBasisError('Source-exact Treasury state interval changed')
        originally_eligible=(canonical_lower is not None and canonical_upper is not None
                             and 100*(canonical_lower-baseline)>=-100.-1e-9
                             and 100*(canonical_upper-baseline)<=100.+1e-9)
        if (original is not None)!=originally_eligible:
            raise CurveBasisError('Original R42 linear-domain admission was altered')
        output={'source_treasury_10y_yield_interval_pct':interval,
                'original_r42_covered_ief_parallel_stress_usd_interval':original,
                'historical_proxy_basis_tolerance_bp':tol,
                '7y_proxy_ief_stress_usd_interval':None}
        if original is None:
            output['admission']='R42_ORIGINAL_PARALLEL_STRESS_UNAVAILABLE'
        else:
            a=interval.get('lower_exclusive')
            z=interval.get('upper_inclusive')
            if (type(a) not in (int,float) or type(z) not in (int,float)
                    or not isfinite(a) or not isfinite(z) or a>=z):
                raise CurveBasisError('Broken exact threshold interval')
            expected=[per_bp*100*(z-baseline),per_bp*100*(a-baseline)]
            if (not isinstance(original,list) or len(original)!=2 or any(
                    type(v) not in (int,float) or not isfinite(v) or
                    abs(v-ev)>1e-8 for v,ev in zip(original,expected))):
                raise CurveBasisError('Underlying R42 original-price stress was altered')
            if 100*(a-baseline)-tol < -100.-1e-8 or 100*(z-baseline)+tol > 100.+1e-8:
                output['admission']='PROXY_BASIS_STRESS_EXCEEDS_R42_LOCAL_MODEL_DOMAIN'
            else:
                delta=abs(per_bp)*tol
                output['7y_proxy_ief_stress_usd_interval']=[original[0]-delta,original[1]+delta]
                output['admission']='HISTORICAL_BASIS_SENSITIVITY_NOT_A_FUTURE_COVERAGE_GUARANTEE'
                output['endpoints']=['CLOSED_MINIMUM','OPEN_SUPREMUM']
        rows.append(output)
    if len(rows)!=9 or sum(r['7y_proxy_ief_stress_usd_interval'] is not None for r in rows)>7:
        raise CurveBasisError('Missing source bands or unexpected local model admission')
    return {'schema':'castle.r43_historical_curve_proxy_sensitivity.v1',
            'status':'HISTORICAL_BASIS_WIDENED_HYPOTHETICAL_IEF_STRESS_ONLY',
            'market_event_id':r42['event_id'],
            'official_10y_baseline_pct':baseline,
            'historical_90pct_basis_tolerance_bp':tol,
            'heldout_coverage_not_prospective':r43['heldout_proxy_basis_coverage'],
            'states':rows,
            'future_scenario_coverage_guarantee':False,
            'physical_event_probability':None,
            'whole_portfolio_conditional_pnl':None,
            'release_allowed':False}

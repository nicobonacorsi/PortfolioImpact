"""Separate bond ETF signed source composition from plain-equity identity look-through.

A valid source-class ledger is NOT approval of the original equity-style holdings
validator, a security master, a whole-NAV reconciliation, or a bond price model.
The signed source records, their exact multiplicity and their original hashes
remain intact. Never clip cash, merge repeated bond tickers or normalize NAV.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import json
import math
from pathlib import Path
from urllib.parse import urlsplit

from .adapters import _normalize_constituent
from .fund_composition import (FundCompositionError, _identity as _source_identity, analyze_holdings_snapshot,
                               selected_rows_from_holdings)

SCHEMA = 'castle.option1.source_class_quality.v1'

# Declared tolerances of the second fixed-income tier. Each covers a structure
# that bond-fund holdings files show (r8c, 2026-10-23: BND); every use is listed
# with its measured value, and anything beyond a limit keeps the structural gap.
LISTED_NAV_SHORTFALL_LIMIT = 0.02      # cash and other assets/liabilities a file may not list
ADDITIVE_SAME_IDENTITY_LIMIT = 0.001   # repeated rows of one security, summed
SELF_TICKER_COLLISION_LIMIT = 0.001    # rows whose ticker field equals the fund, kept unresolved
CASH_LIKE_UNIDENTIFIED_LIMIT = 0.05    # rows without identifier that the provider classes as cash or short-term
SINGLE_LINE_NAV_BAND = 0.001           # exchange-traded vehicle: its one line within 0.1% of NAV
# Further lines of such a vehicle, in absolute NAV summed, set aside as rounding dust. The GLD files of
# 25 Sep to 23 Oct 2026 carry, beside the metal line, one of 0 to 2.8e-8 of NAV (r14, 2026-10-24 run).
SINGLE_LINE_DUST_LIMIT = 1e-6
_CASH_LIKE_CLASSES = {'cash', 'short_term', 'short term'}
EQUITY_CLASS_MIN_NAV = 0.98
# FX forwards and non-USD cash overdrafts in an equity fund (IEMG on 2026-10-29: under 0.01% of NAV).
EQUITY_CLASS_SHORT_LIMIT = 0.001
ACCEPTED_STATUSES = frozenset({
    'PLAIN_EQUITY_HOLDINGS_ALREADY_ADMITTED', 'SIGNED_FIXED_INCOME_SOURCE_CLASS_ACCEPTED',
    'SIGNED_FIXED_INCOME_ACCEPTED_WITH_DECLARED_TOLERANCES', 'SINGLE_LINE_EXCHANGE_TRADED_VEHICLE_PRICED_AS_A_WHOLE'})


def _identity(row):
    """Byte-stable normalized identity; duplicates are counted with multiplicity."""
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _declared_tolerances(rows, signed, *, signed_total, negative_cash, negative_cash_nonusd, fixed_income):
    """Measured uses of the second-tier tolerances, or None when any limit is exceeded."""
    if fixed_income < 0.90 or negative_cash < -0.001 or negative_cash_nonusd or signed['conflicting_identity_group_count']:
        return None
    if not signed['repeated_ticker_groups_all_distinguished_by_identifier']:
        return None
    if not 1 - LISTED_NAV_SHORTFALL_LIMIT <= signed_total <= 1.000001:
        return None
    groups = {}
    for row in rows:
        if row.get('weight') is not None:
            groups.setdefault(_source_identity(row), []).append(float(row['weight']))
    used = []
    shorts = [row for row in rows if row.get('weight') is not None and float(row['weight']) < 0
              and str(row.get('asset_class') or '').strip().lower() != 'cash']
    # A short row is accepted only inside a same-identifier group that is not
    # net short (a TBA sale netted against a purchase of the same CUSIP).
    for row in shorts:
        members = groups[_source_identity(row)]
        if not any(w > 0 for w in members) or math.fsum(members) < -1e-12:
            return None
    if shorts:
        used.append({'tolerance': 'SHORT_ROWS_NETTED_BY_SAME_IDENTIFIER_LONGS', 'limit': 'no group net short',
                     'measured': {'rows': len(shorts), 'gross_short_nav': math.fsum(float(r['weight']) for r in shorts)},
                     'treatment': 'Kept with their sign; each nets against a long row of the same identifier.'})
    additive = math.fsum(math.fsum(ws) for ws in groups.values() if len(ws) > 1 and all(w > 0 for w in ws))
    if additive > ADDITIVE_SAME_IDENTITY_LIMIT:
        return None
    if additive:
        used.append({'tolerance': 'REPEATED_ROWS_OF_ONE_IDENTIFIER_SUMMED', 'limit': ADDITIVE_SAME_IDENTITY_LIMIT,
                     'measured': {'nav': additive}, 'treatment': 'Summed as lots of one security; at most this much NAV could be double counted.'})
    collisions = [r for r in signed['self_reference_rows'] if r['status'] != 'TICKER_FIELD_COLLISION_RESOLVED_BY_AUTHORITATIVE_FUND_IDENTIFIER']
    collision_nav = math.fsum(abs(float(r['weight'])) for r in collisions)
    if (any(r['status'] != 'UNRESOLVED_TICKER_FIELD_COLLISION_NO_AUTHORITATIVE_FUND_IDENTIFIER' for r in collisions)
            or collision_nav > SELF_TICKER_COLLISION_LIMIT):
        return None
    if collisions:
        used.append({'tolerance': 'TICKER_FIELD_EQUALS_FUND_KEPT_UNRESOLVED', 'limit': SELF_TICKER_COLLISION_LIMIT,
                     'measured': {'rows': len(collisions), 'nav': collision_nav},
                     'treatment': 'The row carries its own check-digit-valid identifier; it is kept in its own segment and given no duration.'})
    unidentified = [r for r in rows if r.get('weight') is not None and float(r['weight']) > 0
                    and not str(r.get('constituent_ticker') or '').strip() and not str(r.get('isin') or '').strip()
                    and not str(r.get('figi') or '').strip()]
    cash_like = math.fsum(float(r['weight']) for r in unidentified
                          if str(r.get('asset_class') or '').strip().lower() in _CASH_LIKE_CLASSES)
    other = math.fsum(float(r['weight']) for r in unidentified) - cash_like
    if other > 0.001 or cash_like > CASH_LIKE_UNIDENTIFIED_LIMIT:
        return None
    if cash_like:
        used.append({'tolerance': 'UNIDENTIFIED_CASH_OR_SHORT_TERM_ROWS', 'limit': CASH_LIKE_UNIDENTIFIED_LIMIT,
                     'measured': {'nav': cash_like}, 'treatment': 'Kept as cash and equivalents; no identity or duration is inferred.'})
    if signed_total < 0.999:
        used.append({'tolerance': 'LISTED_NAV_SHORTFALL', 'limit': LISTED_NAV_SHORTFALL_LIMIT,
                     'measured': {'unlisted_nav': 1.0 - signed_total},
                     'treatment': 'Not renormalized; the duration range already spans the unlisted share at zero duration.'})
    return used if used else None


def _single_line_vehicle(rows, instrument):
    """One unidentified line holding the whole NAV of an exchange-traded vehicle, or None.

    Returns the line's NAV weight, and the count and absolute NAV of dust lines set aside.
    """
    if str((instrument or {}).get('type') or '').upper() != 'ETV':
        return None
    if any(r.get('weight') is None for r in rows):
        return None
    lines = [r for r in rows if abs(float(r['weight'])) > SINGLE_LINE_DUST_LIMIT]
    dust = [abs(float(r['weight'])) for r in rows if 0 < abs(float(r['weight'])) <= SINGLE_LINE_DUST_LIMIT]
    if (len(lines) != 1 or math.fsum(dust) > SINGLE_LINE_DUST_LIMIT
            or not 1 - SINGLE_LINE_NAV_BAND <= float(lines[0]['weight']) <= 1 + SINGLE_LINE_NAV_BAND):
        return None
    norm = _normalize_constituent(lines[0], '')
    if norm['asset_type'] in {'stock', 'etf', 'cash'} or str(lines[0].get('constituent_ticker') or '').strip():
        return None
    return float(lines[0]['weight']), len(dust), math.fsum(dust)


def review_source_class_snapshot(snapshot: Mapping, *, symbol: str | None = None,
                                 instrument: Mapping | None = None) -> dict:
    """Review a captured source batch, preserving distinct validity domains.

    This intentionally does not rewrite ``reliable`` or ``quality_issues`` in
    the original stock/ETF holdings parser. Fixed-income rows only support a
    signed source-class allocation after a separate class-aware source replay.
    ``instrument`` is the provider's ticker metadata; only its ``type`` is read.
    """
    sym = str(symbol or (snapshot or {}).get('fund_symbol') or '').upper()
    out = {
        'schema': SCHEMA, 'fund_symbol': sym,
        'status': 'SOURCE_CLASS_UNVERIFIED',
        'holdings_accepted_for_assessment': False,
        'declared_tolerances_used': [],
        'eligible_for_source_class_attribution': False,
        'eligible_for_equity_issuer_lookthrough': False,
        'original_plain_holdings_reliable': (snapshot or {}).get('reliable') is True if isinstance(snapshot, Mapping) else False,
        'original_issues_retained': list(((snapshot or {}).get('metadata') or {}).get('quality_issues') or [])
            if isinstance(snapshot, Mapping) else [],
        'weights_reweighted_or_clipped': False,
        'source_identity_verification': 'NOT_A_SECURITY_MASTER',
        'limitations': [
            'Signed provider composition is not full fund NAV or exact issue identity.',
            'Cash liabilities are not equity return exposures; bond-level causal returns are not inferred.',
            'This review never upgrades the existing plain long-equity look-through validator.',
        ],
    }
    if not isinstance(snapshot, Mapping) or snapshot.get('fund_symbol') != sym:
        out['reason'] = 'Missing or fund-mismatched snapshot.'
        return out
    try:
        metadata = snapshot['metadata']
        if (snapshot.get('weight_basis') != 'fraction_of_nav' or snapshot.get('complete') is not True
                or metadata.get('selection_provisional') is True
                or not isinstance(metadata.get('selected_batch'), list)):
            raise ValueError('Incomplete, provisional or differently weighted source batch')
        # Recheck every wire byte and the archived transport receipt. A saved
        # hash alone is integrity evidence, not cryptographic provider authorship.
        for page in metadata.get('pages', []):
            raw_path = Path(page['raw_path'])
            meta_path = raw_path.with_suffix('.meta.json')
            receipt = json.loads(meta_path.read_text(encoding='utf-8'))
            uri = urlsplit(str(receipt['url']))
            if (receipt.get('status') != 200 or receipt.get('fixture') is not False
                    or receipt.get('sha256') != page.get('sha256')
                    or receipt.get('retrieved_at') != page.get('retrieved_at')
                    or uri.scheme != 'https' or uri.hostname != 'api.massive.com'
                    or uri.path != '/etf-global/v1/constituents'):
                raise ValueError('Recorded HTTPS constituent provenance is not accepted')
        rows = selected_rows_from_holdings(snapshot)
        norm = [_normalize_constituent(row, sym) for row in rows]
        kept = Counter(_identity(row) for row in snapshot.get('holdings', []))
        kept.update(_identity(row) for row in metadata.get('excluded_negligible_offset_rows') or [])
        if Counter(_identity(row) for row in norm) != kept:
            raise ValueError('Normalized selected records do not equal the archived raw source rows')
        signed = analyze_holdings_snapshot(snapshot)
        weights = [float(row['weight']) for row in rows if row.get('weight') is not None]
        # The source class is read from the provider, not guessed from a ticker.
        fixed_income = math.fsum(float(row['weight']) for row in rows
                                 if row.get('weight') is not None
                                 and str(row.get('asset_class') or '').strip().lower() in {'fixed income', 'fixed_income'})
        negative_noncash = [row for row in rows if row.get('weight') is not None and float(row['weight']) < 0
                            and str(row.get('asset_class') or '').strip().lower() != 'cash']
        negative_cash_nonusd = [row for row in rows if row.get('weight') is not None and float(row['weight']) < 0
                                 and str(row.get('asset_class') or '').strip().lower() == 'cash'
                                 and str(row.get('currency_traded') or '').upper() != 'USD']
        negative_cash = math.fsum(float(row['weight']) for row in rows if row.get('weight') is not None
                                  and float(row['weight']) < 0
                                  and str(row.get('asset_class') or '').strip().lower() == 'cash')
        unidentified = math.fsum(float(row['weight']) for row in rows if row.get('weight') is not None
                                 and float(row['weight']) > 0 and not str(row.get('constituent_ticker') or '').strip()
                                 and not str(row.get('isin') or '').strip()
                                 and not str(row.get('figi') or '').strip())
        signed_total = math.fsum(weights)
        if not math.isclose(signed_total, float(signed['reported_weight_sum']), rel_tol=0, abs_tol=1e-9):
            raise ValueError('Signed NAV sum differs from independently analyzed raw constituents')
        # This is a *weaker*, independent information tier. The asset-class
        # labels are source facts and signed class totals can be audited even
        # where CUSIP / TBA offsetting and generic tickers do not establish
        # individual issuer identities. Never promote the aggregate tier into
        # security look-through or source-class attribution for economic P&L.
        class_weights = {}
        for row in rows:
            if row.get('weight') is None:
                continue
            label = str(row.get('asset_class') or 'UNCLASSIFIED').strip() or 'UNCLASSIFIED'
            class_weights.setdefault(label, []).append(float(row['weight']))
        signed_class_total = {label: math.fsum(series) for label, series in sorted(class_weights.items())}
        class_only_eligible = (len(weights) == len(rows)
                               and abs(signed_total - 1.0) <= 0.001
                               and 'UNCLASSIFIED' not in signed_class_total)
        out.update(source_effective_date=snapshot.get('as_of'), source_processed_date=snapshot.get('processed_date'),
                   source_pages_verified=len(metadata['pages']), selected_rows_verified=len(rows),
                   source_signed_class_total_by_provider_label=signed_class_total,
                   source_class_aggregate_reconciles_with_nav=class_only_eligible,
                   source_class_aggregate_does_not_admit_issuer_lookthrough=True,
                   source_signed_nav_sum=signed_total, signed_usd_cash_liability_nav=negative_cash,
                   source_fixed_income_nav=fixed_income, noncash_short_rows=len(negative_noncash),
                   repeated_ticker_groups=signed['repeated_ticker_groups'],
                   repeated_ticker_groups_distinguished_by_identifiers=signed['repeated_ticker_groups_all_distinguished_by_identifier'],
                   offsetting_same_id_groups=signed['identity_group_reviews'].get('OFFSETTING_LONG_SHORT_SAME_IDENTITY', 0),
                   ambiguous_additive_same_id_groups=signed['identity_group_reviews'].get('MULTIPLE_ROWS_SAME_IDENTITY_ADDITIVE', 0),
                   negative_nonusd_cash_rows=len(negative_cash_nonusd),
                   conflicting_identity_groups=signed['conflicting_identity_group_count'],
                   unresolved_self_ticker_rows=len(signed['self_reference_rows']),
                   positive_unidentified_source_identity_nav=unidentified,
                   original_issues_retained=list(metadata.get('quality_issues') or []),
                   source_record_hashes_verified=True,
                   recorded_fund_nav_residual=1.0-signed_total)
        tolerances = _declared_tolerances(rows, signed, signed_total=signed_total, negative_cash=negative_cash,
                                          negative_cash_nonusd=negative_cash_nonusd, fixed_income=fixed_income)
        single_line = _single_line_vehicle(rows, instrument)
        small_shorts = math.fsum(abs(float(row['weight'])) for row in [*negative_noncash, *negative_cash_nonusd])
        if snapshot.get('reliable') is True:
            out.update(status='PLAIN_EQUITY_HOLDINGS_ALREADY_ADMITTED',
                       eligible_for_equity_issuer_lookthrough=True,
                       reason='Original plain-instrument validator admitted the source batch. No override made.')
        elif (fixed_income >= 0.90 and 0.999 <= signed_total <= 1.000001
              and negative_cash >= -0.001 and not negative_noncash and not negative_cash_nonusd
              and signed['repeated_ticker_groups_all_distinguished_by_identifier']
              and not signed['conflicting_identity_group_count']
              and not signed['self_reference_rows']
              and signed['identity_group_reviews'].get('OFFSETTING_LONG_SHORT_SAME_IDENTITY', 0) == 0
              and signed['identity_group_reviews'].get('MULTIPLE_ROWS_SAME_IDENTITY_ADDITIVE', 0) == 0
              and unidentified <= 0.001):
            out.update(status='SIGNED_FIXED_INCOME_SOURCE_CLASS_ACCEPTED',
                       eligible_for_source_class_attribution=True,
                       reason=('Provider fixed-income classifications and signed rows are internally source-consistent; '
                               'repeated generic bond tickers refer to different structured identifiers. '
                               'Negative cash is retained as a liability. Individual equity look-through remains blocked.'))
        elif tolerances is not None:
            out.update(status='SIGNED_FIXED_INCOME_ACCEPTED_WITH_DECLARED_TOLERANCES',
                       eligible_for_source_class_attribution=True, declared_tolerances_used=tolerances,
                       reason=('Provider fixed-income classifications and signed rows are source-consistent within the declared '
                               'tolerances listed with their measured values. Nothing is clipped or renormalized. '
                               'Individual equity look-through remains blocked.'))
        elif single_line is not None:
            line_nav, dust_rows, dust_nav = single_line
            used = [{'tolerance': 'SINGLE_LINE_WITHIN_NAV_BAND', 'limit': SINGLE_LINE_NAV_BAND,
                     'measured': {'line_nav': line_nav},
                     'treatment': 'Unlisted liabilities such as accrued fees can put the line slightly above NAV.'}]
            if dust_rows:
                used.append({'tolerance': 'DUST_LINES_SET_ASIDE', 'limit': SINGLE_LINE_DUST_LIMIT,
                             'measured': {'rows': dust_rows, 'abs_nav': dust_nav},
                             'treatment': 'Set aside, not netted, reweighted or looked through: under one dollar per million of NAV.'})
            out.update(status='SINGLE_LINE_EXCHANGE_TRADED_VEHICLE_PRICED_AS_A_WHOLE', single_line_nav=line_nav,
                       declared_tolerances_used=used,
                       reason=('The provider types this fund as an exchange-traded vehicle and lists one holding line, without '
                               'ticker, identifier or class, for the whole NAV. No listed company or fund is named, so there is '
                               'nothing to look through; the fund is priced through its own measured beta. What the line is '
                               '(for example a metal) is not inferred.'))
        elif (class_only_eligible and signed_class_total.get('Equity', 0.0) >= EQUITY_CLASS_MIN_NAV
              and fixed_income <= 0.01 and small_shorts <= EQUITY_CLASS_SHORT_LIMIT
              and negative_cash >= -0.01 and unidentified <= 0.001
              and not signed['conflicting_identity_group_count'] and not signed['self_reference_rows']):
            # Class-level evidence only: the provider labels at least 98% of a NAV
            # that reconciles as Equity. Repeated tickers (e.g. one issuer's local
            # and depositary lines) block issuer look-through, not the fund class.
            if small_shorts:
                out['declared_tolerances_used'] = [{
                    'tolerance': 'SMALL_FX_OR_NONUSD_CASH_SHORTS', 'limit': EQUITY_CLASS_SHORT_LIMIT,
                    'measured': {'gross_short_nav': small_shorts},
                    'treatment': 'Kept with their sign; too small to change the fund class.'}]
            out.update(status='SIGNED_EQUITY_SOURCE_CLASS_AGGREGATE_ACCEPTED',
                       eligible_for_equity_class_composition=True,
                       reason=('Provider asset classes put at least 98% of a reconciled NAV in Equity, with no noncash '
                               'short rows. This classifies the fund as an equity fund priced through its own measured '
                               'returns; individual issuer look-through remains blocked.'))
        else:
            out.update(status='SOURCE_CLASS_STRUCTURAL_GAPS_RETAINED',
                       reason=('Unidentified source identity, noncash short/offsetting positions, conflicting or self-referential '
                               'identifiers, inadequate fixed-income classification, or material NAV-basis disagreement.'))
        out['holdings_accepted_for_assessment'] = out['status'] in ACCEPTED_STATUSES
        return out
    except (OSError, KeyError, TypeError, ValueError, OverflowError, FundCompositionError, json.JSONDecodeError):
        out['reason'] = 'Recorded batch, its authenticated byte receipts, or original normalized rows could not be reconciled.'
        return out


def review_portfolio_source_classes(holdings: Mapping, instruments: Mapping | None = None) -> list[dict]:
    return [review_source_class_snapshot(value, symbol=symbol, instrument=(instruments or {}).get(symbol))
            for symbol, value in sorted((holdings or {}).items())]

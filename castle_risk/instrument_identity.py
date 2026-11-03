"""Source-bound original-position identity, before portfolio model routing.

The UI's type selector is a declaration, not a security master. Only verified
provider CS/ETF records can automatically correct it. No ticker registry,
marketing-name heuristic or fresh network call is used here. Original source
bytes are checked locally; only safe identity/provenance fields are returned.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
from typing import Mapping
from urllib.parse import parse_qsl, urlsplit

from .instrument_exposures import _metadata_fields, evaluation_instant
from .portfolio import parse_portfolio


class InstrumentIdentityError(ValueError):
    """A safe user-facing error, containing only validated portfolio symbols."""


def _dated_metadata(record, symbol, now):
    """Bind the as-of claim to the exact source request and receipt clock."""
    if not isinstance(record, Mapping):
        return None
    checked = _metadata_fields(record, symbol, now)
    if checked.get('status') == 'METADATA_REJECTED':
        return checked
    try:
        query = dict(parse_qsl(urlsplit(record['source_url']).query))
        requested = query.get('date')
        declared = record.get('metadata_as_of')
        if declared is not None and declared != requested:
            raise ValueError('Normalized as-of date differs from the source request.')
        receipt_day = evaluation_instant(record['retrieved_at']).date()
        as_of = date.fromisoformat(requested) if requested is not None else receipt_day
        # Live acquisition asks for the current date. Permit the previous UTC
        # date solely so a run crossing midnight keeps its original request.
        if not now.date() - timedelta(days=1) <= as_of <= min(now.date(), receipt_day):
            raise ValueError('Metadata is not from the capture date or preceding UTC date.')
    except (TypeError, ValueError, KeyError):
        return {'status': 'METADATA_REJECTED', 'reason': 'Metadata as-of date, request or capture chronology is invalid.'}
    return {**checked, 'metadata_as_of': as_of.isoformat(),
            'as_of_basis': 'SOURCE_REQUEST_DATE' if requested else 'CURRENT_REFERENCE_RECEIPT_DATE'}


def resolve_portfolio_types(portfolio_rows, *, instrument_metadata=None,
                            evaluated_at, require_auto_resolution=True):
    """Resolve supported types while retaining declarations and blocked weight.

    The server encodes automatic input as the temporary parser-compatible
    ``asset_type='stock', input_asset_type='auto'``. That temporary value is
    never authorization to model a stock: unresolved automatic rows either
    raise before research/acquisition continues or remain in ``blocked_symbols``
    for the read-only replay/model guard. Manual declarations with no metadata
    preserve the established explicitly-unverified behavior.
    """
    original = deepcopy(list(portfolio_rows))
    portfolio = parse_portfolio(original)
    now = evaluation_instant(evaluated_at)
    metadata = instrument_metadata if isinstance(instrument_metadata, Mapping) else {}
    rows, audits, blocked, unresolved_auto = [], [], [], []
    for position, supplied in zip(portfolio.positions, original):
        declared = supplied.get('input_asset_type', position.asset_type)
        if declared not in {'auto', 'stock', 'etf', 'cash'}:
            raise InstrumentIdentityError('Choose automatic detection, stock, fund or USD cash.')
        if (declared == 'cash') != (position.asset_type == 'cash'):
            raise InstrumentIdentityError('A cash declaration cannot be substituted for a stock or fund identity.')
        row = {'symbol': position.symbol, 'weight': position.weight,
               'asset_type': position.asset_type}
        if 'input_asset_type' in supplied:
            row['input_asset_type'] = declared
        item = {'symbol': position.symbol, 'weight': position.weight,
                'declared_asset_type': declared, 'resolved_asset_type': position.asset_type,
                'status': 'DECLARED_TYPE_NOT_SOURCE_VERIFIED', 'blocked': False,
                'source_verified': False, 'provider_instrument_type': None,
                'source': None}
        if position.asset_type == 'cash' and declared == 'cash':
            item.update(status='EXPLICIT_USD_CASH_DECLARATION')
            # The public form additionally restricts cash symbols. The core
            # parser's historical explicit cash declarations stay compatible.
            rows.append(row)
            audits.append(item)
            continue
        raw = metadata.get(position.symbol)
        checked = _dated_metadata(raw, position.symbol, now)
        accepted = checked is not None and checked.get('status') != 'METADATA_REJECTED'
        raw_kind = str((raw or {}).get('instrument_type', (raw or {}).get('type', ''))).strip().upper() if isinstance(raw, Mapping) else ''
        kind = checked.get('instrument_type') if accepted else raw_kind or None
        item['provider_instrument_type'] = kind
        verified = accepted and checked.get('local_response_verified') is True
        if accepted:
            item['source'] = {key: checked.get(key) for key in (
                'provider', 'source_url', 'retrieved_at', 'source_response_sha256',
                'metadata_as_of', 'as_of_basis')}
        # ADRC: a USD-listed American depositary receipt of common stock (e.g. TSM)
        # is valued like a stock; its home-market risk enters through events.
        supported = {'CS': 'stock', 'ADRC': 'stock', 'ETF': 'etf'}.get(kind)
        # Rejected/unverified provider claims do not supply an automatic type.
        # A conflicting claim still blocks a declared equity fallback.
        conflict = bool(kind and (supported != position.asset_type))
        bad_scope = accepted and (checked.get('active') is False or
            checked.get('currency_name') is not None and
            str(checked['currency_name']).strip().lower() not in {'usd', 'us dollar', 'united states dollar'})
        missing_auto_currency = declared == 'auto' and accepted and checked.get('currency_name') is None
        if verified and supported and not bad_scope and not missing_auto_currency:
            row['asset_type'] = supported
            if declared == 'auto':
                status = 'AUTO_RESOLVED_FROM_VERIFIED_SOURCE'
            elif declared != supported:
                row['input_asset_type'] = declared
                status = 'DECLARATION_CORRECTED_FROM_VERIFIED_SOURCE'
            else:
                status = 'DECLARATION_CONFIRMED_BY_VERIFIED_SOURCE'
            item.update(status=status, resolved_asset_type=supported, source_verified=True)
        elif kind == 'ETV' and (position.asset_type == 'etf' and declared != 'auto' or
                                declared == 'auto' and verified and not missing_auto_currency) and not bad_scope:
            # Preserve the existing source-class/holdings review of a fund or
            # exchange-traded vehicle (e.g. a gold trust). ETV is not a verified
            # ETF: it cannot correct explicit stock input or bypass its separate
            # class gate, but automatic detection routes it to that same review.
            if declared == 'auto':
                row['asset_type'] = 'etf'
            item.update(status='DECLARED_FUND_REQUIRES_SOURCE_CLASS_REVIEW' if declared != 'auto'
                        else 'AUTO_DETECTED_VEHICLE_REQUIRES_SOURCE_CLASS_REVIEW',
                        resolved_asset_type='etf' if declared == 'auto' else item.get('resolved_asset_type'))
        elif bad_scope or accepted and not supported or conflict or declared == 'auto':
            reason = ('INACTIVE_OR_NON_USD_INSTRUMENT' if bad_scope else
                      'USD_CURRENCY_NOT_CONFIRMED' if missing_auto_currency else
                      'UNSUPPORTED_PROVIDER_INSTRUMENT_TYPE' if accepted and not supported else
                      'CONFLICTING_IDENTITY_NOT_SOURCE_VERIFIED' if conflict else
                      'AUTOMATIC_IDENTITY_NOT_SOURCE_VERIFIED')
            item.update(status=reason, blocked=True, resolved_asset_type=None)
            blocked.append(position.symbol)
            if declared == 'auto':
                unresolved_auto.append(position.symbol)
        elif checked is not None and checked.get('status') == 'METADATA_REJECTED':
            item.update(status='DECLARED_TYPE_METADATA_REJECTED')
        rows.append(row)
        audits.append(item)
    audit = {
        'status': ('IDENTITY_REVIEW_REQUIRED' if blocked else
                   'SOURCE_TYPES_RESOLVED' if all(r['source_verified'] or r['status'] == 'EXPLICIT_USD_CASH_DECLARATION' for r in audits)
                   else 'PARTIAL_IDENTITY_VERIFICATION'),
        'positions': audits,
        'corrections': [deepcopy(row) for row in audits if row['status'] == 'DECLARATION_CORRECTED_FROM_VERIFIED_SOURCE'],
        'blocked_symbols': list(blocked),
        'no_symbol_name_inference': True,
        'weights_changed': False,
        'scope': 'Original instrument identity only; ETF type does not establish underlying asset class, holdings coverage or duration.',
    }
    if unresolved_auto and require_auto_resolution:
        names = ', '.join(unresolved_auto)
        raise InstrumentIdentityError(
            f'Could not verify a supported USD stock or ETF for {names}. Check the ticker and provider access; no stock type was assumed.')
    return {'portfolio_rows': rows, 'audit': audit, 'blocked_symbols': blocked}

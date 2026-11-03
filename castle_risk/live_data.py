"""Read-only acquisition for the portfolio assessment; raw Massive data stays private.

The existing secret-safe transport, retry limits and receipt format are reused.
Provider fields are preserved, not inferred from a fund name or ticker whitelist.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

from .adapters import DataAccessError, MASSIVE_BASE, MassiveClient, sanitize_url
from .portfolio import parse_portfolio
from .instrument_identity import resolve_portfolio_types
from .source_class_quality import review_source_class_snapshot

# Same 0.1%-of-NAV tolerance the source-class review applies to negative cash.
NEGLIGIBLE_OFFSET_NAV = 0.001
ACCEPTED_HOLDINGS_STATUSES = frozenset({'RELIABLE_SNAPSHOT', 'ACCEPTED_BY_DECLARED_HOLDINGS_REVIEW'})
_WEIGHT_ISSUE = 'NEGATIVE_LEVERED_OR_PERCENT_SCALED_WEIGHT'


def exclude_negligible_offsets(snapshot: dict) -> dict:
    """Keep tiny negative derivative or unclassified rows out of look-through, and say so.

    The provider parser fails a whole fund on any negative weight. An index-futures
    mark of -0.0014% of NAV (QQQ, 2026-10-22) then removes every constituent from
    the look-through. When negative non-cash, non-stock rows are the fund's only
    issue and total at most 0.1% of NAV, they are listed under ``metadata`` and left
    out; the fund is then accepted. Negative cash, short stocks, weights above one,
    larger offsets or any other issue still fail closed. Nothing is rescaled.
    """
    metadata = snapshot.get('metadata') or {}
    holdings = snapshot.get('holdings') or []
    metadata.setdefault('excluded_negligible_offset_rows', [])
    metadata.setdefault('excluded_offset_nav', 0.0)
    if (snapshot.get('reliable') is True or snapshot.get('complete') is not True
            or set(metadata.get('quality_issues') or []) != {_WEIGHT_ISSUE}):
        return snapshot
    flagged = [row for row in holdings if _WEIGHT_ISSUE in (row.get('quality_issues') or [])]
    eligible = [row for row in flagged if row.get('weight') is not None and row['weight'] < 0
                and row.get('asset_type') == 'other' and str(row.get('asset_class') or '').strip().lower() != 'cash'
                and row.get('quality_issues') == [_WEIGHT_ISSUE]]
    offset_nav = -math.fsum(row['weight'] for row in eligible)
    if not eligible or len(eligible) != len(flagged) or offset_nav > NEGLIGIBLE_OFFSET_NAV:
        return snapshot
    kept = [row for row in holdings if row not in eligible]
    if not kept:
        return snapshot
    weight_sum = math.fsum(row['weight'] for row in kept if row.get('weight') is not None)
    if weight_sum > 1 + 1e-12:
        return snapshot
    metadata.update(excluded_negligible_offset_rows=eligible, excluded_offset_nav=offset_nav,
                    original_quality_issues=list(metadata.get('quality_issues') or []), quality_issues=[],
                    weight_sum=weight_sum, unreported_nav_weight=max(0.0, 1.0 - weight_sum))
    metadata['quality_warnings'] = list(metadata.get('quality_warnings') or []) + [
        f'{len(eligible)} negative derivative or unclassified row(s), {offset_nav:.4%} of NAV in total, '
        'kept out of look-through as a negligible offset']
    return {**snapshot, 'holdings': kept, 'reliable': bool(kept), 'metadata': metadata}


def _symbol(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r'[A-Z0-9][A-Z0-9.\-]{0,19}', value.upper()):
        raise ValueError('Unsupported provider ticker syntax')
    return value.upper()


def _provenance(meta: dict) -> dict:
    return {'provider': 'Massive', 'source_url': meta['url'],
            'retrieved_at': meta['retrieved_at'], 'source_response_sha256': meta['sha256'],
            'raw_path': meta['raw_path'], 'fixture': meta['fixture']}


class _DatedHoldingsView:
    """Reuse the original normalizer with a narrower request, never its budget.

    The parent owns transport, receipt saving, retries, timeout and request count.
    A processed-date filter is attached to the first request; provider cursors
    are followed unchanged and every returned row must respect the exact pair.
    """

    def __init__(self, parent, *, effective=None, processed=None, reserve_requests=0):
        self.parent, self.effective, self.processed = parent, effective, processed
        self.data_dir = parent.data_dir
        self.reserve_requests = reserve_requests

    @property
    def request_count(self):
        return self.parent.request_count

    def _authenticated_get(self, url):
        if self.parent.request_count + self.parent.max_retries + 1 > self.parent.max_requests - self.reserve_requests:
            raise DataAccessError('Holdings search stopped at its existing request budget and reserved remaining operations')
        parts = urlsplit(url)
        query = dict(parse_qsl(parts.query))
        if self.processed is not None and 'cursor' not in query:
            query['processed_date'] = self.processed
            url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ''))
        data, meta = self.parent._authenticated_get(url)
        rows = data.get('results')
        if isinstance(rows, list) and self.processed is not None:
            if any(not isinstance(row, dict) or row.get('effective_date') != self.effective
                   or row.get('processed_date') != self.processed for row in rows):
                raise DataAccessError('Provider response did not honor the exact ETF effective/processed date pair')
        return data, meta


class MassiveAssessmentClient(MassiveClient):
    """Adds documented metadata, ETF taxonomy and adjusted daily bars to holdings."""

    def ticker_details(self, ticker: str, *, as_of: date) -> dict:
        ticker = _symbol(ticker)
        url = MASSIVE_BASE + '/v3/reference/tickers/' + quote(ticker, safe='')
        data, meta = self._authenticated_get(url + '?' + urlencode({'date': as_of.isoformat()}))
        row = data.get('results')
        if data.get('status', 'OK') != 'OK' or not isinstance(row, dict) or row.get('ticker') != ticker:
            raise DataAccessError('Ticker metadata status, identity mismatch or missing result')
        return {**{k: row.get(k) for k in ('ticker', 'name', 'type', 'currency_name',
                                         'market', 'locale', 'sic_description', 'description',
                                         'cik', 'composite_figi', 'share_class_figi')},
                **{k: row[k] for k in ('active', 'asset_class') if k in row},
                **_provenance(meta), 'metadata_as_of': as_of.isoformat()}

    def issuer_siblings(self, cik: str, *, as_of: date, limit: int = 50) -> dict:
        """US-listed securities sharing one SEC registrant CIK (reference data, not a security master).

        Used only to annotate same-issuer share classes (for example GOOGL and
        GOOG) that are already accepted ledger rows. It never adds NAV.
        """
        if not isinstance(cik, str) or not cik.isdigit() or len(cik) != 10:
            raise ValueError('A ten-digit SEC CIK string is required')
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError('Invalid sibling limit')
        url = MASSIVE_BASE + '/v3/reference/tickers?' + urlencode(
            {'cik': cik, 'active': 'true', 'market': 'stocks', 'limit': limit, 'date': as_of.isoformat()})
        data, meta = self._authenticated_get(url)
        rows = data.get('results')
        if data.get('status', 'OK') != 'OK' or not isinstance(rows, list):
            raise DataAccessError('Issuer sibling lookup status or shape rejected')
        tickers = []
        for row in rows:
            if (isinstance(row, dict) and row.get('cik') == cik and isinstance(row.get('ticker'), str)
                    and row['ticker'].strip()):
                tickers.append({'ticker': row['ticker'].strip().upper(), 'type': row.get('type'),
                                'name': row.get('name'), 'primary_exchange': row.get('primary_exchange')})
        return {'cik': cik, 'tickers': sorted(tickers, key=lambda r: r['ticker']),
                'complete': not data.get('next_url'), **_provenance(meta),
                'metadata_as_of': as_of.isoformat(),
                'identity_scope': 'MASSIVE_REFERENCE_SAME_SEC_CIK_NOT_INDEPENDENT_SECURITY_MASTER'}

    def holdings_for_assessment(self, ticker: str, *, as_of_date: str, max_pages: int = 5,
                                lookback_days: int = 30, max_date_probes: int = 3,
                                reserve_requests: int = 0,
                                allow_expanded_pages: bool = False) -> dict:
        """Search bounded dates after a one-page range cannot finish.

        Complete short range responses keep the unchanged original path. A large
        response is an inventory probe, not a usable snapshot. The fallback
        probes recent effective dates, then observed dated candidates, and reads
        one exact effective/processed pair within the selected page budget.
        Bounded observed selection is never described as globally latest.
        """
        ticker = _symbol(ticker)
        if (type(allow_expanded_pages) is not bool
                or isinstance(max_pages, bool) or not isinstance(max_pages, int)
                or not 1 <= max_pages <= (20 if allow_expanded_pages else 5)
                or isinstance(lookback_days, bool) or not isinstance(lookback_days, int) or not 1 <= lookback_days <= 30
                or isinstance(max_date_probes, bool) or not isinstance(max_date_probes, int) or not 1 <= max_date_probes <= lookback_days
                or isinstance(reserve_requests, bool) or not isinstance(reserve_requests, int) or reserve_requests < 0):
            raise ValueError('Invalid bounded holdings search limits')
        cutoff = date.fromisoformat(as_of_date)
        lower = cutoff - timedelta(days=lookback_days - 1)
        before = self.request_count
        probes = []

        def read(*, effective=None, processed=None, pages=1):
            view = _DatedHoldingsView(self, effective=effective, processed=processed,
                                     reserve_requests=reserve_requests)
            attempt_before = self.request_count
            try:
                result = MassiveClient.holdings(view, ticker, as_of_date=as_of_date,
                    effective_date=effective, max_pages=pages, lookback_days=lookback_days)
            except DataAccessError as exc:
                # Transport errors are already secret-safe. Retain the failed
                # pair and actual attempts even when the initial provisional
                # range is the only snapshot that can be returned.
                probes.append({'effective_date': effective, 'processed_date': processed,
                    'status': 'UNAVAILABLE', 'reason': str(exc),
                    'requests': self.request_count - attempt_before})
                raise
            if processed is not None:
                result['metadata']['query']['processed_date'] = processed
            probes.append({'effective_date': effective, 'processed_date': processed,
                'complete': result['complete'], 'reliable': result['reliable'],
                'observed_selected_batch': result['metadata']['selected_batch'],
                'selected_row_count': result['metadata']['selected_raw_row_count'],
                'quality_issues': result['metadata']['quality_issues'],
                'requests': result['metadata']['request_count']})
            return result

        def finish(result, *, reason, exhaustive):
            result['metadata']['acquisition_search'] = {
                'strategy': 'BOUNDED_EXACT_DATE_PAIR_AFTER_RANGE_PROBE',
                'window_start': lower.isoformat(), 'window_end': cutoff.isoformat(),
                'date_probes': probes, 'max_recent_effective_dates': max_date_probes,
                'exact_pair_page_cap': max_pages, 'requests_used': self.request_count - before,
                'existing_request_cap': self.max_requests, 'reserved_remaining_requests': reserve_requests,
                'termination': reason, 'latest_within_complete_range_verified': exhaustive,
                'latest_global_verified': False,
                'selection_scope': ('Latest eligible batch in the completed bounded range; no historical publication-time guarantee.'
                    if exhaustive else 'Newest selected pair among the bounded observed/probed candidates; unqueried dates or revisions may exist.'),
                'quality_failures_retained': True,
            }
            result = exclude_negligible_offsets(result)
            stamp = result['retrieved_at'].replace(':', '').replace('.', '')
            digest = hashlib.sha256(json.dumps(result['metadata']['query'], sort_keys=True).encode()).hexdigest()[:12]
            path = self.data_dir / f'holdings_{ticker}_{stamp}_{digest}.normalized.json'
            result['metadata']['normalized_path'] = str(path)
            path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
            return result

        initial = read()
        if initial['complete']:
            # An invalid but complete snapshot is retained as invalid; do not
            # hide its negative weights/conflicts by searching an older date.
            return finish(initial, reason='BOUNDED_RANGE_COMPLETED', exhaustive=True)
        if set(initial['metadata']['quality_issues']) - {
                'INCOMPLETE_QUERY', 'CONFLICTING_DUPLICATE_SECURITY', 'CONFLICTING_CONSTITUENT_RANK',
                'NEGATIVE_LEVERED_OR_PERCENT_SCALED_WEIGHT', 'SELF_REFERENTIAL_FUND_HOLDING',
                'REPORTED_NAV_WEIGHT_EXCEEDS_ONE', 'NO_ELIGIBLE_DATED_BATCH_IN_BOUNDED_WINDOW'}:
            return finish(initial, reason='RANGE_TRANSPORT_OR_SHAPE_FAILURE_RETAINED', exhaustive=False)
        observed = initial['metadata']['candidate_batches']
        observed_dates = sorted({row['effective_date'] for row in observed}, reverse=True)
        recent = [(cutoff - timedelta(days=offset)).isoformat() for offset in range(max_date_probes)]
        effective_candidates = list(dict.fromkeys([*recent, *observed_dates]))
        # A batch still being written can appear in the range probe yet return no
        # rows when read exactly; fall back to the next effective date.
        unreproduced = None
        try:
            for effective in effective_candidates:
                if not lower <= date.fromisoformat(effective) <= cutoff:
                    continue
                if effective in observed_dates:
                    revision_candidates = sorted({row['processed_date'] for row in observed
                                                  if row['effective_date'] == effective}, reverse=True)
                    current = None
                else:
                    current = read(effective=effective)
                    if current['complete']:
                        if current['metadata']['selected_raw_row_count']:
                            return finish(current, reason='OBSERVED_EFFECTIVE_DATE_COMPLETED', exhaustive=False)
                        continue
                    revision_candidates = sorted({row['processed_date'] for row in current['metadata']['candidate_batches']}, reverse=True)
                if not revision_candidates:
                    return finish(current or initial, reason='NO_COMPLETE_DATED_CANDIDATE', exhaustive=False)
                # Revisions can lag effective dates. Probe a few current dates
                # before the latest observed revision, with no sorting request.
                processed_candidates = list(dict.fromkeys([
                    *[day for day in recent if day >= effective and day >= revision_candidates[0]],
                    revision_candidates[0],
                ]))
                for processed in processed_candidates:
                    exact = read(effective=effective, processed=processed, pages=max_pages)
                    if exact['complete'] and not exact['metadata']['selected_raw_row_count']:
                        continue
                    return finish(exact, reason=('EXACT_OBSERVED_PAIR_COMPLETED' if exact['complete']
                        else 'EXACT_PAIR_REMAINS_INCOMPLETE'), exhaustive=False)
                unreproduced = unreproduced or (current or initial)
        except DataAccessError:
            return finish(initial, reason='EXACT_SEARCH_STOPPED_BY_BUDGET_OR_PROVIDER_FAILURE', exhaustive=False)
        if unreproduced is not None:
            return finish(unreproduced, reason='OBSERVED_REVISION_NOT_REPRODUCED', exhaustive=False)
        return finish(initial, reason='NO_ELIGIBLE_BATCH_IN_BOUNDED_PROBES', exhaustive=False)

    def taxonomy(self, ticker: str, *, as_of: date, max_pages: int = 3) -> dict:
        ticker = _symbol(ticker)
        if not 1 <= max_pages <= 20:
            raise ValueError('Invalid taxonomy page limit')
        endpoint = '/etf-global/v1/taxonomies'
        window_start = as_of - timedelta(days=89)
        url = MASSIVE_BASE + endpoint + '?' + urlencode({
            'composite_ticker': ticker, 'effective_date.gte': window_start.isoformat(),
            'effective_date.lte': as_of.isoformat(), 'limit': 5000})
        candidates, pages, seen = [], [], set()
        for _ in range(max_pages):
            if url in seen:
                raise DataAccessError('Taxonomy pagination cycle')
            seen.add(url)
            data, meta = self._authenticated_get(url)
            pages.append(_provenance(meta))
            rows = data.get('results')
            if data.get('status', 'OK') != 'OK' or not isinstance(rows, list):
                raise DataAccessError('Invalid ETF taxonomy response')
            for row in rows:
                if not isinstance(row, dict) or row.get('composite_ticker') != ticker:
                    raise DataAccessError('ETF taxonomy identity mismatch')
                try:
                    effective = date.fromisoformat(row['effective_date'])
                    processed = date.fromisoformat(row['processed_date'])
                except (KeyError, TypeError, ValueError):
                    raise DataAccessError('ETF taxonomy is missing dated provenance') from None
                if window_start <= effective <= processed <= as_of:
                    candidates.append((effective, processed, row, meta))
            next_url = data.get('next_url')
            if not next_url:
                break
            url = self._safe_next(next_url, endpoint)
        else:
            raise DataAccessError('ETF taxonomy pagination incomplete')
        if not candidates:
            raise DataAccessError('No eligible ETF taxonomy batch')
        latest = max((x[0], x[1]) for x in candidates)
        selected = [x for x in candidates if (x[0], x[1]) == latest]
        if len({json.dumps(x[2], sort_keys=True) for x in selected}) != 1:
            raise DataAccessError('Conflicting ETF taxonomy records for the same dated batch')
        _, _, row, meta = selected[0]
        return {**{k: row.get(k) for k in ('composite_ticker', 'asset_class', 'effective_date',
                    'processed_date', 'leverage_style', 'levered_amount', 'focus', 'duration', 'maturity')},
                **_provenance(meta), 'pages': pages}

    @staticmethod
    def _safe_next(value, endpoint):
        if not isinstance(value, str):
            raise DataAccessError('Invalid provider pagination link')
        p = urlsplit(value)
        try:
            safe = (p.scheme == 'https' and p.hostname == 'api.massive.com' and
                    p.port in (None, 443) and not p.username and not p.password and p.path == endpoint)
        except ValueError:
            safe = False
        if not safe:
            raise DataAccessError('Untrusted provider pagination link')
        return sanitize_url(value)

    def treasury_yields(self, *, start: date, end: date, max_pages: int = 8) -> dict:
        """Daily 10-year constant-maturity Treasury yields (percent) from the provider."""
        if start >= end or not 1 <= max_pages <= 20:
            raise ValueError('Invalid yield range or page limit')
        endpoint = '/fed/v1/treasury-yields'
        url = MASSIVE_BASE + endpoint + '?' + urlencode({'date.gte': start.isoformat(), 'date.lte': end.isoformat(),
                                                         'sort': 'date.asc', 'limit': 50000})
        by_day, receipts, seen = {}, [], set()
        for _ in range(max_pages):
            if url in seen:
                raise DataAccessError('Treasury-yield pagination cycle')
            seen.add(url)
            data, meta = self._authenticated_get(url)
            receipts.append(_provenance(meta))
            rows = data.get('results')
            if data.get('status', 'OK') != 'OK' or not isinstance(rows, list):
                raise DataAccessError('Treasury-yield status or shape mismatch')
            for row in rows:
                try:
                    day = date.fromisoformat(str(row['date']))
                    value = row.get('yield_10_year')
                    if value is None:
                        continue
                    value = float(value)
                    if isinstance(row.get('yield_10_year'), bool) or not math.isfinite(value) or not start <= day <= end:
                        raise ValueError
                except (KeyError, TypeError, ValueError):
                    raise DataAccessError('Invalid or out-of-range Treasury-yield row') from None
                if day.isoformat() in by_day and by_day[day.isoformat()] != value:
                    raise DataAccessError('Conflicting Treasury yield for the same date')
                by_day[day.isoformat()] = value
            next_url = data.get('next_url')
            if not next_url:
                break
            url = self._safe_next(next_url, endpoint)
        else:
            raise DataAccessError('Treasury-yield pagination incomplete')
        if not by_day:
            raise DataAccessError('Empty Treasury-yield history')
        return {'series': 'US_TREASURY_10Y_CONSTANT_MATURITY_PERCENT',
                'yields': [{'date': k, 'yield_pct': by_day[k]} for k in sorted(by_day)],
                'start': start.isoformat(), 'end': end.isoformat(), 'receipts': receipts}

    def daily_prices(self, ticker: str, *, start: date, end: date, max_pages: int = 4) -> dict:
        ticker = _symbol(ticker)
        if start >= end or not 1 <= max_pages <= 20:
            raise ValueError('Invalid daily-price range or page limit')
        endpoint = f'/v2/aggs/ticker/{quote(ticker, safe="")}/range/1/day/{start.isoformat()}/{end.isoformat()}'
        url = MASSIVE_BASE + endpoint + '?' + urlencode({'adjusted': 'true', 'sort': 'asc', 'limit': 50000})
        by_day, receipts, seen = {}, [], set()
        for _ in range(max_pages):
            if url in seen:
                raise DataAccessError('Daily-price pagination cycle')
            seen.add(url)
            data, meta = self._authenticated_get(url)
            receipts.append(_provenance(meta))
            if data.get('status', 'OK') != 'OK' or data.get('ticker') != ticker or data.get('adjusted') is not True:
                raise DataAccessError('Daily-price status, identity or adjustment mismatch')
            rows = data.get('results')
            if not isinstance(rows, list):
                raise DataAccessError('No daily prices in provider response')
            count = data.get('resultsCount')
            if count is not None and (isinstance(count, bool) or not isinstance(count, int)
                                      or count != len(rows)):
                raise DataAccessError('Daily-price result count does not match returned bars')
            for row in rows:
                try:
                    if isinstance(row['t'], bool) or isinstance(row['c'], bool):
                        raise ValueError
                    day = datetime.fromtimestamp(float(row['t']) / 1000, tz=timezone.utc).date()
                    close = float(row['c'])
                    if not math.isfinite(close) or close <= 0 or not start <= day <= end:
                        raise ValueError
                except (KeyError, TypeError, ValueError, OverflowError, OSError):
                    raise DataAccessError('Invalid or out-of-range daily bar') from None
                key = day.isoformat()
                if key in by_day and by_day[key]['close'] != close:
                    raise DataAccessError('Conflicting adjusted daily bar for the same session')
                by_day[key] = {'date': key, 'close': close}
            next_url = data.get('next_url')
            if not next_url:
                break
            # The documented aggregate cursor may advance the start path from
            # an ISO date to epoch milliseconds. Keep the ticker/unit fixed;
            # every returned bar still has to satisfy the original date range.
            if not isinstance(next_url, str):
                raise DataAccessError('Invalid provider pagination link')
            next_path = urlsplit(next_url).path
            prefix = '/v2/aggs/ticker/' + quote(ticker, safe='') + '/range/1/day/'
            date_token = r'(?:\d{4}-\d{2}-\d{2}|\d{10,16})'
            if not re.fullmatch(re.escape(prefix) + date_token + '/' + date_token, next_path):
                raise DataAccessError('Untrusted aggregate ticker or interval in pagination link')
            url = self._safe_next(next_url, next_path)
        else:
            raise DataAccessError('Daily-price pagination incomplete')
        if not by_day:
            raise DataAccessError('Empty daily-price history')
        return {'symbol': ticker, 'prices': [by_day[k] for k in sorted(by_day)],
                'start': start.isoformat(), 'end': end.isoformat(), 'receipts': receipts,
                'adjustment': 'split_adjusted_close_not_dividend_total_return',
                'scope': 'Daily association data, not intraday event-window causal identification'}


def acquire_portfolio_inputs(portfolio_rows, *, notional, client, evaluated_at,
                             benchmark='SPY', history_days=1095, prior_holdings=None,
                             holdings_max_pages=5, include_scenario_proxies=False):
    """Attempt every owned ETF; never silently substitute archived data for live data."""
    original_rows = deepcopy(list(portfolio_rows))
    portfolio = parse_portfolio(original_rows, notional=notional)
    if not isinstance(evaluated_at, datetime) or evaluated_at.utcoffset() is None:
        raise ValueError('Timezone-aware evaluation time required')
    if (type(holdings_max_pages) is not int or not 1 <= holdings_max_pages <= 20):
        raise ValueError('Invalid holdings page allowance')
    day = evaluated_at.astimezone(timezone.utc).date()
    # Exclude the current session even if a provider publishes a partial daily bar.
    through = day - timedelta(days=1)
    start = through - timedelta(days=history_days)
    metadata, holdings, prices, price_sources, attempts = {}, {}, {}, {}, []
    symbols = sorted({p.symbol for p in portfolio.positions if p.asset_type != 'cash'} | {_symbol(benchmark)})
    # Metadata is a first pass. A default UI 'stock' declaration must not skip
    # the ETF holdings path after the provider has identified the same symbol.
    for position in portfolio.positions:
        if position.asset_type == 'cash':
            continue
        symbol = position.symbol
        try:
            metadata[symbol] = client.ticker_details(symbol, as_of=day)
            attempts.append({'symbol': symbol, 'operation': 'ticker_details', 'status': 'OK'})
        except (DataAccessError, ValueError) as exc:
            attempts.append({'symbol': symbol, 'operation': 'ticker_details', 'status': 'UNAVAILABLE', 'reason': str(exc)})
    # Receipt time follows acquisition start. Use the actual completion clock
    # for source verification, never rewrite the record to the earlier start.
    identity_checked_at = max(evaluated_at, datetime.now(timezone.utc))
    identity = resolve_portfolio_types(original_rows, instrument_metadata=metadata,
        evaluated_at=identity_checked_at)
    resolved_rows = identity['portfolio_rows']
    portfolio = parse_portfolio(resolved_rows, notional=notional)
    # R122: same-issuer share classes for directly held stocks (optional
    # annotation input; a failed lookup leaves exact-ticker grouping only).
    issuer_siblings = {}
    sibling_reader = getattr(client, 'issuer_siblings', None)
    if callable(sibling_reader):
        for position in portfolio.positions:
            cik = (metadata.get(position.symbol) or {}).get('cik')
            if position.asset_type != 'stock' or not isinstance(cik, str) or not cik.isdigit() or len(cik) != 10:
                continue
            try:
                issuer_siblings[position.symbol] = sibling_reader(cik, as_of=day)
                attempts.append({'symbol': position.symbol, 'operation': 'issuer_siblings', 'status': 'OK',
                                 'tickers': len(issuer_siblings[position.symbol].get('tickers') or [])})
            except (DataAccessError, ValueError) as exc:
                attempts.append({'symbol': position.symbol, 'operation': 'issuer_siblings',
                                 'status': 'UNAVAILABLE', 'reason': str(exc)})
    for position_index, position in enumerate(portfolio.positions):
        symbol = position.symbol
        if position.asset_type == 'etf':
            for operation in ('holdings', 'taxonomy'):
                try:
                    if operation == 'holdings':
                        dated_reader = getattr(client, 'holdings_for_assessment', None)
                        if callable(dated_reader):
                            remaining = [p for p in portfolio.positions[position_index + 1:] if p.asset_type != 'cash']
                            minimum_remaining = (1 + len(symbols)
                                + 2 * sum(p.asset_type == 'etf' for p in remaining))
                            reserve = minimum_remaining * (getattr(client, 'max_retries', 0) + 1)
                            holdings[symbol] = dated_reader(symbol, as_of_date=day.isoformat(),
                                max_pages=holdings_max_pages, reserve_requests=reserve,
                                **({'allow_expanded_pages': True} if holdings_max_pages > 5 else {}))
                        else:
                            # Existing injected clients retain their established
                            # interface; their result still receives quality diagnostics.
                            holdings[symbol] = client.holdings(symbol, as_of_date=day.isoformat(), max_pages=holdings_max_pages)
                    else:
                        metadata.setdefault(symbol, {})['taxonomy'] = client.taxonomy(symbol, as_of=day)
                    attempt = {'symbol': symbol, 'operation': operation, 'status': 'OK'}
                    if operation == 'holdings':
                        snapshot = holdings[symbol]
                        quality = snapshot.get('metadata', {})
                        plain = (snapshot.get('reliable') is True and snapshot.get('complete') is True
                                 and not quality.get('quality_issues'))
                        # Bond funds and single-line vehicles fail the plain equity
                        # validator by construction; the class-aware review accepts
                        # them only within its declared, measured tolerances.
                        review = None if plain else review_source_class_snapshot(
                            snapshot, symbol=symbol, instrument=metadata.get(symbol))
                        accepted = bool(review and review.get('holdings_accepted_for_assessment') is True)
                        attempt.update(transport_status='OK',
                            data_quality_status=('RELIABLE_SNAPSHOT' if plain
                                else 'ACCEPTED_BY_DECLARED_HOLDINGS_REVIEW' if accepted
                                else 'UNRELIABLE_OR_INCOMPLETE_SNAPSHOT'),
                            holdings_review_status=review['status'] if review else None,
                            query_complete=snapshot.get('complete') is True,
                            snapshot_reliable=snapshot.get('reliable') is True,
                            quality_issues=quality.get('quality_issues', []),
                            selected_batch=quality.get('selected_batch'),
                            quality_scope='Ingestion quality only; HTTP success and a complete query do not prove full NAV disclosure or resolved constituents.')
                    attempts.append(attempt)
                except (DataAccessError, ValueError) as exc:
                    attempts.append({'symbol': symbol, 'operation': operation, 'status': 'UNAVAILABLE', 'reason': str(exc)})
    for symbol in symbols:
        try:
            result = client.daily_prices(symbol, start=start, end=through)
            prices[symbol] = result['prices']
            price_sources[symbol] = {k: v for k, v in result.items() if k != 'prices'}
            attempts.append({'symbol': symbol, 'operation': 'daily_prices', 'status': 'OK', 'sessions': len(result['prices'])})
        except (DataAccessError, ValueError) as exc:
            attempts.append({'symbol': symbol, 'operation': 'daily_prices', 'status': 'UNAVAILABLE', 'reason': str(exc)})
    treasury_10y, treasury_source = [], {}
    if callable(getattr(client, 'treasury_yields', None)):
        # Second factor of the measured two-factor fund response; optional.
        try:
            treasury_result = client.treasury_yields(start=start, end=through)
            treasury_10y = treasury_result['yields']
            treasury_source = {k: v for k, v in treasury_result.items() if k != 'yields'}
            attempts.append({'symbol': 'US10Y', 'operation': 'treasury_yields', 'status': 'OK', 'days': len(treasury_10y)})
        except (DataAccessError, ValueError) as exc:
            attempts.append({'symbol': 'US10Y', 'operation': 'treasury_yields', 'status': 'UNAVAILABLE', 'reason': str(exc)})
    proxy_prices, proxy_sources = {}, {}
    if include_scenario_proxies:
        from .scenario_proxy_response import PROXY_SYMBOLS
        for symbol in PROXY_SYMBOLS:
            if symbol in prices:
                proxy_prices[symbol] = prices[symbol]
                proxy_sources[symbol] = price_sources[symbol]
                continue
            try:
                result = client.daily_prices(symbol, start=start, end=through)
                proxy_prices[symbol] = result['prices']
                proxy_sources[symbol] = {k: v for k, v in result.items() if k != 'prices'}
                attempts.append({'symbol': symbol, 'operation': 'scenario_proxy_prices',
                                 'status': 'OK', 'sessions': len(result['prices'])})
            except (DataAccessError, ValueError) as exc:
                attempts.append({'symbol': symbol, 'operation': 'scenario_proxy_prices',
                                 'status': 'UNAVAILABLE', 'reason': str(exc)})
    required = [attempt for attempt in attempts if attempt['operation'] not in ('taxonomy', 'treasury_yields', 'scenario_proxy_prices', 'issuer_siblings')]
    holdings_attempts = [attempt for attempt in attempts if attempt['operation'] == 'holdings']
    quality_summary = {
        'required_operations': ['ticker_details', 'holdings', 'daily_prices'],
        'optional_metadata_operations': ['taxonomy'],
        'all_required_transport_operations_succeeded': bool(required) and all(a['status'] == 'OK' for a in required),
        'all_required_holdings_quality_passed': all(a.get('data_quality_status') in ACCEPTED_HOLDINGS_STATUSES for a in holdings_attempts),
        'holdings_quality_gap_symbols': [a['symbol'] for a in holdings_attempts if a.get('data_quality_status') not in ACCEPTED_HOLDINGS_STATUSES],
        'holdings_accepted_by_declared_review_symbols': [a['symbol'] for a in holdings_attempts
                                                         if a.get('data_quality_status') == 'ACCEPTED_BY_DECLARED_HOLDINGS_REVIEW'],
        'optional_taxonomy_unavailable_symbols': [a['symbol'] for a in attempts if a['operation'] == 'taxonomy' and a['status'] != 'OK'],
        'scope': 'Operational diagnostics only. Taxonomy availability, empirical model validation and full authenticated acceptance retain their separate gates.',
        'full_acceptance_redefined': False,
    }
    # The prior snapshot is deliberately not promoted to a fresh acquisition.
    return {'schema': 'castle.live_inputs.v1', 'evaluated_at_utc': evaluated_at.isoformat(),
            'instrument_metadata': metadata, 'holdings': holdings, 'prices_by_symbol': prices,
            'issuer_siblings': issuer_siblings,
            'treasury_10y': treasury_10y, 'treasury_source': treasury_source,
            'scenario_proxy_prices': proxy_prices, 'scenario_proxy_sources': proxy_sources,
            'original_portfolio_rows': original_rows, 'resolved_portfolio_rows': resolved_rows,
            'instrument_identity': identity['audit'], 'identity_checked_at_utc': identity_checked_at.isoformat(),
            'price_sources': price_sources, 'benchmark_symbol': benchmark,
            'attempts': attempts, 'provider_request_count': client.request_count,
            'acquisition_quality': quality_summary,
            'prior_holdings_not_substituted_for_live': sorted((prior_holdings or {}).keys()),
            'raw_massive_distribution': 'PRIVATE_RUNTIME_ONLY',
            'limitations': ['Provider entitlement and instrument coverage can leave gaps.',
                'ETF batch dates do not prove intraday historical public availability.',
                'Daily prices are split adjusted; dividend total returns and FX effects are not modeled.']}


def verify_saved_responses(directory: Path) -> list[dict]:
    """Independently hash saved responses; a hash proves bytes, not provider authorship."""
    records = []
    for meta_path in sorted(Path(directory).rglob('*.meta.json')):
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        raw = meta_path.with_name(meta_path.name.removesuffix('.meta.json') + '.json')
        if not raw.is_file():
            raw = meta_path.with_name(meta_path.name.removesuffix('.meta.json') + '.html')
        actual = hashlib.sha256(raw.read_bytes()).hexdigest() if raw.is_file() else None
        records.append({'response': str(raw.relative_to(directory)), 'url': meta.get('url'),
                        'retrieved_at': meta.get('retrieved_at'), 'status': meta.get('status'),
                        'sha256': actual, 'matches_receipt': actual is not None and actual == meta.get('sha256'),
                        'fixture': meta.get('fixture')})
    return records

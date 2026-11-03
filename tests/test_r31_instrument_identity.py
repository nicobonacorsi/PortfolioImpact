"""Offline R31 identity tests; every generated response is synthetic test data.

Receipt fixture flags exercise the production byte-binding gate, not a claim
that the named securities or any provider were contacted in these tests.
"""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import socket

import pytest

from castle_risk.adapters import DataAccessError
from castle_risk.instrument_identity import InstrumentIdentityError, resolve_portfolio_types
from castle_risk.live_data import acquire_portfolio_inputs


AT = datetime(2026, 10, 28, 1, 2, 27, tzinfo=timezone.utc)
RECEIVED = '2026-10-28T01:00:00+00:00'


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('R31 identity tests may not contact providers')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


def source_metadata(tmp_path, symbol, kind, *, query_date='2026-10-28',
                    retrieved_at=RECEIVED, fixture=False, payload_status='OK', **changes):
    row = {'ticker': symbol, 'type': kind, 'currency_name': 'usd',
           'active': True, **changes}
    body = json.dumps({'status': payload_status, 'results': row}, sort_keys=True).encode()
    path = tmp_path / (symbol + '.json')
    path.write_bytes(body)
    url = 'https://api.massive.com/v3/reference/tickers/' + symbol
    if query_date is not None:
        url += '?date=' + query_date
    digest = sha256(body).hexdigest()
    path.with_suffix('.meta.json').write_text(json.dumps({
        'url': url, 'retrieved_at': retrieved_at, 'sha256': digest,
        'status': 200, 'fixture': fixture}))
    result = {**row, 'provider': 'Massive', 'source_url': url,
              'retrieved_at': retrieved_at, 'source_response_sha256': digest,
              'raw_path': str(path), 'fixture': fixture}
    if query_date is not None:
        result['metadata_as_of'] = query_date
    return result


def row(symbol='SYNTHETF', declared='stock', *, automatic=False):
    return {'symbol': symbol, 'asset_type': declared, 'weight': 1.0,
            **({'input_asset_type': 'auto'} if automatic else {})}


def resolve(rows, records, **kwargs):
    return resolve_portfolio_types(rows, instrument_metadata=records,
                                   evaluated_at=AT, **kwargs)


@pytest.mark.parametrize('symbol', ['IEF', 'ANOTHERFUND', 'UNKNOWN123'])
def test_verified_etf_corrects_stock_without_symbol_registry(tmp_path, symbol):
    original = [row(symbol)]
    before = deepcopy(original)
    result = resolve(original, {symbol: source_metadata(tmp_path, symbol, 'ETF')})
    assert original == before
    assert result['portfolio_rows'] == [{'symbol': symbol, 'asset_type': 'etf',
                                      'weight': 1.0, 'input_asset_type': 'stock'}]
    assert result['blocked_symbols'] == []
    audit = result['audit']['positions'][0]
    assert audit['status'] == 'DECLARATION_CORRECTED_FROM_VERIFIED_SOURCE'
    assert audit['source_verified'] and audit['source']['metadata_as_of'] == '2026-10-28'
    assert 'raw_path' not in json.dumps(result['audit'])
    again = resolve(result['portfolio_rows'], {symbol: source_metadata(tmp_path, symbol, 'ETF')})
    assert again['portfolio_rows'] == result['portfolio_rows']
    assert again['audit']['corrections'] == result['audit']['corrections']


def test_verified_common_stock_corrects_wrong_fund_declaration(tmp_path):
    result = resolve([row('SYNTHCO', 'etf')], {'SYNTHCO': source_metadata(tmp_path, 'SYNTHCO', 'CS')})
    assert result['portfolio_rows'][0]['asset_type'] == 'stock'
    assert result['portfolio_rows'][0]['input_asset_type'] == 'etf'


@pytest.mark.parametrize('provider_kind,expected', [('ETF', 'etf'), ('CS', 'stock')])
def test_auto_requires_verified_source_and_resolves_supported_types(tmp_path, provider_kind, expected):
    result = resolve([row(automatic=True)], {'SYNTHETF': source_metadata(tmp_path, 'SYNTHETF', provider_kind)})
    assert result['portfolio_rows'][0]['asset_type'] == expected
    assert result['audit']['positions'][0]['status'] == 'AUTO_RESOLVED_FROM_VERIFIED_SOURCE'


@pytest.mark.parametrize('failure', ['missing', 'no_bytes', 'fixture', 'wrong_hash',
                                   'future_receipt', 'wrong_date', 'future_date',
                                   'stale_date', 'unsupported', 'inactive', 'eur', 'unknown_currency'])
def test_auto_never_silently_falls_back_to_stock(tmp_path, failure):
    record = source_metadata(tmp_path, 'SYNTHETF',
        'ETN' if failure == 'unsupported' else 'ETF',
        fixture=failure == 'fixture',
        active=False if failure == 'inactive' else True,
        currency_name='eur' if failure == 'eur' else None if failure == 'unknown_currency' else 'usd',
        retrieved_at='2026-10-29T01:00:00+00:00' if failure == 'future_receipt' else RECEIVED,
        query_date='2026-10-29' if failure == 'future_date' else '2026-10-10' if failure == 'stale_date' else '2026-10-28')
    if failure == 'no_bytes':
        record.pop('raw_path')
    if failure == 'wrong_hash':
        record['source_response_sha256'] = '0' * 64
    if failure == 'wrong_date':
        record['metadata_as_of'] = '2026-10-27'
    records = {} if failure == 'missing' else {'SYNTHETF': record}
    with pytest.raises(InstrumentIdentityError, match='SYNTHETF'):
        resolve([row(automatic=True)], records)
    safe = resolve([row(automatic=True)], records, require_auto_resolution=False)
    assert safe['blocked_symbols'] == ['SYNTHETF']
    assert safe['audit']['positions'][0]['resolved_asset_type'] is None


def test_unverified_conflicting_type_blocks_even_manual_equity_input(tmp_path):
    record = source_metadata(tmp_path, 'SYNTHETF', 'ETF')
    record.pop('raw_path')
    result = resolve([row()], {'SYNTHETF': record})
    assert result['portfolio_rows'][0]['asset_type'] == 'stock'
    assert result['blocked_symbols'] == ['SYNTHETF']
    assert result['audit']['positions'][0]['status'] == 'CONFLICTING_IDENTITY_NOT_SOURCE_VERIFIED'


@pytest.mark.parametrize('failure', ['normalized_fixture', 'provider_error'])
def test_source_verifier_rejects_explicit_fixture_or_error_envelope(tmp_path, failure):
    from castle_risk.instrument_exposures import _metadata_fields
    record = source_metadata(tmp_path, 'SYNTHETF', 'ETF',
        payload_status='ERROR' if failure == 'provider_error' else 'OK')
    # Keep the hashed local receipt non-fixture and HTTP-successful: a
    # contradictory normalized flag or failed provider envelope still vetoes
    # identity, even if the results object itself looks usable.
    if failure == 'normalized_fixture':
        record['fixture'] = True
    checked = _metadata_fields(record, 'SYNTHETF', AT)
    assert checked['status'] == 'METADATA_REJECTED'
    assert not checked.get('local_response_verified')
    with pytest.raises(InstrumentIdentityError, match='SYNTHETF'):
        resolve([row(automatic=True)], {'SYNTHETF': record})
    manual = resolve([row()], {'SYNTHETF': record})
    assert manual['blocked_symbols'] == ['SYNTHETF']
    assert manual['audit']['positions'][0]['resolved_asset_type'] is None


def test_missing_manual_metadata_retains_unverified_declaration_not_verified_stock():
    result = resolve([row()], {})
    assert result['blocked_symbols'] == []
    assert result['audit']['status'] == 'PARTIAL_IDENTITY_VERIFICATION'
    assert not result['audit']['positions'][0]['source_verified']


def test_cash_marker_cannot_disagree_with_the_actual_position_type():
    with pytest.raises(InstrumentIdentityError, match='cash declaration'):
        resolve([{'symbol': 'SYNTHCO', 'weight': 1., 'asset_type': 'cash',
                  'input_asset_type': 'stock'}], {})


def test_manual_etv_fund_keeps_separate_legacy_source_class_gate(tmp_path):
    record = source_metadata(tmp_path, 'SYNTHETF', 'ETV')
    result = resolve([row(declared='etf')], {'SYNTHETF': record})
    assert result['blocked_symbols'] == []
    assert result['audit']['positions'][0]['status'] == 'DECLARED_FUND_REQUIRES_SOURCE_CLASS_REVIEW'
    assert resolve([row()], {'SYNTHETF': record})['blocked_symbols'] == ['SYNTHETF']
    # R85 policy change: automatic detection routes a verified exchange-traded
    # vehicle (e.g. a gold trust) to the same fund source-class review instead of
    # stopping the whole analysis. It is never treated as an equity ETF.
    automatic = resolve([row(automatic=True)], {'SYNTHETF': record})
    assert automatic['blocked_symbols'] == []
    assert automatic['portfolio_rows'][0]['asset_type'] == 'etf'
    assert automatic['audit']['positions'][0]['status'] == 'AUTO_DETECTED_VEHICLE_REQUIRES_SOURCE_CLASS_REVIEW'


def test_source_request_from_previous_utc_date_survives_midnight(tmp_path):
    record = source_metadata(tmp_path, 'SYNTHETF', 'ETF', query_date='2026-10-27')
    assert resolve([row(automatic=True)], {'SYNTHETF': record})['portfolio_rows'][0]['asset_type'] == 'etf'


class FixedClock(datetime):
    @classmethod
    def now(cls, tz=None):
        return AT.astimezone(tz) if tz else AT.replace(tzinfo=None)


class SyntheticClient:
    """Source-shaped byte-bound identity with explicitly synthetic other inputs."""
    max_retries = 0
    request_count = 0

    def __init__(self, records):
        self.records, self.calls = records, []

    def ticker_details(self, ticker, *, as_of):
        self.calls.append(('metadata', ticker))
        self.request_count += 1
        return deepcopy(self.records[ticker])

    def holdings(self, ticker, **kwargs):
        self.calls.append(('holdings', ticker))
        self.request_count += 1
        return {'fund_symbol': ticker, 'reliable': True, 'complete': True,
                'holdings': [], 'metadata': {'quality_issues': []}}

    def taxonomy(self, ticker, **kwargs):
        self.calls.append(('taxonomy', ticker))
        self.request_count += 1
        raise DataAccessError('Synthetic unavailable optional taxonomy')

    def daily_prices(self, ticker, **kwargs):
        self.calls.append(('prices', ticker))
        self.request_count += 1
        return {'symbol': ticker, 'prices': [{'date': '2026-10-27', 'close': 100.0}]}


def test_acquisition_resolves_all_metadata_before_holdings_and_preserves_input(tmp_path, monkeypatch):
    import castle_risk.live_data as module
    monkeypatch.setattr(module, 'datetime', FixedClock)
    records = {'IEF': source_metadata(tmp_path, 'IEF', 'ETF'),
               'SYNTHCO': source_metadata(tmp_path, 'SYNTHCO', 'CS')}
    service = SyntheticClient(records)
    original = [{'symbol': 'IEF', 'weight': .2, 'asset_type': 'stock'},
                {'symbol': 'SYNTHCO', 'weight': .8, 'asset_type': 'stock', 'input_asset_type': 'auto'}]
    before = deepcopy(original)
    result = acquire_portfolio_inputs(original, notional=100000, client=service,
        evaluated_at=FixedClock.fromisoformat(AT.isoformat()).replace(hour=0, minute=59))
    assert original == before
    assert service.calls[:3] == [('metadata', 'IEF'), ('metadata', 'SYNTHCO'), ('holdings', 'IEF')]
    assert result['resolved_portfolio_rows'][0]['asset_type'] == 'etf'
    assert result['original_portfolio_rows'] == before
    assert result['instrument_identity']['corrections'][0]['symbol'] == 'IEF'
    assert 'IEF' in result['holdings']
    assert result['identity_checked_at_utc'] == AT.isoformat()
    assert result['acquisition_quality']['optional_taxonomy_unavailable_symbols'] == ['IEF']


def test_unresolved_auto_stops_before_holdings_prices_or_research(tmp_path, monkeypatch):
    import castle_risk.live_data as module
    monkeypatch.setattr(module, 'datetime', FixedClock)
    service = SyntheticClient({'SYNTHETF': {'ticker': 'SYNTHETF'}})
    with pytest.raises(InstrumentIdentityError):
        acquire_portfolio_inputs([row(automatic=True)], notional=100000,
            client=service, evaluated_at=FixedClock.fromisoformat(AT.isoformat()))
    assert service.calls == [('metadata', 'SYNTHETF')]


def test_deterministic_builder_uses_resolved_portfolio_and_correct_fingerprint(tmp_path):
    from castle_risk.live_assessment import build_assessment_from_evidence
    from castle_risk.option1_brief import portfolio_input_fingerprint
    metadata = source_metadata(tmp_path, 'SYNTHETF', 'ETF', asset_class='Fixed Income')
    inputs = {'instrument_metadata': {'SYNTHETF': metadata}, 'holdings': {}, 'prices_by_symbol': {}}
    original = [row()]
    report = build_assessment_from_evidence(portfolio_rows=original, notional=100000,
        inputs=inputs, discovery={'events': [], 'markets_by_event': {}, 'matches_by_event': {}},
        market_objects={}, evaluated_at=AT, release_dates={}, event_observations=[])
    assert original[0]['asset_type'] == 'stock'
    assert report['portfolio']['positions'][0]['asset_type'] == 'etf'
    assert report['exposures']['positions'][0]['instrument_class'] == 'bond_etf'
    assert report['instrument_identity']['corrections'][0]['symbol'] == 'SYNTHETF'
    corrected = [{'symbol': 'SYNTHETF', 'asset_type': 'etf', 'weight': 1.0}]
    assert report['input_fingerprint'] == portfolio_input_fingerprint(corrected, 100000, {})


def test_live_orchestration_and_saved_report_share_corrected_types(tmp_path, monkeypatch):
    import castle_risk.live_assessment as module
    monkeypatch.setattr(module, 'datetime', FixedClock)
    for key in ('MASSIVE_API_KEY', 'POLYGON_API_KEY', 'ANTHROPIC_API_KEY'):
        monkeypatch.delenv(key, raising=False)
    metadata = source_metadata(tmp_path, 'SYNTHETF', 'ETF', asset_class='Fixed Income')
    original = [row()]
    inputs = {'instrument_metadata': {'SYNTHETF': metadata}, 'holdings': {},
              'prices_by_symbol': {}, 'price_sources': {}, 'attempts': [],
              'original_portfolio_rows': deepcopy(original),
              'resolved_portfolio_rows': [{'symbol': 'SYNTHETF', 'asset_type': 'etf',
                                          'weight': 1.0, 'input_asset_type': 'stock'}]}
    monkeypatch.setattr(module, 'acquire_portfolio_inputs', lambda *args, **kwargs: deepcopy(inputs))
    class EmptyKalshi:
        request_count = 0
        def list_markets(self, **kwargs):
            return []
    report = module.run_live_assessment(portfolio_rows=original, root=tmp_path / 'run',
        massive_client=SyntheticClient({}), kalshi_client=EmptyKalshi(), release_dates={},
        event_observations=[], include_geopolitical_search=False, include_payrolls=False)
    assert original[0]['asset_type'] == 'stock'
    assert report['portfolio']['positions'][0]['asset_type'] == 'etf'
    assert report['instrument_identity']['positions'][0]['declared_asset_type'] == 'stock'
    saved = json.loads((Path(report['run_directory']) / 'derived_assessment.json').read_text())
    private = json.loads((Path(report['run_directory']) / 'private_replay_inputs.json').read_text())
    assert saved['portfolio']['positions'] == list(report['portfolio']['positions'])
    assert private['inputs']['original_portfolio_rows'][0]['asset_type'] == 'stock'
    assert private['inputs']['resolved_portfolio_rows'][0]['asset_type'] == 'etf'


def test_complete_ief_stock_input_to_acquired_bonds_to_rates_table_chain(tmp_path, monkeypatch):
    """Synthetic end-to-end boundary regression, not an empirical IEF rerun."""
    from castle_risk import adapters
    import castle_risk.live_data as data_module
    from castle_risk.default_assessment import build_default_assessment
    from castle_risk.live_assessment import build_assessment_from_evidence
    from castle_risk.portfolio_impact_adapter import live_report_to_case
    from test_round8_default_integration import synthetic_distribution

    monkeypatch.setattr(data_module, 'datetime', FixedClock)
    monkeypatch.setattr(adapters, 'datetime', FixedClock)
    monkeypatch.setattr(adapters, 'utc_now', lambda: AT.isoformat())
    constituent = {
        'composite_ticker': 'IEF', 'constituent_ticker': 'T 4 02/15/36',
        'constituent_name': 'SYNTHETIC TEST TREASURY', 'weight': 1.0,
        'isin': 'US91282CJZ59', 'security_type': 'US GOVERNMENT',
        'asset_class': 'Fixed Income', 'currency_traded': 'USD',
        'effective_date': '2026-10-27', 'processed_date': '2026-10-27',
        'market_value': 1_000_000.0, 'shares_held': 1_000_000.0,
    }
    # As in the source-composition tests, a local native-transport replacement
    # exercises receipt binding; no HTTP request is made.
    monkeypatch.setattr(adapters, '_transport', lambda *args, **kwargs:
        (200, {}, json.dumps({'status': 'OK', 'results': [constituent]}).encode()))
    service = adapters.MassiveClient(tmp_path / 'synthetic_holdings',
        api_key='SYNTHETIC_TEST_VALUE', min_interval=0, max_retries=0, cache_ttl=0)
    snapshot = service.holdings('IEF', as_of_date='2026-10-28')
    metadata = source_metadata(tmp_path, 'IEF', 'ETF', asset_class='Fixed Income')

    class CompleteBondClient(SyntheticClient):
        def holdings(self, ticker, **kwargs):
            assert ticker == 'IEF'
            self.calls.append(('holdings', ticker))
            self.request_count += 1
            return deepcopy(snapshot)

    acquisition = CompleteBondClient({'IEF': metadata})
    original = [row('IEF')]
    inputs = acquire_portfolio_inputs(original, notional=100000, client=acquisition,
        evaluated_at=FixedClock.fromisoformat(AT.isoformat()).replace(hour=0, minute=59))
    assert ('holdings', 'IEF') in acquisition.calls
    assert inputs['resolved_portfolio_rows'][0]['asset_type'] == 'etf'
    report = build_assessment_from_evidence(portfolio_rows=original, notional=100000,
        inputs=inputs, discovery={'events': [], 'markets_by_event': {}, 'matches_by_event': {}},
        market_objects={}, evaluated_at=AT, release_dates={}, event_observations=[])
    assert report['portfolio']['positions'][0]['asset_type'] == 'etf'
    distribution = synthetic_distribution(q=(.4, .5, .1))
    distribution.update(event_date='2026-12-09', evaluated_at=AT.isoformat(),
        status='CONDITIONAL_SOURCE_DISTRIBUTION',
        fixture_scope='SYNTHETIC_TEST_POLYTOPE_NOT_ACQUIRED_MARKET_EVIDENCE')
    for atom in distribution['atoms']:
        atom['q_interval'] = [atom['q_reference'], atom['q_reference']]
    report['default_analysis'] = build_default_assessment(report, market_objects={},
        inputs=inputs, evaluated_at=AT,
        distribution_overrides={'SYNTHETIC-FED': distribution})
    event = next(e for e in report['default_analysis']['events']
                 if e['family'] == 'FED_POLICY' and e['event_date'] == '2026-12-09')
    table = event['outcome_table']
    assert table['status'] == 'COMPLETE_DECLARED_MODEL'
    assert table['positions'][0]['channel'] == 'RATES'
    assert table['positions'][0]['duration']['duration'] > 0
    assert all(r['position_contributions'][0]['channel'] == 'RATES' for r in table['outcomes'])
    assert any(abs(r['position_contributions'][0]['pnl_usd']) > 0 for r in table['outcomes'])
    from castle_risk.live_assessment import public_assessment_view
    case = live_report_to_case(public_assessment_view(report))
    displayed = next(e for e in case['risks'] if e['date'] == '2026-12-09' and e['family'] == 'FED_POLICY')
    assert displayed['outcomes']
    assert all(o['position_contributions'][0]['channel'] == 'RATES' for o in displayed['outcomes'])
    assert original[0]['asset_type'] == 'stock'


def test_unresolved_auto_identity_survives_builder_serialization_and_default_projection(monkeypatch):
    import castle_risk.default_assessment as defaults
    from castle_risk.live_assessment import build_assessment_from_evidence, public_assessment_view
    from castle_risk.portfolio_impact_adapter import live_report_to_case
    from test_live_assessment_integration import synthetic_inputs
    from test_round8_default_integration import synthetic_distribution

    inputs, releases = synthetic_inputs()
    distribution = synthetic_distribution(q=(.4, .5, .1))
    distribution.update(event_date='2026-12-09', evaluated_at=AT.isoformat(),
        status='CONDITIONAL_SOURCE_DISTRIBUTION',
        fixture_scope='SYNTHETIC_TEST_POLYTOPE_NOT_ACQUIRED_MARKET_EVIDENCE')
    for atom in distribution['atoms']:
        atom['q_interval'] = [atom['q_reference'], atom['q_reference']]
    monkeypatch.setattr(defaults, 'build_distributions_from_markets',
                        lambda *args, **kwargs: {'SYNTHETIC-FED': deepcopy(distribution)})
    report = build_assessment_from_evidence(portfolio_rows=[row('AAPL', automatic=True)],
        notional=100000, inputs=inputs,
        discovery={'events': [], 'markets_by_event': {}, 'matches_by_event': {}},
        market_objects={}, evaluated_at=AT, release_dates=releases,
        event_observations=[], default_scenarios=True)
    assert report['instrument_identity']['blocked_symbols'] == ['AAPL']
    # Portfolio dataclass serialization drops the UI hint. The explicit audit
    # must carry the restriction across that boundary, even with fitted betas.
    assert 'input_asset_type' not in report['portfolio']['positions'][0]
    event = next(e for e in report['default_analysis']['events']
                 if e['family'] == 'FED_POLICY' and e['event_date'] == '2026-12-09')
    assert event['instrument_identity_gate']['blocked_symbols'] == ['AAPL']
    assert event['outcome_table']['status'] != 'COMPLETE_DECLARED_MODEL'
    assert event['legacy_fallback_allowed'] is False
    case = live_report_to_case(public_assessment_view(report))
    shown = next(e for e in case['risks'] if e['date'] == '2026-12-09' and e['family'] == 'FED_POLICY')
    assert shown['outcomes'] == []

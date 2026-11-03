"""R63: the documented two-venue read profile must be real, not prose.

R85 raised Polymarket to 150 reads so every Gamma query is read to its last page."""

import json

import pytest

from castle_risk.adapters import DataAccessError
from castle_risk.geopolitical_market_search import GammaSearchClient
from castle_risk.live_assessment import _geopolitical_capture_clients


def test_expanded_capture_gets_60_kalshi_and_150_polymarket_reads(tmp_path):
    kalshi, gamma = _geopolitical_capture_clients(tmp_path, expanded_live=True)
    assert kalshi.max_requests == 60
    assert gamma.max_requests == 150
    assert kalshi.cache_ttl == gamma.cache_ttl == 0
    assert kalshi.request_count == gamma.request_count == 0
    assert kalshi.data_dir != gamma.data_dir


def test_legacy_capture_keeps_original_caps(tmp_path):
    kalshi, gamma = _geopolitical_capture_clients(tmp_path, expanded_live=False)
    assert kalshi is None and gamma is None
    assert GammaSearchClient(tmp_path / 'legacy_gamma').max_requests == 30


def test_caller_injection_keeps_its_own_budget_and_transport(tmp_path):
    own_gamma = GammaSearchClient(tmp_path / 'own', max_requests=7, cache_ttl=123)
    kalshi, gamma = _geopolitical_capture_clients(
        tmp_path / 'expanded', expanded_live=True, gamma_client=own_gamma)
    assert kalshi.max_requests == 60
    assert gamma is own_gamma
    assert gamma.max_requests == 7 and gamma.cache_ttl == 123
    assert _geopolitical_capture_clients(
        tmp_path / 'legacy', expanded_live=False, gamma_client=own_gamma) == (None, own_gamma)


def test_gamma_expanded_allows_31st_read_and_refuses_151st_without_network(tmp_path):
    _, gamma = _geopolitical_capture_clients(tmp_path, expanded_live=True)
    calls = []

    def local_transport(url, _headers, _timeout):
        calls.append(url)
        return 200, {}, json.dumps({'events': []}).encode('utf-8')

    gamma.transport = local_transport
    gamma.min_interval = 0
    for i in range(150):
        result, _ = gamma._get(f'https://gamma-api.polymarket.com/public-search?q=R63-{i}')
        assert result == {'events': []}
    assert len(calls) == 150 and gamma.request_count == 150
    with pytest.raises(DataAccessError, match='budget exhausted'):
        gamma._get('https://gamma-api.polymarket.com/public-search?q=R63-151')
    assert len(calls) == 150 and gamma.request_count == 150

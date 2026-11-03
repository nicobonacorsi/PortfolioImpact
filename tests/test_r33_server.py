"""Read-only job reconnection never starts another paid analysis."""
import http.client
import json
import threading
import time

import pytest

from scripts.serve_portfolio_impact import PortfolioImpactServer


@pytest.fixture
def running_server(tmp_path):
    release = threading.Event()
    started = threading.Event()
    calls = []

    def runner(rows, total, progress):
        calls.append((rows, total))
        progress('Checking retained inputs')
        started.set()
        assert release.wait(5)
        return {'synthetic': True, 'total': total}

    server = PortfolioImpactServer(('127.0.0.1', 0), runner=runner,
        converter=lambda report: report, saved_runs_root=tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(method, path, payload=None, extra_headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=5)
        headers = {'Content-Type': 'application/json', 'X-Portfolio-Impact': '1'}
        headers.update(extra_headers or {})
        connection.request(method, path, body=json.dumps(payload) if payload is not None else None,
                           headers=headers)
        response = connection.getresponse()
        body = response.read()
        code = response.status
        connection.close()
        return code, json.loads(body)

    try:
        yield server, request, release, started, calls
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_status_exposes_only_safe_reconnection_metadata(running_server):
    server, request, release, started, calls = running_server
    server.provider_keys = {'massive': 'SYNTHETIC_PRIVATE_MASSIVE',
                            'anthropic': 'SYNTHETIC_PRIVATE_ANTHROPIC'}
    code, idle = request('GET', '/api/status')
    assert code == 200
    assert idle['busy'] is False
    assert idle['active_job_id'] is None
    assert idle['active_job_progress'] is None
    payload = {'positions': [{'symbol': 'CAT', 'type': 'auto', 'amount': 1000}]}
    code, submission = request('POST', '/api/analyze', payload)
    assert code == 202 and started.wait(3)
    identifier = submission['job_id']
    for _ in range(3):
        code, status = request('GET', '/api/status')
        assert code == 200 and status['busy'] is True
        assert status['active_job_id'] == identifier
        assert status['active_job_progress'] == 'Checking retained inputs'
        assert 'SYNTHETIC_PRIVATE' not in json.dumps(status)
        code, task = request('GET', '/api/jobs/' + status['active_job_id'])
        assert code == 200 and task['status'] == 'running'
    assert len(calls) == 1
    assert request('POST', '/api/analyze', payload)[0] == 409
    assert len(calls) == 1
    release.set()
    for _ in range(100):
        code, status = request('GET', '/api/status')
        if not status['busy']:
            break
        time.sleep(.01)
    assert status['active_job_id'] is None
    assert status['active_job_progress'] is None
    assert request('GET', '/api/jobs/' + identifier)[1]['status'] == 'complete'
    assert len(calls) == 1


def test_reconnection_metadata_still_rejects_cross_origin_access(running_server):
    _, request, _, _, calls = running_server
    code, _ = request('GET', '/api/status', extra_headers={'Origin': 'https://untrusted.example'})
    assert code == 403
    assert not calls

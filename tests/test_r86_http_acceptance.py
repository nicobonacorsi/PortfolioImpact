"""HTTP acceptance of the actual shipped recorded-analysis and export routes."""
from io import BytesIO
from html.parser import HTMLParser
import json
import threading
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen
from zipfile import ZipFile

from scripts.serve_portfolio_impact import PortfolioImpactServer, RECORDED_CASE


def test_every_declared_local_stylesheet_is_served_as_css():
    """A deployable page must serve the stylesheets it actually references."""
    class Stylesheets(HTMLParser):
        def __init__(self):
            super().__init__()
            self.hrefs = []

        def handle_starttag(self, tag, attributes):
            attrs = dict(attributes)
            if tag == 'link' and 'stylesheet' in attrs.get('rel', '').split():
                self.hrefs.append(attrs.get('href', ''))

    app = PortfolioImpactServer(('127.0.0.1', 0))
    thread = threading.Thread(target=app.serve_forever, daemon=True)
    thread.start()
    origin = f'http://127.0.0.1:{app.server_port}'
    try:
        with urlopen(origin + '/', timeout=10) as response:
            page = response.read().decode('utf-8')
        parser = Stylesheets()
        parser.feed(page)
        local_stylesheets = [urljoin(origin + '/', href) for href in parser.hrefs
                             if href and urlsplit(urljoin(origin + '/', href)).netloc
                             == urlsplit(origin).netloc]
        assert local_stylesheets, 'The page declares no local stylesheet.'
        for url in local_stylesheets:
            with urlopen(url, timeout=10) as response:
                assert response.status == 200, url
                assert response.headers.get_content_type() == 'text/css', url
                assert response.read().strip(), url
    finally:
        app.shutdown()
        app.server_close()
        thread.join(timeout=10)


def test_recorded_portfolio_and_corrected_report_are_served_end_to_end(tmp_path):
    # Live benchmark runs may be active in the shared working tree. This
    # historical HTTP fixture must load the shipped example, not those runs.
    app = PortfolioImpactServer(('127.0.0.1', 0), saved_runs_root=tmp_path / 'empty_saved_runs')
    app.recorded_case_path = RECORDED_CASE
    thread = threading.Thread(target=app.serve_forever, daemon=True)
    thread.start()
    host = f'http://127.0.0.1:{app.server_port}'
    try:
        with urlopen(host + '/api/status', timeout=10) as response:
            status = json.load(response)
        assert status['available'] is True
        with urlopen(host + '/studio.js', timeout=10) as response:
            source = response.read().decode('utf-8')
        assert 'function analogueMeasuredRobustness(a)' in source
        with urlopen(host + '/assessment_audit.js', timeout=10) as response:
            assessment_source = response.read().decode('utf-8')
        assert 'function scenarioAccounting(c, o)' in assessment_source
        with urlopen(host + '/api/latest', timeout=10) as response:
            case = json.load(response)['case']
        assert case['recorded_release_example'] is True
        assert len(case['risks']) >= 10
        # The new custom-scenario route must reuse the already available case,
        # without vendor keys, fresh analysis jobs or ETF double counting.
        explicit = {p['symbol']: (1 if i == 0 else 0) for i, p in enumerate(case['positions'])}
        scenario = {'hypothesis': 'What if equities rise after CPI?',
                    'mode': 'binary_gap', 'position_shocks': explicit,
                    'pricing_weight': .25, 'pricing_weight_range': [.1, .7]}
        request = Request(host + '/api/scenarios/evaluate',
                          data=json.dumps(scenario).encode('utf-8'),
                          headers={'Content-Type': 'application/json', 'X-Portfolio-Impact': '1'},
                          method='POST')
        with urlopen(request, timeout=10) as response:
            custom = json.load(response)
        gap = case['notional'] * case['positions'][0]['weight'] * .01
        assert custom['schema'] == 'castle.scenario_lab.v1'
        assert custom['complete'] is True
        assert abs(custom['yes_pnl_usd'] - .75 * gap) < 1e-6
        assert abs(custom['no_pnl_usd'] + .25 * gap) < 1e-6
        assert abs(.25 * custom['yes_pnl_usd'] + .75 * custom['no_pnl_usd']) < 1e-8
        assert custom['pricing']['q_empirically_identified'] is False
        with urlopen(host + '/api/export/latest', timeout=10) as response:
            archive = response.read()
            assert response.headers['Content-Type'] == 'application/zip'
        with ZipFile(BytesIO(archive)) as zipfile:
            html = zipfile.read('report.html').decode('utf-8')
        assert 'Excluding catalogued overlaps' in html
        assert '$7,135' in html
        assert 'Leave-one-episode-out coverage' not in html
    finally:
        app.shutdown()
        app.server_close()
        thread.join(timeout=10)

"""Local Portfolio Impact frontend backed by the actual Castle R21 engine.

Run from the repository root: python -m scripts.serve_portfolio_impact
Analysis is initiated ONLY by a real POST from the portfolio editor. The
server listens on loopback; API credentials never reach browser JavaScript.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import secrets
import threading
from urllib.parse import urlsplit
import webbrowser

from castle_risk.portfolio_impact_adapter import live_report_to_case
from castle_risk.portfolio_impact_export import make_report_zip
from castle_risk.restart_recovery import (
    find_matching_legacy_run, load_job_records, portfolio_fingerprint, save_job_records,
)


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'web' / 'portfolio_impact'
RECORDED_CASE = ROOT / 'examples' / 'recorded_analysis' / 'case.json'
SYMBOL = re.compile(r'^[A-Z][A-Z0-9.]{0,11}$')
MAX_INPUT_BYTES = 12_288
ALLOWED_TYPES = frozenset(('auto', 'stock', 'etf', 'cash'))


class PortfolioInputError(ValueError):
    pass


def parse_positions(payload):
    """Accept user dollar holdings and normalize weights with no hidden holdings."""
    if not isinstance(payload, dict) or not isinstance(payload.get('positions'), list):
        raise PortfolioInputError('Add investments and their dollar values.')
    entries = payload['positions']
    if not 1 <= len(entries) <= 24:
        raise PortfolioInputError('Enter between 1 and 24 holdings.')
    positions, total, used = [], 0., set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise PortfolioInputError('An investment row is incomplete.')
        ticker = entry.get('symbol')
        kind = entry.get('type', 'auto')
        amount = entry.get('amount')
        if not isinstance(ticker, str) or not SYMBOL.fullmatch(ticker.upper()):
            raise PortfolioInputError('Check the investment symbols (for example, AAPL).')
        ticker = ticker.upper()
        if not isinstance(kind, str) or kind not in ALLOWED_TYPES:
            raise PortfolioInputError('Choose automatic detection, stock, fund or cash for each row.')
        if kind == 'cash' and ticker not in {'USD', 'CASH'}:
            raise PortfolioInputError('For cash, use USD or CASH. Investments need their stock or fund code.')
        if ticker in used:
            raise PortfolioInputError(f'{ticker} appears twice. Combine its dollar amounts.')
        if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not math.isfinite(amount) or amount <= 0:
            raise PortfolioInputError('Each investment needs an amount greater than zero.')
        if amount > 1_000_000_000:
            raise PortfolioInputError('An individual amount is outside the supported range.')
        used.add(ticker)
        total += float(amount)
        positions.append({'symbol': ticker, 'asset_type': kind, 'amount': float(amount)})
    if total > 1_000_000_000:
        raise PortfolioInputError('The portfolio value is outside the supported range.')
    # "stock" is only an internal parser placeholder for auto rows. The marker
    # MUST reach acquisition, which resolves source metadata before modeling.
    normalized = [{'symbol': row['symbol'],
                   'asset_type': 'stock' if row['asset_type'] == 'auto' else row['asset_type'],
                   'weight': row['amount'] / total,
                   **({'input_asset_type': 'auto'} if row['asset_type'] == 'auto' else {})}
                  for row in positions]
    return normalized, total


def _real_assessment(rows, total, progress, provider_keys, *, run_directory=None, resume_archives=False):
    # Defer imports until the user clicks Analyze. The UI can load even if the
    # heavyweight, pinned assessment dependencies have not been installed.
    from castle_risk.live_assessment import run_live_assessment
    return run_live_assessment(portfolio_rows=rows, notional=total, root=ROOT,
                               massive_key=provider_keys.get('massive'),
                               anthropic_key=provider_keys.get('anthropic'),
                               include_geopolitical_search=True,
                               include_payrolls=True, progress=progress,
                               run_directory=run_directory, resume_archives=resume_archives)


def _safe_browser_case(report):
    from castle_risk.live_assessment import public_assessment_view
    return live_report_to_case(public_assessment_view(report))


class PortfolioImpactServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, *, asset_root=ASSETS, runner=None, converter=None,
                 saved_runs_root=None):
        self.asset_root = Path(asset_root).resolve()
        self.saved_runs_root = Path(saved_runs_root or ROOT / 'runtime' / 'option1_live').resolve()
        # Only the real application (no injected test runner) shows the shipped example.
        self.recorded_case_path = RECORDED_CASE if runner is None and saved_runs_root is None else None
        self.provider_keys = {}
        self.require_sources = runner is None
        self.runner = runner or (lambda rows, total, progress:
                                 _real_assessment(rows, total, progress, dict(self.provider_keys)))
        self.converter = converter or _safe_browser_case
        self.replay_saved_inputs = converter is None
        self.job_lock = threading.Lock()
        self.persist_jobs = runner is None or saved_runs_root is not None
        self.jobs = {}
        if self.persist_jobs:
            for record in load_job_records(self.saved_runs_root):
                if record['status'] == 'running':
                    record['status'] = 'interrupted'
                    record['progress'] = 'Previous session interrupted. Reconnect providers and resume this portfolio.'
                    record['updated_at'] = datetime.now(timezone.utc).isoformat()
                self.jobs[record['id']] = record
            if self.jobs:
                self._persist_jobs_locked()
        self.active_job = None
        self.latest_case = None
        self._r45_context = None
        super().__init__(address, PortfolioImpactHandler)

    def source_status(self):
        return {'data_provider_configured': bool(self.provider_keys.get('massive') or
                                                 os.environ.get('MASSIVE_API_KEY') or
                                                 os.environ.get('POLYGON_API_KEY')),
                'research_model_configured': bool(self.provider_keys.get('anthropic') or
                                                  os.environ.get('ANTHROPIC_API_KEY'))}

    def configure_sources(self, payload):
        if not isinstance(payload, dict):
            raise PortfolioInputError('Enter valid provider credentials.')
        candidate = {}
        for key in ('massive', 'anthropic'):
            value = payload.get(key)
            if value is not None:
                if not isinstance(value, str) or not 8 <= len(value.strip()) <= 400 or '\n' in value or '\r' in value:
                    raise PortfolioInputError('Check the provider credentials and retry.')
                candidate[key] = value.strip()
        if not candidate:
            raise PortfolioInputError('Enter at least one provider key.')
        with self.job_lock:
            if self.active_job:
                raise PortfolioInputError('Wait for the current analysis to finish before changing sources.')
            self.provider_keys.update(candidate)
        return self.source_status()

    def _persist_jobs_locked(self):
        """Persist sanitized job state atomically, without provider credentials."""
        if not self.persist_jobs:
            return
        fields = ('id', 'fingerprint', 'run_name', 'status', 'progress',
                  'created_at', 'updated_at', 'error')
        records = []
        for identifier, item in list(self.jobs.items())[-50:]:
            if item.get('run_name') and item.get('fingerprint'):
                records.append({key: identifier if key == 'id' else item.get(key)
                                for key in fields})
        save_job_records(self.saved_runs_root, records)

    def start_job(self, rows, total):
        fingerprint = portfolio_fingerprint(rows, total)
        with self.job_lock:
            if self.active_job:
                raise RuntimeError('An analysis is already running. Please wait for its results.')
            previous = next((
                (identifier, item) for identifier, item in reversed(list(self.jobs.items()))
                if item.get('fingerprint') == fingerprint
                and item.get('status') in {'interrupted', 'failed'}
                and isinstance(item.get('run_name'), str)
                and (self.saved_runs_root / item['run_name']).is_dir()
            ), None) if self.persist_jobs else None
            if previous:
                identifier, prior = previous
                run_name = prior['run_name']
                reused = True
            else:
                identifier = secrets.token_urlsafe(21)
                run_name = (
                    find_matching_legacy_run(self.saved_runs_root, {row['symbol'] for row in rows})
                    if self.require_sources else None
                )
                reused = run_name is not None
                run_name = run_name or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            now = datetime.now(timezone.utc).isoformat()
            initial = ('Resuming verified archived provider responses'
                       if reused else 'Checking your portfolio')
            self.jobs[identifier] = {
                'status': 'running', 'progress': initial, 'created_at': now,
                'updated_at': now, 'run_name': run_name, 'fingerprint': fingerprint,
            }
            try:
                self._persist_jobs_locked()
            except OSError as exc:
                if previous:
                    self.jobs[identifier] = prior
                else:
                    self.jobs.pop(identifier, None)
                raise RuntimeError('Could not persist analysis state. No provider requests were started.') from exc
            self.active_job = identifier

        def worker():
            def progress(message):
                with self.job_lock:
                    self.jobs[identifier]['progress'] = str(message)[:160]
                    self.jobs[identifier]['updated_at'] = datetime.now(timezone.utc).isoformat()
                    self._persist_jobs_locked()
            identity_errors = ()
            try:
                # Preserve the injected runner contract for the offline test suite.
                from castle_risk.instrument_identity import InstrumentIdentityError
                identity_errors = (InstrumentIdentityError,)
                if self.require_sources:
                    report = _real_assessment(
                        rows, total, progress, dict(self.provider_keys),
                        run_directory=self.saved_runs_root / run_name,
                        resume_archives=reused,
                    )
                else:
                    report = self.runner(rows, total, progress)
                case = self.converter(report)
                with self.job_lock:
                    self.latest_case = case
                    self._r45_context = None
                    self.jobs[identifier].update(
                        status='complete', progress='Portfolio analysis ready',
                        updated_at=datetime.now(timezone.utc).isoformat(), case=case)
                    self._persist_jobs_locked()
            except Exception as exc:
                # Provider response bodies, model prompts and credentials never reach JS.
                print(f'Portfolio analysis failed ({type(exc).__name__}); see local engine diagnostics.', flush=True)
                message = (str(exc) if isinstance(exc, identity_errors) else
                           'The analysis could not finish. Review the provider setup and try again. Saved provider data can be reused on retry.')
                with self.job_lock:
                    self.jobs[identifier].update(
                        status='failed', progress='Analysis stopped', error=message,
                        updated_at=datetime.now(timezone.utc).isoformat())
                    try:
                        self._persist_jobs_locked()
                    except OSError:
                        print('Could not persist failed-job state.', flush=True)
            finally:
                with self.job_lock:
                    self.active_job = None

        threading.Thread(target=worker, name='castle-live-assessment', daemon=True).start()
        return identifier

    def job(self, identifier):
        with self.job_lock:
            task = self.jobs.get(identifier)
            snapshot = dict(task) if task else None
        if snapshot is None:
            return None
        if snapshot.get('status') == 'complete' and 'case' not in snapshot:
            run_name = snapshot.get('run_name', '')
            report_path = self.saved_runs_root / run_name / 'derived_assessment.json'
            try:
                if (not report_path.is_symlink() and report_path.is_file()
                        and report_path.stat().st_size <= 50_000_000):
                    case = self.converter(json.loads(report_path.read_text(encoding='utf-8')))
                    snapshot['case'] = case
            except (OSError, ValueError, TypeError, KeyError):
                snapshot['status'] = 'interrupted'
                snapshot['progress'] = 'Saved report could not be restored. Retry this portfolio to resume.'
        return {key: snapshot[key] for key in
                ('status', 'progress', 'created_at', 'updated_at', 'case', 'error')
                if key in snapshot}

    def recorded_case(self):
        """The shipped, dated public analysis shown before any local run exists."""
        path = self.recorded_case_path
        if path is None or path.is_symlink() or not path.is_file():
            return None
        try:
            case = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return None
        if not isinstance(case, dict) or case.get('user_supplied') is not True or not case.get('risks'):
            return None
        case['recorded_release_example'] = True
        with self.job_lock:
            if self.latest_case is None:
                self.latest_case = case
            return self.latest_case

    def last_completed_case(self):
        """Recover saved findings and replay eligible local inputs, never fetch."""
        with self.job_lock:
            if self.latest_case is not None:
                return self.latest_case
        if not self.saved_runs_root.is_dir():
            return self.recorded_case()
        # R21 stores one timestamp-named run per completed assessment.
        for folder in sorted(self.saved_runs_root.iterdir(), reverse=True):
            if folder.is_symlink() or not folder.is_dir() or not re.fullmatch(r'\d{8}T\d{12}Z', folder.name):
                continue
            report_file = folder / 'derived_assessment.json'
            if report_file.is_symlink() or not report_file.is_file():
                continue
            try:
                if report_file.stat().st_size > 64 * 1024 * 1024:
                    continue
                report = json.loads(report_file.read_text(encoding='utf-8'))
                if self.replay_saved_inputs:
                    from castle_risk.saved_assessment_replay import upgrade_saved_assessment
                    report = upgrade_saved_assessment(report, folder)
                case = self.converter(report)
                if not case.get('user_supplied') or not case.get('risks'):
                    continue
            except (OSError, ValueError, TypeError, KeyError):
                continue
            if folder.name == '20261028T154706240078Z':
                # R45 is opt-in through genuine original source bytes. A
                # missing/changed market capture hides the add-on altogether.
                try:
                    from castle_risk.treasury_what_if_ui import (
                        append_original_research_event, verify_original_context)
                    context = verify_original_context(report_file)
                    case = append_original_research_event(case, context=context)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    print('Original Treasury what-if not admitted: ' +
                          type(exc).__name__, flush=True)
                    context = None
            else:
                context = None
            with self.job_lock:
                if self.latest_case is None:
                    self.latest_case = case
                    self._r45_context = context
                return self.latest_case
        return self.recorded_case()

    def treasury_what_if(self, payload):
        """User-selected no-network R44 model, only for the verified original."""
        if (not isinstance(payload, dict) or
                set(payload) != {'equity_return_pct', 'portfolio_notional_usd'}):
            raise PortfolioInputError('Choose an explicit stock return and portfolio value.')
        self.last_completed_case()
        with self.job_lock:
            context = self._r45_context
        if context is None:
            raise PortfolioInputError('Source-verified Treasury what-if is not available for this portfolio.')
        from castle_risk.treasury_what_if_ui import evaluate_original_what_if
        try:
            return evaluate_original_what_if(
                context, equity_return_pct=payload['equity_return_pct'],
                notional_usd=payload['portfolio_notional_usd'])
        except (ValueError, TypeError, KeyError) as exc:
            raise PortfolioInputError('Invalid source-defined stock stress request.') from exc


class PortfolioImpactHandler(BaseHTTPRequestHandler):
    server: PortfolioImpactServer

    def _send(self, code, payload):
        body = json.dumps(payload, allow_nan=False, separators=(',', ':')).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.end_headers()
        self.wfile.write(body)

    def _send_export(self, case):
        body = make_report_zip(case)
        symbols = '-'.join(p.get('symbol', '') for p in case.get('positions', []))[:68]
        symbols = re.sub(r'[^A-Za-z0-9_-]', '', symbols) or 'portfolio'
        date = re.sub(r'[^0-9]', '', (case.get('capture') or '')[:10]) or 'analysis'
        self.send_response(200)
        self.send_header('Content-Type', 'application/zip')
        self.send_header('Content-Disposition',
                         f'attachment; filename="Portfolio_Impact_{symbols}_{date}.zip"')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.end_headers()
        self.wfile.write(body)

    def _local_request(self):
        host = self.headers.get('Host', '')
        allowed = {f'127.0.0.1:{self.server.server_port}',
                   f'localhost:{self.server.server_port}'}
        if host not in allowed:
            return False
        origin = self.headers.get('Origin')
        return not origin or origin in {f'http://{address}' for address in allowed}

    def _static(self, path):
        allowed = path in {
            '/', '/index.html', '/studio.js', '/assessment_audit.js', '/assessment_audit_r126.js',
            '/studio.css', '/identity.css', '/r99.css', '/recorded_portfolios.css',
            '/data.json', '/r53_payroll_evidence.json', '/r54_macro_research_summary.json',
            # Exact original v4 design assets; names are allowlisted, not directory-wildcarded.
            '/app.js', '/styles.css', '/workspace.css', '/recovery.js',
            '/market_scenarios_20261025.json',
            '/r125_defense_semis_japan_credit_acceptance_summary_20261102.json',
            # Source-scoped, derived numeric aggregates used by the browser.
            # Raw issuer/API receipts are deliberately not served.
            '/r55_rate_ief_summary.json', '/r56_bond_portfolio_summary.json',
            '/r57_bond_weight_generalization.json',
            '/r74_verified_tlt_rate_case.json',
            '/r78_nasdaq_conditional_portfolio_with_quote.json',
        }
        allowed = allowed or (path.startswith(('/fonts/', '/licenses/')) and '..' not in path)
        if not allowed:
            self._send(404, {'error': 'Not found'})
            return
        relative = 'index.html' if path == '/' else path.lstrip('/')
        target = (self.server.asset_root / relative).resolve()
        try:
            target.relative_to(self.server.asset_root)
        except ValueError:
            self._send(404, {'error': 'Not found'})
            return
        if not target.is_file():
            self._send(404, {'error': 'Not found'})
            return
        content = target.read_bytes()
        # The approved v4 assets remain byte-for-byte unchanged. The private
        # server adds this same-origin, optional restart adapter at response time.
        if relative == 'index.html' and self.server.asset_root.name == 'portfolio_impact_v4':
            original_tag = b'<script src="./app.js" defer></script>'
            if original_tag in content:
                content = content.replace(original_tag, original_tag +
                    b'\n  <script src="./recovery.js" defer></script>', 1)
        mime = mimetypes.guess_type(str(target))[0] or 'application/octet-stream'
        self.send_response(200)
        self.send_header('Content-Type', mime + ('; charset=utf-8' if mime.startswith(('text/', 'application/javascript', 'application/json')) else ''))
        self.send_header('Content-Length', str(len(content)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        if not self._local_request():
            return self._send(403, {'error': 'Local access only'})
        path = urlsplit(self.path).path
        if path == '/api/latest':
            case = self.server.last_completed_case()
            return self._send(200, {'case': case}) if case is not None else self._send(404, {'error': 'No saved analysis'})
        if path == '/api/export/latest':
            case = self.server.last_completed_case()
            return self._send_export(case) if case is not None else self._send(404, {'error': 'No completed analysis to download'})
        if path == '/api/status':
            with self.server.job_lock:
                active_job_id = self.server.active_job
                active_job = self.server.jobs.get(active_job_id, {})
                active_job_progress = active_job.get('progress') if active_job_id else None
            return self._send(200, {'schema': 'castle.portfolio_impact.api.v1',
                                    'presentation_build': 'R33',
                                    'available': True, 'busy': active_job_id is not None,
                                    'active_job_id': active_job_id,
                                    'active_job_progress': active_job_progress,
                                    **self.server.source_status()})
        prefix = '/api/jobs/'
        if path.startswith(prefix):
            task = self.server.job(path[len(prefix):])
            return self._send(200, task) if task else self._send(404, {'error': 'Unknown analysis'})
        return self._static(path)

    def do_POST(self):
        if not self._local_request():
            return self._send(403, {'error': 'Local access only'})
        path = urlsplit(self.path).path
        if path not in ('/api/analyze', '/api/providers', '/api/what-if/treasury', '/api/scenarios/evaluate'):
            return self._send(404, {'error': 'Not found'})
        if self.headers.get('Content-Type', '').split(';')[0].strip() != 'application/json' or self.headers.get('X-Portfolio-Impact') != '1':
            return self._send(415, {'error': 'Expected the app’s JSON request'})
        length = self.headers.get('Content-Length', '')
        if not length.isdigit() or not 0 < int(length) <= MAX_INPUT_BYTES:
            return self._send(413, {'error': 'Portfolio input is too large or empty'})
        try:
            payload = json.loads(self.rfile.read(int(length)))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self._send(400, {'error': 'Invalid portfolio JSON'})
        if path == '/api/what-if/treasury':
            try:
                result = self.server.treasury_what_if(payload)
            except PortfolioInputError as exc:
                return self._send(422, {'error': str(exc)})
            return self._send(200, result)
        if path == '/api/scenarios/evaluate':
            case = self.server.last_completed_case()
            if case is None:
                return self._send(409, {'error': 'Run a portfolio analysis before building a personal scenario.'})
            from castle_risk.scenario_lab import ScenarioInputError, evaluate_scenario
            try:
                return self._send(200, evaluate_scenario(case, payload))
            except ScenarioInputError as exc:
                return self._send(422, {'error': str(exc)})
        if path == '/api/providers':
            try:
                status = self.server.configure_sources(payload)
            except PortfolioInputError as exc:
                return self._send(422, {'error': str(exc)})
            return self._send(200, {'configured': status})
        try:
            rows, total = parse_positions(payload)
        except PortfolioInputError as exc:
            return self._send(422, {'error': str(exc)})
        if self.server.require_sources and not all(self.server.source_status().values()):
            return self._send(428, {'error': 'Configure both market-data and research provider keys to run the live assessment.'})
        try:
            identifier = self.server.start_job(rows, total)
        except RuntimeError as exc:
            return self._send(409, {'error': str(exc)})
        return self._send(202, {'job_id': identifier, 'status': 'running'})

    def log_message(self, format, *args):
        # Suppress URL paths and payloads in routinely shared terminal logs.
        pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--open', action='store_true', help='Open the local interface in a browser')
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error('Choose a port from 1024 to 65535')
    server = PortfolioImpactServer(('127.0.0.1', args.port))
    url = f'http://127.0.0.1:{args.port}/'
    print(f'Portfolio Impact ready at {url}', flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()

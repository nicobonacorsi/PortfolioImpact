"""Round 28: partial acquisition evidence must not crash or imply full P&L.

This test intentionally uses only synthetic provider output and fake credentials.
It runs without Streamlit, a network connection, or paid inference.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest

from castle_risk import live_assessment_ui


ROOT = Path(__file__).resolve().parents[1]
PORTFOLIO = [{'symbol': 'QQQ', 'weight': 1., 'asset_type': 'etf'}]


class Recorder:
    def __init__(self):
        self.session_state = {}
        self.metrics = []
        self.warnings = []
        self.infos = []
        self.tables = []
        self.downloads = []

    def expander(self, *_args, **_kwargs):
        return self

    def rerun(self):
        # Unit probe continues into the report render; native AppTest exercises
        # Streamlit's real rerun and the collapsed input screen.
        return None

    def spinner(self, *_args, **_kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def caption(self, *_args, **_kwargs):
        return None

    def info(self, message, **_kwargs):
        self.infos.append(str(message))

    def warning(self, message, **_kwargs):
        self.warnings.append(str(message))

    def write(self, *_args, **_kwargs):
        return None

    def json(self, *_args, **_kwargs):
        return None

    def number_input(self, _label, **kwargs):
        return kwargs['value']

    def button(self, *_args, **kwargs):
        return kwargs.get('key') == 'opt1_live_run'

    def columns(self, count):
        return [self] * count

    def metric(self, name, value, **_kwargs):
        self.metrics.append((name, value))

    def empty(self):
        return self

    def dataframe(self, data, **_kwargs):
        self.tables.append(data)

    def download_button(self, name, data, **_kwargs):
        self.downloads.append((name, data))


def _partial_report(exposures):
    return {
        'evaluated_at_utc': datetime.now(timezone.utc).isoformat(),
        'verification_status': 'PARTIAL_SYNTHETIC_UI_FIXTURE',
        'stages': {'Massive': {'status': 'PARTIAL'}},
        'exposures': deepcopy(exposures),
        'events': [],
        'default_analysis': None,
    }


COVERAGE = {'unresolved_etf_identity_weight': .1,
            'taiwan_identified_nvda_channel_weight': .08}


@pytest.mark.parametrize('exposures', [
    {'coverage': COVERAGE},
    {'coverage': COVERAGE, 'positions': []},
    {'coverage': COVERAGE, 'positions': [{'original_symbol': 'QQQ'}]},
    {'coverage': COVERAGE, 'positions': [{'original_symbol': 'QQQ'}],
     'exposure_ledger': {'entries': []}},
    {'coverage': COVERAGE, 'positions': [{'original_symbol': 'QQQ'}],
     'exposure_ledger': {'entries': 'incorrect_type'}},
])
def test_partial_exposure_report_renders_diagnostics_without_claiming_complete_impact(
    exposures, monkeypatch,
):
    report = _partial_report(exposures)
    original = deepcopy(report)
    invocations = []

    def synthetic_run(**kwargs):
        invocations.append(kwargs)
        return report

    monkeypatch.setattr(live_assessment_ui, 'run_live_assessment', synthetic_run)
    st = Recorder()
    live_assessment_ui.render_live_assessment(
        st, portfolio_rows=PORTFOLIO, notional=100000., holdings={}, root=ROOT,
        massive_key='SYNTHETIC_KEY_A', anthropic_key='SYNTHETIC_KEY_B',
    )
    assert len(invocations) == 1
    assert any(name == 'Portfolio value' for name, _ in st.metrics)
    assert any('Position-level exposure evidence is missing or incomplete' in w
               for w in st.warnings)
    assert any('Partial run' in w for w in st.warnings)
    assert st.downloads == []
    assert st.tables == []
    assert report == original, 'Displaying the incomplete report must not alter evidence'
    assert not any('SYNTHETIC_KEY_' in msg for msg in st.warnings + st.infos)


def test_complete_original_ledger_still_allows_an_explicitly_partial_export(monkeypatch):
    exposures = {
        'coverage': COVERAGE,
        'positions': [{'original_symbol': 'QQQ', 'status': 'unresolved_etf'}],
        'exposure_ledger': {'entries': [
            {'original_symbol': 'QQQ', 'portfolio_weight': 1., 'status': 'unresolved_etf'}]},
    }
    report = _partial_report(exposures)
    report['limitations'] = ['SYNTHETIC_PARTIAL_REPORT_NOT_AGENT_ACCEPTANCE']
    monkeypatch.setattr(live_assessment_ui, 'run_live_assessment', lambda **_: report)
    st = Recorder()
    live_assessment_ui.render_live_assessment(
        st, portfolio_rows=PORTFOLIO, notional=100000., holdings={}, root=ROOT,
        massive_key='SYNTHETIC_KEY_A', anthropic_key='SYNTHETIC_KEY_B',
    )
    assert not any('Position-level exposure evidence is missing' in w for w in st.warnings)
    assert len(st.downloads) == 1
    assert 'SYNTHETIC_PARTIAL_REPORT_NOT_AGENT_ACCEPTANCE' in st.downloads[0][1]
    assert 'SYNTHETIC_KEY_' not in st.downloads[0][1]

@pytest.mark.parametrize('default_analysis', [None, {'synthetic_case': True}])
def test_missing_original_position_ledger_blocks_all_event_impact_and_export(
    default_analysis, monkeypatch,
):
    report = _partial_report({'coverage': COVERAGE})
    report['default_analysis'] = default_analysis
    report['events'] = [{'title': 'SYNTHETIC_NON_ADMISSIBLE_EVENT',
                         'impact': {'delta_pnl_usd': 999999.}}]
    monkeypatch.setattr(live_assessment_ui, 'run_live_assessment', lambda **_: report)
    st = Recorder()
    live_assessment_ui.render_live_assessment(
        st, portfolio_rows=PORTFOLIO, notional=100000., holdings={}, root=ROOT,
        massive_key='SYNTHETIC_KEY_A', anthropic_key='SYNTHETIC_KEY_B',
    )
    assert any('cannot certify ETF look-through' in w for w in st.warnings)
    assert not st.downloads
    assert len(st.metrics) == 3  # Aggregate figures, not event-P&L metrics.
    assert not st.tables

"""Concise live Option 1 screen; keys stay in sidebar/session memory."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import secrets
from pathlib import Path

from .live_assessment import public_assessment_view, run_live_assessment
from .option1_brief import portfolio_input_fingerprint
from .economic_binding import BOUND


def _money(value):
    return 'Unknown' if value is None else f'${value:,.2f}'


def _range(value, *, probability=False):
    if value is None:
        return 'Unknown'
    if probability:
        return f'{value[0]:.1%} to {value[1]:.1%}'
    return f'${value[0]:,.2f} to ${value[1]:,.2f}'


def _evidence_rows(rows):
    """Nested evidence as JSON text: a field such as a tolerance limit can be a number in one row and words in another."""
    return [{key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
             for key, value in row.items()} for row in rows]


def _live_stage_notice(report):
    """Describe operational results without changing any coverage or acceptance gate."""
    stages = report.get('stages', {})
    failure_statuses = {'BLOCKED_MISSING_CREDENTIAL', 'FAILED', 'PARTIAL'}
    has_failure = any(stage.get('status') in failure_statuses for stage in stages.values())
    if not report.get('verification_status', '').startswith('PARTIAL') and not has_failure:
        return None
    # A completed search may retain no admissible contract. This is different
    # from an acquisition error: rejected candidates and unknown outcomes stay
    # in the report, including its original conservative verification_status.
    no_supported_contract = (
        report.get('verification_status') == 'PARTIAL_LIVE_RUN_COVERAGE_OR_REVIEW_GAPS'
        and stages.get('Massive', {}).get('status') == 'ACQUIRED'
        and stages.get('Claude', {}).get('status') == 'NO_RETRIEVED_CONTRACTS_WITHIN_BOUNDED_SEARCH'
        and stages.get('Kalshi', {}).get('status') == 'READS_COMPLETED'
        and ('GeopoliticalSearch' not in stages or stages['GeopoliticalSearch'].get('status') in {
            'RECORDED_BOUNDED_TWO_VENUE_SEARCH', 'NOT_APPLICABLE_NO_GEOPOLITICAL_TARGETS'})
        and not has_failure
    )
    if no_supported_contract:
        return ('info', 'The analysis returned results. No contracts passed the available source checks '
                'for the discovered risks in this search. Available macro scenarios and unpriced risks '
                'are shown below.')
    return ('warning', 'Partial run: one or more acquisition or review stages did not complete. '
            'Public contract inspection is not a verified Massive + Claude assessment.')



# The nonce and digest never enter a report, JSON export, repository or provider
# receipt. A browser session's two nonempty keys are not interchangeable merely
# because both remain present: swapping an account must invalidate past results.
_SESSION_NONCE_KEY = '_opt1_live_private_context_nonce'


def _connection_context_digest(session_state, *, massive_key, anthropic_key):
    """Privacy-preserving, session-local binding to the *actual* credentials.

    This is NOT an authentication attempt and does not claim a provider is
    reachable. The random keyed digest is used only to discard a previously
    displayed assessment if an input credential changes in the same session.
    """
    nonce = session_state.get(_SESSION_NONCE_KEY)
    if not isinstance(nonce, bytes) or len(nonce) != 32:
        nonce = secrets.token_bytes(32)
        session_state[_SESSION_NONCE_KEY] = nonce
    digest = hashlib.blake2b(key=nonce, digest_size=32, person=b'castle-live-v1')
    for name, value in (('Massive', massive_key), ('Anthropic', anthropic_key)):
        if value is not None and not isinstance(value, str):
            raise ValueError('Connection credentials must be text or absent')
        name_bytes = name.encode('ascii')
        credential = (value or '').encode('utf-8')
        for data in (name_bytes, credential):
            digest.update(len(data).to_bytes(8, 'big'))
            digest.update(data)
    return digest.hexdigest()


def _live_input_signature(session_state, *, portfolio_rows, notional, holdings,
                          scenarios, horizon, massive_key, anthropic_key, model):
    """Bind displayed source data to holdings, settings and source credentials.

    Do not save the returned signature in exported evidence: it is private to
    the local UI session and cannot authenticate the vendor's underlying data.
    """
    return (
        portfolio_input_fingerprint(portfolio_rows, notional, holdings),
        json.dumps(scenarios, sort_keys=True, separators=(',', ':'), allow_nan=False),
        int(horizon),
        _connection_context_digest(session_state, massive_key=massive_key,
                                   anthropic_key=anthropic_key),
        model,
    )

def _has_reconcilable_exposure_details(exposure):
    """Require original-position and constituent-ledger rows before showing impacts.

    A source can return aggregate coverage without position-level evidence.
    Neither an absent nor an empty ledger establishes zero positions or zero
    exposure. In that case the UI must fail closed rather than manufacture rows
    or export a seemingly complete portfolio scenario.
    """
    if not isinstance(exposure, dict):
        return False
    original_positions = exposure.get('positions')
    ledger = exposure.get('exposure_ledger')
    # The exposure engine intentionally freezes its constituent ledger into a
    # tuple. An immutable tuple is valid evidence, not a missing source.
    # Reject absent/empty entries and scalar/dict impostors as before.
    return (isinstance(original_positions, (list, tuple)) and bool(original_positions)
            and isinstance(ledger, dict)
            and isinstance(ledger.get('entries'), (list, tuple)) and bool(ledger['entries']))


def render_live_assessment(st, *, portfolio_rows, notional, holdings, root: Path,
                           massive_key=None, anthropic_key=None, model=None):
    missing = [name for name, value in (('Massive', massive_key), ('Claude', anthropic_key)) if not value]
    if missing:
        st.info('Add ' + ', '.join(missing) + ' credentials under Provider settings for the full analysis. '
                'Without them, only public market data are available.')
    # Keep assumptions across workspace navigation; Streamlit removes widget
    # keys while their screen is not rendered.
    settings = st.session_state.get('_castle_analysis_settings', {})
    action = st.empty()
    with st.expander('Analysis settings', expanded=False):
        st.caption('Optional. The defaults are ready to use. Stress assumptions are separate from the '
                   'main outcome model; a contract needs its own economic mapping.')
        horizon = st.number_input('Live investigation window (days)', min_value=1, max_value=366,
                                  value=settings.get('horizon', 90), key='opt1_live_horizon')
        fed = st.number_input('FOMC family: assumed SPY stress contrast (%)',
                              min_value=-50., max_value=50., value=settings.get('fed', -1.), step=.25, key='opt1_live_fed') / 100
        cpi = st.number_input('CPI family: assumed SPY stress contrast (%)',
                              min_value=-50., max_value=50., value=settings.get('cpi', -.75), step=.25, key='opt1_live_cpi') / 100
        spread = st.number_input('Benchmark scenario range around each point (percentage points)',
                                  min_value=0., max_value=20., value=settings.get('spread', 1.), step=.25, key='opt1_live_spread') / 100
        nvda = st.number_input('Taiwan: assumed identified NVDA YES-minus-NO return (%)',
                               min_value=-100., max_value=100., value=settings.get('nvda', -20.), step=5., key='opt1_live_taiwan') / 100
        st.caption('Taiwan uses direct NVDA plus verified NVDA holdings inside ETFs. The rest of the '
                   'portfolio remains unknown and receives a break-even sensitivity, not a zero-return assumption. '
                   'An unmatched broad event has no quote-implied priced-in weight.')
    st.session_state['_castle_analysis_settings'] = {
        'horizon': int(horizon), 'fed': fed * 100, 'cpi': cpi * 100,
        'spread': spread * 100, 'nvda': nvda * 100,
    }
    scenarios = {
        'FED_POLICY': {'benchmark_return_contrast': fed, 'benchmark_contrast_interval': [fed-spread, fed+spread]},
        'US_INFLATION': {'benchmark_return_contrast': cpi, 'benchmark_contrast_interval': [cpi-spread, cpi+spread]},
        'TAIWAN_SUPPLY': {'nvda_return_contrast': nvda, 'nvda_contrast_interval': [nvda, nvda]},
    }
    signature = _live_input_signature(
        st.session_state, portfolio_rows=portfolio_rows, notional=notional,
        holdings=holdings, scenarios=scenarios, horizon=horizon,
        massive_key=massive_key, anthropic_key=anthropic_key, model=model,
    )
    has_current_result = (st.session_state.get('option1_live_report') is not None
                          and signature == st.session_state.get('option1_live_signature'))
    if action.button('Refresh analysis' if has_current_result else 'Analyze portfolio',
                     key='opt1_live_run', type='secondary' if has_current_result else 'primary'):
        st.session_state.pop('option1_live_report', None)
        st.session_state.pop('option1_live_signature', None)
        status = st.empty()
        try:
            with st.spinner('Looking through holdings, finding risks and checking markets...'):
                report = run_live_assessment(portfolio_rows=portfolio_rows, notional=notional,
                    root=root, massive_key=massive_key, anthropic_key=anthropic_key, model=model,
                    scenario_settings=scenarios, horizon_days=int(horizon), progress=status.info,
                    include_geopolitical_search=True, include_payrolls=True)
            st.session_state['option1_live_report'] = report
            st.session_state['option1_live_signature'] = signature
            st.session_state.pop('opt1_live_event', None)
            # Rerender the entry with its editor collapsed and results in front.
            st.rerun()
        except (ValueError, TypeError, KeyError, OSError, RuntimeError):
            st.error('The analysis could not finish. Check provider connections and try again. '
                     'Details are available in the local provider receipts.')
        finally:
            status.empty()
    report = st.session_state.get('option1_live_report')
    if report is None:
        st.caption('Uses your connected providers. Allow a few minutes; progress appears here.')
        return
    if signature != st.session_state.get('option1_live_signature'):
        st.info('Portfolio, ETF holdings, provider connection or scenario settings changed. '
                'Run a new assessment; earlier quotes and impacts are hidden.')
        return
    st.caption('Analysis completed · ' + report['evaluated_at_utc'])
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(report['evaluated_at_utc'])).total_seconds()
    if age > 300:
        st.warning('This recorded live run is over five minutes old. Its quotes are dated evidence; '
                   'run again before treating them as current.')
    stages = report.get('stages', {})
    notice = _live_stage_notice(report)
    if notice is not None:
        level, message = notice
        if level == 'info':
            st.caption(message)
        else:
            st.warning(message)
    exposure = report['exposures']
    if not report.get('default_analysis') or not _has_reconcilable_exposure_details(exposure):
        left, middle, right = st.columns(3)
        left.metric('Portfolio value', _money(notional))
        middle.metric('Unresolved ETF identity', f"{exposure['coverage']['unresolved_etf_identity_weight']:.1%}")
        right.metric('Identified NVDA channel', _money(notional * exposure['coverage']['taiwan_identified_nvda_channel_weight']))
    if not _has_reconcilable_exposure_details(exposure):
        st.warning('Position-level exposure evidence is missing or incomplete in this assessment. '
                   'The aggregate coverage above is partial and cannot certify ETF look-through, '
                   'event P&L, or a derived export. No missing positions have been assigned zero.')
        with st.expander('Available acquisition-stage diagnostics', expanded=False):
            st.json(stages)
            for key in ('acquisition_attempts', 'holdings_quality_review', 'provider_receipts'):
                rows = report.get(key)
                if isinstance(rows, list) and rows:
                    st.dataframe(rows, hide_index=True, width='stretch')
        return
    if report.get('default_analysis'):
        from .default_assessment_ui import render_reviewer_dashboard
        render_reviewer_dashboard(st, report)
        st.download_button('Export derived assessment (JSON)',
            json.dumps(public_assessment_view(report), indent=2, allow_nan=False), file_name='castle_live_assessment.json',
            mime='application/json', key='opt1_live_export')
        return
    events = sorted(report.get('events', []),
                    key=lambda event: event.get('event_ranking', event['ranking'])['display_order'])
    if not events:
        st.warning('No source-admissible candidate was retained by this bounded investigation. '
                   'This is not evidence that the portfolio has no event risk.')
    else:
        st.dataframe([{
            'Order': event.get('event_ranking', event['ranking'])['display_order'], 'Event': event['title'],
            'Pricing interval Q': _range(event['probability']['pricing_measure_Q_interval'], probability=True),
            '|YES − NO| (USD)': _money(event.get('event_ranking', event['ranking'])['abs_delta_pnl_usd']),
            'Magnitude range (USD)': _range(event.get('event_ranking', event['ranking'])['abs_delta_pnl_interval_usd']),
            'Impact scope': event['impact']['scope'],
            'Question origin': 'Source-defined settlement question' if event.get('parent_event_id') else 'Original research question',
            'Family stress contrast (USD)': _money(event.get('family_stress', {}).get('delta_pnl_usd')),
        } for event in events], hide_index=True, width='stretch')
        st.caption('Order uses modeled absolute YES-minus-NO movement, with unknown totals last. '
                   'Family stress is a separate sensitivity and does not determine the event ranking. '
                   'Related parent and source-defined questions describe overlapping risks and must not be added.')
        by_id = {event['event_id']: event for event in events}
        chosen = st.selectbox('Inspect event evidence and economic scenario', list(by_id),
                              format_func=lambda key: by_id[key]['title'], key='opt1_live_event')
        event = by_id[chosen]
        st.markdown('#### What event are we evaluating?')
        st.write(event['question_we_care_about'])
        st.caption(event['event_definition_scope'] + ' · ' + event.get('candidate_origin', 'CLAUDE_RESEARCH_PROPOSAL'))
        if event.get('parent_event_id'):
            st.info('Source-defined binary question · local derivation. The original research hypothesis remains '
                    'unmatched; this contract asks a different, narrower question.')
            st.write('Original hypothesis: ' + event['original_macro_question'])
            with st.expander('How this source question was selected', expanded=False):
                st.write(event['refinement_difference'])
                st.write(event['scenario_scope']['NO'])
                st.caption('Selection is deterministic among already-associated source-compatible contracts; '
                           'it does not select the most attractive price or loss.')
                st.json(event['scenario_scope'])
        st.dataframe([{
            'Question we care about': c.get('question_we_care_about'),
            'Market question': c.get('market_question'),
            'Difference': c.get('difference'), 'Decision': c.get('decision'),
        } for c in event['matching_comparisons']], hide_index=True, width='stretch')
        st.markdown('#### What probability does the evidence support?')
        q = event['probability']['pricing_measure_Q_interval']
        if q is None:
            st.info('Event probability is unknown. A related contract price is not automatically the probability '
                    'of this economic question; the matching decision above explains the gap.')
        else:
            st.metric('Quote-implied pricing interval Q', _range(q, probability=True))
        st.caption('Physical probability P is not estimated. Bid/ask bounds and receipt freshness describe '
                   'the exact settlement claim; they are not a statistical probability confidence interval.')
        impact = event['impact']
        st.markdown('#### What would the portfolio gain or lose?')
        binding = event.get('economic_binding', {})
        if binding and binding.get('status') != BOUND:
            st.info('Economic event impact is not identified: no complete declared YES/NO mapping is bound to this '
                    'specific settlement question. Family sensitivities are shown separately below.')
        if binding.get('legacy_numeric_payload_retained'):
            # Legacy API consumers retain numeric sensitivities, but the v2 UI
            # and acceptance never promote them to identified event effects.
            impact = {'status': 'EVENT_SPECIFIC_ASSUMPTIONS_REQUIRED',
                      'scope': 'NO_EVENT_CONDITIONAL_PORTFOLIO_RETURN_IDENTIFIED',
                      'delta_pnl_usd': None, 'delta_pnl_interval_usd': None,
                      'priced_in': {}, 'positions': []}
        label = ('Declared NVDA channel sensitivity: ' if impact.get('event_conditional_total_identified') is False
                 else 'Conditional scenario: ')
        st.caption(label + impact['scope'] + ' · ' + impact['status'])
        left, right = st.columns(2)
        left.metric('Full portfolio YES-minus-NO contrast', _money(impact.get('delta_pnl_usd')))
        right.metric('Model contrast range', _range(impact.get('delta_pnl_interval_usd')))
        if impact.get('delta_pnl_usd') is None:
            st.write('Known component contrast: **' + _money(impact.get('known_component_delta_pnl_usd')) + '**. '
                     'The total remains unknown; missing positions are not assigned zero.')
        st.dataframe(impact.get('positions', []), hide_index=True, width='stretch')
        if impact.get('transfer_status') == 'FAIL':
            st.warning('This estimated association failed its chronological transfer check. The numbers are '
                       'a declared sensitivity scenario, not a validated forward forecast.')
        if impact.get('price_domain_status') == 'RANGE_EXTENDS_BELOW_MINUS_100_PERCENT':
            st.warning('The linear sensitivity range includes a branch return below -100%. '
                       'Those endpoints are outside the nonnegative-price domain and do not form '
                       'an economically feasible prediction envelope. They have not been clipped.')
        if 'break_even' in impact:
            breakeven = impact['break_even'].get('remaining_return_contrast')
            st.write('Return contrast required on the remaining portfolio to offset the identified NVDA channel: '
                     + ('**unknown**.' if breakeven is None else f'**{breakeven:+.2%}**.'))
        stress = event.get('family_stress')
        if stress:
            with st.expander('Declared family/channel sensitivity — separate from this event', expanded=False):
                st.write(stress['interpretation'])
                st.write('Full portfolio stress contrast: **' + _money(stress.get('delta_pnl_usd')) +
                         '**; range: ' + _range(stress.get('delta_pnl_interval_usd')) + '.')
                st.write('Identified component: **' + _money(stress.get('known_component_delta_pnl_usd')) + '**.')
                st.dataframe(stress.get('positions', []), hide_index=True, width='stretch')
                if stress.get('transfer_status') == 'FAIL':
                    st.warning('The sensitivity includes an association that failed the chronological transfer check. '
                               'The failure remains visible and no forward forecast is certified.')
        st.markdown('#### What is already priced in?')
        priced = impact.get('priced_in', {})
        if priced.get('yes_pnl_usd') is None:
            st.info('A full priced-in residual requires both an admitted weight for this exact event and a '
                    'complete economic contrast. The missing input is left unknown.')
            component = impact.get('known_component_priced_in', {})
            if component.get('yes_pnl_usd') is not None:
                st.write('Known component only: YES ' + _money(component['yes_pnl_usd']) +
                         '; NO ' + _money(component['no_pnl_usd']) + '.')
        else:
            left, right = st.columns(2)
            left.metric('YES residual: (1 − q) × ΔV', _money(priced['yes_pnl_usd']))
            right.metric('NO residual: −q × ΔV', _money(priced['no_pnl_usd']))
            st.write('YES range: ' + _range(priced['yes_pnl_interval_usd']) +
                     '; NO range: ' + _range(priced['no_pnl_interval_usd']) + '.')
        st.caption('Transferring the prediction-market pricing weight q to these asset-price branches is '
                   'an explicit assumption. Centering at q removes the assumed expected component by construction; '
                   'it does not measure how much an equity market has already incorporated.')
        with st.expander('Estimated instrument sensitivities and chronological checks', expanded=False):
            st.warning('An earlier forward test of transferring daily coefficients to later announcements failed and stays on record. Daily close-to-close '
                       'associations include other news and are not isolated causal policy/inflation shocks.')
            st.caption('Training cutoff: ' + report['empirical']['train_end'] +
                       '. Later observations are assessed without refitting. HC3 ranges describe parameter '
                       'uncertainty, not prediction intervals.')
            for name, model_report in [('Daily beta', report['empirical']['daily_betas']),
                *[(family + ' event days', study) for family, study in report['empirical']['release_day_studies'].items()]]:
                st.write('**' + name + '**')
                st.dataframe([{'Instrument': symbol, 'Beta': fit['beta'],
                    'Parameter interval': str(fit['beta_interval']), 'Training sessions': fit['n_train'],
                    'Held-out sessions': fit['n_holdout'], 'Transfer check': fit['transfer_check']['status']}
                    for symbol, fit in model_report['instruments'].items()], hide_index=True, width='stretch')
    with st.expander('Acquisition receipts, ownership and limitations', expanded=False):
        st.json(stages)
        st.dataframe(_evidence_rows(exposure['positions']), hide_index=True, width='stretch')
        st.dataframe(exposure['exposure_ledger']['entries'], hide_index=True, width='stretch')
        st.caption('A successful HTTP response does not establish complete or usable holdings. '
                   'Taxonomy access, source-weighted composition, snapshot completeness, and unresolved identities are separate checks.')
        st.dataframe(report.get('acquisition_attempts', []), hide_index=True, width='stretch')
        st.dataframe(report.get('holdings_quality_review', []), hide_index=True, width='stretch')
        st.dataframe(report.get('provider_receipts', []), hide_index=True, width='stretch')
        for limitation in report['limitations']:
            st.write('- ' + limitation)
    st.download_button('Export derived assessment (JSON)',
        json.dumps(public_assessment_view(report), indent=2, allow_nan=False), file_name='castle_live_assessment.json',
        mime='application/json', key='opt1_live_export')

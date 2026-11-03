"""Read an independently rederived, source-hashed, DERIVED-ONLY historic case.

The underlying private Massive/Claude/provider archive is never redistributed.
This case cannot be applied to a different user portfolio and cannot be
presented as a fresh quote or an empirically validated event effect.
"""
from __future__ import annotations

from hashlib import sha256
import math
import json
from pathlib import Path

from .reviewer_decision import _portfolio_key

CASE_PORTFOLIO = (100000.0, (('BND', 'etf', .6), ('IEF', 'etf', .4)))


class Round22RecordedEvidenceError(ValueError):
    pass


def verified_round22_case(root: Path, *, portfolio: dict | None = None) -> dict:
    folder = Path(root)/'data'/'round22_verified'
    try:
        manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        if (manifest.get('schema') != 'castle.round22.derived_only_manifest.v1'
                or manifest.get('file') != 'derived_receipt.json'
                or manifest.get('original_provider_archive_included') is not False):
            raise ValueError('Unrecognized derived-only source inventory')
        body = (folder/'derived_receipt.json').read_bytes()
        if sha256(body).hexdigest() != manifest['sha256']:
            raise ValueError('Recorded derived case SHA-256 mismatch')
        receipt = json.loads(body)
        if (receipt.get('schema') != 'castle.option1.round22.private_verification.v1'
                or receipt.get('inventory_members_sha256_verified') != 251
                or receipt.get('original_full_authenticated_acceptance') is not False
                or receipt.get('total_exact_scenarios_completed') != 1):
            raise ValueError('Private evidence and acceptance must keep their independent statuses')
        cases = [x for x in receipt['portfolio_cases'] if x.get('case') == 'bonds']
        if len(cases) != 1 or cases[0]['new_exact_scenarios_complete'] != 1:
            raise ValueError('One exact bond scenario was not reproduced')
        case = cases[0]['scenarios'][0]
        if (case['contract_id'] != 'KXFEDDECISION-26OCT-C25'
                or case.get('q_interval') != [0.0, .01]
                or case.get('portfolio_notional_usd') != 100000.0
                or case.get('causal_effect_identified') is not False
                or case.get('physical_event_probability_identified') is not False
                or case.get('asset_implied_pricing_weight_identified') is not False):
            raise ValueError('Settlement and non-identification constraints did not reproduce')
        if portfolio is not None and _portfolio_key(portfolio) != CASE_PORTFOLIO:
            raise ValueError('The dated source case cannot be transferred to another portfolio')
        by_symbol = {p['symbol']: p for p in case['positions']}
        if set(by_symbol) != {'BND', 'IEF'} or any(by_symbol[s]['weight'] != w for s,w in [('BND',.6),('IEF',.4)]):
            raise ValueError('Portfolio weights do not reconcile to the source case')
        return {'scenario':case,'fund_reviews':cases[0]['fund_class_reviews'],
                'original_evidence_sha256':receipt['original_archive_sha256'],
                'source_capture_date':manifest['capture_day'],
                'scope':'RECORDED_VERIFIED_DERIVED_ONLY_NOT_CURRENT_MARKET'}
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        raise Round22RecordedEvidenceError('Recorded independent Round 22 case cannot be authenticated') from exc


def priced_in_quote_sensitivity(case: dict) -> dict:
    """Compute *assumed* priced-in sensitivity, not an inferred asset pricing weight.

    Quote Q is only a one-sided, source-contract pricing interval. The actual
    asset-implied mixture weight pi is unknown, so the Q-based bands below are
    counterfactuals under the *additional* assumption pi is inside source Q.
    This deliberately does not revise any saved event or model estimate.
    """
    def bounds(value, domain=None):
        if (not isinstance(value, (list, tuple)) or len(value) != 2
                or any(isinstance(x, bool) or not isinstance(x, (int, float))
                       or not math.isfinite(x) for x in value)
                or value[0] > value[1]):
            raise Round22RecordedEvidenceError('Invalid model or source-quote sensitivity interval')
        if domain is not None and (value[0] < domain[0] or value[1] > domain[1]):
            raise Round22RecordedEvidenceError('Quote interval outside probability domain')
        return tuple(float(x) for x in value)

    qlo, qhi = bounds(case['q_interval'], (0., 1.))
    low, high = bounds(case['yes_minus_no_pnl_interval_usd'])
    central = float(case['yes_minus_no_pnl_usd'])
    reference = float(case['q_reference'])
    if (not math.isfinite(central) or not math.isfinite(reference)
            or not low <= central <= high or not qlo <= reference <= qhi):
        raise Round22RecordedEvidenceError('Scenario centre or quote reference outside its admissible range')

    def product(a, b):
        products = [x * y for x in a for y in b]
        if not all(math.isfinite(x) for x in products):
            raise Round22RecordedEvidenceError('Nonfinite quoted repricing sensitivity')
        return [min(products), max(products)]

    result = {
        'status': 'EXPLICIT_PI_IN_SOURCE_Q_INTERVAL_ASSUMPTION_ONLY',
        'q_interval': [qlo, qhi], 'model_contrast_interval_usd': [low, high],
        'reference_quote_q': reference,
        'central_yes_residual_interval_usd': product((1-qhi, 1-qlo), (central,)),
        'central_no_residual_interval_usd': product((-qhi, -qlo), (central,)),
        'all_model_yes_residual_interval_usd': product((1-qhi, 1-qlo), (low,high)),
        'all_model_no_residual_interval_usd': product((-qhi, -qlo), (low,high)),
        'physical_probability_P_identified': False,
        'actual_asset_implied_pi_identified': False,
    }
    return result


def render_recorded_case(st, *, root: Path, portfolio: dict):
    """Keep source clocks and acceptance semantics separate from Round 11."""
    if _portfolio_key(portfolio) != CASE_PORTFOLIO:
        return
    try:
        record = verified_round22_case(root, portfolio=portfolio)
    except Round22RecordedEvidenceError:
        st.warning('The newer exact-event case cannot be verified and was not shown.')
        return
    case = record['scenario']
    try:
        sensitivity = priced_in_quote_sensitivity(case)
    except Round22RecordedEvidenceError:
        st.warning('The dated model and quote intervals cannot be reconciled; no scenario was shown.')
        return
    st.markdown('### One complete *conditional* YES / NO scenario — 22 October source capture')
    st.caption('SEPARATE OCTOBER 22 AUTHENTICATED CAPTURE, rederived offline with 251 matching source-inventory hashes. '
               'Do not combine its probability with October 21 or with later provider-refresh attempts. '
               'Opening this case does not call Massive, Claude, Kalshi or Polymarket.')
    st.dataframe([{'Contract': case['contract_id'],
                   'Exact settlement': case['settlement_question'],
                   'Q from one-sided quote': '0.0–1.0% (not physical P)',
                   'YES minus NO portfolio scenario': f"${case['yes_minus_no_pnl_usd']:+,.2f}",
                   'Declared model range (not confidence)': (
                       f"${sensitivity['model_contrast_interval_usd'][0]:+,.2f} to "
                       f"${sensitivity['model_contrast_interval_usd'][1]:+,.2f}"),
                   'YES residual (assumed π=Q)': f"${case['yes_residual_usd_under_assumed_pi_equals_q']:+,.2f}",
                   'NO residual (assumed π=Q)': f"${case['no_residual_usd_under_assumed_pi_equals_q']:+,.2f}",
                   'Method': 'Explicit historical sensitivity transfer — NOT causally certified'}],
                 hide_index=True, width='stretch')
    # A reviewer must be able to export exactly the displayed, independently
    # verified case without copying a grid or conflating separate source dates.
    # Only derived evidence is exported: no raw vendor payload or API credential.
    download = getattr(st, 'download_button', None)
    if callable(download):
        reviewer_export = {
            'schema': 'castle.option1.verified_exact_contract_review.v1',
            'evidence_mode': record['scope'],
            'source_capture_date': record['source_capture_date'],
            'original_evidence_sha256': record['original_evidence_sha256'],
            'scenario': case,
            'assumed_pi_sensitivity': sensitivity,
            'fund_source_quality': record['fund_reviews'],
            'evidence_limitations': {
                'physical_event_probability_identified': False,
                'asset_implied_pricing_weight_identified': False,
                'event_effect_causally_identified': False,
                'new_live_provider_request': False,
                'provider_raw_payload_included': False,
            },
        }
        download('Export verified exact-event scenario (JSON)',
                 data=json.dumps(reviewer_export, indent=2, sort_keys=True, allow_nan=False),
                 file_name='castle-bnd-ief-exact-event-2026-10-22.json',
                 mime='application/json', key='round25_exact_case_export')
    def dollars(bounds):
        return f"${bounds[0]:+,.2f} to ${bounds[1]:+,.2f}"
    st.caption('QUOTE SENSITIVITY, NOT IDENTIFIED PRICING: only under the extra assumption that asset-implied '
               'π lies within this source-contract Q interval, the central scenario has YES residual '
               f"{dollars(sensitivity['central_yes_residual_interval_usd'])} and NO residual "
               f"{dollars(sensitivity['central_no_residual_interval_usd'])}. "
               'The actual asset-implied π and physical event probability P are NOT identified. '
               'Model uncertainty is separate and much wider; no confidence or execution claim is made.')
    st.caption('NO means every other settlement outcome, including source exceptions, with its mixture bounded '
               'by the admitted joint source distribution. The previous adverse forward-transfer audit is retained; '
               'these numbers are scenario assumptions, not predicted returns or a trade recommendation.')
    with st.expander('Position reconciliation, source quality and falsification limits', expanded=False):
        st.dataframe([{'Position': row['symbol'], 'Original weight': f"{row['weight']:.0%}",
                       'Assumed YES-minus-NO return': f"{row['return_contrast']:+.3%}",
                       'Scenario contribution': f"${row['delta_pnl_usd']:+,.2f}",
                       'Basis': row['basis']} for row in case['positions']],
                     hide_index=True, width='stretch')
        st.dataframe([{'ETF': row['fund_symbol'], 'Source-class status': row['status'],
                       'Signed NAV': row.get('source_signed_nav_sum'),
                       'Non-cash short rows': row.get('noncash_short_rows'),
                       'Equity issuer look-through': row.get('eligible_for_equity_issuer_lookthrough')}
                      for row in record['fund_reviews']], hide_index=True, width='stretch')
        st.caption('BND source-identity and short-position anomalies remain unresolved; IEF source bond-class composition '
                   'is accepted but original equity-style look-through is not. The scenario uses each ETF’s OWN market return once, '
                   'not unverified additive constituent shocks.')
        # The production Streamlit UI has .json; recorder-only test shims may not.
        # The substantive case, assumptions and provenance are already visible.
        if callable(getattr(st, 'json', None)):
            st.json({'quote_basis':case['q_interval'], 'no_branch':case['no_conditional_mixture'],
                     'model_scope':case['interpretation'], 'archive_sha256':record['original_evidence_sha256']})

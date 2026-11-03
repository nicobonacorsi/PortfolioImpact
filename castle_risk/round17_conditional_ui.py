"""Small reviewer-facing panel for an independently source-checked Taiwan stress.

The derived numerical snapshot is readable without redistributing licensed
Massive original responses. It is NOT a live current quote, an invasion forecast
or a complete YES-event expected return. It never alters the original engine.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path


def verified_record(root: Path) -> dict:
    folder=Path(root)/'data'/'round17_geopolitical_stress'
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    name=manifest['file']
    if name!='tech_taiwan_reference.json' or manifest['schema']!='CASTLE_R17_GEO_DERIVED_MANIFEST_V1':
        raise ValueError('Unrecognized source-derived stress manifest')
    raw=(folder/name).read_bytes()
    if sha256(raw).hexdigest()!=manifest['sha256']:
        raise ValueError('Derived historical stress has changed since its source audit')
    result=json.loads(raw)
    if (result['schema']!='CASTLE_ROUND17_GEO_CONDITIONAL_STRESS_V1'
            or result['status']!='DECLARED_HISTORICALLY_ANCHORED_CONDITIONAL_SCENARIO_NOT_CAUSAL_ESTIMATE'
            or result['public_contract_source']['contract_id']!='567621'
            or result['notional_usd']!=100000
            or result['portfolio_weights']!={'NVDA':.35,'QQQ':.45,'TLT':.2}):
        raise ValueError('Unexpected stress definition or application scope')
    return result


def render_recorded_taiwan_stress(st, *, root: Path) -> None:
    st.markdown('#### Taiwan: explicit one-day portfolio stress, not an identified invasion loss')
    try:
        x=verified_record(root)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        st.error('Round 17 historical reference stress is unavailable or failed its integrity checks.')
        return
    src=x['public_contract_source']; xq=x['market_price']; stress=x['assumed_yes_minus_no_incremental_pnl_usd']
    st.caption('Recorded public YES question: '+src['question']+' · Polymarket market '+src['contract_id']+
               ' · Gamma retrieved '+src['gamma_receipt_at_utc']+'. This market is not equivalent to any blockade or general supply-chain disruption.')
    st.caption('Conditional story: IF a market downside day and an additional NVIDIA supply-specific shock accompany YES, '
               'what does this reference portfolio do in one trading session? No event-conditional economic parameter was identified from actual invasions.')
    l,m,r=st.columns(3)
    l.metric('Recorded market YES price (Q proxy)',f"{xq['reference']:.1%}")
    m.metric('Assumed stress YES−NO (not causal)',f"${stress['reference_median']:+,.0f}")
    r.metric('Assumed YES residual if pi = Q',f"${x['priced_in_under_assumed_transfer']['YES_residual_usd']:+,.0f}")
    limits=stress['historical_cohort_p10_p90_plus_declared_increment']
    st.warning('Observed-risk-off reference + declared NVIDIA increment: '
               f"${limits[0]:+,.0f} to ${limits[1]:+,.0f} (cohort p10/p90, NOT a confidence interval or an invasion-loss bound).")
    st.caption('Q is a *market price proxy*, not a physical probability. The price-to-asset assumption '
               'pi=Q yields the displayed identity; it does not measure the amount actually priced into NVDA, QQQ or TLT.')
    c=x['calibration']
    st.dataframe([
        {'Channel':'Joint, observed risk-off sessions','Evidence / assumption':f"{c['coherent_historical_market_tail_days']} matched training sessions; SPY at or below {c['SPY_training_return_threshold']:.2%}",
         'Effect on reference median (USD)':round(stress['unshifted_riskoff_reference_median'],2)},
        {'Channel':'Additional NVIDIA issuer stress','Evidence / assumption':f"Historical NVDA residual p25 − median: {c['additional_NVDA_issuer_return_assumption']:.2%}; assumed supplier channel",
         'Effect on reference median (USD)':round(stress['issuer_addon'],2)},
    ],hide_index=True,width='stretch')
    st.caption('Direct NVDA exposure 35%, plus the independently verified 8.17082268% NVDA slice of QQQ: '
               f"{c['effective_NVDA_weight_in_original_portfolio']:.3%} total. QQQ parent return is counted once; only its NVIDIA slice receives the extra *increment*. ")
    st.caption(f"Frozen historical calibration: {x['train_sessions']} daily training sessions, "
               f"{x['holdout_sessions']} later held-out sessions. "
               'Holdout checks market-stress regime behavior, NOT a Taiwan YES outcome.')
    st.markdown('##### Real historical regulatory announcements — analogues, not event-effect estimates')
    st.dataframe([{'BIS announcement':a['announcement_date'],
                   'Same-day portfolio P&L (USD)':round(a['portfolio_same_day_pnl_usd'],2),
                   'Interpretation':'Observed, confounded close-to-close move'}
                  for a in x['BIS_regulatory_announcement_analogs']],hide_index=True,width='stretch')
    with st.expander('Evidence, economic assumptions and unresolved identification',expanded=False):
        st.write('Price source SHA-256 (Gamma): '+src['gamma_raw_sha256'])
        st.write('Settlement rules SHA-256: '+src['full_rules_sha256'])
        st.write('Recorded quote: '+str(xq['indicative_bid_ask_band'])+' bid/ask; historical token observations were point prices only.')
        st.write('The pricing-weight transfer pi = Q is an explicit assumption; exact binary economic contrast is unknown.')
        for s in x['limitations']:st.write('- '+s)
        st.json({'holdout_diagnostic':x['holdout_diagnostic'],'source_documents':x['source_documents']})
    st.download_button('Export derived conditional stress JSON',json.dumps(x,indent=2),
                       'castle_taiwan_conditional_reference.json','application/json',
                       key='round17_export_geopolitical_stress')

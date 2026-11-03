"""Bounded public-primary-page nonmacro desk case, never a provider capture.

This is deliberately NOT a replacement for the raw-receipt P1 agent or the
source-native Kalshi verifier. The source-page content was reviewed separately;
no live API, exchange signature, authenticated quote or independent source
snapshot is implied. It is an auditable, explicitly human-reviewed illustration
of how similar geopolitical questions have different settlement semantics.
"""
from __future__ import annotations

import math
from typing import Any

from .portfolio import parse_portfolio

BLOCKADE_URL = 'https://polymarket.com/event/will-china-blockade-taiwan-by-in-2026'
OFFENSIVE_URL = 'https://polymarket.com/event/will-china-invade-taiwan-before-2027'
ISSUER_URL = 'https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm'


def taiwan_primary_page_desk_case(*, portfolio_rows: list[dict[str, Any]],
                                  notional: float = 100_000.0) -> dict[str, Any]:
    """Evaluate a known NON-EQUIVALENCE case without claiming agent discovery.

    Blockade requires a qualifying enforced >=24h commercial access denial to
    the main island. Territorial-offensive YES requires a military offensive
    intended to take qualifying inhabited ROC-controlled territory. A blockade
    need not attempt territorial control, and the offensive can concern an
    inhabited offshore island without a qualifying main-island blockade.

    These are logical *countermodels* interpreted from official page excerpts,
    not observed military events, legal adjudication of full external terms,
    price evidence, or predictions of either event's impact on equity prices.
    """
    portfolio = parse_portfolio(portfolio_rows, notional=notional)
    direct_nvda = math.fsum(p.weight for p in portfolio.positions
                             if p.symbol == 'NVDA' and p.asset_type == 'stock')
    unresolved_etf = math.fsum(p.weight for p in portfolio.positions
                               if p.asset_type == 'etf')
    examples = [
        {
            'id': 'BLOCKADE_WITHOUT_TERRITORIAL_OFFENSIVE',
            'assumed_facts': ('PRC enforces a 30-hour restriction on normal foreign '
                              'commercial port access to Taiwan main island, without '
                              'commencing an offensive intended to seize territory.'),
            'blockade_yes_if_rules_apply': True,
            'territorial_offensive_yes_if_rules_apply': False,
            'scenario_is_fictional_counterexample': True,
        },
        {
            'id': 'TERRITORIAL_OFFENSIVE_WITHOUT_MAIN_ISLAND_BLOCKADE',
            'assumed_facts': ('PRC commences an offensive intended to take an '
                              'inhabited offshore island administered by Taiwan, '
                              'without enforced denial of normal foreign commercial '
                              'access to the Taiwan main island for 24 hours.'),
            'blockade_yes_if_rules_apply': False,
            'territorial_offensive_yes_if_rules_apply': True,
            'scenario_is_fictional_counterexample': True,
        },
        {
            'id': 'MILITARY_DRILLS_WITHOUT_ENFORCEMENT_OR_OFFENSIVE',
            'assumed_facts': ('PLA performs a warned blockade exercise that does not '
                              'interdict foreign commercial traffic and does not '
                              'commence an offensive to take territory.'),
            'blockade_yes_if_rules_apply': False,
            'territorial_offensive_yes_if_rules_apply': False,
            'scenario_is_fictional_counterexample': True,
        },
    ]
    assert any(x['blockade_yes_if_rules_apply'] and not x['territorial_offensive_yes_if_rules_apply']
               for x in examples)
    assert any(x['territorial_offensive_yes_if_rules_apply'] and not x['blockade_yes_if_rules_apply']
               for x in examples)
    return {
        'status': 'PRIMARY_PUBLIC_PAGE_MANUAL_DESK_CASE_NOT_SOURCE_NATIVE_PROVIDER_ADMISSION',
        'case_id': 'TAIWAN_BLOCKADE_VS_TERRITORIAL_OFFENSIVE_2026',
        'research_review_as_of': '2026-10-19',
        'sources': [
            {'role': 'blockade settlement terms', 'url': BLOCKADE_URL,
             'provenance': 'OFFICIAL_PUBLIC_MARKET_PAGE_TEXT_REVIEWED_NOT_RAW_HTTP_RECEIPT'},
            {'role': 'territorial-offensive settlement terms', 'url': OFFENSIVE_URL,
             'provenance': 'OFFICIAL_PUBLIC_MARKET_PAGE_TEXT_REVIEWED_NOT_RAW_HTTP_RECEIPT'},
            {'role': 'foundry dependence and supply concentration', 'url': ISSUER_URL,
             'provenance': 'NVIDIA_FY2026_PRIMARY_SEC_FILING_WEB_REVIEW_NOT_LOCAL_RAW_CAPTURE'},
        ],
        'source_event_outcomes': [
            {'source_url': BLOCKADE_URL, 'id': 'TAIWAN_MAIN_ISLAND_COMMERCIAL_BLOCKADE',
             'reviewed_conditions': ('An announced or de facto enforced PRC air/naval '
                'commercial-access restriction affecting the main island for at least 24 hours.'),
             'nonqualifying_examples': ['mere military exercises', 'voluntary rerouting',
                                        'economic sanctions without physical interdiction']},
            {'source_url': OFFENSIVE_URL, 'id': 'ROC_TERRITORIAL_CONTROL_OFFENSIVE',
             'reviewed_conditions': ('PRC commences a military offensive intending to '
                'establish control over ROC territory, including inhabited offshore islands.'),
             'nonqualifying_examples': ['uninhabited island only', 'commercial blockade without such an offensive']},
        ],
        'pairwise_settlement_review': {
            'relation': 'NEITHER_DIRECTION_IMPLIED_BY_REVIEWED_TERMS',
            'source_page_terms_complete_locally_verified': False,
            'countermodels': examples,
            'scope': 'REVIEWED_PAGE_TEXT_LOGIC_NOT_INDEPENDENT_VENUE_RULE_ADJUDICATION',
            'certified_contract_equivalence': False,
        },
        'original_portfolio_notional_usd': portfolio.notional,
        'direct_nvda_position_usd': portfolio.notional * direct_nvda,
        'direct_nvda_weight': direct_nvda,
        'etf_lookthrough_identity_unverified_original_weight': unresolved_etf,
        'known_pathway': ('NVIDIA filing names TSMC among semiconductor foundries; '
                          'Taiwan-related foundry disruption is a plausible exposure channel.'),
        'risk_size_and_return_sign': 'NOT_IDENTIFIED_FROM_ISSUER_RELATIONSHIP_OR_CONTRACT_RULES',
        'candidate_economic_loss_ranking': None,
        'current_contract_quote_Q': None,
        'physical_event_probability_P': None,
        'identified_yes_no_portfolio_pnl': None,
        'no_new_provider_or_model_requests': True,
        'limitations': [
            'The two source-page rule excerpts were manually reviewed; no raw provider API receipts were captured.',
            'Countermodels prove lack of implication under the reviewed predicates, not universal legal interpretation.',
            'A primary issuer disclosure establishes a plausible supply channel, not a conditional stock return.',
            'ETF component identities are not reconstructed from issuer disclosures in this case.',
            'No current prediction-market Q/P, probability-weighted loss, or event-induced equity effect is inferred.',
            'This static research case is NOT an autonomous discovery run or an independent blind holdout.',
        ],
    }

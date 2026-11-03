"""Derive usable binary outcomes and sound relations from inspected Kalshi rules.

LLM proposals are deliberately not inputs to the logical proofs. Source receipts
are checked before this function is called. The source-defined YES settlement
is always an exact binary question; a *numeric* predicate is certified only by
an independently reviewed rule grammar. CPI strike implications can additionally
be proved CONDITIONAL on a single common scalar settlement value, with identical
full secondary rules, payout and timing, verified directly from locally archived
original HTTP bytes. No quote freshness or pricing-measure hypothesis is required
for a logical implication. No conversion to P, q_equity or equity P&L is made.
"""
from __future__ import annotations

from itertools import combinations
from hashlib import sha256
from datetime import datetime
from typing import Any, Mapping, Sequence

from .adapters import Market
from .contract_questions import _FED_2026_SECONDARY_SHA256, source_defined_yes_question
from .agentic_bridge import _receipt, AgenticEvidenceError
from .semantics import deterministic_relation, FED_PRIMARY_RE, FED_SECONDARY
from .strike_lattice import _parse_source, StrikeLatticeError

# Each condition concerns only the inference between two contracts, never a
# claim about an LLM-proposed release event or stock/ETF price response.
_CPI_CONDITIONS = [
    'The two contracts settle against the same scalar CPI value, including any delayed or fallback determination.',
    'The identical full secondary terms preserve their strict-threshold unit payouts without a contract-specific override.',
    'No priority between differing external contract documents is asserted; conclusions apply to this common-settlement model.',
]


def _source_cpi_strikes(markets: Sequence[Market]) -> dict[str, float]:
    """Check an entire cohort before deriving any CPI cross-strike inference."""
    parsed: dict[str, float] = {}
    common = None
    for market in markets:
        try:
            threshold, scope = _parse_source(market)
        except (StrikeLatticeError, ValueError, TypeError, KeyError):
            return {}
        if common is None:
            common = scope
        elif scope != common:
            return {}
        if market.ticker in parsed or threshold in parsed.values():
            return {}
        parsed[market.ticker] = threshold
    return parsed if len(parsed) >= 2 else {}



def _fed_exclusive_pair(first: Market, second: Market) -> bool:
    """Prove only mutual exclusion, not the ambiguous numeric strike.

    Full version-pinned secondary terms explicitly say that at most one
    outcome of the FOMC event can resolve YES. This argument remains valid
    even when source primary text drops a '>' comparator in an individual
    strike; such a market still MUST NOT acquire an exact numeric predicate.
    """
    if (first.ticker == second.ticker or
            not first.event_ticker.startswith('KXFEDDECISION-') or
            first.event_ticker != second.event_ticker):
        return False
    pinned = {_FED_2026_SECONDARY_SHA256,
              sha256(FED_SECONDARY.encode('utf-8')).hexdigest()}
    if (sha256(first.rules_secondary.encode('utf-8')).hexdigest() not in pinned or
            first.rules_secondary != second.rules_secondary):
        return False
    expected_date = None
    for market in (first, second):
        raw = market.metadata.get('raw')
        if (not isinstance(raw, dict) or
                raw.get('event_ticker') != market.event_ticker or
                raw.get('ticker') != market.ticker or
                raw.get('market_type') != 'binary' or
                market.status not in ('active', 'open') or
                str(raw.get('notional_value_dollars')) not in ('1.0000', '1.00', '1.0', '1')):
            return False
        found = FED_PRIMARY_RE.fullmatch(market.rules_primary)
        if found is None:
            return False
        action, modifier, amount, day_text = found.groups()
        if not amount.isdigit():
            return False
        try:
            day = datetime.strptime(day_text, '%B %d, %Y').date()
            closes = datetime.fromisoformat(market.close_time.replace('Z', '+00:00'))
            expects = datetime.fromisoformat(str(raw['expected_expiration_time']).replace('Z', '+00:00'))
            # This condition deliberately avoids inferring the missing > sign.
            if (closes.utcoffset() is None or expects.utcoffset() is None or
                    closes.date() != day or expects.date() != day or closes > expects):
                return False
        except (ValueError, TypeError, KeyError):
            return False
        if expected_date is None:
            expected_date = day
        if expected_date != day or (market.event_ticker !=
                f"KXFEDDECISION-{day.strftime('%y%b').upper()}"):
            return False
        # The strike ticker and action MUST agree. A missing comparator in the
        # primary text is acceptable for EXCLUSIVITY ONLY when structured
        # source fields and the label both explicitly encode the other reading.
        raw_strike = raw.get('custom_strike')
        if not isinstance(raw_strike, dict) or set(raw_strike) != {action}:
            return False
        if str(raw_strike[action]) not in (amount, '>'+amount):
            return False
        if modifier == '>' and str(raw_strike[action]) != '>'+amount:
            return False
        expected_suffix = ('H' if action == 'Hike' else 'C')
        if market.ticker != f"{market.event_ticker}-{expected_suffix}{amount}":
            # An ambiguous >25 contract uses C26 rather than C25. This is
            # NOT a numerical inequality proof: only event membership.
            if not (str(raw_strike[action]) == '>'+amount and
                    market.ticker == f"{market.event_ticker}-{expected_suffix}{int(float(amount))+1}"):
                return False
        subtitle = str(raw.get('yes_sub_title', ''))
        if subtitle not in (f'{action} {amount}bps', f'{action} >{amount}bps') and not (
                action == 'Hike' and amount == '0' and subtitle == 'Fed maintains rate'):
            return False
        # Standalone calls cannot bypass the original local-source integrity.
        try:
            _receipt(market)
        except (AgenticEvidenceError, ValueError, TypeError, OSError):
            return False
    return True


def build_source_native_graph(
    markets: Sequence[Market],
    questions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build an evidence-scoped outcome graph, without a model or provider call.

    Callers MUST admit each original Market through the exact receipt and
    portfolio bridge first. This function deliberately does not create an
    additional trust channel from model-visible copied dictionaries.
    """
    if len(markets) > 24 or len({m.ticker for m in markets}) != len(markets):
        raise ValueError('Source-native outcomes require distinct bounded contract identities')
    by_id = {m.ticker: m for m in markets}
    if set(by_id) != set(questions):
        raise ValueError('Source questions must correspond exactly to admitted contracts')
    outcomes: list[dict[str, Any]] = []
    for market in markets:
        # Recheck original source bytes in this public helper as well. Never
        # accept model-provided questions or a forged 'certified' status.
        try:
            _receipt(market)
        except (AgenticEvidenceError, ValueError, TypeError, OSError) as exc:
            raise ValueError('Original source receipt is required for outcome graph') from exc
        question = questions[market.ticker]
        if question.get('contract_id') != market.ticker:
            raise ValueError('Source question does not match original market identity')
        if dict(question) != source_defined_yes_question(market):
            raise ValueError('Source question does not reproduce from reviewed original market rules')
        outcomes.append({
            'event_id': 'kalshi-contract-yes:' + market.ticker,
            'contract_id': market.ticker,
            'yes_question': question['yes_question'],
            'definition': 'THIS_EXACT_CONTRACT_SETTLES_YES_UNDER_FULL_SOURCE_RULES',
            'identity_status': 'SOURCE_RECEIPT_VERIFIED_CONTRACT_YES_IDENTITY',
            'numeric_predicate': question['typed_predicate'],
            'numeric_review_status': question['numeric_rule_status'],
            'additional_model_assumptions': list(question['additional_model_assumptions']),
            'exception_flags': list(question['exception_flags']),
            'rules_sha256': question['verified_rules_sha256'],
            'original_macro_topic_equivalence': 'NOT_ESTABLISHED',
            'equity_yes_no_impact_usd': None,
        })
    edges: list[dict[str, Any]] = []

    # The guarded CPI parser checks the original receipt for every source
    # record, identical secondary text, identical expiry and co-observation.
    cpi_candidates = [m for m in markets if m.event_ticker.startswith('KXCPI-')]
    cpi = _source_cpi_strikes(cpi_candidates) if len(cpi_candidates) >= 2 else {}
    for left, right in combinations(markets, 2):
        left_q, right_q = questions[left.ticker], questions[right.ticker]
        # Even if both rules are source-verified, different event identifiers
        # must never be linked merely on the basis of similar words.
        if left.event_ticker != right.event_ticker:
            continue
        relation, basis, conditions = 'NONE', None, []
        if _fed_exclusive_pair(left, right):
            edges.append({
                'from_yes': left.ticker, 'to_yes': right.ticker,
                'relationship': 'EXCLUDES',
                'assurance': 'SOURCE_RULE_PREDICATE_CERTIFIED',
                'source_basis': 'PINNED_FED_SECONDARY_EXPLICIT_MUTUAL_EXCLUSIVITY',
                'required_assumptions': [],
                'reference_event_id': left.event_ticker,
                'not_macro_equivalence': True,
            })
            continue
        if left.ticker in cpi and right.ticker in cpi:
            # Threshold T_h > T_l on the SAME final scalar X implies
            # X > T_l. This statement is independent of Kalshi prices.
            high, low = ((left, right) if cpi[left.ticker] > cpi[right.ticker]
                         else (right, left))
            edges.append({
                'from_yes': high.ticker, 'to_yes': low.ticker,
                'relationship': 'IMPLIES',
                'assurance': 'CONDITIONAL_SHARED_SETTLEMENT_SCALAR',
                'source_basis': 'ORIGINAL_RECEIPTS_IDENTICAL_CPI_FULL_SECONDARY_AND_SETTLEMENT_FIELDS',
                'required_assumptions': list(_CPI_CONDITIONS),
                'reference_event_id': high.event_ticker,
                'not_macro_equivalence': True,
            })
            continue
        # Fed actions can be compared ONLY when both have fully and
        # independently parsed typed source-native rules, including the
        # identical cancellation convention, period, source, unit and vintage.
        accepted = ('ORIGINAL_REVIEWED_RULE_TEMPLATE_CERTIFIED',
                    'SEPARATE_PINNED_EXACT_FED_2026_SOURCE_VARIANT')
        if (left_q['numeric_rule_status'] in accepted and
                right_q['numeric_rule_status'] in accepted and
                isinstance(left_q['typed_predicate'], dict) and
                isinstance(right_q['typed_predicate'], dict)):
            relation = deterministic_relation(right_q['typed_predicate'], left_q['typed_predicate'])
            basis = 'INDEPENDENTLY_PARSED_FULL_FED_SETTLEMENT_RULES'
            conditions = []
        if relation == 'NONE':
            continue
        source, target = left.ticker, right.ticker
        if relation == 'IMPLIED_BY':
            source, target = target, source
            relation = 'IMPLIES'
        edges.append({
            'from_yes': source, 'to_yes': target,
            'relationship': relation,
            'assurance': 'SOURCE_RULE_PREDICATE_CERTIFIED',
            'source_basis': basis,
            'required_assumptions': conditions,
            'reference_event_id': left.event_ticker,
            'not_macro_equivalence': True,
        })
    return {
        'status': 'SOURCE_NATIVE_BINARY_OUTCOME_GRAPH_NO_CLAUDE_MACRO_CERTIFICATION',
        'outcomes': outcomes,
        'relations': edges,
        'source_binary_outcomes': len(outcomes),
        'source_relationships_proved': sum(e['assurance'] == 'SOURCE_RULE_PREDICATE_CERTIFIED' for e in edges),
        'conditional_relationships_proved': sum(e['assurance'] == 'CONDITIONAL_SHARED_SETTLEMENT_SCALAR' for e in edges),
        'physical_P': None,
        'equity_priced_in_q': None,
        'contract_yes_no_equity_PnL': None,
    }

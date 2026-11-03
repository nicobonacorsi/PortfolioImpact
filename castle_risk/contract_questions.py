"""Independently sourced binary contract questions for the final integrated investigation.

A quote concerns the contract YES settlement only. Typed macro event assertions are
admitted solely under exact, previously reviewed versioned source-rule templates.
The original semantic verifier is never modified or bypassed for cross-contract LP.
No external API calls or model text can confer source-verification status here.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import re
from typing import Any

from .adapters import Market
from .semantics import parse_market_predicate

# A version-pinned Kalshi Fed secondary-rules variant present byte-for-byte
# on independently archived October 28, 2026 source records. It is not the template
# supported by the original, hash-protected Fed semantic engine. Different
# secondary text FAILS CLOSED; '>' strikes remain UNCERTIFIED because source
# primary/secondary prose loses the operator.
_FED_2026_SECONDARY_SHA256 = (
    "42ed5fc9491c83df65f41a636c6b12407d28644ef39f36edb2171ed8dad07882"
)
_FED_EXACT = re.compile(
    r"If the Federal Reserve does a (Hike|Cut) of ([0-9]+)bps on "
    r"([A-Z][a-z]+ [0-9]{1,2}, [0-9]{4}), then the market resolves to Yes\."
)


def _alternate_fed_exact(market: Market) -> dict[str, Any] | None:
    """Qualify only literal EXACT actions in an independently pinned rule variant.

    Never repair missing strict '>' signs with title, metadata or assumptions.
    Do not generalize this source-version review to other venues or templates.
    """
    raw = market.metadata.get("raw", {})
    if not isinstance(raw, dict):
        return None
    if sha256(market.rules_secondary.encode('utf-8')).hexdigest() != _FED_2026_SECONDARY_SHA256:
        return None
    match = _FED_EXACT.fullmatch(market.rules_primary)
    if match is None:
        return None
    action, amount_txt, when_txt = match.groups()
    try:
        when = datetime.strptime(when_txt, "%B %d, %Y").date()
    except ValueError:
        return None
    # The complete version-pinned wording is reusable across scheduled meetings,
    # but only when the date, parent event, full rules, strike, label and timing
    # independently agree.  No calendar date is inferred from Claude text.
    if when.year < 2025:
        return None
    expected_event = f"KXFEDDECISION-{when.strftime('%y%b').upper()}"
    if raw.get('event_ticker') != expected_event or market.event_ticker != expected_event:
        return None
    # The source label and structured strike MUST agree with the literal primary rule.
    # This does not authorize an inference from a garbled inequality in the title.
    if raw.get('custom_strike') != {action: amount_txt}:
        return None
    if market.status not in ('active', 'open') or raw.get('market_type') != 'binary':
        return None
    if str(raw.get('notional_value_dollars')) != '1.0000':
        return None
    try:
        closes = datetime.fromisoformat(market.close_time.replace('Z', '+00:00'))
        expects = datetime.fromisoformat(str(raw['expected_expiration_time']).replace('Z', '+00:00'))
    except (ValueError, KeyError, TypeError):
        return None
    if (closes.tzinfo is None or expects.tzinfo is None or
            closes.astimezone(timezone.utc).date() != when or
            expects.astimezone(timezone.utc).date() != when or
            closes > expects):
        return None
    try:
        amount = int(amount_txt)
    except ValueError:
        return None
    if amount > 10000 or (action == 'Cut' and amount == 0):
        return None
    expected_ticker = f"{expected_event}-{'H' if action == 'Hike' else 'C'}{amount}"
    if market.ticker != expected_ticker:
        return None
    label = str(raw.get('yes_sub_title', ''))
    if label != f'{action} {amount}bps' and not (action == 'Hike' and amount == 0 and label == 'Fed maintains rate'):
        return None
    return {
        'metric': 'fed_target_rate_change',
        'source': 'Federal Reserve / Kalshi resolution terms',
        'period': when.isoformat(),
        'unit': 'basis_points',
        'operator': 'eq',
        'threshold': amount * (1 if action == 'Hike' else -1),
        'vintage': 'scheduled_meeting_decision',
        'cancellation_policy': 'a canceled scheduled meeting is treated as no change',
    }


def source_defined_yes_question(market: Market) -> dict[str, Any]:
    """Describe a receipt-verified contract, never a model-defined macro event.

    Caller MUST complete the original `inspect_exact_contract` source receipt and
    exact-identity checks first. This function does not establish source integrity.
    Any unreviewed numeric rule is left opaque, rather than guessed from title.
    """
    if not isinstance(market, Market):
        raise ValueError('Verified normalized Kalshi market expected')
    raw = market.metadata.get('raw')
    if not isinstance(raw, dict) or raw.get('ticker') != market.ticker:
        raise ValueError('Missing consistent original source row')
    parsed = parse_market_predicate({'venue': 'kalshi', 'ticker': market.ticker, 'raw': raw})
    status = parsed.get('status')
    if status == 'CERTIFIED':
        predicate = parsed.get('predicate')
        scope = 'ORIGINAL_REVIEWED_RULE_TEMPLATE_CERTIFIED'
        conditions = []
    elif status == 'CONDITIONAL_VERIFIED':
        predicate = parsed.get('predicate')
        scope = 'REVIEWED_RULES_CONDITIONAL_ON_COMMON_SETTLEMENT_VALUE'
        conditions = parsed.get('required_model_assumptions', [])
    else:
        predicate = _alternate_fed_exact(market)
        scope = ('SEPARATE_PINNED_EXACT_FED_2026_SOURCE_VARIANT'
                 if predicate is not None else 'EXACT_CONTRACT_YES_IDENTITY_ONLY_NUMERIC_RULE_UNVERIFIED')
        conditions = []
    if predicate is not None and not isinstance(predicate, dict):
        raise ValueError('Non-object predicate from reviewed parser')
    secondary = market.rules_secondary.lower()
    exceptional = [name for name, condition in (
        ('CANCELLATION', 'cancel' in secondary),
        ('SHUTDOWN_OR_DELAY', 'shutdown' in secondary or 'delay' in secondary),
        ('SETTLEMENT_EXTENSION', 'extend' in secondary),
    ) if condition]
    return {
        'contract_id': market.ticker,
        # This field is source evidence, not a card heading. A silent prefix
        # limit can remove a deadline, qualifier or exception and therefore
        # describe a different proposition from the recorded contract.
        'yes_question': market.title,
        'binary_definition': 'THIS_EXACT_CONTRACT_SETTLES_YES_AS_DEFINED_BY_FULL_RULES',
        'numeric_rule_status': scope,
        'typed_predicate': predicate,
        'additional_model_assumptions': conditions,
        'exception_flags': exceptional,
        'verified_rules_sha256': sha256((market.rules_primary + '\n' + market.rules_secondary).encode('utf-8')).hexdigest(),
        'macro_hypothesis_equivalence': 'NOT_ESTABLISHED',
        'equity_impact_conditional_on_yes': 'NOT_IDENTIFIED',
    }

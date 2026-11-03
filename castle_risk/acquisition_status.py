"""Operational acquisition diagnostics, separate from historical acceptance.

Taxonomy is optional metadata. Required transport and holdings data quality are
independent conditions; neither is inferred from a successful HTTP request alone.
"""
from collections.abc import Mapping

# A bond fund or single-line vehicle accepted by the class-aware holdings review
# (declared, measured tolerances) counts like a plain reliable snapshot.
ACCEPTED_HOLDINGS_STATUSES = frozenset({'RELIABLE_SNAPSHOT', 'ACCEPTED_BY_DECLARED_HOLDINGS_REVIEW'})


REQUIRED_OPERATIONS = frozenset({'ticker_details', 'holdings', 'daily_prices'})
# The 10-year yield series feeds the measured two-factor fund response; optional.
# Scenario-proxy histories are optional to original portfolio metadata/prices.
# Live acquisition logs their attempts under this distinct operation name.
# SEC issuer-sibling metadata is advisory look-through enrichment, not a
# required quote/holdings transport. Massive's real R124 capture includes this
# operation; rejecting it invalidated the *whole* otherwise-complete attempt
# log and incorrectly labelled the acquisition PARTIAL.
OPTIONAL_OPERATIONS = frozenset({'taxonomy', 'treasury_yields', 'scenario_proxy_prices',
                                 'issuer_siblings'})


def massive_acquisition_stage(inputs, *, request_count):
    """Build live stage status without promoting optional metadata to a blocker.

Require both the ingestion summary and its underlying required-operation log.
Missing, malformed or contradictory diagnostics cannot produce ACQUIRED. This
does not change the historical all-operations/three-portfolio acceptance gate.
    """
    raw_attempts = inputs.get('attempts')
    raw_quality = inputs.get('acquisition_quality')
    quality = dict(raw_quality) if isinstance(raw_quality, Mapping) else {}
    attempts_valid = isinstance(raw_attempts, (list, tuple)) and bool(raw_attempts)
    attempts = raw_attempts if attempts_valid else []
    attempts_valid = attempts_valid and all(
        isinstance(a, Mapping)
        and isinstance(a.get('symbol'), str) and bool(a['symbol'])
        and isinstance(a.get('operation'), str)
        and a['operation'] in REQUIRED_OPERATIONS | OPTIONAL_OPERATIONS
        and isinstance(a.get('status'), str) and bool(a['status'])
        for a in attempts)
    rows = attempts if attempts_valid else []
    required = [a for a in rows if a['operation'] in REQUIRED_OPERATIONS]
    required_transport_complete = (
        attempts_valid and bool(required)
        and quality.get('all_required_transport_operations_succeeded') is True
        and all(a['status'] == 'OK' for a in required))
    holdings_quality_complete = (
        quality.get('all_required_holdings_quality_passed') is True
        and all(a.get('data_quality_status') in ACCEPTED_HOLDINGS_STATUSES
                for a in rows if a['operation'] == 'holdings'))
    optional_failures = [
        {'symbol': a['symbol'], 'operation': a['operation'], 'status': a['status']}
        for a in rows if a['operation'] in OPTIONAL_OPERATIONS and a['status'] != 'OK'
    ]
    return {
        'status': 'ACQUIRED' if required_transport_complete and holdings_quality_complete else 'PARTIAL',
        'request_count': request_count,
        'successful_operations': sum(a['status'] == 'OK' for a in rows),
        'required_transport_complete': required_transport_complete,
        'holdings_quality_complete': holdings_quality_complete,
        'optional_metadata_failures': optional_failures,
        'attempt_diagnostics_valid': attempts_valid,
        'acquisition_quality': quality,
        'status_scope': 'REQUIRED_TRANSPORT_AND_HOLDINGS_QUALITY_NOT_FULL_AUTHENTICATED_ACCEPTANCE',
    }

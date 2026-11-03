"""Historical-analogue stress: what today's portfolio did in comparable past episodes.

For an event family without an admitted response model (Hormuz, Taiwan, tariffs,
export controls, ...), the conditional question "what if it happens?" can still be
answered with evidence instead of "no estimate": hold today's positions fixed and
replay each instrument's own adjusted daily closes over the first sessions after
dated past episodes of the same kind. ETFs are replayed through their own prices,
so fund look-through is implicit and nothing is double counted.

This is a measured historical range, not a forecast. Past episodes were milder or
different (no past Taiwan episode was a blockade), markets moved for other reasons
in the same window, and a handful of episodes cannot pin down a tail. The output
therefore always carries (a) every episode with its own dollar move, (b) a
non-event-window baseline, and (c) an endpoint-influence diagnostic showing
which single episode controls the historical range. A leave-one-out min/max
"coverage" ratio is an order-statistic identity, not a validation test.

Offline: callers supply prices; this module makes no request.
"""
from __future__ import annotations

from datetime import date
import math
import re

WINDOW_SESSIONS = 5
MIN_EPISODES = 3
SCHEMA = 'castle.option1.historical_analogue_stress.v1'

# First U.S. trading session on or after the moment the news became public.
# Each entry: (label, onset session, note). Weekend events start on Monday.
EPISODE_SETS = {
    'OIL_SUPPLY_SHOCK': {
        'title': 'Middle East / oil-supply shocks',
        'episodes': (
            ('Drone attack on Saudi Abqaiq and Khurais facilities', '2019-09-16', 'Attack on Sat 14 Sep 2019.'),
            ('U.S. strike kills Iranian general Soleimani', '2020-01-03', ''),
            ('Russia invades Ukraine', '2022-02-24', ''),
            ('Surprise OPEC+ production cut', '2023-04-03', 'Announced Sun 2 Apr 2023.'),
            ('Hamas attack on Israel', '2023-10-09', 'Attack on Sat 7 Oct 2023.'),
            ('Iranian missile and drone attack on Israel', '2024-04-15', 'Attack on Sat 13 Apr 2024.'),
            ('Israeli strikes on Iran', '2025-06-13', ''),
            ('U.S. strikes on Iranian nuclear sites', '2025-06-23', 'Strikes on Sat 21 Jun 2025.'),
        ),
        'caveat': 'None was a sustained loss of Gulf supply such as a Strait of Hormuz closure; a larger supply shock would move oil and energy stocks more than any analogue.',
    },
    'TAIWAN_STRAIT': {
        'title': 'Taiwan Strait military escalations',
        'episodes': (
            ('U.S. House Speaker visits Taiwan; PLA live-fire drills', '2022-08-02', ''),
            ('PLA "Joint Sword" drills around Taiwan', '2023-04-10', 'Drills began Sat 8 Apr 2023.'),
            ('PLA "Joint Sword-2024A" drills', '2024-05-23', ''),
            ('PLA "Joint Sword-2024B" drills', '2024-10-14', ''),
            ('PLA "Strait Thunder-2025A" drills', '2025-04-01', ''),
        ),
        'caveat': 'No past episode was a blockade, invasion or chip-supply stoppage; these are mild escalations, so the range understates a real disruption.',
    },
    'US_CHINA_TARIFFS': {
        'title': 'U.S.-China tariff escalations',
        'episodes': (
            ('Section 301 tariff action on China announced', '2018-03-22', ''),
            ('U.S. raises China tariffs to 25% (announced by tweet)', '2019-05-06', 'Announced Sun 5 May 2019.'),
            ('U.S. announces 10% tariff on remaining Chinese imports', '2019-08-01', ''),
            ('China retaliates; U.S. raises tariffs again', '2019-08-23', ''),
            ('U.S. "reciprocal" tariffs announced', '2025-04-03', 'Announced after the close on 2 Apr 2025.'),
            ('U.S. threatens additional 100% tariff on China', '2025-10-10', ''),
        ),
        'caveat': 'Episodes mix announcements, threats and implementations; the 2025 reciprocal tariffs also hit non-China trade.',
    },
    'US_CHIP_EXPORT_CONTROLS': {
        'title': 'U.S. advanced-chip export restrictions',
        'episodes': (
            ('BIS advanced-computing export rule', '2022-10-07', ''),
            ('BIS updated advanced-computing rule', '2023-10-17', ''),
            ('BIS semiconductor-equipment and HBM rule', '2024-12-02', ''),
            ('AI diffusion rule published', '2025-01-13', ''),
            ('NVIDIA discloses H20 China license requirement', '2025-04-16', 'Disclosed after the close on 15 Apr 2025.'),
        ),
        'caveat': 'Rules differ in scope; the moves are dominated by chip stocks and say little about other holdings.',
    },
    'CHINA_CRITICAL_MINERALS': {
        'title': 'Chinese critical-mineral export restrictions',
        'episodes': (
            ('China announces gallium and germanium export controls', '2023-07-03', ''),
            ('China bans gallium, germanium and antimony exports to the U.S.', '2024-12-03', ''),
            ('China imposes rare-earth export controls', '2025-04-04', 'Announced with retaliatory tariffs on the same day.'),
            ('China expands rare-earth export controls', '2025-10-09', ''),
        ),
        'caveat': 'Two of the four windows also contain major tariff news; the moves mix both shocks.',
    },
    'US_SOVEREIGN_DOWNGRADE': {
        'title': 'U.S. sovereign rating downgrades',
        'episodes': (
            ('Fitch downgrades the U.S. to AA+', '2023-08-02', 'Announced after the close on 1 Aug 2023.'),
            ("Moody's downgrades the U.S. to Aa1", '2025-05-19', 'Announced after the close on Fri 16 May 2025.'),
        ),
        'caveat': 'Only two episodes since 2018; too few for a range.',
    },
    'US_GOVERNMENT_SHUTDOWN': {
        'title': 'U.S. federal government shutdowns',
        'episodes': (
            ('Federal shutdown begins', '2018-01-22', 'Lapse began Sat 20 Jan 2018.'),
            ('Federal shutdown begins', '2018-12-24', 'Lapse began Sat 22 Dec 2018; the window overlaps a broad sell-off.'),
            ('Federal shutdown begins', '2025-10-01', ''),
        ),
        'caveat': 'Shutdown windows coincide with other market news; a small effect is not separable from noise.',
    },
}

# R87: a genuinely disrupted Gulf-shipping regime has now been observed.
# Federal Reserve, Oct 9 2026: the closure of Hormuz early in the conflict
# beginning Feb 28 caused an abrupt oil-supply shock. The first US trading
# session was Monday Mar 2. OPEC supply *cuts* must retain their original
# eight-event set: they are not equivalent to an actual transit disruption.
# The 2019-25 rows are contextual comparators, not closure observations.
EPISODE_SETS['HORMUZ_TRANSIT_SHOCK'] = {
    'title': 'Strait of Hormuz transit disruptions and contextual oil shocks',
    'episodes': EPISODE_SETS['OIL_SUPPLY_SHOCK']['episodes'] + (
        ('2026 Gulf conflict: Strait of Hormuz closure / severe supply disruption',
         '2026-03-02', 'Conflict began Sat 28 Feb 2026; first US session Mon 2 Mar. '
                       'Federal Reserve 2026-10-09 confirms the closure. '
                       'EIA 2026 historical shipping volumes document its magnitude. '
                       'Multiple macro/policy shocks coexist; not an isolated causal treatment.'),
    ),
    'caveat': ('The eight 2019-25 events were milder and were not a closure. '
               'The actual 2026 Hormuz disruption provides one materially relevant '
               'five-session historical stress, but it occurred amid a complex war '
               'and policy response; its net asset returns are not the isolated '
               'causal effect of closure. Any new 2027 event must be specified as '
               'an incremental change from the then-current priced-in state. '
               'Neither this nine-row mixture nor the 2026 observation is a '
               'validated forecast of a future closure.'),
    'source_reference': (
        'https://www.federalreserve.gov/econres/notes/feds-notes/'
        'convenience-yields-risk-premiums-and-the-interpretation-of-oil-futures-prices-'
        'during-geo-conflict-20261009.html'),
    'direct_disruption_onset_session': '2026-03-02',
}

_FAMILY_SETS = {
    'HORMUZ_ENERGY': 'HORMUZ_TRANSIT_SHOCK', 'OPEC_SUPPLY': 'OIL_SUPPLY_SHOCK',
    'MIDDLE_EAST_ENERGY': 'OIL_SUPPLY_SHOCK', 'OIL_SUPPLY': 'OIL_SUPPLY_SHOCK',
    'TAIWAN_SUPPLY': 'TAIWAN_STRAIT', 'TAIWAN_STRAIT': 'TAIWAN_STRAIT',
    'TRADE_TARIFFS': 'US_CHINA_TARIFFS', 'EXPORT_CONTROLS': 'US_CHIP_EXPORT_CONTROLS',
}
# The event wording is checked first: a family label can be coarse (a Chinese
# mineral export ban is not a U.S. chip export rule). Order matters.
_TEXT_SETS = (
    # Natural disasters have no political-escalation analogue here.
    (re.compile(r'earthquake|typhoon|tsunami|hurricane', re.I), None),
    (re.compile(r'rare[- ]earth|critical mineral|gallium|germanium|antimony|graphite|mineral export', re.I), 'CHINA_CRITICAL_MINERALS'),
    (re.compile(r'\bhormuz\b', re.I), 'HORMUZ_TRANSIT_SHOCK'),
    (re.compile(r'\bopec\b|crude oil|oil (supply|production|export)', re.I), 'OIL_SUPPLY_SHOCK'),
    (re.compile(r'\btaiwan\b', re.I), 'TAIWAN_STRAIT'),
    (re.compile(r'\bgovernment shutdown\b|\blapse in (federal )?appropriations\b', re.I), 'US_GOVERNMENT_SHUTDOWN'),
    (re.compile(r'\b(sovereign|credit) rating\b.*\b(lower|downgrad)|downgrad\w*.*\b(u\.?s\.?|united states)\b.*rating', re.I), 'US_SOVEREIGN_DOWNGRADE'),
    (re.compile(r'(chip|semiconductor|accelerator|GPU)s?\b.*export|export.*\b(chip|semiconductor|accelerator|GPU)|\bBIS\b', re.I), 'US_CHIP_EXPORT_CONTROLS'),
    (re.compile(r'\btariff', re.I), 'US_CHINA_TARIFFS'),
)


def episode_set_for(family, text=''):
    """Episode set for an event, from its wording first and its family second."""
    for pattern, name in _TEXT_SETS:
        if pattern.search(text or ''):
            return name
    return _FAMILY_SETS.get(str(family or '').upper())


def _closes(series):
    rows = series.get('prices') if isinstance(series, dict) else series
    out = {}
    for row in rows or ():
        close = float(row['close'])
        if math.isfinite(close) and close > 0:
            out[str(row['date'])] = close
    return out


def _quantile(values, q):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def analogue_stress(positions, prices, *, notional, set_name, window=WINDOW_SESSIONS):
    """Replay today's weights over each episode window; never fills missing prices.

    positions: [{'symbol', 'weight', 'asset_type'}]; cash earns zero.
    prices: {symbol: {'prices': [{'date', 'close'}, ...]}} adjusted daily closes.
    """
    spec = EPISODE_SETS[set_name]
    invested = [p for p in positions if p.get('asset_type') != 'cash' and p['weight'] > 0]
    closes = {p['symbol']: _closes(prices.get(p['symbol']) or {}) for p in invested}
    calendar = sorted(set.intersection(*(set(c) for c in closes.values()))) if closes else []
    episodes = []
    for label, onset, note in spec['episodes']:
        row = {'episode': label, 'onset_session': onset, 'note': note}
        if onset == spec.get('direct_disruption_onset_session'):
            row['direct_disruption_observation_not_causal_estimate'] = True
            row['source_reference'] = spec['source_reference']
        first = next((i for i, d in enumerate(calendar) if d >= onset), None)
        if first is None or first == 0 or first + window - 1 >= len(calendar) or calendar[first] > _shift(onset, 6):
            missing = [s for s, c in closes.items() if not c or min(c) > onset]
            row.update(status='OUTSIDE_COMMON_PRICE_HISTORY',
                       reason=('No price history for ' + ', '.join(sorted(missing)) + ' at this date.') if missing
                       else 'The window is not inside the common price history.')
            episodes.append(row)
            continue
        base, end = calendar[first - 1], calendar[first + window - 1]
        contributions = []
        for p in invested:
            r = closes[p['symbol']][end] / closes[p['symbol']][base] - 1
            contributions.append({'symbol': p['symbol'], 'weight': p['weight'], 'return': r,
                                  'pnl_usd': notional * p['weight'] * r})
        ret = math.fsum(c['return'] * c['weight'] for c in contributions)
        overlaps = sorted({f'{other_label} ({other_onset})' for other_name, other in EPISODE_SETS.items()
                           if other_name != set_name for other_label, other_onset, _ in other['episodes']
                           if (other_label, other_onset) != (label, onset) and base < other_onset <= end})
        row.update(status='MEASURED', base_session=base, end_session=end, portfolio_return=ret,
                   pnl_usd=notional * ret, contributions=contributions, overlapping_episodes=overlaps)
        episodes.append(row)
    measured = [e for e in episodes if e['status'] == 'MEASURED']
    result = {'schema': SCHEMA, 'episode_set': set_name, 'episode_set_title': spec['title'],
              'window_sessions': window, 'notional_usd': notional, 'episodes': episodes,
              'caveat': spec['caveat'], 'measured_count': len(measured),
              'method': ('Today\'s weights applied to each holding\'s own adjusted close-to-close return from the '
                         f'session before onset through {window} sessions after it. Funds are replayed through '
                         'their own prices. No beta, factor model or renormalization.'),
              'interpretation': 'Measured past-episode range for this portfolio, not a forecast or a probability.'}
    if 'direct_disruption_onset_session' in spec:
        result['direct_disruption_source_reference'] = spec['source_reference']
        result['direct_disruption_onset_session'] = spec['direct_disruption_onset_session']
        result['direct_disruption_measured_count'] = sum(
            row.get('direct_disruption_observation_not_causal_estimate') is True
            and row['status'] == 'MEASURED' for row in episodes)
    if len(measured) < MIN_EPISODES:
        result.update(status='TOO_FEW_EPISODES',
                      reason=f'{len(measured)} measured episode(s); at least {MIN_EPISODES} are needed for a range.')
        return result
    rets = [e['portfolio_return'] for e in measured]
    # Baseline: exclude ANY window touching one of the dated event windows.
    # Excluding only the episode start allowed the same shock to leak into up
    # to 2*window neighbouring "ordinary" windows, contaminating the control.
    event_sessions = set()
    for episode in measured:
        first = calendar.index(episode['base_session'])
        last = calendar.index(episode['end_session'])
        event_sessions.update(calendar[first:last + 1])
    ordinary = []
    for i in range(1, len(calendar) - window + 1):
        if not event_sessions.isdisjoint(calendar[i - 1:i + window]):
            continue
        ordinary.append(math.fsum(p['weight'] * (closes[p['symbol']][calendar[i + window - 1]] /
                                                 closes[p['symbol']][calendar[i - 1]] - 1) for p in invested))
    lo5, hi95 = (_quantile(ordinary, .05), _quantile(ordinary, .95)) if ordinary else (None, None)
    # This count is preserved solely for backward-compatible machine readers.
    # For n distinct values it ALWAYS equals n-2, regardless of predictability.
    covered = 0
    for k, r in enumerate(rets):
        others = rets[:k] + rets[k + 1:]
        covered += min(others) <= r <= max(others)
    # Actual sample fragility: the change in the historical envelope's endpoints
    # upon removal of each observation. Unlike min/max containment this varies
    # with the magnitude of outliers and identifies the range-setting episode.
    envelope_influence = []
    for k, episode in enumerate(measured):
        others = rets[:k] + rets[k + 1:]
        lower_change = min(others) - min(rets)
        upper_change = max(rets) - max(others)
        envelope_influence.append({
            'episode': episode['episode'], 'onset_session': episode['onset_session'],
            'lower_endpoint_change': lower_change,
            'upper_endpoint_change': upper_change,
            'max_abs_endpoint_change_usd': notional * max(abs(lower_change), abs(upper_change)),
        })
    most_influential = max(envelope_influence, key=lambda e: e['max_abs_endpoint_change_usd'])
    # Event-study sensitivity: report a second, explicitly smaller sample
    # excluding windows with an OTHER catalogued major-event onset. This is
    # not a causal identification: unrecorded news can still overlap.
    without_catalogued_overlap = [e for e in measured if not e['overlapping_episodes']]
    clean_rets = [e['portfolio_return'] for e in without_catalogued_overlap]
    # Descriptive ranking against NON-OVERLAPPING event windows (but the
    # ordinary rolling windows overlap each other). Neither a p-value nor
    # event-attributed causal effect; the +1 prevents an artificial zero.
    median_ordinary = _quantile(ordinary, .5) if ordinary else None
    for episode in measured:
        episode['ordinary_window_two_sided_tail_rank'] = (
            (1 + sum(abs(r - median_ordinary) >= abs(episode['portfolio_return'] - median_ordinary)
                     for r in ordinary)) / (len(ordinary) + 1)
            if ordinary else None)
    worst = min(measured, key=lambda e: e['portfolio_return'])
    result.update(
        status='MEASURED_RANGE',
        return_range=[min(rets), max(rets)], median_return=_quantile(rets, .5),
        pnl_range_usd=[notional * min(rets), notional * max(rets)], median_pnl_usd=notional * _quantile(rets, .5),
        worst_episode=worst['episode'],
        ordinary_window_5_95=[lo5, hi95] if ordinary else None,
        ordinary_window_count=len(ordinary),
        ordinary_baseline_excludes_full_episode_windows=True,
        episodes_outside_ordinary_range=sum(not lo5 <= r <= hi95 for r in rets) if ordinary else None,
        leave_one_out_covered=covered, leave_one_out_total=len(rets),
        leave_one_out_coverage_is_order_statistic_identity=True,
        envelope_influence=envelope_influence,
        max_envelope_endpoint_influence_usd=most_influential['max_abs_endpoint_change_usd'],
        most_influential_episode=most_influential['episode'],
        unconfounded_by_catalogued_events_count=len(clean_rets),
        catalogued_overlap_excluded_count=len(rets) - len(clean_rets),
        no_catalogued_overlap_return_range=[min(clean_rets), max(clean_rets)] if len(clean_rets) >= MIN_EPISODES else None,
        no_catalogued_overlap_median_return=_quantile(clean_rets, .5) if len(clean_rets) >= MIN_EPISODES else None,
        common_history=[calendar[0], calendar[-1]] if calendar else None)
    return result


def _shift(day, days):
    from datetime import timedelta
    return (date.fromisoformat(day) + timedelta(days=days)).isoformat()


def analogue_stress_for_events(events, positions, prices, *, notional):
    """Attach one analogue result per event id whose family has an episode set."""
    out = {}
    for event in events or ():
        name = episode_set_for(event.get('family'), ' '.join(str(event.get(k) or '') for k in ('title', 'original_question')))
        if name and event.get('event_id'):
            out[event['event_id']] = analogue_stress(positions, prices, notional=notional, set_name=name)
    return out

"""Official past release dates for descriptive daily FOMC/CPI event studies.

Dates identify sessions only. They supply no expectations, surprise magnitude,
causal attribution, or evidence that a daily close was known before release.
"""
from datetime import date, datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, build_opener

from .adapters import DataAccessError, _NoRedirect

SOURCES = {
    'FOMC': 'https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm',
    'CPI': 'https://www.bls.gov/bls/news-release/cpi.htm',
}


def _safe_official_url(value, expected):
    try:
        url, anchor = urlsplit(value), urlsplit(expected)
        return (url.scheme == 'https' and url.hostname == anchor.hostname and
                url.port in (None, 443) and not url.username and not url.password)
    except ValueError:
        return False


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self._divs = []
        self._strong = None
        self._anchor = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'div':
            self._divs.append({'meeting': 'fomc-meeting' in (attrs.get('class') or '').split(),
                               'statement': False})
        elif tag == 'strong':
            self._strong = {'text': [], 'container': self._divs[-1] if self._divs else None}
        elif tag == 'a' and isinstance(attrs.get('href'), str):
            self._anchor = {'href': attrs['href'], 'text': [],
                            'meeting_statement': bool(self._divs and self._divs[-1]['statement']
                                                      and any(d['meeting'] for d in self._divs))}

    def handle_data(self, data):
        if self._strong is not None:
            self._strong['text'].append(data)
        if self._anchor is not None:
            self._anchor['text'].append(data)

    def handle_endtag(self, tag):
        if tag == 'strong' and self._strong is not None:
            container = self._strong['container']
            if container is not None:
                container['statement'] = ''.join(self._strong['text']).strip().lower() == 'statement:'
            self._strong = None
        elif tag == 'a' and self._anchor is not None:
            self._anchor['text'] = ' '.join(''.join(self._anchor['text']).split())
            self.links.append(self._anchor)
            self._anchor = None
        elif tag == 'div' and self._divs:
            self._divs.pop()


def parse_release_dates(raw: bytes, *, family: str, through: date) -> list[str]:
    if family not in SOURCES or not 0 < len(raw) <= 2_000_000:
        raise ValueError('Invalid official release-calendar input')
    text = raw.decode('utf-8')
    title = 'FOMC' if family == 'FOMC' else 'Consumer Price Index'
    if title.lower() not in text.lower():
        raise ValueError('Official page does not identify the requested series')
    parser = _Links()
    parser.feed(text)
    dates = set()
    for link in parser.links:
        if family == 'FOMC' and not (link['meeting_statement'] and link['text'].lower() == 'html'):
            # The calendar also links strategy notation votes and other monetary
            # releases. Only the HTML link under a meeting's Statement label is
            # a decision-session observation; a matching filename is not enough.
            continue
        url = urlsplit(urljoin(SOURCES[family], link['href']))
        if not _safe_official_url(url.geturl(), SOURCES[family]):
            continue
        if family == 'FOMC':
            match = re.fullmatch(r'/newsevents/pressreleases/monetary(\d{8})a\.htm', url.path)
            fmt = '%Y%m%d'
        else:
            match = re.fullmatch(r'/news\.release/archives/cpi_(\d{8})\.htm', url.path)
            fmt = '%m%d%Y'
        if match:
            try:
                event_date = datetime.strptime(match.group(1), fmt).date()
            except ValueError:
                continue
            if event_date <= through:
                dates.add(event_date.isoformat())
    if len(dates) < 4:
        raise ValueError('Insufficient independently linked release dates; source layout may have changed')
    return sorted(dates)


def fetch_release_dates(output_dir: Path, *, through: date, opener=None) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    open_url = opener or build_opener(_NoRedirect()).open
    for family, url in SOURCES.items():
        try:
            request = Request(url, headers={'User-Agent': 'CastleOption1/1.0 (read-only)', 'Accept': 'text/html'})
            with open_url(request, timeout=12) as response:
                if (response.status != 200 or not _safe_official_url(response.url, url)
                        or urlsplit(response.url).path != urlsplit(url).path):
                    raise DataAccessError('Official calendar host or status mismatch')
                raw = response.read(2_000_001)
            dates = parse_release_dates(raw, family=family, through=through)
            retrieved = datetime.now(timezone.utc).isoformat()
            stamp = retrieved.replace(':', '').replace('.', '')
            path = output_dir / f'{family}_{stamp}.html'
            path.write_bytes(raw)
            meta = {'url': url, 'retrieved_at': retrieved, 'sha256': hashlib.sha256(raw).hexdigest(),
                    'status': 200, 'fixture': opener is not None, 'raw_path': str(path)}
            path.with_suffix('.meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
            results[family] = {**meta, 'status': 'OFFICIAL_LINK_DATES', 'http_status': meta['status'], 'dates': dates,
                'scope': 'Release-linked trading sessions for descriptive daily association; not causal surprises'}
        except Exception:
            # No arbitrary HTTP exception text is persisted.
            results[family] = {'status': 'UNAVAILABLE', 'dates': [], 'source_url': url,
                               'reason': 'Official release-date retrieval or layout validation failed'}
    return results

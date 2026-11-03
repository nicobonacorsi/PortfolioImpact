"""Strict portfolio accounting and conservative, one-level ETF look-through.

Look-through identifies instruments.  It does not estimate betas, factor loads,
event sensitivities, or the fraction of an event already reflected in equities.
All weights are fractions, never percentages.  No weights are renormalized.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
import math
import re
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse


WEIGHT_TOLERANCE = 1e-8
_ASSET_TYPES = frozenset({"stock", "etf", "cash"})


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number, not a boolean.")
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{field} must be a finite number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite.")
    return number


def _symbol(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("Instrument symbol must be a string.")
    symbol = value.strip().upper()
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9.\-_/]{0,31}", symbol):
        raise ValueError(f"Invalid instrument symbol: {value!r}.")
    return symbol


@dataclass(frozen=True)
class Position:
    symbol: str
    weight: float
    asset_type: str = "stock"

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        object.__setattr__(self, "weight", _finite_number(self.weight, "Position weight"))
        if self.weight < 0 or self.weight > 1:
            raise ValueError("Position weights must be between zero and one.")
        if self.asset_type not in _ASSET_TYPES:
            raise ValueError("Position asset_type must be stock, etf, or cash.")


@dataclass(frozen=True)
class Portfolio:
    positions: tuple[Position, ...]
    notional: float = 100_000.0
    currency: str = "USD"

    def __post_init__(self) -> None:
        positions = tuple(self.positions)
        object.__setattr__(self, "positions", positions)
        if not positions or not all(isinstance(p, Position) for p in positions):
            raise ValueError("A portfolio needs at least one valid Position.")
        symbols = [p.symbol for p in positions]
        if len(set(symbols)) != len(symbols):
            raise ValueError("Duplicate portfolio symbols must be aggregated explicitly.")
        total = math.fsum(p.weight for p in positions)
        if abs(total - 1.0) > WEIGHT_TOLERANCE:
            raise ValueError(f"Portfolio weights must sum to 1; received {total:.12g}. No normalization is applied.")
        object.__setattr__(self, "notional", _finite_number(self.notional, "Portfolio notional"))
        if self.notional <= 0:
            raise ValueError("Portfolio notional must be positive.")
        if self.currency != "USD":
            raise ValueError("The first slice supports USD portfolios only; FX exposure is not modeled.")

    @property
    def total_weight(self) -> float:
        return math.fsum(position.weight for position in self.positions)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_portfolio(rows: Iterable[Mapping[str, Any]], notional: float = 100_000.0) -> Portfolio:
    """Create a portfolio from UI/JSON rows; weights must already be fractions."""
    positions = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Every portfolio row must be a mapping.")
        if "symbol" not in row or "weight" not in row or "asset_type" not in row:
            raise ValueError("Every portfolio row requires symbol, weight, and asset_type.")
        positions.append(Position(row["symbol"], row["weight"], row["asset_type"]))
    return Portfolio(tuple(positions), notional=notional)


@dataclass(frozen=True)
class Holding:
    symbol: str
    weight: float
    asset_type: str = "stock"


@dataclass(frozen=True)
class HoldingsSnapshot:
    fund_symbol: str
    as_of: str | date
    retrieved_at: str | datetime
    source_url: str
    holdings: tuple[Holding, ...]
    reliable: bool = True
    weight_basis: str = "fraction_of_nav"
    complete: bool = True


@dataclass(frozen=True)
class Exposure:
    symbol: str
    portfolio_weight: float
    original_symbol: str
    asset_type: str
    status: str
    reason: str
    as_of: str | None = None
    retrieved_at: str | None = None
    source_url: str | None = None


@dataclass(frozen=True)
class ExposureReport:
    entries: tuple[Exposure, ...]
    identified_by_symbol: dict[str, float]
    unresolved_weight: float
    total_weight: float
    max_holdings_age_days: int
    evaluated_at: str
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_timestamp(value: str | datetime) -> datetime:
    dt = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("retrieved_at must include a timezone.")
    return dt.astimezone(timezone.utc)


def _snapshot_mapping(snapshot: HoldingsSnapshot | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(snapshot, HoldingsSnapshot):
        return asdict(snapshot)
    if isinstance(snapshot, Mapping):
        return dict(snapshot)
    raise ValueError("Holdings snapshot must be a HoldingsSnapshot or mapping.")


def _validate_snapshot(
    snapshot: HoldingsSnapshot | Mapping[str, Any],
    fund_symbol: str,
    now: datetime,
    max_holdings_age_days: int,
) -> tuple[dict[str, Any], list[Holding], date, datetime]:
    data = _snapshot_mapping(snapshot)
    if _symbol(data.get("fund_symbol", "")) != fund_symbol:
        raise ValueError("Holdings fund symbol does not match the portfolio ETF.")
    if data.get("reliable") is not True:
        raise ValueError("The data adapter has not marked this snapshot reliable.")
    if data.get("complete", True) is not True:
        raise ValueError("The holdings retrieval is incomplete; pagination or retrieval failed.")
    if data.get("weight_basis") != "fraction_of_nav":
        raise ValueError("Holdings weights lack a verified fraction-of-NAV basis.")
    source = urlparse(str(data.get("source_url", "")))
    if source.scheme not in {"http", "https"} or not source.netloc:
        raise ValueError("A valid holdings source URL is required.")
    if source.username is not None or source.password is not None or source.query:
        raise ValueError("Holdings source URLs must not contain credentials or query parameters.")
    retrieved = _utc_timestamp(data["retrieved_at"])
    as_of = data["as_of"]
    if isinstance(as_of, datetime):
        as_of = as_of.date()
    elif not isinstance(as_of, date):
        as_of = date.fromisoformat(str(as_of))
    if retrieved > now:
        raise ValueError("Holdings retrieval timestamp is in the future.")
    if as_of > retrieved.date() or as_of > now.date():
        raise ValueError("Holdings effective date is in the future.")
    if (now.date() - as_of).days > max_holdings_age_days:
        raise ValueError(f"Holdings are stale under the explicit {max_holdings_age_days}-day limit.")
    raw_holdings = data.get("holdings")
    if not isinstance(raw_holdings, (list, tuple)) or not raw_holdings:
        raise ValueError("No usable holdings records were supplied.")
    holdings = []
    seen = set()
    for item in raw_holdings:
        item = asdict(item) if isinstance(item, Holding) else item
        if not isinstance(item, Mapping):
            raise ValueError("Every holding must be a record.")
        symbol = _symbol(item.get("symbol", ""))
        if symbol == fund_symbol:
            raise ValueError("Self-referential ETF holdings are unsupported.")
        if symbol in seen:
            raise ValueError("Duplicate holdings must be resolved by the data adapter.")
        seen.add(symbol)
        if item.get("weight") is None:
            # Upstream unknown weights never become zero-valued known holdings.
            # Their unidentified share remains in the unallocated NAV residual.
            continue
        weight = _finite_number(item.get("weight"), "Holding weight")
        if weight < 0 or weight > 1:
            raise ValueError("Negative, levered, or percent-scaled holdings are unsupported.")
        asset_type = item.get("asset_type")
        if asset_type not in _ASSET_TYPES | {"other"}:
            raise ValueError("Derivative or unclassified holdings require a richer exposure model.")
        holdings.append(Holding(symbol, weight, asset_type))
    # Issuer files routinely sum to slightly above 100% of NAV (rounding,
    # accruals). Up to 1% excess is scaled back so the fund's portfolio weight
    # is conserved exactly; a larger excess still rejects the snapshot.
    reported = math.fsum(h.weight for h in holdings)
    if reported > 1.0 + 0.01:
        raise ValueError("Reported ETF holdings exceed 100% of NAV.")
    if reported > 1.0 + 1e-12:
        holdings = [Holding(h.symbol, h.weight / reported, h.asset_type) for h in holdings]
    return data, holdings, as_of, retrieved


def recover_exposures(
    portfolio: Portfolio,
    snapshots: Mapping[str, HoldingsSnapshot | Mapping[str, Any]] | None = None,
    *,
    now: datetime | None = None,
    max_holdings_age_days: int = 90,
) -> ExposureReport:
    """Preserve allocation mass, including missing/rejected ETF look-through.

    The age threshold is an explicit application policy, not a guarantee that a
    fund's holdings remained unchanged. A complete API response can still expose
    less than 100% of NAV; its residual remains unidentified. Invalid snapshots
    are rejected as a whole. Nested ETFs remain unresolved at this one-level cut.
    """
    if isinstance(max_holdings_age_days, bool) or not isinstance(max_holdings_age_days, int) or max_holdings_age_days < 0:
        raise ValueError("max_holdings_age_days must be a nonnegative integer.")
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware.")
    now = now.astimezone(timezone.utc)
    snapshots = snapshots or {}
    entries = []
    for position in portfolio.positions:
        if position.asset_type != "etf":
            entries.append(Exposure(position.symbol, position.weight, position.symbol, position.asset_type, "direct", "Direct portfolio position."))
            continue
        snapshot = snapshots.get(position.symbol)
        if snapshot is None:
            entries.append(Exposure(position.symbol, position.weight, position.symbol, "etf", "unresolved_etf", "No verified ETF holdings snapshot is available."))
            continue
        try:
            data, holdings, as_of, retrieved = _validate_snapshot(snapshot, position.symbol, now, max_holdings_age_days)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            # A failed lookup is uncertainty in exposure, not permission to drop it.
            entries.append(Exposure(position.symbol, position.weight, position.symbol, "etf", "unresolved_etf", f"Snapshot rejected: {exc}"))
            continue
        provenance = {"as_of": as_of.isoformat(), "retrieved_at": retrieved.isoformat(), "source_url": data["source_url"]}
        for holding in holdings:
            nested = holding.asset_type == "etf"
            unsupported = holding.asset_type == "other"
            entries.append(Exposure(
                holding.symbol, position.weight * holding.weight,
                position.symbol, holding.asset_type,
                "unresolved_etf" if nested or unsupported else "look_through",
                "Unclassified or derivative holding: reported weight preserved as unresolved." if unsupported else "Nested ETF: underlying identity remains unresolved at one level." if nested else "Reported fund holding; weight retained without renormalization.",
                **provenance,
            ))
        residual = max(0.0, 1.0 - math.fsum(h.weight for h in holdings))
        if residual > 0:
            entries.append(Exposure(position.symbol, position.weight * residual, position.symbol, "etf", "unresolved_etf", "Unidentified share of fund NAV; reported holdings were not renormalized.", **provenance))
    identified: dict[str, list[float]] = {}
    for entry in entries:
        if entry.status != "unresolved_etf":
            identified.setdefault(entry.symbol, []).append(entry.portfolio_weight)
    return ExposureReport(
        entries=tuple(entries),
        identified_by_symbol={symbol: math.fsum(weights) for symbol, weights in sorted(identified.items())},
        unresolved_weight=math.fsum(e.portfolio_weight for e in entries if e.status == "unresolved_etf"),
        total_weight=math.fsum(e.portfolio_weight for e in entries),
        max_holdings_age_days=max_holdings_age_days,
        evaluated_at=now.isoformat(),
        limitations=(
            "Look-through is a dated instrument-identity decomposition, not a factor or event-sensitivity model.",
            "Fund changes after the effective date, derivatives, leverage, fees, and tracking differences are not modeled.",
            "Rejected, missing, unclassified, residual, and nested-ETF exposure remains explicitly unresolved.",
            "Effective/processed dates do not prove historical public availability; this is not a point-in-time ledger.",
        ),
    )

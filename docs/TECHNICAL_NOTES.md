# Technical notes

## Portfolio and exposure

User positions are normalized to a common portfolio notional. For ETFs, the engine resolves fund composition when source data is available and aggregates overlapping issuer exposure across direct holdings and funds. Each event remains associated with the positions and economic channels used to rank its relevance.

## Event definition and probability

An event has an explicit trigger and date or terminal window. Prediction-market research checks the actual contract question and resolution criteria before associating a quote with an event. A related contract is not silently treated as an exact match. Structural model estimates retain their own origin and uncertainty interval; market-implied contract prices and model probabilities are not interchangeable observations.

## Conditional portfolio impact

Let `V₀` denote the portfolio notional, `wᵢ` the normalized holding weight and `rᵢ(e,o,T)` the modeled return for holding `i` under event `e`, outcome `o` and common future endpoint `T`. The aggregation is

```text
Portfolio change(e, o, T) = V0 × Σᵢ [ wᵢ × rᵢ(e, o, T) ]
```

The user can inspect YES and NO as separate conditional outcomes. Event probability, uncertainty in assumptions and overlap through ETFs are kept explicit rather than conflated with realized P&L.

## Application components

- `castle_risk/`: portfolio normalization, exposure recovery, event research, exact contract matching, probability calculations and impact estimation.
- `scripts/serve_portfolio_impact.py`: HTTP API and job lifecycle.
- `castle_risk/codespaces_server.py`: private Codespaces application server and provider connection handling.
- `web/portfolio_impact_v4/`: responsive UI, event comparison and recorded scenario browser.
- `examples/recorded_analysis/case.json`: saved example for repeatable walkthroughs.
- `tests/`: calculation and HTTP regression tests.

## Running modes

Recorded examples are bundled with the application. Fresh research uses configured market-data and model providers and records source context to support inspection. User-supplied credentials are held in server memory; the application is intended to run on a private Codespaces port or on the user's local machine.

## Reproducibility

Dependencies are pinned for Python 3.12 in `requirements-lock.txt`. The test workflow checks the calculation and HTTP suites as well as JavaScript syntax. Reproducible saved scenarios and prospective estimates should be interpreted according to their event definitions, as-of timestamps and model inputs.

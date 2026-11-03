# Portfolio Impact

**Scenario analysis for portfolios.**

Portfolio Impact connects stock and ETF holdings to dated economic and geopolitical events. It researches event probabilities, maps portfolio exposures, and estimates the value change associated with each defined outcome.

**[Launch Portfolio Impact in GitHub Codespaces](https://codespaces.new/nicobonacorsi/PortfolioImpact)**

## Explore the application

1. Open the launch link and select **Create codespace**. The application starts automatically.
2. Choose a recorded portfolio, or select **Edit portfolio** to enter stock and ETF positions or import a CSV.
3. Explore the event list, ordered by modeled portfolio impact.
4. Select an event and switch between **YES** and **NO** to compare its conditional portfolio outcomes.
5. Expand **Definitions & sources** for the settlement rule, evidence, assumptions, ETF exposures, and calculation inputs.

Recorded examples are available without API keys. Researching a new portfolio uses Massive market data and Anthropic models, configured by the user in the connection screen; provider charges may apply.

## Recorded portfolio

The November 3, 2026 demonstration covers an illustrative **$100,000** portfolio:

| Holding | Weight |
| --- | ---: |
| Nvidia (NVDA) | 30% |
| Taiwan Semiconductor (TSM) | 25% |
| Lockheed Martin (LMT) | 25% |
| iShares 20+ Year Treasury Bond ETF (TLT) | 20% |

The saved analysis includes eight dated events, probability estimates and uncertainty ranges, and paired YES/NO portfolio scenarios. Its evaluation window ends February 1, 2027.

For one Taiwan port-blockade scenario, the model reports a 5.69% event probability, with conditional portfolio changes of −$9,420 for YES and +$3,200 for NO, measured against a common future-date reference. These are **scenario model outputs**, not observed returns or a claim of calibrated predictive accuracy.

## Analysis workflow

| Stage | Capability |
| --- | --- |
| Holdings | Resolve tickers, values, ETF constituents, and overlapping exposures. |
| Discovery | Identify economic and geopolitical events relevant to the actual portfolio. |
| Probability | Inspect dated research and related Kalshi or Polymarket contracts; distinguish matched market quotes from model estimates. |
| Impact | Construct event-outcome return assumptions, calculate holding-level contributions, and aggregate portfolio-dollar scenarios on one comparison date. |
| Inspection | Expose event definitions, market evidence, timestamps, uncertainty ranges, and model inputs. |

## Run locally

**Windows:** Install Python 3.12 and run `START_V4_WINDOWS.cmd`.

**macOS / Linux:** Install Python 3.12 and run:

```bash
bash scripts/start_v4_codespaces.sh --open
```

Open `http://127.0.0.1:8765`. To run the regression suite:

```bash
python -m pytest -q
```

Source: [analysis engine](castle_risk/) · [browser interface](web/portfolio_impact_v4/) · [tests](tests/) · [technical notes](docs/TECHNICAL_NOTES.md) · [original assignment](docs/CASTLE_ORIGINAL_ASSIGNMENT.md)

---

Nicolò Bonacorsi

# Portfolio Impact

**Event-driven scenario analysis for investment portfolios.**

Portfolio Impact connects stocks and ETFs to dated macroeconomic and geopolitical events, researches their probabilities and estimates portfolio-level outcomes. The interface puts the most consequential scenarios first and lets users inspect the event definition, supporting evidence and holding-level exposure.

**[Launch Portfolio Impact in GitHub Codespaces](https://codespaces.new/nicobonacorsi/PortfolioImpact)**

## Open the application

1. Select **Create codespace** from the link above. The Python environment is configured automatically and the private application port opens in your browser.
2. On the connection screen, choose **Open Portfolio Impact** to explore recorded analyses without entering credentials.
3. Select an example portfolio, or use **Edit portfolio** to enter holdings, dollar amounts and weights or import CSV data.
4. Review dated scenarios ranked by portfolio impact. Compare **YES** and **NO** when a paired contract is available; otherwise inspect the recorded economic outcome categories.
5. Expand **Definitions & sources** to review the event rule, exposed holdings, probability evidence and conditional portfolio model.

For a *new* portfolio, connect Massive market data and Anthropic research in the private connection screen. Provider charges may apply. Credentials are held in server memory, not committed to the repository; keep the Codespaces port **Private**.

## Included demonstration data

- **NVDA 20% / QQQ 50% / TLT 30%** — $250,000 demonstration portfolio, including dated Taiwan-related YES/NO contract scenarios and US macroeconomic outcomes.
- **BND 60% / IEF 40%** — $100,000 bond portfolio with interest-rate and economic scenarios.
- **AAPL 35% / JPM 35% / GLD 30%** — $100,000 multi-asset portfolio with geopolitical and macroeconomic scenarios.

All dollar figures displayed in these examples come from the saved scenario data, including outcome-specific P&L ranges; no dollar impact is injected through interface constants. These are conditional scenario estimates, not realized returns. Contract prices and modeled scenario weights are identified separately. The examples are dated research snapshots, not live quotes.

## What the system does

| Stage | Function |
| --- | --- |
| Portfolio | Resolve positions, ETF constituents and overlapping issuer exposures. |
| Event discovery | Identify dated economic, regulatory and geopolitical scenarios relevant to the holdings. |
| Probability | Compare the exact event question with prediction-market contract terms; use a modeled estimate when appropriate. |
| Conditional impact | Model each outcome's holding-level changes and aggregate them into portfolio-dollar effects at a common endpoint. |
| Inspection | Show the event trigger, venue and source information, estimation inputs and calculation details. |

## Run locally

**Windows:** Install Python 3.12 and run `START_V4_WINDOWS.cmd`.

**macOS / Linux:** Install Python 3.12 and run:

```bash
bash scripts/start_v4_codespaces.sh --open
```

The local application runs at `http://127.0.0.1:8765`.

Run the Python test suite:

```bash
python -m pip install -r requirements-lock.txt
python -m pytest -q
```

Project files: [analysis engine](castle_risk/) · [application UI](web/portfolio_impact_v4/) · [test suite](tests/) · [methodology](docs/TECHNICAL_NOTES.md) · [assessment brief](docs/CASTLE_ORIGINAL_ASSIGNMENT.md).

---

Nicolò Bonacorsi

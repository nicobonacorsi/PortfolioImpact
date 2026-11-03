# Portfolio Impact

**Event-driven scenario analysis for investment portfolios.**

Portfolio Impact connects stocks and ETFs to dated macroeconomic and geopolitical events, researches their probabilities and estimates portfolio-level outcomes. The interface puts the most consequential scenarios first and lets users inspect the event definition, supporting evidence and holding-level exposure.

**[Launch Portfolio Impact in GitHub Codespaces](https://codespaces.new/nicobonacorsi/PortfolioImpact)**

## Open the application

1. Select **Create codespace** from the link above. The Python environment is configured automatically and the private application port opens in your browser.
2. On the connection screen, choose **Open Portfolio Impact** to explore recorded analyses without entering credentials.
3. Select an example portfolio, or use **Edit portfolio** to enter holdings, dollar amounts and weights or import CSV data.
4. Review events ranked by modeled portfolio impact. Open any event to compare **YES** and **NO** outcomes on the same evaluation date.
5. Expand **Definitions & sources** for exact event terms, related prediction-market contracts, assumptions, ETF exposures and research provenance.

For a *new* portfolio, connect Massive market data and Anthropic research in the private connection screen. Provider charges may apply. Credentials are held in server memory, not committed to the repository; keep the Codespaces port **Private**.

## Included demonstration data

- A recorded $100,000 portfolio from October 30, 2026: XOM 20%, TSM 25%, LMT 15%, SPY 25%, TLT 15%.
- A browser scenario collection containing, among other cases, an NVDA / QQQ / TLT portfolio and paired contract-outcome scenarios.
- A November 2 research snapshot for LMT, SMH, EWJ and LQD, including modeled event probabilities and conditional portfolio outcomes.

Saved outcomes are reproducible *scenario calculations*, not realized investment returns. Probability intervals and market quotes are shown with their definitions and evidence status.

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

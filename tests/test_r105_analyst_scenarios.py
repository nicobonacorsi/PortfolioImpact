"""Acceptance of the Castle-specific, separate analyst-assumption layer."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import random
import unittest

from castle_risk.analyst_scenarios import (
    SCHEMA, TLT_DURATION, LONG_AUCTION_YIELD_SHOCK_BP,
    attach_analyst_scenarios, propose_analyst_scenario,
)
from castle_risk.scenario_lab import evaluate_scenario


def portfolio(symbols=("MSFT", "JPM", "TLT", "GLD"), weights=(.30, .25, .25, .20), nav=100_000):
    kinds = {"TLT": "etf", "GLD": "etf", "IEF": "etf", "QQQ": "etf",
             "VTI": "etf", "BND": "etf", "FUND": "etf", "USD": "cash"}
    return {"id": "synthetic-live", "capture": "2026-10-31T12:00:00Z", "notional": nav,
            "positions": [{"symbol": s, "weight": w, "type": kinds.get(s, "stock")}
                          for s, w in zip(symbols, weights)],
            "lookthrough": []}


EVENTS = [
    ("shutdown", "us_government_shutdown_by_2027_01_29", "U.S. government shutdown by 29 January 2027"),
    ("china_tariff", "new_us_tariffs_on_china_by_2027_01_29", "New U.S. tariffs on China by 29 January 2027"),
    ("long_auction", "treasury_20y_auction_sizes_increase", "Increase in nominal Treasury 20-year auction sizes"),
    ("cyber", "MSFT_CYBER_8K_ITEM_1_05", "Microsoft 8-K Item 1.05 cybersecurity incident"),
    ("antitrust", "MSFT_DOJ_ANTITRUST_SUIT", "FTC or DOJ antitrust lawsuit against MSFT"),
    ("payroll", "payrolls-2027-01-08-scheduled", "December payrolls release on 8 January 2027"),
    ("basel", "basel_iii_endgame_final", "Basel III endgame finalization"),
]


def risk(identifier, title, *, existing=None):
    return {"id": identifier, "title": title, "outcomes": existing or [],
            "quality": {"definition_review": {"status": "DEFINED_FOR_REVIEW"}}}


class AnalystScenariosForCastle(unittest.TestCase):
    def test_three_shipped_examples_recompute_11_assumptions_without_provider_receipts(self):
        path = Path(__file__).resolve().parents[1] / "web" / "portfolio_impact" / "data.json"
        cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
        self.assertEqual(len(cases), 3)
        count = 0
        for c in cases:
            self.assertEqual(c["user_supplied"] if "user_supplied" in c else None, None)
            for event in c["risks"]:
                proposal = event.get("analyst_assumption")
                if proposal is None:
                    continue
                count += 1
                request = {"hypothesis": proposal["scenario_condition"], "mode": "binary_gap",
                           "position_shocks": proposal["position_return_gap_pct"],
                           "pricing_weight": .5, "pricing_weight_range": [.25, .75]}
                replay = evaluate_scenario(c, request)
                self.assertEqual(proposal["status"], "COMPLETE")
                self.assertAlmostEqual(replay["yes_pnl_usd"], proposal["result"]["yes_pnl_usd"])
                self.assertAlmostEqual(replay["no_pnl_usd"], proposal["result"]["no_pnl_usd"])
                self.assertFalse(proposal["result"]["pricing"]["prediction_market_quote_used_as_equity_weight"])
        self.assertEqual(count, 11)

    def test_all_seven_original_gaps_have_an_editable_separate_analyst_scenario(self):
        c = portfolio()
        for expected_key, identifier, title in EVENTS:
            with self.subTest(key=expected_key):
                event = risk(identifier, title)
                before = deepcopy(event)
                proposal = propose_analyst_scenario(c, event)
                self.assertIsNotNone(proposal)
                self.assertEqual(proposal["schema"], SCHEMA)
                self.assertEqual(proposal["key"], expected_key)
                self.assertEqual(proposal["status"], "COMPLETE")
                self.assertEqual(proposal["unmodeled_symbols"], [])
                self.assertFalse(proposal["prediction_market_probability_used"])
                self.assertEqual(proposal["result"]["covered_weight"], 1)
                self.assertTrue(proposal["result"]["complete"])
                self.assertTrue(math.isfinite(proposal["result"]["yes_pnl_usd"]))
                self.assertTrue(math.isfinite(proposal["result"]["no_pnl_usd"]))
                self.assertAlmostEqual(proposal["result"]["yes_pnl_usd"] +
                                       proposal["result"]["no_pnl_usd"], 0)
                self.assertEqual(proposal["result"]["pricing"]["equity_weight_q"], .5)
                self.assertFalse(proposal["result"]["pricing"]["q_empirically_identified"])
                self.assertFalse(proposal["result"]["event_effect_estimated"])
                self.assertEqual(event, before)  # never reclassify source evidence

    def test_economic_channels_and_original_nav_give_exact_dollars(self):
        c = portfolio()
        p = propose_analyst_scenario(c, risk("basel_iii_endgame", "Basel III final bank capital rule"))
        self.assertAlmostEqual(p["result"]["known_event_gap_usd"], -375)
        self.assertAlmostEqual(p["result"]["yes_pnl_usd"], -187.5)
        self.assertEqual(p["position_return_gap_pct"], {"MSFT": 0, "JPM": -1.5, "TLT": 0, "GLD": 0})
        a = propose_analyst_scenario(c, risk("treasury_20y_auction_size", "Increase in 20-year auction sizes"))
        # MSFT -$90, JPM -$25, TLT -$372.50 (observed duration), GLD -$80.
        self.assertAlmostEqual(a["result"]["known_event_gap_usd"], -567.5)
        self.assertAlmostEqual(a["result"]["yes_pnl_usd"], -283.75)
        self.assertAlmostEqual(a["position_return_gap_pct"]["TLT"], -1.49)
        observed = a["position_parameter_basis"]["TLT"]
        self.assertEqual(observed["kind"], "PUBLISHED_DURATION_WITH_ASSUMED_YIELD_MOVE")
        self.assertEqual(observed["effective_duration_years"], 14.90)
        self.assertEqual(observed["parameter_as_of"], "2026-10-29")
        self.assertEqual(observed["assumed_yield_change_bp"], 10.)
        self.assertEqual(TLT_DURATION["effective_years"], 14.90)
        self.assertEqual(LONG_AUCTION_YIELD_SHOCK_BP, 10.)
        self.assertIsNone(a["event_probability"]["value"])
        self.assertIn("same assumed two-state q", a["pricing_q_limitation"])

    def test_corporate_bond_funds_are_not_aggregate_treasury_proxies(self):
        for fund in ("LQD", "HYG"):
            with self.subTest(fund=fund):
                c = portfolio(("MSFT", fund), (.4, .6))
                c["positions"][1]["type"] = "etf"
                p = propose_analyst_scenario(c, risk("treasury_20y_auction", "20-year Treasury auction increases"))
                self.assertIsNotNone(p)
                self.assertEqual(p["status"], "PARTIAL")
                self.assertIn(fund, p["unmodeled_symbols"])
                self.assertEqual(p["position_parameter_basis"][fund]["kind"],
                                 "UNMODELED_NO_JUSTIFIED_INSTRUMENT_CHANNEL")
                self.assertIsNone(p["result"]["yes_pnl_usd"])

    def test_payroll_admits_competing_macro_regimes_not_forced_equity_sign(self):
        p = propose_analyst_scenario(portfolio(), risk("payrolls-2027-01-08", "Payrolls report 8 Jan 2027"))
        self.assertIn("lift some equities", p["alternative_economic_regime"])
        self.assertEqual(p["position_parameter_basis"]["MSFT"]["kind"],
                         "DECLARED_ANALYST_RETURN_GAP_NOT_CALIBRATED")

    def test_company_specific_shocks_can_trace_through_a_qqq_holding_without_double_counting(self):
        c = portfolio(("MSFT", "QQQ", "TLT"), (.2, .5, .3))
        c["lookthrough"] = [{"symbol": "MSFT", "direct": .2, "via": {"QQQ": .055},
                            "weight": .255, "unresolved": False}]
        p = propose_analyst_scenario(c, risk("MSFT_CYBER_8K_ITEM_1_05", "MSFT 8-K Item 1.05 cyber incident"))
        self.assertEqual(p["status"], "COMPLETE")
        self.assertAlmostEqual(p["position_return_gap_pct"]["MSFT"], -3)
        self.assertAlmostEqual(p["position_return_gap_pct"]["QQQ"], -.33)
        self.assertAlmostEqual(p["result"]["known_event_gap_usd"], -765)
        self.assertAlmostEqual(p["result"]["yes_pnl_usd"], -382.5)
        self.assertEqual(p["channels"]["QQQ"], "VERIFIED_ISSUER_LOOKTHROUGH_ASSUMPTION")

    def test_unknown_fund_is_not_falsely_declared_fully_modeled(self):
        c = portfolio(("MSFT", "FUND"), (.4, .6))
        p = propose_analyst_scenario(c, risk("MSFT_CYBER_8K_ITEM_1_05", "MSFT 8-K Item 1.05 cyber incident"))
        self.assertEqual(p["status"], "PARTIAL")
        self.assertEqual(p["unmodeled_symbols"], ["FUND"])
        self.assertIsNone(p["result"]["yes_pnl_usd"])
        self.assertAlmostEqual(p["result"]["known_yes_pnl_usd"], -600)
        p2 = propose_analyst_scenario(c, risk("shutdown", "Government shutdown"))
        self.assertEqual(p2["status"], "PARTIAL")

    def test_new_issuer_specific_stock_works_with_generic_rule_without_ticker_specific_patch(self):
        c = portfolio(("UNH", "JPM", "USD"), (.4, .4, .2))
        p = propose_analyst_scenario(c, risk("unh_antitrust_suit_by_2027", "DOJ antitrust suit against UnitedHealth Group"))
        self.assertEqual(p["key"], "antitrust")
        self.assertAlmostEqual(p["result"]["known_event_gap_usd"], -800)
        self.assertEqual(p["position_return_gap_pct"], {"UNH": -2, "JPM": 0, "USD": 0})

    def test_evidence_admission_unchanged_after_attachment(self):
        event = risk("MSFT_CYBER_8K_ITEM_1_05", "MSFT 8-K Item 1.05 cyber incident",
                     existing=[{"id": "source-1", "pnl": 12.5}])
        event["quality"]["model_admission"] = {"state": "CONDITIONAL_SCENARIO", "primary_impact_allowed": True}
        c = {**portfolio(), "risks": [event]}
        before = deepcopy(event)
        attach_analyst_scenarios(c)
        self.assertEqual(c["risks"][0]["outcomes"], before["outcomes"])
        self.assertEqual(c["risks"][0]["quality"], before["quality"])
        self.assertIn("analyst_assumption", c["risks"][0])

    def test_unknown_event_and_blocked_identity_and_bad_definition_do_not_auto_price(self):
        c = portfolio()
        self.assertIsNone(propose_analyst_scenario(c, risk("custom_unclassified", "Entirely unknown event")))
        c["instrument_identity"] = {"blocked_symbols": ["MSFT"]}
        self.assertIsNone(propose_analyst_scenario(c, risk("MSFT_CYBER_8K_ITEM_1_05", "MSFT cyber")))
        c.pop("instrument_identity")
        e = risk("MSFT_CYBER_8K_ITEM_1_05", "MSFT cyber")
        e["quality"]["definition_review"] = {"status": "UNDEFINED"}
        self.assertIsNone(propose_analyst_scenario(c, e))
        e["quality"]["definition_review"] = {"status": "DEFINED_FOR_REVIEW"}
        e["source_defined_scenario"] = True
        self.assertIsNone(propose_analyst_scenario(c, e))

    def test_twenty_unseen_random_portfolios_accounting_and_price_in_math(self):
        rng = random.Random(105)
        assets = ["MSFT", "JPM", "CAT", "NEE", "UNH", "XOM", "AAPL", "GLD", "TLT", "IEF", "BND", "QQQ", "VTI"]
        fund_symbols = {"GLD", "TLT", "IEF", "BND", "QQQ", "VTI"}
        for case_id in range(20):
            subset = rng.sample(assets, rng.randrange(2, 9))
            weights = [rng.uniform(.1, 1.8) for _ in subset]
            total = math.fsum(weights)
            c = {"id": str(case_id), "notional": rng.uniform(1000, 10_000_000),
                 "positions": [{"symbol": s, "weight": w / total,
                                "type": "etf" if s in fund_symbols else "stock"}
                               for s, w in zip(subset, weights)]}
            for key, identifier, title in EVENTS:
                if key in {"cyber", "antitrust"} and "MSFT" not in subset:
                    continue
                if key == "basel" and "JPM" not in subset:
                    continue
                p = propose_analyst_scenario(c, risk(identifier, title))
                self.assertIsNotNone(p)
                self.assertTrue(0 <= p["result"]["covered_weight"] <= 1 + 1e-9)
                self.assertAlmostEqual(p["result"]["known_event_gap_usd"],
                    math.fsum(row["original_value_usd"] * row["return_gap_pct"] / 100
                              for row in p["result"]["positions"] if row["modeled"]), places=7)
                self.assertAlmostEqual(.5 * p["result"]["known_yes_pnl_usd"] +
                                       .5 * p["result"]["known_no_pnl_usd"], 0, places=7)
                if p["result"]["complete"]:
                    self.assertAlmostEqual(p["result"]["yes_pnl_usd"] +
                                           p["result"]["no_pnl_usd"], 0, places=7)


if __name__ == "__main__":
    unittest.main()

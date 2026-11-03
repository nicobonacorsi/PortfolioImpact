"""R107 acceptance: real source counts, numerical replay, no evidence promotion."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import unittest

from castle_risk.analyst_scenarios import propose_analyst_scenario
from castle_risk.r107_evidence import (
    DATA_FILE, MARKET_SYMBOLS, _ols_hc3_split, empirical_miss_reference,
    load_study, nested_threshold_probability_bounds,
    payroll_context_for_positions, wilson_95,
)


ROOT = Path(__file__).resolve().parents[1]
TAPE_FILE = ROOT / "data" / "payrolls" / "payroll_surprise_tape.json"


def event():
    return {"id": "payrolls-2027-01-08-scheduled",
            "title": "December jobs report on January 8, 2027",
            "quality": {"definition_review": {"status": "DEFINED_FOR_REVIEW"}}}


def case():
    return {"id": "r107-audit", "notional": 100_000,
            "positions": [{"symbol":"MSFT", "weight": .3, "type":"stock"},
                          {"symbol":"JPM", "weight":.25,"type":"stock"},
                          {"symbol":"TLT", "weight":.25,"type":"etf"},
                          {"symbol":"GLD", "weight":.2,"type":"etf"}],
            "lookthrough": []}


class R107EmpiricalEvidence(unittest.TestCase):
    def test_reference_class_is_different_from_actual_consensus_question(self):
        tape = json.loads(TAPE_FILE.read_text(encoding="utf-8"))
        record = empirical_miss_reference(tape)
        self.assertEqual((record["count"], record["denominator"]), (3, 39))
        self.assertAlmostEqual(record["observed_frequency"], 3/39)
        self.assertEqual(record["wilson_95"], wilson_95(3, 39))
        self.assertLess(record["wilson_95"][0], 3/39)
        self.assertGreater(record["wilson_95"][1], 3/39)
        self.assertIsNone(record["forecast_probability"])
        self.assertIn("Kalshi", record["definition"])
        self.assertIn("consensus", record["not_equivalent_to"])

    def test_derived_data_recomputes_every_frozen_historical_coefficient(self):
        data = load_study()
        self.assertIsNotNone(data)
        self.assertEqual(data["provenance"]["kalshi_tape"]["path"],
                         "data/payrolls/payroll_surprise_tape.json")
        self.assertEqual(data["event_study"]["event_dates_n"], 31)
        self.assertEqual(set(data["event_study"]["fits"]), set(MARKET_SYMBOLS))
        self.assertFalse(data["used_in_portfolio_pnl"])
        self.assertTrue(DATA_FILE.is_file())
        rows = data["event_study"]["observations"]
        self.assertEqual(len(set(r["date"] for r in rows)), 31)
        self.assertEqual([r["date"] for r in rows], sorted(r["date"] for r in rows))
        for symbol in MARKET_SYMBOLS:
            fit = _ols_hc3_split([{"date": r["date"], "surprise_jobs": r["surprise_jobs"],
                                  "return_fraction": r["return_fraction_by_symbol"][symbol]} for r in rows])
            saved = data["event_study"]["fits"][symbol]
            self.assertEqual(fit, saved)
            self.assertEqual((fit["train_n"], fit["test_n"]), (22, 9))
            self.assertFalse(fit["admitted_as_forecast"])
            self.assertEqual(data["provenance"][symbol]["provider"], "Massive")
            self.assertEqual(data["provenance"][symbol]["daily_closes"], 753)

    def test_gld_observation_does_not_pass_causal_identification(self):
        gld = load_study()["event_study"]["fits"]["GLD"]
        self.assertLess(gld["slope_fraction_per_100k_jobs"], 0)
        self.assertLess(gld["slope_hc3_normal_95"][0], 0)
        self.assertGreater(gld["slope_hc3_normal_95"][1], 0)
        self.assertLess(gld["test_rmse_fraction"]["regression"],
                        gld["test_rmse_fraction"]["zero"])
        self.assertFalse(gld["mechanical_screen_passed"])

    def test_holding_scenario_and_market_probability_are_not_promoted(self):
        p = propose_analyst_scenario(case(), event())
        self.assertEqual(p["status"], "COMPLETE")
        self.assertEqual(p["empirical_context"]["status"], "HISTORICAL_CONTEXT_ONLY")
        self.assertIn("GLD", p["empirical_context"]["holding_studies"])
        self.assertNotIn("MSFT", p["empirical_context"]["holding_studies"])
        self.assertEqual(p["position_parameter_basis"]["GLD"]["kind"],
                         "DECLARED_ANALYST_RETURN_GAP_NOT_CALIBRATED")
        self.assertAlmostEqual(p["position_return_gap_pct"]["GLD"], .5)
        self.assertAlmostEqual(p["result"]["known_event_gap_usd"], -190)
        self.assertIsNone(p["event_probability"]["value"])
        self.assertFalse(p["empirical_context"]["event_probability_updated"])
        self.assertFalse(p["empirical_context"]["portfolio_pnl_updated"])
        self.assertFalse(p["prediction_market_probability_used"])

    def test_chronological_split_never_refits_on_heldout_outcomes(self):
        rows = [{"date": f"2025-{1+i//3:02d}-{2+i%3:02d}",
                 "surprise_jobs": (i % 7 - 3)*10_000,
                 "return_fraction": .0002 * (i-10)} for i in range(22)]
        rows += [{"date": f"2026-{i+1:02d}-03", "surprise_jobs": i*10_000,
                  "return_fraction": .001*i} for i in range(9)]
        first = _ols_hc3_split(rows)
        changed = deepcopy(rows)
        for row in changed:
            if row["date"].startswith("2026"):
                row["return_fraction"] += .2
        second = _ols_hc3_split(changed)
        for key in ("intercept_fraction", "slope_fraction_per_100k_jobs",
                    "slope_hc3_normal_95", "train_n"):
            self.assertEqual(first[key], second[key])
        self.assertNotEqual(first["test_rmse_fraction"], second["test_rmse_fraction"])

    def test_reference_and_regression_guardrails(self):
        for values in ((-1,3), (4,3), (0,0)):
            with self.assertRaises(ValueError):
                wilson_95(*values)
        sample = json.loads(TAPE_FILE.read_text(encoding="utf-8"))
        x = next(x for x in sample["events"] if x.get("surprise") is not None)
        sample["events"].append(deepcopy(x))
        with self.assertRaises(ValueError):
            empirical_miss_reference(sample)
        with self.assertRaises(ValueError):
            _ols_hc3_split([{"date":"2025-01-01", "surprise_jobs": 2.,
                             "return_fraction": float("nan")}])

    def test_verified_nested_threshold_bounds_and_no_fit(self):
        basis = {"metric_id":"BLS_PAYROLL_FIRST_RELEASE", "unit":"jobs",
                 "measurement_rule":"BLS first preliminary print, total nonfarm",
                 "settlement_source":"BLS", "settlement_date":"2027-01-08",
                 "operator":"ge", "threshold":100_000}
        source = {**basis, "contract_id":"KXPAYROLLS-EXAMPLE", "identity_verified":True,
                  "quote_status":"OBSERVED_CONTRACT_SNAPSHOT",
                  "quote_observed_at":"2026-10-31T16:00:00Z",
                  "source_url":"https://kalshi.com/markets/example",
                  "verified_quote_interval":[.33,.37]}
        exact = nested_threshold_probability_bounds(basis,source)
        self.assertEqual(exact["interval"], [.33,.37])
        subset = nested_threshold_probability_bounds({**basis, "threshold":150_000},source)
        self.assertEqual(subset["status"], "LOGICAL_SUBSET_BOUND")
        self.assertEqual(subset["interval"], [0.,.37])
        superset = nested_threshold_probability_bounds({**basis,"threshold":50_000},source)
        self.assertEqual(superset["interval"], [.33,1.])
        for modified in ({"settlement_date":"2027-02-08"}, {"measurement_rule":"revised data"},
                         {"operator":"le"}, {"unit":"thousands"}):
            self.assertIsNone(nested_threshold_probability_bounds({**basis,**modified},source))
        self.assertIsNone(nested_threshold_probability_bounds(basis,{**source,"identity_verified":False}))
        self.assertIsNone(nested_threshold_probability_bounds(basis,{**source,"verified_quote_interval":[.6,.4]}))
        self.assertIsNone(nested_threshold_probability_bounds(basis,{**source,"source_url":"https://example.com"}))
        self.assertIsNone(nested_threshold_probability_bounds(basis,{**source,"quote_observed_at":"undated"}))

    def test_unknown_symbol_does_not_acquire_fitted_return(self):
        context = payroll_context_for_positions([{"symbol":"UNKNOWN123", "type":"etf"}])
        self.assertEqual(context["holding_studies"], {})
        self.assertFalse(context["portfolio_pnl_updated"])
        self.assertFalse(context["event_probability_updated"])


if __name__ == "__main__":
    unittest.main()

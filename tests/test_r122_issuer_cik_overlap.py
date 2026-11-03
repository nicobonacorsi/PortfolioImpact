"""R122 point 1: same-issuer share classes grouped by SEC registrant CIK (synthetic fixtures)."""
from __future__ import annotations

from datetime import datetime, timezone
import math
import unittest

from castle_risk.instrument_exposures import build_instrument_exposures
from castle_risk.issuer_overlap import audit_issuer_overlap
from castle_risk.portfolio import Portfolio, Position, recover_exposures

NOW = datetime(2026, 11, 2, 4, 0, tzinfo=timezone.utc)
ALPHABET = "0001652044"


def snapshot(fund, rows):
    return {"fund_symbol": fund, "as_of": "2026-10-29", "retrieved_at": "2026-11-02T00:23:41+00:00",
            "source_url": "https://api.massive.com/etf-global/v1/constituents", "reliable": True,
            "complete": True, "weight_basis": "fraction_of_nav",
            "holdings": [{"symbol": s, "source_ticker": s, "weight": w, "asset_type": "stock",
                          "identifiers": {"isin": isin}} for s, w, isin in rows]}


def siblings(direct="GOOGL", cik=ALPHABET, tickers=("GOOG", "GOOGL"), complete=True):
    return {direct: {"cik": cik, "complete": complete,
                     "tickers": [{"ticker": t, "type": "CS"} for t in tickers],
                     "identity_scope": "MASSIVE_REFERENCE_SAME_SEC_CIK_NOT_INDEPENDENT_SECURITY_MASTER"}}


QQQ = snapshot("QQQ", [("GOOGL", .0255, "US02079K3059"), ("GOOG", .0245, "US02079K1079"), ("NVDA", .0807650198, "US67066G1040")])


class TestCikShareClassOverlap(unittest.TestCase):
    def setUp(self):
        self.portfolio = Portfolio((Position("GOOGL", .20), Position("QQQ", .40, "etf"),
                                    Position("NVDA", .10), Position("USD", .30, "cash")))
        self.snaps = {"QQQ": QQQ}
        self.ledger = recover_exposures(self.portfolio, self.snaps, now=NOW)

    def test_without_sibling_records_only_exact_tickers_group(self):
        audit = audit_issuer_overlap(self.portfolio, self.ledger, self.snaps)
        ids = {g["issuer_group_id"]: g for g in audit["groups"]}
        self.assertEqual(set(ids), {"EXACT_SYMBOL:GOOGL", "EXACT_SYMBOL:NVDA"})
        self.assertAlmostEqual(ids["EXACT_SYMBOL:GOOGL"]["combined_portfolio_nav_fraction"], .20 + .40 * .0255, places=12)
        self.assertEqual(audit["sec_cik_registrant_groups"], [])

    def test_cik_record_groups_both_share_classes_without_new_nav(self):
        audit = audit_issuer_overlap(self.portfolio, self.ledger, self.snaps, issuer_siblings=siblings())
        ids = {g["issuer_group_id"]: g for g in audit["groups"]}
        self.assertEqual(set(ids), {"CIK:" + ALPHABET, "EXACT_SYMBOL:NVDA"})
        group = ids["CIK:" + ALPHABET]
        self.assertEqual(group["scope"], "SAME_SEC_REGISTRANT_SHARE_CLASSES")
        self.assertEqual(group["issuer_cik"], ALPHABET)
        self.assertEqual(group["distinct_security_tickers"], ["GOOG", "GOOGL"])
        self.assertEqual(group["contributing_original_positions"], ["GOOGL", "QQQ"])
        self.assertAlmostEqual(group["combined_portfolio_nav_fraction"], .20 + .40 * (.0255 + .0245), places=12)
        self.assertAlmostEqual(group["direct_portfolio_nav_fraction"], .20, places=12)
        self.assertAlmostEqual(group["etf_lookthrough_portfolio_nav_fraction"], .40 * .05, places=12)
        self.assertEqual(group["new_nav_created"], 0.)
        self.assertEqual(audit["sec_cik_registrant_groups"], [ALPHABET])
        self.assertEqual(audit["issuer_sibling_records_used"], ["GOOGL"])
        self.assertTrue(math.isclose(self.ledger.total_weight, 1.0, abs_tol=1e-12))
        # The authoritative ledger is unchanged: GOOG stays a separate look-through row.
        self.assertEqual(sum(e.symbol == "GOOG" for e in self.ledger.entries), 1)

    def test_records_that_do_not_name_the_direct_holding_or_are_incomplete_fail_closed(self):
        for bad in (siblings(tickers=("GOOG",)), siblings(complete=False), siblings(cik="1652044"),
                    {"GOOGL": "not-a-record"}, {"QQQ": siblings()["GOOGL"]}):
            audit = audit_issuer_overlap(self.portfolio, self.ledger, self.snaps, issuer_siblings=bad)
            self.assertEqual({g["issuer_group_id"] for g in audit["groups"]},
                             {"EXACT_SYMBOL:GOOGL", "EXACT_SYMBOL:NVDA"}, bad)

    def test_conflicting_registrant_records_disable_all_cik_grouping(self):
        records = {**siblings(), **siblings(direct="NVDA", cik="0001045810", tickers=("NVDA", "GOOG"))}
        audit = audit_issuer_overlap(self.portfolio, self.ledger, self.snaps, issuer_siblings=records)
        self.assertEqual({g["issuer_group_id"] for g in audit["groups"]}, {"EXACT_SYMBOL:GOOGL", "EXACT_SYMBOL:NVDA"})

    def test_single_share_class_registrant_keeps_exact_symbol_group(self):
        records = siblings(direct="NVDA", cik="0001045810", tickers=("NVDA",))
        audit = audit_issuer_overlap(self.portfolio, self.ledger, self.snaps, issuer_siblings=records)
        self.assertIn("EXACT_SYMBOL:NVDA", {g["issuer_group_id"] for g in audit["groups"]})

    def test_tsmc_rule_is_unaffected_by_a_single_ticker_cik_record(self):
        portfolio = Portfolio((Position("TSM", .40), Position("EWT", .60, "etf")))
        snaps = {"EWT": snapshot("EWT", [("2330", .221146, "TW0002330008")])}
        ledger = recover_exposures(portfolio, snaps, now=NOW)
        audit = audit_issuer_overlap(portfolio, ledger, snaps,
                                     issuer_siblings={"TSM": {"cik": "0001046179", "complete": True,
                                                              "tickers": [{"ticker": "TSM", "type": "ADRC"}]}})
        self.assertEqual([g["issuer_group_id"] for g in audit["groups"]], ["TSMC:TW0002330008"])

    def test_end_to_end_exposure_brief_carries_the_cik_group(self):
        integrated = build_instrument_exposures([
            {"symbol": "GOOGL", "weight": .20, "asset_type": "stock"},
            {"symbol": "QQQ", "weight": .40, "asset_type": "etf"},
            {"symbol": "NVDA", "weight": .10, "asset_type": "stock"},
            {"symbol": "USD", "weight": .30, "asset_type": "cash"},
        ], holdings=self.snaps, evaluated_at=NOW.isoformat(), issuer_siblings=siblings())
        ids = [g["issuer_group_id"] for g in integrated["issuer_overlap_audit"]["groups"]]
        self.assertIn("CIK:" + ALPHABET, ids)
        self.assertEqual(integrated["agent_exposure_input"]["source_supported_issuer_overlap"],
                         integrated["issuer_overlap_audit"])
        self.assertAlmostEqual(integrated["exposure_ledger"]["total_weight"], 1.0, places=12)
        with self.assertRaises(ValueError):
            build_instrument_exposures([{"symbol": "GOOGL", "weight": 1., "asset_type": "stock"}],
                                       evaluated_at=NOW.isoformat(), issuer_siblings="bad")


if __name__ == "__main__":
    unittest.main()

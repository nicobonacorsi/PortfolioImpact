"""Regression gates for source-bound ETF issuer exposures (R120).

Fixture weights and exact local ISIN were independently observed from Massive's
authenticated 2026-10-30 processed / 2026-10-29 effective holdings on
2026-11-02; tests do not call the provider or treat fixtures as new observations.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
import unittest

from castle_risk.instrument_exposures import build_instrument_exposures
from castle_risk.issuer_overlap import audit_issuer_overlap
from castle_risk.portfolio import Portfolio, Position, recover_exposures


NOW = datetime(2026, 11, 2, 1, 0, tzinfo=timezone.utc)


def snapshot(fund: str, constituent: str, weight: float, *, isin: str,
             source_ticker: str | None = None, complete: bool = True) -> dict:
    return {
        "fund_symbol": fund,
        "as_of": "2026-10-29",
        "retrieved_at": "2026-11-02T00:23:41+00:00",
        "source_url": "https://api.massive.com/etf-global/v1/constituents",
        "reliable": True,
        "complete": complete,
        "weight_basis": "fraction_of_nav",
        "holdings": [{
            "symbol": constituent,
            "source_ticker": constituent if source_ticker is None else source_ticker,
            "weight": weight,
            "asset_type": "stock",
            "identifiers": {"isin": isin},
        }],
    }


class TestR120IssuerOverlap(unittest.TestCase):
    def test_qqq_nvda_same_nav_instrument_and_issuer_views(self):
        portfolio = Portfolio((
            Position("NVDA", 0.35), Position("QQQ", 0.30, "etf"),
            Position("TLT", 0.20, "etf"), Position("USD", 0.15, "cash"),
        ))
        snaps = {"QQQ": snapshot("QQQ", "NVDA", 0.0807650198, isin="US67066G1040")}
        ledger = recover_exposures(portfolio, snaps, now=NOW)
        audit = audit_issuer_overlap(portfolio, ledger, snaps)
        self.assertEqual(len(audit["groups"]), 1)
        group = audit["groups"][0]
        self.assertEqual(group["issuer_group_id"], "EXACT_SYMBOL:NVDA")
        self.assertEqual(group["contributing_original_positions"], ["NVDA", "QQQ"])
        self.assertAlmostEqual(group["combined_portfolio_nav_fraction"], 0.37422950594, places=12)
        self.assertAlmostEqual(group["direct_portfolio_nav_fraction"], 0.35, places=12)
        self.assertAlmostEqual(group["etf_lookthrough_portfolio_nav_fraction"], 0.02422950594, places=12)
        self.assertTrue(math.isclose(ledger.total_weight, 1.0, abs_tol=1e-12))
        self.assertAlmostEqual(ledger.unresolved_weight, .20 + .30 * (1. - .0807650198), places=12)
        self.assertEqual(audit["new_nav_created"], 0.)
        self.assertFalse(audit["authoritative_ledger_modified"])

        integrated = build_instrument_exposures([
            {"symbol": "NVDA", "weight": .35, "asset_type": "stock"},
            {"symbol": "QQQ", "weight": .30, "asset_type": "etf"},
            {"symbol": "TLT", "weight": .20, "asset_type": "etf"},
            {"symbol": "USD", "weight": .15, "asset_type": "cash"},
        ], holdings=snaps, evaluated_at=NOW.isoformat())
        self.assertEqual(integrated["issuer_overlap_audit"]["groups"][0]["issuer_group_id"], "EXACT_SYMBOL:NVDA")
        self.assertEqual(integrated["agent_exposure_input"]["source_supported_issuer_overlap"],
                         integrated["issuer_overlap_audit"])
        self.assertAlmostEqual(integrated["exposure_ledger"]["total_weight"], 1.0, places=12)

    def test_ewt_tsmc_local_ordinary_and_us_adr_same_issuer_only(self):
        portfolio = Portfolio((Position("TSM", .40), Position("EWT", .60, "etf")))
        snaps = {"EWT": snapshot("EWT", "2330", .221146, isin="TW0002330008")}
        ledger = recover_exposures(portfolio, snaps, now=NOW)
        audit = audit_issuer_overlap(portfolio, ledger, snaps)
        self.assertEqual(len(audit["groups"]), 1)
        group = audit["groups"][0]
        self.assertEqual(group["issuer_group_id"], "TSMC:TW0002330008")
        self.assertEqual(group["scope"], "SAME_ISSUER_DIFFERENT_SECURITIES")
        self.assertEqual(group["contributing_original_positions"], ["EWT", "TSM"])
        self.assertAlmostEqual(group["combined_portfolio_nav_fraction"], .5326876, places=12)
        self.assertAlmostEqual(group["etf_lookthrough_portfolio_nav_fraction"], .1326876, places=12)
        self.assertIn("NOT price-return-equivalent", group["identity_support"])
        self.assertAlmostEqual(ledger.total_weight, 1., places=12)
        self.assertAlmostEqual(ledger.unresolved_weight, .60 * (1. - .221146), places=12)
        self.assertNotIn("TSM", {entry.symbol for entry in ledger.entries if entry.original_symbol == "EWT"})
        self.assertIn("2330", {entry.symbol for entry in ledger.entries if entry.original_symbol == "EWT"})

    def test_cross_listed_alias_fails_closed_on_isin_and_source_mismatch(self):
        p = Portfolio((Position("TSM", .40), Position("EWT", .60, "etf")))
        for wrong in (
            snapshot("EWT", "2330", .221146, isin="US0000000000"),
            snapshot("EWT", "2330", .221146, isin=""),
            snapshot("EWT", "2330", .221146, isin="TW0002330008", source_ticker="TSM"),
            snapshot("EWT", "2330", .221146, isin="TW0002330008", complete=False),
        ):
            with self.subTest(wrong=wrong):
                ledger = recover_exposures(p, {"EWT": wrong}, now=NOW)
                result = audit_issuer_overlap(p, ledger, {"EWT": wrong})
                self.assertEqual(result["groups"], [])
                self.assertAlmostEqual(ledger.total_weight, 1., places=12)
        self.assertEqual(audit_issuer_overlap(p, recover_exposures(p, {}, now=NOW), {})["groups"], [])

    def test_standalone_issuer_lookthrough_does_not_invent_direct_stock(self):
        p = Portfolio((Position("EWT", .60, "etf"), Position("USD", .40, "cash")))
        snap = {"EWT": snapshot("EWT", "2330", .221146, isin="TW0002330008")}
        audit = audit_issuer_overlap(p, recover_exposures(p, snap, now=NOW), snap)
        self.assertEqual(audit["groups"], [])

    def test_two_fund_nvda_overlap_without_direct_holding(self):
        p = Portfolio((Position("QQQ", .6, "etf"), Position("SOXX", .4, "etf")))
        snap = {
            "QQQ": snapshot("QQQ", "NVDA", .0807650198, isin="US67066G1040"),
            "SOXX": snapshot("SOXX", "NVDA", .0764839, isin="US67066G1040"),
        }
        report = recover_exposures(p, snap, now=NOW)
        audit = audit_issuer_overlap(p, report, snap)
        self.assertEqual(len(audit["groups"]), 1)
        g = audit["groups"][0]
        self.assertEqual(g["issuer_group_id"], "EXACT_SYMBOL:NVDA")
        self.assertAlmostEqual(g["combined_portfolio_nav_fraction"],
                               .6 * .0807650198 + .4 * .0764839, places=12)
        self.assertEqual(g["direct_portfolio_nav_fraction"], 0.)

    def test_overlay_does_not_accept_invalid_max_groups(self):
        p = Portfolio((Position("NVDA", 1.),))
        report = recover_exposures(p, now=NOW)
        for invalid in (-1, 0, 65, True, 1.5):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    audit_issuer_overlap(p, report, max_groups=invalid)


    def test_tsmc_issuer_identity_reaches_taiwan_event_discovery(self):
        portfolio_rows = [
            {"symbol": "TSM", "weight": .40, "asset_type": "stock"},
            {"symbol": "EWT", "weight": .60, "asset_type": "etf"},
        ]
        record = snapshot("EWT", "2330", .221146, isin="TW0002330008")
        positive = build_instrument_exposures(
            portfolio_rows, holdings={"EWT": record}, evaluated_at=NOW.isoformat())
        tsmc_channels = [c for c in positive["channels_by_family"]["TAIWAN_SUPPLY"]
                         if c["instrument_symbol"] in ("TSM", "2330")]
        self.assertEqual({c["instrument_symbol"] for c in tsmc_channels}, {"TSM", "2330"})
        self.assertTrue(all(c["estimated_return_sensitivity"] is None for c in tsmc_channels))
        self.assertTrue(all(c["channel_status"] == "PRIMARY_ISSUER_LISTING_LINK_NOT_ESTIMATED_RETURN"
                            for c in tsmc_channels))
        self.assertAlmostEqual(positive["coverage"]["taiwan_source_supported_tsmc_channel_weight"],
                               .5326876, places=12)
        self.assertAlmostEqual(positive["coverage"]["taiwan_identified_nvda_channel_weight"],
                               0., places=12)
        self.assertAlmostEqual(positive["exposure_ledger"]["total_weight"], 1., places=12)
        record["holdings"][0]["identifiers"]["isin"] = "US0000000000"
        negative = build_instrument_exposures(
            portfolio_rows, holdings={"EWT": record}, evaluated_at=NOW.isoformat())
        self.assertEqual({c["instrument_symbol"] for c in negative["channels_by_family"]["TAIWAN_SUPPLY"]},
                         {"TSM"})
        self.assertAlmostEqual(negative["coverage"]["taiwan_source_supported_tsmc_channel_weight"],
                               .40, places=12)

    def test_ewt_tsmc_channel_without_direct_adr_still_relevant(self):
        rows = [
            {"symbol": "EWT", "weight": .60, "asset_type": "etf"},
            {"symbol": "USD", "weight": .40, "asset_type": "cash"},
        ]
        snap = {"EWT": snapshot("EWT", "2330", .221146, isin="TW0002330008")}
        result = build_instrument_exposures(rows, holdings=snap, evaluated_at=NOW.isoformat())
        channels = result["channels_by_family"]["TAIWAN_SUPPLY"]
        self.assertEqual([c["instrument_symbol"] for c in channels], ["2330"])
        self.assertAlmostEqual(channels[0]["portfolio_weight"], .1326876, places=12)
        self.assertEqual(result["issuer_overlap_audit"]["groups"], [])
        self.assertAlmostEqual(result["exposure_ledger"]["total_weight"], 1., places=12)



    def test_local_tsmc_isin_mapping_generalizes_without_ticker_specific_fund_registry(self):
        # Hypothetical ETF symbol: verifies an algorithmic rule, not a new provider observation.
        p = Portfolio((Position("TSM", .4), Position("FUNDX", .6, "etf")))
        snap = {"FUNDX": snapshot("FUNDX", "2330", .25, isin="TW0002330008")}
        report = recover_exposures(p, snap, now=NOW)
        groups = audit_issuer_overlap(p, report, snap)["groups"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["issuer_group_id"], "TSMC:TW0002330008")
        self.assertAlmostEqual(groups[0]["combined_portfolio_nav_fraction"], .55, places=12)
        invalid = {"FUNDX": {**snap["FUNDX"], "fund_symbol": "EWT"}}
        bad_report = recover_exposures(p, invalid, now=NOW)
        self.assertEqual(audit_issuer_overlap(p, bad_report, invalid)["groups"], [])
        self.assertAlmostEqual(bad_report.total_weight, 1., places=12)



if __name__ == "__main__":
    unittest.main()

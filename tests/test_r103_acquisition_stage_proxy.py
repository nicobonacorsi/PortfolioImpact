"""Live acquisition classification for optional scenario proxy histories."""
from castle_risk.acquisition_status import massive_acquisition_stage


def test_successful_proxy_history_does_not_invalidate_required_transport():
    inputs = {
        "attempts": [
            {"symbol": "UNH", "operation": "ticker_details", "status": "OK"},
            {"symbol": "UNH", "operation": "daily_prices", "status": "OK"},
            {"symbol": "SPY", "operation": "daily_prices", "status": "OK"},
            {"symbol": "USO", "operation": "scenario_proxy_prices", "status": "OK"},
        ],
        "acquisition_quality": {
            "all_required_transport_operations_succeeded": True,
            "all_required_holdings_quality_passed": True,
        },
    }
    result = massive_acquisition_stage(inputs, request_count=4)
    assert result["status"] == "ACQUIRED"
    assert result["attempt_diagnostics_valid"] is True
    assert result["required_transport_complete"] is True
    assert result["successful_operations"] == 4


def test_unavailable_optional_proxy_is_reported_but_not_required():
    inputs = {
        "attempts": [
            {"symbol": "JPM", "operation": "ticker_details", "status": "OK"},
            {"symbol": "JPM", "operation": "daily_prices", "status": "OK"},
            {"symbol": "DBC", "operation": "scenario_proxy_prices", "status": "UNAVAILABLE"},
        ],
        "acquisition_quality": {
            "all_required_transport_operations_succeeded": True,
            "all_required_holdings_quality_passed": True,
        },
    }
    result = massive_acquisition_stage(inputs, request_count=3)
    assert result["status"] == "ACQUIRED"
    assert result["optional_metadata_failures"] == [
        {"symbol": "DBC", "operation": "scenario_proxy_prices", "status": "UNAVAILABLE"}
    ]


def test_unknown_attempt_operation_still_blocks_acquired_status():
    inputs = {
        "attempts": [
            {"symbol": "JPM", "operation": "daily_prices", "status": "OK"},
            {"symbol": "JPM", "operation": "unexpected_operation", "status": "OK"},
        ],
        "acquisition_quality": {
            "all_required_transport_operations_succeeded": True,
            "all_required_holdings_quality_passed": True,
        },
    }
    result = massive_acquisition_stage(inputs, request_count=2)
    assert result["status"] == "PARTIAL"
    assert result["attempt_diagnostics_valid"] is False


def test_real_capture_issuer_siblings_is_optional_and_does_not_spoil_verified_attempts():
    """R124: the provider's extra SEC sibling lookup is not a failed quote."""
    inputs = {
        "attempts": [
            {"symbol": "MSFT", "operation": "ticker_details", "status": "OK"},
            {"symbol": "MSFT", "operation": "daily_prices", "status": "OK"},
            {"symbol": "QQQ", "operation": "holdings", "status": "OK",
             "data_quality_status": "RELIABLE_SNAPSHOT"},
            {"symbol": "MSFT", "operation": "issuer_siblings", "status": "OK"},
        ],
        "acquisition_quality": {
            "all_required_transport_operations_succeeded": True,
            "all_required_holdings_quality_passed": True,
        },
    }
    stage = massive_acquisition_stage(inputs, request_count=4)
    assert stage["status"] == "ACQUIRED"
    assert stage["attempt_diagnostics_valid"] is True
    assert stage["required_transport_complete"] is True
    assert stage["successful_operations"] == 4

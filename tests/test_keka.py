"""Unit tests for the mock 'Keka' HR layer.

These exercise the layer with NO LLM and NO network: seeding, the four
operations, ID generation, input validation, and category-to-team routing.
"""
import pytest

from mock.keka import LEAVE_TYPES, LEAVE_YEAR, new_state, route_category


# --- seeding ----------------------------------------------------------------
def test_seed_state_has_three_leave_balances():
    s = new_state()
    assert set(s.balances) == set(LEAVE_TYPES)
    assert s.balances == {"casual": 8, "sick": 10, "earned": 15}
    assert all(isinstance(v, int) and v >= 0 for v in s.balances.values())


def test_seed_counters_continue_past_seeded_records():
    s = new_state()
    # Two seeded leave records and one seeded ticket -> next ids are 0003 / 0002.
    assert s.apply_leave("casual", 1, "2030-06-25")["reference"] == "LEAVE-0003"
    assert s.raise_ticket("it", "printer down")["reference"] == "TICKET-0002"


# --- check_leave (read) -----------------------------------------------------
def test_check_leave_returns_balance_copy_not_live_object():
    s = new_state()
    out = s.check_leave()
    assert out["ok"] is True
    assert out["leave_year"] == LEAVE_YEAR
    assert out["balances"] == s.balances
    # Mutating the returned copy must not affect live state.
    out["balances"]["casual"] = 999
    assert s.balances["casual"] == 8


# --- apply_leave (write): happy path ----------------------------------------
def test_apply_leave_valid_deducts_and_records():
    s = new_state()
    before = s.balances["casual"]
    out = s.apply_leave("casual", 2, "2030-06-25")
    assert out["ok"] is True
    assert out["reference"].startswith("LEAVE-")
    assert out["leave_type"] == "casual"
    assert out["days"] == 2
    assert out["balance_after"] == before - 2
    assert s.balances["casual"] == before - 2
    assert any(r["reference"] == out["reference"] for r in s.leave_records)


def test_apply_leave_accepts_numeric_string_days():
    s = new_state()
    out = s.apply_leave("sick", "3", "2030-06-25")
    assert out["ok"] is True
    assert out["days"] == 3


def test_apply_leave_synonym_privilege_maps_to_earned():
    s = new_state()
    before = s.balances["earned"]
    out = s.apply_leave("privilege leave", 2, "2030-06-25")
    assert out["ok"] is True
    assert out["leave_type"] == "earned"
    assert s.balances["earned"] == before - 2


def test_reference_ids_are_sequential_and_zero_padded():
    s = new_state()
    r1 = s.apply_leave("casual", 1, "2030-01-01")["reference"]
    r2 = s.apply_leave("sick", 1, "2030-01-02")["reference"]
    suffix1, suffix2 = r1.split("-")[1], r2.split("-")[1]
    assert len(suffix1) == 4 and len(suffix2) == 4
    assert int(suffix2) == int(suffix1) + 1


# --- apply_leave (write): validation, never mutates on failure --------------
def test_apply_leave_insufficient_balance_does_not_mutate():
    s = new_state()
    n_records = len(s.leave_records)
    out = s.apply_leave("casual", 999, "2030-06-25")
    assert out["ok"] is False
    assert out["code"] == "insufficient_balance"
    assert "8 casual days left" in out["error"]
    assert s.balances["casual"] == 8           # unchanged
    assert len(s.leave_records) == n_records   # no record added


def test_apply_leave_unknown_type():
    s = new_state()
    out = s.apply_leave("vacation", 1, "2030-06-25")
    assert out["ok"] is False
    assert out["code"] == "unknown_leave_type"


@pytest.mark.parametrize("bad_days", [0, -1, 1.5, "two", True, None])
def test_apply_leave_invalid_days(bad_days):
    s = new_state()
    out = s.apply_leave("casual", bad_days, "2030-06-25")
    assert out["ok"] is False
    assert out["code"] == "invalid_days"
    assert s.balances["casual"] == 8


def test_apply_leave_invalid_date_format():
    s = new_state()
    out = s.apply_leave("casual", 1, "25-06-2030")
    assert out["ok"] is False
    assert out["code"] == "invalid_date"


def test_apply_leave_past_date_rejected():
    s = new_state()
    out = s.apply_leave("casual", 1, "2020-01-01")
    assert out["ok"] is False
    assert out["code"] == "past_date"
    assert s.balances["casual"] == 8


# --- raise_ticket (write) + routing -----------------------------------------
def test_raise_ticket_returns_reference_and_records():
    s = new_state()
    n_tickets = len(s.tickets)
    out = s.raise_ticket("policy", "Question about notice period")
    assert out["ok"] is True
    assert out["reference"].startswith("TICKET-")
    assert out["team"] == "HR Generalist"
    assert out["confidential"] is False
    assert len(s.tickets) == n_tickets + 1
    assert any(t["reference"] == out["reference"] for t in s.tickets)


def test_raise_ticket_harassment_routes_confidential_people_ops():
    s = new_state()
    out = s.raise_ticket("harassment", "Concern about my manager's behaviour")
    assert out["team"] == "People Ops (Confidential)"
    assert out["confidential"] is True


def test_raise_ticket_payroll_routes_finance():
    s = new_state()
    assert s.raise_ticket("payroll", "Salary not credited")["team"] == "Finance"


def test_raise_ticket_it_routes_it():
    s = new_state()
    assert s.raise_ticket("it", "Laptop will not boot")["team"] == "IT"


def test_sensitive_summary_overrides_benign_category():
    # Even with a benign category, sensitive content must route confidentially.
    s = new_state()
    out = s.raise_ticket("general", "I am being harassed by a colleague")
    assert out["team"] == "People Ops (Confidential)"
    assert out["confidential"] is True


@pytest.mark.parametrize("category,team,confidential", [
    ("harassment", "People Ops (Confidential)", True),
    ("discrimination", "People Ops (Confidential)", True),
    ("payroll", "Finance", False),
    ("it", "IT", False),
    ("policy", "HR Generalist", False),
    ("something-unrecognised", "HR Generalist", False),
])
def test_route_category(category, team, confidential):
    assert route_category(category) == (team, confidential)


# --- get_payslip (read) -----------------------------------------------------
def test_get_payslip_latest_when_no_month():
    s = new_state()
    out = s.get_payslip()
    assert out["ok"] is True
    assert out["month"] == s.payslips[-1]["month"]
    assert out["net"] == out["gross"] - out["deductions"]


def test_get_payslip_specific_month():
    s = new_state()
    month = s.payslips[0]["month"]
    out = s.get_payslip(month)
    assert out["ok"] is True
    assert out["month"] == month


def test_get_payslip_unknown_month_returns_not_found():
    s = new_state()
    out = s.get_payslip("2019-01")
    assert out["ok"] is False
    assert out["code"] == "not_found"
    assert "Available months" in out["error"]


# --- counters independent, snapshot isolation -------------------------------
def test_leave_and_ticket_counters_are_independent():
    s = new_state()
    lref = s.apply_leave("casual", 1, "2030-06-25")["reference"]
    tref = s.raise_ticket("it", "printer issue")["reference"]
    assert lref.startswith("LEAVE-")
    assert tref.startswith("TICKET-")


def test_snapshot_is_deep_copy():
    s = new_state()
    snap = s.snapshot()
    snap["balances"]["casual"] = 0
    snap["leave_records"].append({"reference": "LEAVE-9999"})
    assert s.balances["casual"] == 8
    assert all(r["reference"] != "LEAVE-9999" for r in s.leave_records)

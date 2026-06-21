"""In-memory mock of the 'Keka' HR backend.

Everything here is simulated — there is no network, no database, and no real HR
system. The layer owns all state and, crucially, **mints every reference id**
(``LEAVE-000n`` / ``TICKET-000n``). The agent only ever relays ids this layer
returns; it can never fabricate one.

Write operations (``apply_leave``, ``raise_ticket``) validate their input first
and return a structured ``{"ok": False, "code": ..., "error": ...}`` result
rather than mutating state on invalid input. Read operations (``check_leave``,
``get_payslip``) never mutate.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

# Canonical leave types. These MUST match the leave types named in the policy
# docs and surfaced in the UI side panel.
LEAVE_TYPES: tuple[str, ...] = ("casual", "sick", "earned")

# The leave year these mock balances belong to.
LEAVE_YEAR: int = 2026

# Accepted spellings/synonyms mapped to a canonical leave type, so the agent can
# pass a reasonable variant without the write being rejected on a technicality.
_LEAVE_SYNONYMS: dict[str, str] = {
    "casual": "casual", "casual leave": "casual", "cl": "casual",
    "sick": "sick", "sick leave": "sick", "medical": "sick",
    "medical leave": "sick", "sl": "sick",
    "earned": "earned", "earned leave": "earned", "privilege": "earned",
    "privilege leave": "earned", "annual": "earned", "annual leave": "earned",
    "pl": "earned", "el": "earned",
}

# Routing destinations. (team, confidential).
_PEOPLE_OPS = ("People Ops (Confidential)", True)
_FINANCE = ("Finance", False)
_IT = ("IT", False)
_HR_GENERALIST = ("HR Generalist", False)

# Exact category -> destination. The agent is prompted to pass one of these
# tokens; anything else falls through to the keyword scan below.
_CATEGORY_TEAM: dict[str, tuple[str, bool]] = {
    "harassment": _PEOPLE_OPS,
    "discrimination": _PEOPLE_OPS,
    "misconduct": _PEOPLE_OPS,
    "posh": _PEOPLE_OPS,
    "retaliation": _PEOPLE_OPS,
    "payroll": _FINANCE,
    "pay": _FINANCE,
    "salary": _FINANCE,
    "compensation": _FINANCE,
    "reimbursement": _FINANCE,
    "expense": _FINANCE,
    "tax": _FINANCE,
    "it": _IT,
    "access": _IT,
    "hardware": _IT,
    "software": _IT,
    "policy": _HR_GENERALIST,
    "leave": _HR_GENERALIST,
    "benefits": _HR_GENERALIST,
    "general": _HR_GENERALIST,
    "hr": _HR_GENERALIST,
    "other": _HR_GENERALIST,
}

# Substrings that force confidential People Ops routing, regardless of the
# category supplied. Sensitive matters must never be downgraded to another team.
_SENSITIVE_KEYWORDS: tuple[str, ...] = (
    "harass", "discriminat", "posh", "retaliat", "bully", "sexual",
)
# Substrings for the defensive fallback scan (only used when the category is
# unrecognised). Kept specific to avoid false positives.
_FINANCE_KEYWORDS: tuple[str, ...] = (
    "payroll", "salary", "payslip", "reimburs", "compensation", "provident",
)
_IT_KEYWORDS: tuple[str, ...] = (
    "laptop", "vpn", "login", "wifi", "wi-fi", "hardware", "software", "password",
)


def route_category(category: str, summary: str = "") -> tuple[str, bool]:
    """Resolve a ticket category to a ``(team, confidential)`` destination.

    Resolution happens entirely in code so the model can never choose where a
    sensitive ticket is routed. Precedence:

    1. Sensitive keywords anywhere in ``category``/``summary`` force confidential
       People Ops routing (a harassment ticket can never be downgraded).
    2. An exact recognised category maps directly.
    3. A defensive keyword scan (Finance, then IT) for unrecognised categories.
    4. Default: HR generalist.
    """
    text = f"{category} {summary}".lower()
    if any(kw in text for kw in _SENSITIVE_KEYWORDS):
        return _PEOPLE_OPS

    key = (category or "").strip().lower()
    if key in _CATEGORY_TEAM:
        return _CATEGORY_TEAM[key]

    if any(kw in text for kw in _FINANCE_KEYWORDS):
        return _FINANCE
    if any(kw in text for kw in _IT_KEYWORDS):
        return _IT
    return _HR_GENERALIST


def _coerce_days(value: Any) -> Optional[int]:
    """Coerce a 'days' argument to a positive-capable int, or ``None`` if it is
    not a whole number. ``bool`` is rejected (``True``/``False`` are not days)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        text = value.strip()
        if text.lstrip("+").isdigit():
            return int(text)
    return None


@dataclass
class KekaState:
    """All mock HR state for one employee session, plus the four operations.

    Held in Streamlit ``session_state`` in the app, or instantiated directly in
    tests and the CLI harness — it has no dependency on Streamlit or the network.
    """

    employee: dict[str, Any]
    balances: dict[str, int]
    leave_records: list[dict[str, Any]]
    tickets: list[dict[str, Any]]
    payslips: list[dict[str, Any]]
    _counters: dict[str, int] = field(default_factory=lambda: {"LEAVE": 0, "TICKET": 0})

    # -- internal helpers ----------------------------------------------------
    def _next_reference(self, prefix: str) -> str:
        """Mint the next zero-padded reference id for ``prefix`` (LEAVE/TICKET)."""
        self._counters[prefix] = self._counters.get(prefix, 0) + 1
        return f"{prefix}-{self._counters[prefix]:04d}"

    # -- read operations -----------------------------------------------------
    def check_leave(self) -> dict[str, Any]:
        """Return current leave balances (read-only)."""
        return {"ok": True, "leave_year": LEAVE_YEAR, "balances": dict(self.balances)}

    def get_payslip(self, month: Optional[str] = None) -> dict[str, Any]:
        """Return the payslip for ``month`` ('YYYY-MM'), or the latest if omitted.

        Read-only. Returns a structured ``not_found`` error listing the available
        months if the requested month has no payslip.
        """
        if not self.payslips:
            return {"ok": False, "code": "not_found", "error": "No payslips are available."}

        if month is None or str(month).strip().lower() in ("", "latest", "current"):
            slip = self.payslips[-1]
            return {"ok": True, **slip}

        wanted = str(month).strip()
        for slip in self.payslips:
            if slip["month"] == wanted:
                return {"ok": True, **slip}

        available = ", ".join(s["month"] for s in self.payslips)
        return {
            "ok": False,
            "code": "not_found",
            "error": f"No payslip found for {wanted}. Available months: {available}.",
        }

    # -- write operations ----------------------------------------------------
    def apply_leave(self, leave_type: Any, days: Any, start_date: Any) -> dict[str, Any]:
        """Validate then apply a leave request.

        Validates leave type, day count, date validity, that the date is not in
        the past, and that the balance is sufficient — in that order. On any
        failure it returns a structured error and does **not** mutate state. On
        success it deducts the balance, records the leave, and returns the
        mock-generated reference id and the new balance.
        """
        # 1. Leave type
        if not isinstance(leave_type, str):
            return {"ok": False, "code": "unknown_leave_type",
                    "error": "Leave type must be one of: casual, sick, earned."}
        canonical = _LEAVE_SYNONYMS.get(leave_type.strip().lower())
        if canonical is None:
            return {"ok": False, "code": "unknown_leave_type",
                    "error": f"Unknown leave type '{leave_type}'. "
                             "Valid types: casual, sick, earned."}

        # 2. Day count
        n_days = _coerce_days(days)
        if n_days is None or n_days < 1:
            return {"ok": False, "code": "invalid_days",
                    "error": "Number of days must be a whole number of at least 1."}

        # 3. Date validity
        try:
            requested = date.fromisoformat(str(start_date))
        except (TypeError, ValueError):
            return {"ok": False, "code": "invalid_date",
                    "error": "Start date must be a valid date in YYYY-MM-DD format."}
        if requested < date.today():
            return {"ok": False, "code": "past_date",
                    "error": f"Start date {requested.isoformat()} is in the past."}

        # 4. Balance
        available = self.balances.get(canonical, 0)
        if n_days > available:
            unit = "day" if available == 1 else "days"
            return {"ok": False, "code": "insufficient_balance",
                    "error": f"Only {available} {canonical} {unit} left; "
                             f"you requested {n_days}."}

        # All checks passed — mutate and record.
        self.balances[canonical] = available - n_days
        reference = self._next_reference("LEAVE")
        record = {
            "reference": reference,
            "leave_type": canonical,
            "days": n_days,
            "start_date": requested.isoformat(),
            "status": "approved",
            "applied_on": date.today().isoformat(),
        }
        self.leave_records.append(record)
        return {
            "ok": True,
            "reference": reference,
            "leave_type": canonical,
            "days": n_days,
            "start_date": requested.isoformat(),
            "balance_after": self.balances[canonical],
        }

    def raise_ticket(self, category: Any, summary: Any) -> dict[str, Any]:
        """Create a routed support ticket and return its mock-generated id.

        The category is resolved to a team **in code** via :func:`route_category`;
        the model cannot influence the routing of a sensitive ticket.
        """
        category_str = (category if isinstance(category, str) else "general").strip() or "general"
        summary_str = (summary if isinstance(summary, str) else "").strip() or "(no summary provided)"
        team, confidential = route_category(category_str, summary_str)

        reference = self._next_reference("TICKET")
        ticket = {
            "reference": reference,
            "category": category_str.lower(),
            "summary": summary_str,
            "team": team,
            "confidential": confidential,
            "status": "open",
            "raised_on": date.today().isoformat(),
        }
        self.tickets.append(ticket)
        return {
            "ok": True,
            "reference": reference,
            "category": category_str.lower(),
            "team": team,
            "confidential": confidential,
            "status": "open",
        }

    # -- convenience for the UI ----------------------------------------------
    def recent_leave_records(self, limit: int = 5) -> list[dict[str, Any]]:
        """Most recent leave records first (for the side panel)."""
        return list(reversed(self.leave_records))[:limit]

    def snapshot(self) -> dict[str, Any]:
        """Deep copy of the visible state, for read-only rendering in the UI."""
        return {
            "employee": copy.deepcopy(self.employee),
            "balances": dict(self.balances),
            "leave_records": copy.deepcopy(self.leave_records),
            "tickets": copy.deepcopy(self.tickets),
            "payslips": copy.deepcopy(self.payslips),
        }


def new_state() -> KekaState:
    """Build a freshly seeded :class:`KekaState`.

    Seed data is deterministic so the demo and tests are reproducible. Balances
    are 'remaining' figures; the policy docs hold the annual entitlements.
    """
    employee = {
        "employee_id": "EMP-1042",
        "name": "Jawahar Jeeva",
        "designation": "Software Engineer II",
        "department": "Engineering",
        "manager": "Regina Joseph",
        "location": "Bengaluru",
        "date_of_joining": "2023-07-15",
    }
    balances = {"casual": 8, "sick": 10, "earned": 15}
    leave_records = [
        {"reference": "LEAVE-0001", "leave_type": "earned", "days": 3,
         "start_date": "2026-03-10", "status": "approved", "applied_on": "2026-02-20"},
        {"reference": "LEAVE-0002", "leave_type": "sick", "days": 2,
         "start_date": "2026-05-02", "status": "approved", "applied_on": "2026-05-02"},
    ]
    tickets = [
        {"reference": "TICKET-0001", "category": "it", "summary": "VPN access reset",
         "team": "IT", "confidential": False, "status": "resolved",
         "raised_on": "2026-04-18"},
    ]
    payslips = [
        _payslip("2026-03", gross=118000, basic=59000, hra=29500, allowances=29500,
                 pf=7080, tax=17000, other=2920),
        _payslip("2026-04", gross=120000, basic=60000, hra=30000, allowances=30000,
                 pf=7200, tax=18000, other=2800),
        _payslip("2026-05", gross=122000, basic=61000, hra=30500, allowances=30500,
                 pf=7320, tax=18680, other=2500),
    ]
    # Continue id counters past the seeded records so new ids never collide.
    counters = {"LEAVE": len(leave_records), "TICKET": len(tickets)}
    return KekaState(
        employee=employee,
        balances=balances,
        leave_records=leave_records,
        tickets=tickets,
        payslips=payslips,
        _counters=counters,
    )


def _payslip(month: str, *, gross: int, basic: int, hra: int, allowances: int,
             pf: int, tax: int, other: int) -> dict[str, Any]:
    """Build a single mock payslip dict with a consistent net = gross - deductions."""
    deductions = pf + tax + other
    return {
        "month": month,
        "currency": "INR",
        "gross": gross,
        "deductions": deductions,
        "net": gross - deductions,
        "components": {
            "basic": basic, "hra": hra, "allowances": allowances,
            "pf": pf, "tax": tax, "other_deductions": other,
        },
    }

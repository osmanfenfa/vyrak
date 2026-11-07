"""
Synthetic identity behaviour and red-team attack scenarios.

These are NOT real data. They exist so detection can be measured honestly
before a design partner provides real logs: every number produced from them
must be reported as "on synthetic scenarios".

Normal identity archetypes
--------------------------
analyst   human; reads/searches, some writes, occasional small exports
admin     human; like analyst plus occasional config/admin work
agent     AI agent (e.g. ticket triage); fast, fairly regular read/write loop
service   integration account; periodic export of a stable record count

Attack scenarios (run on an identity after it has a clean history)
------------------------------------------------------------------
token_theft_bulk_export   stolen session bulk-exports at speed
low_and_slow_exfil        human-paced exports, each a little large
rogue_agent               prompt-injected agent starts deleting/exporting/admin
undeclared_bot            script drives a human account (scraping cadence)
privilege_escalation      analyst account starts doing admin/config work
new_identity_smash_grab   brand-new identity goes straight for sensitive data
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List

from core.action import UserAction


@dataclass
class SimEvent:
    action_type: str
    resource_id: str
    timestamp: float
    records: int = 1

    def to_action(self, session_id: str) -> UserAction:
        return UserAction(
            self.action_type, self.resource_id, session_id=session_id,
            timestamp=self.timestamp, metadata={"records": self.records},
        )


def _human_gap(rng: random.Random, mean: float = 6.0) -> float:
    # Log-normal think time with occasional longer pauses
    gap = rng.lognormvariate(math.log(mean), 0.55)
    if rng.random() < 0.05:
        gap += rng.uniform(20, 90)
    return max(0.6, gap)


def _choice(rng: random.Random, weights: Dict[str, float]) -> str:
    r = rng.random() * sum(weights.values())
    for k, w in weights.items():
        r -= w
        if r <= 0:
            return k
    return next(iter(weights))


# ─────────────────────────────────────────────────────────────────────────────
# Normal sessions
# ─────────────────────────────────────────────────────────────────────────────

ARCHETYPE_KIND = {"analyst": "human", "admin": "human", "agent": "agent", "service": "service"}


def normal_session(archetype: str, rng: random.Random, start: float, persona: Dict) -> List[SimEvent]:
    """One clean session for an identity of the given archetype."""
    ev: List[SimEvent] = []
    t = start
    if archetype in ("analyst", "admin"):
        n = rng.randint(25, 90)
        mix = dict(persona["mix"])
        for i in range(n):
            at = "login" if i == 0 else _choice(rng, mix)
            rec = 1
            if at in ("read", "search"):
                rec = rng.randint(1, persona["read_size"])
            elif at == "export":
                rec = rng.randint(5, persona["export_size"])
            ev.append(SimEvent(at, f"{persona['area']}_{rng.randint(1, 60)}", t, rec))
            t += _human_gap(rng, persona["pace"])
    elif archetype == "agent":
        n = rng.randint(40, 120)
        loop = persona["loop"]
        for i in range(n):
            at = loop[i % len(loop)] if rng.random() > 0.1 else "search"
            ev.append(SimEvent(at, f"ticket_{rng.randint(1, 500)}", t, rng.randint(1, 8)))
            t += max(0.2, rng.gauss(persona["pace"], persona["pace"] * 0.2))
    elif archetype == "service":
        for i in range(rng.randint(8, 14)):
            at = "read" if i % 3 else "export"
            rec = int(rng.gauss(persona["batch"], persona["batch"] * 0.1)) if at == "export" else rng.randint(10, 40)
            ev.append(SimEvent(at, f"sync_{i}", t, max(1, rec)))
            t += max(0.5, rng.gauss(30.0, 4.0))
    return ev


def make_persona(archetype: str, rng: random.Random) -> Dict:
    if archetype == "analyst":
        return {
            "mix": {"read": 0.5, "search": 0.25, "write": 0.18, "export": 0.04, "delete": 0.03},
            "read_size": rng.randint(5, 40), "export_size": rng.randint(20, 120),
            "pace": rng.uniform(4.0, 9.0), "area": rng.choice(["crm", "tickets", "finance"]),
        }
    if archetype == "admin":
        return {
            "mix": {"read": 0.4, "search": 0.2, "write": 0.15, "config": 0.1, "admin": 0.1, "export": 0.03, "delete": 0.02},
            "read_size": rng.randint(5, 30), "export_size": rng.randint(20, 80),
            "pace": rng.uniform(5.0, 10.0), "area": "settings",
        }
    if archetype == "agent":
        return {"loop": rng.choice([["read", "read", "write"], ["read", "search", "write"]]), "pace": rng.uniform(0.8, 2.0)}
    return {"batch": rng.randint(150, 400)}


# ─────────────────────────────────────────────────────────────────────────────
# Attacks — each returns the events of the hostile part of a session
# ─────────────────────────────────────────────────────────────────────────────

def token_theft_bulk_export(rng, start, persona) -> List[SimEvent]:
    ev, t = [], start
    for i in range(rng.randint(3, 8)):          # quick look around
        ev.append(SimEvent("search", f"x_{i}", t, 5)); t += rng.uniform(0.8, 2.0)
    for i in range(40):
        ev.append(SimEvent("export", f"customers_{i % 4}", t, rng.randint(500, 2000))); t += rng.uniform(0.3, 1.5)
    return ev


def low_and_slow_exfil(rng, start, persona) -> List[SimEvent]:
    ev, t = [], start
    size = persona.get("export_size", 100)
    for i in range(60):
        at = "export" if i % 3 == 0 else "read"
        rec = rng.randint(size * 2, size * 4) if at == "export" else rng.randint(1, 20)
        ev.append(SimEvent(at, f"acct_{rng.randint(1, 200)}", t, rec))
        t += _human_gap(rng, persona.get("pace", 6.0))
    return ev


def rogue_agent(rng, start, persona) -> List[SimEvent]:
    ev, t = [], start
    for i in range(30):
        at = rng.choice(["delete", "export", "admin", "config", "delete", "read"])
        ev.append(SimEvent(at, f"ticket_{rng.randint(1, 500)}", t, rng.randint(1, 50)))
        t += max(0.2, rng.gauss(persona.get("pace", 1.2), 0.2))
    return ev


def undeclared_bot(rng, start, persona) -> List[SimEvent]:
    ev, t = [], start
    for i in range(80):
        ev.append(SimEvent("read", f"profile_{i}", t, 25)); t += 1.5 + rng.uniform(-0.03, 0.03)
    return ev


def privilege_escalation(rng, start, persona) -> List[SimEvent]:
    ev, t = [], start
    for i in range(25):
        at = rng.choice(["admin", "config", "read", "admin", "export"])
        ev.append(SimEvent(at, f"settings_{i}", t, rng.randint(1, 30)))
        t += _human_gap(rng, persona.get("pace", 6.0))
    return ev


def new_identity_smash_grab(rng, start, persona) -> List[SimEvent]:
    ev, t = [], start
    for i in range(30):
        at = rng.choice(["export", "export", "delete", "admin", "config"])
        ev.append(SimEvent(at, f"core_{i % 5}", t, rng.randint(50, 400)))
        t += rng.uniform(0.1, 0.6)
    return ev


ATTACKS: Dict[str, Dict] = {
    "token_theft_bulk_export": {"fn": token_theft_bulk_export, "victims": ("analyst", "admin"), "warm": True},
    "low_and_slow_exfil":      {"fn": low_and_slow_exfil, "victims": ("analyst",), "warm": True},
    "rogue_agent":             {"fn": rogue_agent, "victims": ("agent",), "warm": True},
    "undeclared_bot":          {"fn": undeclared_bot, "victims": ("analyst",), "warm": True},
    "privilege_escalation":    {"fn": privilege_escalation, "victims": ("analyst",), "warm": True},
    "new_identity_smash_grab": {"fn": new_identity_smash_grab, "victims": ("analyst", "agent"), "warm": False},
}

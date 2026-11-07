"""
scripts/evaluate_detection.py
==============================
Measure detection quality on synthetic identities and red-team scenarios.

No real data required. Results must be presented as *synthetic* results —
they show the engine behaves as designed, not how it performs on a real
customer's traffic. Replace with shadow-mode results from a design partner
as soon as you have them.

Reports, per scenario:
  detection rate          share of attacked sessions that reached CHALLENGE/BLOCK
  median actions to catch hostile actions before the first CHALLENGE/BLOCK
  median records leaked   records exported before detection (exfil scenarios)
and for normal traffic:
  false-challenge rate    share of clean sessions with any CHALLENGE/BLOCK
                          (established identities and new identities separately)
  anomaly rate            anomalous actions per 1,000 clean actions

Usage:
  python scripts/evaluate_detection.py
  python scripts/evaluate_detection.py --victims 30 --warm-sessions 15 --seed 7
  python scripts/evaluate_detection.py --with-ml       # include the RF classifier
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.engine import ABSAv2Engine                       # noqa: E402
from core.profile import BehavioralProfile                 # noqa: E402
from detectors.peer import PeerGroupBaseline               # noqa: E402
from simulation.scenarios import (                         # noqa: E402
    ARCHETYPE_KIND, ATTACKS, make_persona, normal_session,
)

DETECTED = ("CHALLENGE", "BLOCK")


class Population:
    def __init__(self, rng: random.Random, classifier=None, warm_sessions: int = 12) -> None:
        self.rng = rng
        self.classifier = classifier
        self.warm_sessions = warm_sessions
        self.peers: Dict[str, PeerGroupBaseline] = {a: PeerGroupBaseline() for a in ARCHETYPE_KIND}
        self.normal = {"established": [], "new": []}   # per-session dicts
        self.counter = 0

    def _engine(self, profile, archetype, sid):
        return ABSAv2Engine(
            profile=profile,
            classifier=self.classifier if ARCHETYPE_KIND[archetype] == "human" else None,
            session_id=sid,
            identity_kind=ARCHETYPE_KIND[archetype],
            peer_baseline=self.peers[archetype],
            peer_group=archetype,
        )

    def run_session(self, profile, archetype, events, record=True, phase="established"):
        self.counter += 1
        eng = self._engine(profile, archetype, f"s{self.counter}")
        flagged_actions, challenged = 0, False
        for e in events:
            r = eng.evaluate_action(e.to_action(eng.session_id))
            flagged_actions += bool(r.anomalies)
            challenged = challenged or r.decision in DETECTED
        if record:
            self.normal[phase].append({
                "archetype": archetype, "actions": len(events),
                "anomalous_actions": flagged_actions, "challenged": challenged,
            })
        return eng

    def warm_identity(self, archetype: str, t0: float):
        persona = make_persona(archetype, self.rng)
        profile = BehavioralProfile(user_id=f"{archetype}_{self.counter}", identity_kind=ARCHETYPE_KIND[archetype])
        t = t0
        for s in range(self.warm_sessions):
            events = normal_session(archetype, self.rng, t, persona)
            self.run_session(profile, archetype, events, phase="new" if s < 3 else "established")
            t = events[-1].timestamp + self.rng.uniform(4 * 3600, 30 * 3600)
        return profile, persona, t


def evaluate_attack(pop: Population, name: str, spec: Dict, victims: int) -> Dict:
    rows = []
    for i in range(victims):
        archetype = spec["victims"][i % len(spec["victims"])]
        t0 = 1_700_000_000 + pop.rng.uniform(0, 1e6)
        if spec["warm"]:
            profile, persona, t = pop.warm_identity(archetype, t0)
        else:
            persona = make_persona(archetype, pop.rng)
            profile = BehavioralProfile(user_id=f"new_{i}", identity_kind=ARCHETYPE_KIND[archetype])
            t = t0

        # Hijack mid-session: a few normal actions, then the attack
        lead = normal_session(archetype, pop.rng, t, persona)[: pop.rng.randint(3, 8)] if spec["warm"] else []
        t_attack = (lead[-1].timestamp + 2.0) if lead else t
        attack = spec["fn"](pop.rng, t_attack, persona)

        pop.counter += 1
        eng = pop._engine(profile, archetype, f"atk{pop.counter}")
        for e in lead:
            eng.evaluate_action(e.to_action(eng.session_id))
        detected_at: Optional[int] = None
        leaked = 0
        reasons: List[str] = []
        for k, e in enumerate(attack):
            r = eng.evaluate_action(e.to_action(eng.session_id))
            if r.decision in DETECTED:
                detected_at = k
                reasons = r.reasons or [a for a in r.anomalies]
                break
            if e.action_type == "export":
                leaked += e.records
        rows.append({"detected": detected_at is not None, "actions_to_detect": detected_at,
                     "records_leaked": leaked, "reasons": reasons[:2]})

    caught = [r for r in rows if r["detected"]]
    return {
        "scenario": name,
        "victims": victims,
        "detection_rate": len(caught) / len(rows),
        "median_actions_to_detect": statistics.median([r["actions_to_detect"] for r in caught]) if caught else None,
        "median_records_leaked": statistics.median([r["records_leaked"] for r in rows]),
        "example_reason": next((r["reasons"][0] for r in caught if r["reasons"]), None),
    }


def summarize_normal(sessions: List[Dict]) -> Dict:
    if not sessions:
        return {"sessions": 0}
    actions = sum(s["actions"] for s in sessions)
    by_arch = {}
    for a in ARCHETYPE_KIND:
        ss = [s for s in sessions if s["archetype"] == a]
        if ss:
            by_arch[a] = round(sum(s["challenged"] for s in ss) / len(ss), 4)
    return {
        "sessions": len(sessions),
        "actions": actions,
        "false_challenge_rate": round(sum(s["challenged"] for s in sessions) / len(sessions), 4),
        "anomalies_per_1000_actions": round(1000 * sum(s["anomalous_actions"] for s in sessions) / max(actions, 1), 2),
        "false_challenge_rate_by_archetype": by_arch,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--victims", type=int, default=20, help="attacked identities per scenario")
    ap.add_argument("--population", type=int, default=24, help="extra clean identities (peer baselines + false-positive stats)")
    ap.add_argument("--warm-sessions", type=int, default=12)
    ap.add_argument("--with-ml", action="store_true", help="use the active registry model for human identities")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    classifier = None
    if args.with_ml:
        from ml.registry import ModelRegistry
        classifier = ModelRegistry(registry_dir=os.environ.get("MODEL_PATH", "models")).get_active_model()

    rng = random.Random(args.seed)
    pop = Population(rng, classifier, args.warm_sessions)
    t_start = time.time()

    archetypes = list(ARCHETYPE_KIND)
    for i in range(args.population):
        pop.warm_identity(archetypes[i % len(archetypes)], 1_690_000_000 + i * 3600)

    attacks = [evaluate_attack(pop, name, spec, args.victims) for name, spec in ATTACKS.items()]
    report = {
        "synthetic": True,
        "note": "Synthetic identities and attacks. Not evidence of real-world performance.",
        "seed": args.seed,
        "with_ml": bool(classifier),
        "attacks": attacks,
        "normal_established": summarize_normal(pop.normal["established"]),
        "normal_new_identities": summarize_normal(pop.normal["new"]),
        "runtime_s": round(time.time() - t_start, 1),
    }

    Path(args.out).mkdir(parents=True, exist_ok=True)
    json_path = Path(args.out) / "detection_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Detection report (SYNTHETIC)",
        "",
        "> Synthetic identities and red-team scenarios. These numbers show the engine",
        "> behaves as designed; they are **not** real-world performance.",
        "",
        f"Seed {args.seed} · {args.victims} victims per scenario · {args.warm_sessions} warm-up sessions · ML {'on' if classifier else 'off'}",
        "",
        "| Scenario | Detected | Median actions to catch | Median records leaked | Example reason |",
        "|---|---|---|---|---|",
    ]
    for a in attacks:
        lines.append(
            f"| {a['scenario']} | {a['detection_rate']:.0%} | "
            f"{a['median_actions_to_detect'] if a['median_actions_to_detect'] is not None else '–'} | "
            f"{a['median_records_leaked']:,.0f} | {(a['example_reason'] or '').replace('|', '/')[:110]} |"
        )
    for label, key in (("Established identities", "normal_established"), ("New identities (first 3 sessions)", "normal_new_identities")):
        n = report[key]
        lines += [
            "",
            f"**{label}:** {n['sessions']} clean sessions, {n['actions']:,} actions · "
            f"false-challenge rate {n['false_challenge_rate']:.1%} · "
            f"{n['anomalies_per_1000_actions']} anomalous actions per 1,000 · by archetype {n['false_challenge_rate_by_archetype']}",
        ]
    md = "\n".join(lines) + "\n"
    (Path(args.out) / "detection_report.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"Wrote {json_path} ({report['runtime_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

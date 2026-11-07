"""
Label-free behavioural detectors.

None of these need labelled attack data: each learns what is normal for an
identity (or for its peer group) from clean traffic and scores deviations.

- SequenceModel     per-identity Markov chain over action transitions
- VolumeBaseline    per-identity data-volume rate baseline (records / bytes)
- PeerGroupBaseline per-tenant, per-group feature statistics for comparing an
                    identity with similar identities (helps cold start)
"""

from detectors.sequence import SequenceModel, SequenceAssessment
from detectors.volume import VolumeBaseline, VolumeAssessment
from detectors.peer import PeerGroupBaseline, PeerAssessment

__all__ = [
    "SequenceModel", "SequenceAssessment",
    "VolumeBaseline", "VolumeAssessment",
    "PeerGroupBaseline", "PeerAssessment",
]

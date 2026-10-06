"""
Personalization on/off assignment for the study of whether adapting to the
student helps (see PROJECT_LOG.md, 2026-10-06).

Without a comparison, an improvement over the term can't be told apart
from the student simply learning the subject or getting used to the
tutor. So within each student, conversations ALTERNATE between
personalization "on" (the tutor uses the stored StudentProfile) and "off"
(the tutor gets an empty profile). Everyone benefits part of the time, and
each student is compared against themselves.

"off" only hides the profile from the tutor - the profiler keeps updating
the real profile from every conversation, so the profile itself evolves
the same way in both arms.

Which arm a student's FIRST conversation lands in is decided by a stable
hash of their matrícula, so half the class starts "on" and half "off"
(otherwise "on" would always coincide with odd session numbers).
"""

import hashlib
import os
from typing import Literal

Arm = Literal["on", "off"]

# always    - every conversation personalized (pre-study behavior)
# alternate - on/off alternating per student, the study design
# off       - never personalized (profiles still built)
MODES = ("always", "alternate", "off")
DEFAULT_MODE = "alternate"


def personalization_mode() -> str:
    mode = os.getenv("PERSONALIZATION_MODE", DEFAULT_MODE).strip().lower()
    return mode if mode in MODES else DEFAULT_MODE


def _starts_on(matricula: str) -> bool:
    # hashlib, not hash(): Python's str hash is salted per process, and the
    # assignment must be identical across container restarts.
    digest = hashlib.sha256(matricula.encode("utf-8")).digest()
    return digest[0] % 2 == 0


def assign_arm(matricula: str | None, session_index: int, mode: str | None = None) -> Arm:
    """
    Arm for a student's `session_index`-th conversation (0-based).

    No matrícula (CLI, tests) -> "on": nothing to compare against anyway.
    """
    mode = mode or personalization_mode()

    if mode == "always" or not matricula:
        return "on"

    if mode == "off":
        return "off"

    on_first = _starts_on(matricula)
    return "on" if (session_index % 2 == 0) == on_first else "off"

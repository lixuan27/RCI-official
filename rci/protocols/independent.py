"""Independent agents (Parallel ReAct): N members with the task's tools and no communication."""
from __future__ import annotations

from rci.protocols.base import Protocol


class IndependentAgents(Protocol):
    name = "independent"

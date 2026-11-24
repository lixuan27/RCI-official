"""Name tables for tasks and protocols."""
from __future__ import annotations

import importlib

TASKS = {
    "hiddenbench": "rci.tasks.hiddenbench.task:HiddenBench",
    "audit": "rci.tasks.audit.task:ContradictionAudit",
    "docsweep": "rci.tasks.docsweep.task:DocSweep",
    "gpqa": "rci.tasks.gpqa.task:GPQADiamond",
    "longbench2": "rci.tasks.longbench2.task:LongBenchV2",
    "mmlongdoc": "rci.tasks.mmlongdoc.task:MMLongBenchDoc",
    "browsecomp": "rci.tasks.browsecomp.task:BrowseCompPlus",
    "swe": "rci.tasks.swe.task:SWEBenchVerified",
    "fate": "rci.tasks.fate.task:FATE",
}

PROTOCOLS = {
    "blackboard": ("rci.protocols.blackboard:BlackboardProtocol", {}),
    "blackboard_private": ("rci.protocols.blackboard:BlackboardProtocol", {"private": True}),
    "independent": ("rci.protocols.independent:IndependentAgents", {}),
    "debate": ("rci.protocols.debate:DebateProtocol", {}),
    "docsweep_board": ("rci.protocols.docsweep_board:DocSweepBoard", {}),
    "sequential": ("rci.protocols.sequential:Sequential", {}),
    "sequential_map": ("rci.protocols.sequential:SequentialMap", {}),
    "oracle": ("rci.protocols.sequential:Oracle", {}),
    "delm": ("rci.protocols.delm:DeLMProtocol", {}),
    "aorchestra": ("rci.protocols.aorchestra:AOrchestraProtocol", {}),
}


def _load(path: str):
    mod, attr = path.split(":")
    return getattr(importlib.import_module(mod), attr)


def make_task(name: str, **kwargs):
    """Build a task; `confidence=True` makes every submission state its confidence."""
    if name not in TASKS:
        raise SystemExit(f"unknown task {name!r}; one of {', '.join(sorted(TASKS))}")
    conf = bool(kwargs.pop("confidence", False))
    task = _load(TASKS[name])(**kwargs)
    task.stake_field = "confidence" if conf else None
    return task


def make_protocol(name: str, **kwargs):
    if name not in PROTOCOLS:
        raise SystemExit(f"unknown protocol {name!r}; one of {', '.join(sorted(PROTOCOLS))}")
    path, defaults = PROTOCOLS[name]
    return _load(path)(**{**defaults, **kwargs})


def protocol_dir(name: str, **kwargs) -> str:
    """The directory name under which `rci.run` writes the records of a protocol with these kwargs."""
    return make_protocol(name, **kwargs).name

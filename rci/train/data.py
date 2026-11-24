"""Question sources for training: a deterministic, resumable stream of examples."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

from rci.paths import data_dir


def _perm(n: int, seed: int, epoch: int) -> list[int]:
    order = list(range(n))
    random.Random(f"{seed}/{epoch}").shuffle(order)
    return order


def build_audit_instance(project: str, snips: list[dict], contradictions: list[dict], level: int, dseed: int,
                      max_contradictions: int = 20) -> dict:
    con = contradictions[:max_contradictions]
    keep = {x for c in con for x in (c["edited"], c["partner"])}
    edited_text = {c["edited"]: c["new_text"] for c in con}
    rng = random.Random(dseed)
    pool_whole = [s["idx"] for s in snips if s["idx"] not in keep and s.get("whole_file")]
    pool_rest = [s["idx"] for s in snips if s["idx"] not in keep and not s.get("whole_file")]
    rng.shuffle(pool_whole)
    rng.shuffle(pool_rest)
    extra = (pool_whole + pool_rest)[: len(keep) * (level - 1)]
    chosen = sorted(keep | set(extra))
    remap = {old: new for new, old in enumerate(chosen)}
    by_idx = {s["idx"]: s for s in snips}
    return {"id": f"{project}_x{level}_d{dseed}", "project": project, "level": level,
            "snippets": [{"idx": remap[i], "file": by_idx[i]["file"], "text": edited_text.get(i, by_idx[i]["text"])}
                         for i in chosen],
            "contradictions": [{"edited": remap[c["edited"]], "partner": remap[c["partner"]],
                                "same_file": c["same_file"], "why": c.get("why", "")} for c in con]}


class AuditTrainSource:
    """Contradiction-audit training questions, prebuilt or assembled from the train corpora."""

    def __init__(self, root: str | None = None, level: int = 2, kind: str = "auto", min_contradictions: int = 4,
                 projects: list[str] | None = None, max_contradictions: int = 20, seed: int = 0,
                 exclude: list[str] | None = None):
        self.root, self.level, self.seed = Path(root) if root else data_dir("audit", "train"), int(level), seed
        self.max_contradictions = max_contradictions
        prebuilt = sorted((self.root / f"x{self.level}").glob("*.json")) if (self.root / f"x{self.level}").is_dir() else []
        if projects:
            prebuilt = [p for p in prebuilt if p.stem in projects]
        if exclude:
            prebuilt = [p for p in prebuilt if p.stem not in exclude]
        self.kind = ("prebuilt" if prebuilt else "fly") if kind == "auto" else kind
        if self.kind == "prebuilt":
            self.files = prebuilt
            if not self.files:
                raise FileNotFoundError(f"no instances in {self.root / f'x{self.level}'}")
        else:
            self.projects = []
            for d in sorted(p for p in self.root.iterdir() if (p / "contradictions.json").exists()):
                if (projects and d.name not in projects) or (exclude and d.name in exclude):
                    continue
                n = len(json.loads((d / "contradictions.json").read_text())["contradictions"])
                if n >= min_contradictions:
                    self.projects.append(d.name)
            if not self.projects:
                raise FileNotFoundError(f"no project in {self.root} has >= {min_contradictions} contradictions")
        self._cache: dict[str, tuple[list[dict], list[dict]]] = {}

    def __len__(self) -> int:
        return len(self.files) if self.kind == "prebuilt" else len(self.projects)

    def _project(self, name: str) -> tuple[list[dict], list[dict]]:
        if name not in self._cache:
            d = self.root / name
            snips = [json.loads(line) for line in (d / "snippets.jsonl").read_text().splitlines() if line.strip()]
            con = json.loads((d / "contradictions.json").read_text())["contradictions"]
            self._cache[name] = (snips, con)
        return self._cache[name]

    def example(self, i: int) -> dict:
        n = len(self)
        epoch, j = divmod(i, n)
        k = _perm(n, self.seed, epoch)[j]
        if self.kind == "prebuilt":
            return json.loads(self.files[k].read_text())
        name = self.projects[k]
        snips, con = self._project(name)
        return build_audit_instance(name, snips, con, self.level, dseed=epoch, max_contradictions=self.max_contradictions)

    def describe(self) -> dict:
        return {"kind": self.kind, "level": self.level, "n": len(self),
                "items": [p.stem for p in self.files] if self.kind == "prebuilt" else self.projects}


class TaskSource:
    """Generic: cycle over task.examples() in per-epoch shuffled order."""

    def __init__(self, task, seed: int = 0, limit: int | None = None, refresh_s: float | None = None,
                 exclude: set[str] | None = None):
        self.task, self.limit, self.refresh_s = task, limit, refresh_s
        self.exclude = set(exclude or ())
        self.items = self._read()
        self.seed = seed
        if not self.items:
            raise ValueError("task has no examples")
        self._read_t = time.time()
        self.sizes = [len(self.items)]

    def __len__(self) -> int:
        return len(self.items)

    def _read(self) -> list[dict]:
        items = self.task.examples(limit=self.limit)
        return [e for e in items if str(e["id"]) not in self.exclude] if self.exclude else items

    def _refresh(self) -> None:
        if not self.refresh_s or time.time() - self._read_t < self.refresh_s:
            return
        self._read_t = time.time()
        try:
            items = self._read()
        except Exception as e:
            print(f"[data] {type(self.task).__name__} re-read failed, keeping {len(self.items)}: {e!r}", flush=True)
            return
        if items and len(items) != len(self.items):
            print(f"[data] {type(self.task).__name__} pool {len(self.items)} -> {len(items)}", flush=True)
            self.sizes.append(len(items))
        if items:
            self.items = items

    def example(self, i: int) -> dict:
        self._refresh()
        epoch, j = divmod(i, len(self.items))
        return self.items[_perm(len(self.items), self.seed, epoch)[j]]

    def describe(self) -> dict:
        d = {"kind": "task", "n": len(self.items)}
        if self.exclude:
            d["excluded"] = len(self.exclude)
        if self.refresh_s:
            d["refresh_s"], d["sizes"] = self.refresh_s, self.sizes
        return d


class MixedSource:
    def __init__(self, sources: list, weights: list[int]):
        self.sources, self.weights = sources, [int(w) for w in weights]
        self.pattern = [j for j, w in enumerate(self.weights) for _ in range(w)]
        self.before = [[self.pattern[:p].count(j) for j in range(len(sources))] for p in range(len(self.pattern))]

    def __len__(self) -> int:
        return sum(len(s) for s in self.sources)

    def example(self, i: int) -> dict:
        cycle, p = divmod(i, len(self.pattern))
        j = self.pattern[p]
        return {**self.sources[j].example(cycle * self.weights[j] + self.before[p][j]), "_stream": j}

    def describe(self) -> dict:
        return {"kind": "mixed", "weights": self.weights, "streams": [s.describe() for s in self.sources]}


def make_source(task_name: str, task, task_kwargs: dict, source_cfg: dict, seed: int):
    if task_name == "audit":
        sc = {k: v for k, v in source_cfg.items() if k in ("kind", "min_contradictions", "projects",
                                                           "max_contradictions", "exclude")}
        return AuditTrainSource(root=task_kwargs.get("root"), level=task_kwargs.get("level", 2),
                             seed=seed, **sc)
    return TaskSource(task, seed=seed, limit=source_cfg.get("limit"), refresh_s=source_cfg.get("refresh_s"),
                      exclude=read_ids(source_cfg.get("exclude_ids")))


def read_ids(spec) -> set[str]:
    """Example ids from a list, or from a text file (one id per line; blank lines and '#' comments ignored)."""
    if not spec:
        return set()
    if isinstance(spec, (list, tuple, set)):
        return {str(x) for x in spec}
    lines = (line.split("#", 1)[0].strip() for line in open(spec))
    return {x for x in lines if x}

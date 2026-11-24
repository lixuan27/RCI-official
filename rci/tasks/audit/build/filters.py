"""Auditable file-exclusion rules for the contradiction-audit corpora."""
from __future__ import annotations

import os
import posixpath
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    rid: str
    category: str
    target: str
    pattern: str
    note: str = ""

    def __post_init__(self):
        object.__setattr__(self, "_re", re.compile(self.pattern))

    def match(self, rel: str, rootrel: str, text_head: str) -> bool:
        rx = self._re
        if self.target == "rel":
            return bool(rx.search(rel))
        if self.target == "rootrel":
            return bool(rx.search(rootrel))
        if self.target == "content":
            return bool(rx.search(text_head))
        comps = rel.split("/")
        if self.target == "component":
            return any(rx.fullmatch(c) for c in comps[:-1])
        if self.target == "stem":
            stem = comps[-1].rsplit(".", 1)[0]
            return bool(rx.fullmatch(stem))
        raise ValueError(self.target)


_REL = r"release[-_ ]?notes?|releasenotes|relnotes|releases?|whats[-_]?new|what[-_]s[-_]new|news|changelogs?|change[-_]?log\.d|changes|upcoming[-_]?changes|unreleased(?:[-_]\w+)?|history"
_CONTRIB = r"[\w-]*contribut[\w-]*"
_META_DIR = (_CONTRIB + r"|governance|licen[cs]es?|legal|code[-_]of[-_]conduct|conduct|coc|gitwash|"
             r"maintainers?|credits|sponsors?|funding|doc[-_]guide|_?style[-_]?guides?")
_META_STEM = (_CONTRIB + r"|governance|licen[cs]es?|licen[cs]ing(?:[-_][\w-]+)?|licen[cs]e[-_][\w-]+|copying|"
              r"copyright|code[-_]of[-_]conduct|conduct|coc|authors?|credits|thanks|acknowledg\w*|"
              r"maintainers?|maintaining|maintainer[-_][\w-]+|core[-_]?devs?|[\w-]*emeritus|sponsors?|"
              r"sponsorship|funding|security[-_]policy|bug[-_]triag\w*|triag\w*|reviewer[-_]guidelines?|"
              r"code[-_]?review(?:ing)?|gitwash|git[-_]development|development[-_]workflow|hacking|"
              r"style[-_]?guide|doc[-_]?guide|documentation[-_]?guide|howto[-_]docs?|writing[-_]docs?|"
              r"developer[-_]?policy|cla")

GLOBAL_RULES: list[Rule] = [
    Rule("hidden_dir", "meta", "component", r"\..+", "hidden directories (.github, .vale, ...)"),
    Rule("root_readme", "meta", "rootrel", r"^(readme|contributing|license|security|code_of_conduct)(\.[a-z]+)+$",
         "docs-repo meta files at the corpus root"),
    Rule("release_dir", "release", "component", _REL),
    Rule("release_stem", "release", "stem", r"(?:" + _REL + r")(?:[-_.]?[\w.]*)?|\w*[-_](?:release[-_]?notes?|changelog|whats[-_]?new)"),
    Rule("deprec", "deprec", "rel", r"deprecat"),
    Rule("roadmap", "roadmap", "rel", r"roadmap|wishlist|(^|/)to[-_]?dos?(/|\.|$)"),
    Rule("include_dir", "include", "component", r"_?includes?|_?partials?|fragments|_?snippets"),
    Rule("template_dir", "template", "component", r"_templates?|templates?|archetypes|_template[-_]\w+"),
    Rule("template_stem", "template", "stem", r"(?:\w+[-_])?template(?:[-_]\w+)?|_template"),
    Rule("translations", "lang", "component", r"translations", "non-English translations"),
    Rule("meta_dir", "meta", "component", _META_DIR),
    Rule("meta_stem", "meta", "stem", _META_STEM),
    Rule("jinja_template", "template", "content", r"\{\{\s*(fullname|objname|module)\b|\{%-?\s*(extends|block)\b",
         "Jinja templates (autosummary etc.)"),
]

PROJECT_RULES: dict[str, list[Rule]] = {
    "linux": [Rule("linux_process", "meta", "rel", r"^documentation/process/", "kernel development-process pages")],
}

EXEMPT: dict[str, dict[str, str]] = {}


def rules_for(project: str) -> list[Rule]:
    return GLOBAL_RULES + PROJECT_RULES.get(project, [])


def classify(project: str, rel: str, rootrel: str, text: str) -> Rule | None:
    """rel: repo-relative path; rootrel: path relative to its corpus root."""
    rel_l = rel.lower()
    root_l = rootrel.lower()
    head = text[:4096]
    exempt = EXEMPT.get(project, {})
    for r in rules_for(project):
        if r.rid in exempt and re.search(exempt[r.rid], rel_l):
            continue
        if r.match(rel_l, root_l, head):
            return r
    return None


_INC_PATTERNS = [
    re.compile(r"^\s*\.\.\s+include::\s+(\S+)", re.M),
    re.compile(r"^\s*(?:`{3,}|:{3,})\s*\{include\}\s+(\S+)", re.M),
    re.compile(r"\{\{[<%]\s*include\s+\"?([^\"\s>%]+)\"?", re.M),
    re.compile(r"^\s*--8<--\s+\"([^\"]+)\"", re.M),
    re.compile(r"\{%-?\s*include(?:_relative)?\s+\"?([^\"\s%]+)\"?", re.M),
    re.compile(r"\{\{#include\s+([^\s}:]+)", re.M),
]


def include_targets(text: str) -> list[str]:
    out = []
    for rx in _INC_PATTERNS:
        for m in rx.finditer(text):
            t = m.group(1).strip().strip("'\"")
            if t.startswith("<") or "://" in t or t.startswith("{"):
                continue
            out.append(t)
    return out


def resolve_transcluded(files: dict[str, str], exts: tuple[str, ...]) -> dict[str, str]:
    """files: rel_path -> text (rel to corpus root)."""
    keys = set(files)
    by_base: dict[str, list[str]] = {}
    for k in keys:
        by_base.setdefault(posixpath.basename(k), []).append(k)
    hit: dict[str, str] = {}
    for src, text in files.items():
        for t in include_targets(text):
            if not t.lower().endswith(exts):
                continue
            t = t.split("#")[0]
            cands = []
            d = posixpath.dirname(src)
            if t.startswith("/"):
                parts = d.split("/") if d else []
                for i in range(len(parts), -1, -1):
                    cands.append(posixpath.normpath(posixpath.join(*parts[:i], t.lstrip("/"))) if i else t.lstrip("/"))
            else:
                cands.append(posixpath.normpath(posixpath.join(d, t)))
                cands.append(posixpath.normpath(t))
            found = next((c for c in cands if c in keys and c != src), None)
            if found is None:
                tail = posixpath.normpath(t.lstrip("/"))
                opts = [k for k in by_base.get(posixpath.basename(tail), []) if k.endswith(tail) and k != src]
                if len(opts) == 1:
                    found = opts[0]
            if found is not None and found not in hit:
                hit[found] = src
    return hit


def list_doc_files(base: str, exts: tuple[str, ...]) -> list[str]:
    out = []
    for dp, dn, fn in os.walk(base):
        dn[:] = sorted(d for d in dn if d != ".git")
        for f in sorted(fn):
            if f.lower().endswith(exts) and not f.startswith(".audit_"):
                out.append(os.path.relpath(os.path.join(dp, f), base).replace(os.sep, "/"))
    return out

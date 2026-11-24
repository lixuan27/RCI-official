"""Self-test of the contradiction-audit build steps (stdlib only)."""
from __future__ import annotations

from rci.tasks.audit.build import candidates as C
from rci.tasks.audit.build import filters as F
from rci.tasks.audit.build import snippets as S


def test_paragraphs():
    t = "a\nb\n\n\n  \nc\n\t\nd  \n"
    assert S.split_paragraphs(t) == ["a\nb", "c", "d"], S.split_paragraphs(t)
    assert S.split_paragraphs("") == []


def test_pack_after():
    paras = list("abcde")
    nt = [400, 400, 400, 900, 10]
    g = S.pack(paras, nt, 1024, "after")
    assert [x.paragraphs for x in g] == [["a", "b", "c"], ["d", "e"]], g
    assert g[0].body_tokens == 1202


def test_pack_before():
    paras = list("abcde")
    nt = [400, 400, 400, 900, 10]
    g = S.pack(paras, nt, 1024, "before")
    assert [x.paragraphs for x in g] == [["a", "b"], ["c"], ["d", "e"]], g


def test_render():
    g = S.Packed(["x", "y"], 3)
    assert S.render("Doc/a.rst", g) == "File: Doc/a.rst\n\nx\n\ny"
    assert C.body(S.render("Doc/a.rst", g)) == "\nx\n\ny"


def test_terms():
    ts = C.terms("The NumPy v2 API uses dtype=float64 since 1.26.4, see np.ndarray and a b cc ddd")
    assert {"numpy", "uses", "dtype", "float64", "1.26.4", "since", "ndarray", "v2"} <= ts, ts
    assert "the" not in ts and "ddd" not in ts and "np" not in ts and "api" not in ts


def test_mine():
    bodies = ["alpha bravo charlie common", "alpha bravo common", "charlie delta common", "zzzz"]
    pairs, st = C.mine(bodies, max_df=2, min_shared=2, seed=0)
    assert [(i, j) for i, j, _ in pairs] == [(0, 1)], pairs
    pairs3, _ = C.mine(bodies, max_df=3, min_shared=2, seed=0)
    assert pairs3[0][:2] == (0, 1) and len(pairs3[0][2]) == 3
    assert C.mine(bodies, 3, 2, 7)[0] == C.mine(bodies, 3, 2, 7)[0]


def test_filters():
    cases = {
        "Doc/whatsnew/3.14.rst": "release",
        "Doc/deprecations/index.rst": "deprec",
        "doc/release/upcoming_changes/123.new_feature.rst": "release",
        "doc/source/release/2.0.0-notes.rst": "release",
        "doc/changelog/2.1.0-changelog.rst": "release",
        "llvm/docs/ReleaseNotes.md": "release",
        "doc/source/_templates/autosummary/class.rst": "template",
        "doc/source/development/contributing.rst": "meta",
        "doc/source/dev/governance/people.rst": "meta",
        "Doc/license.rst": "meta",
        "content/includes/compose/services.md": "include",
        ".github/PULL_REQUEST_TEMPLATE.md": "meta",
        "doc/whats_new/v1.5.rst": "release",
        "docs/ROADMAP.md": "roadmap",
    }
    for rel, cat in cases.items():
        r = F.classify("x", rel, rel.split("/", 1)[-1], "")
        assert r is not None and r.category == cat, (rel, r)
    keep = ["Doc/library/os.rst", "docs/jep/9263-typed-keys.md", "doc/source/dev/internals.rst",
            "doc/developers/develop.rst", "docs/source/community/design.rst", "doc/source/user_guide/indexing.rst", "content/en/docs/concepts/overview/_index.md",
            "Documentation/admin-guide/README.rst", "docs/source/en/model_doc/bert.md",
            "doc/source/reference/arrays.ndarray.rst", "tutorials/scripting/gdscript/gdscript_basics.rst",
            "Documentation/process/changes.rst"]
    for rel in keep[:-1]:
        r = F.classify("x", rel, rel.split("/", 1)[-1], "")
        assert r is None, (rel, r)


def test_transclusion():
    files = {
        "Doc/library/a.rst": ".. include:: ../includes/x.rst\n.. include:: /frag/y.rst\n",
        "Doc/includes/x.rst": "x",
        "Doc/frag/y.rst": "y",
        "content/manuals/b.md": '{{% include "compose/services.md" %}}',
        "content/includes/compose/services.md": "z",
        "Doc/library/c.rst": "c",
    }
    hit = F.resolve_transcluded(files, (".rst", ".md"))
    assert set(hit) == {"Doc/includes/x.rst", "Doc/frag/y.rst", "content/includes/compose/services.md"}, hit


def main():
    n = 0
    for k, v in list(globals().items()):
        if k.startswith("test_") and callable(v):
            v()
            n += 1
    print(f"selftest OK ({n} tests)")


if __name__ == "__main__":
    from rci.paths import cli
    cli(main)

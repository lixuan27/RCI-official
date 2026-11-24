import json
import shlex
from fractions import Fraction

import numpy as np
import pytest

from rci.chains import health as HL
from rci.chains import loop as L
from rci.chains import merge as MG
from rci.chains.adapter import Adapter
from rci.chains.util import write_jsonl


class Toy(Adapter):
    """Confirmation gains and health per round come from cfg["script"]: {round: {"d": gain, "healthy": bool}}."""

    def play_argv(self, models, role, r, out):
        return ["echo", role, str(r), next(iter(models.values()))["served"], '"$RCI_BASE_URLS"']

    def accept(self, r, records, rows_path, lin):
        write_jsonl(rows_path, [{"row_id": f"r{r}:{i}", "ids": [1, 2, 3], "spans": [[1, 2]]} for i in range(4)])
        return {"rows": 4, **({"stop": "stagnation"} if self.cfg["script"].get(r, {}).get("stop") else {})}

    def mix(self, st, r, lin, out):
        write_jsonl(out, [])
        return {"rows": 4, "steps": 1}

    def summary(self, records, role, r):
        return {"rate": 0.5, "healthy": self.cfg["script"].get(r, {}).get("healthy", True)}

    def health(self, summ, base):
        return [HL.Line("toy", None, None, summ["healthy"])]

    def rank(self, summ):
        return summ["rate"]

    def scores(self, records, role, r):
        d = self.cfg["script"].get(r, {}).get("d", 0.0)
        gain = d if records.name == "cand" else 0.0
        return {i: (1.0 if i < 50 + 100 * gain else 0.0) + 0.0 * gain for i in range(100)}


def cfg(tmp_path, **kw):
    c = {"results": str(tmp_path / "res"), "ckpt": str(tmp_path / "ckpt"), "base": {"served": "m", "weights": "/w"},
         "alphas": ["1"], "train": {"lr": 1e-5, "batch": 8, "seed": 0}, "release": {"margin": 0.05},
         "stop": {"max_rounds": 5}, "script": {}}
    c.update(kw)
    return c


def drive(ch, limit=200):
    seen = []
    for _ in range(limit):
        d = ch.next()
        if d is None or d["kind"] == "wait":
            return seen
        seen.append(d["step"])
        if d["kind"] == L.CPU:
            ch.run_cpu(d["step"])
        else:
            if d["step"].startswith("train:"):
                out = ch.train_dir(ch.state()["rounds"][-1]["r"], d["step"].split(":")[1])
                (out / "hf").mkdir(parents=True)
                (out / "master").mkdir()
            ch.mark(d["step"])
    raise AssertionError("the chain did not stop")


def test_a_round_runs_every_step_in_order(tmp_path):
    ch = L.Chain("toy", cfg(tmp_path, script={1: {"d": 0.2}}, stop={"max_rounds": 1}), Toy)
    steps = drive(ch)
    assert steps == ["base:select", "harvest", "accept", "mix", "train:a", "select:1", "select", "confirm:cand",
                     "confirm:inc", "verdict"]
    st = ch.state()
    assert st["rounds"][0]["label"] == "released" and st["gens"]["1"]["a"]["served"] == "m-toy-r1-a1"
    assert st["stop"] == "1 rounds decided"
    v = json.loads((ch.rdir(1) / "verdict.json").read_text())
    assert v["confirm"]["d"] == pytest.approx(0.2) and v["label"] == "released"


def test_flat_and_unhealthy_rounds_keep_the_generation(tmp_path):
    script = {1: {"d": 0.2}, 2: {"d": 0.01}, 3: {"healthy": False}, 4: {"d": 0.3}}
    ch = L.Chain("toy", cfg(tmp_path, script=script, stop={"max_rounds": 4}), Toy)
    drive(ch)
    st = ch.state()
    assert [x["label"] for x in st["rounds"]] == ["released", "flat", "unhealthy", "released"]
    assert [x["gen"] for x in st["rounds"]] == [0, 1, 1, 1] and ch.current(st) == 2
    assert "confirm:cand" not in st["rounds"][2]["done"]
    assert ch.cand(4, "1")["a"]["weights"] == st["gens"]["2"]["a"]["weights"]


def test_round_two_trains_from_the_released_generation(tmp_path):
    ch = L.Chain("toy", cfg(tmp_path, script={1: {"d": 0.2}}, stop={"max_rounds": 2}), Toy)
    for _ in range(30):
        d = ch.next()
        if d["step"] == "train:a" and d["round"] == 2:
            break
        if d["kind"] == L.CPU:
            ch.run_cpu(d["step"])
        else:
            if d["step"].startswith("train:"):
                out = ch.train_dir(1, "a")
                (out / "hf").mkdir(parents=True)
                (out / "master").mkdir()
            ch.mark(d["step"])
    argv = shlex.split(d["line"])
    assert argv[argv.index("--model") + 1] == str(ch.train_dir(1, "a") / "master") and "--master" in argv


def test_stop_rules(tmp_path):
    ch = L.Chain("a", cfg(tmp_path / "a", script={1: {"d": 0.2}, 2: {"d": 0.2}}, stop={"max_releases": 2}), Toy)
    drive(ch)
    assert ch.state()["stop"] == "2 generations released" and len(ch.state()["rounds"]) == 2
    ch = L.Chain("b", cfg(tmp_path / "b", stop={"max_rounds": 8, "flat_first_two": True}), Toy)
    drive(ch)
    assert ch.state()["stop"] == "rounds 1 and 2 both flat"
    ch = L.Chain("c", cfg(tmp_path / "c", script={1: {"stop": True}}), Toy)
    drive(ch)
    assert ch.state()["stop"] == "stagnation" and ch.state()["rounds"][0]["label"] == "stopped"


def test_cpu_steps_refuse_out_of_order(tmp_path):
    ch = L.Chain("toy", cfg(tmp_path), Toy)
    with pytest.raises(L.ChainError):
        ch.run_cpu("verdict")


def test_a_forked_chain_waits_then_starts_from_the_parent_generation(tmp_path):
    parent = L.Chain("p", cfg(tmp_path / "p", script={1: {"d": 0.2}, 2: {"d": 0.2}}, stop={"max_rounds": 2}), Toy)
    child_cfg = cfg(tmp_path / "c", fork={"gen": 1, "stop_with_parent_releases": 2}, harvester="start",
                    stop={"max_rounds": 9})
    child = L.Chain("c", child_cfg, Toy, parent)
    assert child.next()["kind"] == "wait"
    drive(parent)
    assert child.fork_round() == 1
    d = None
    for _ in range(3):
        d = child.next()
        child.mark(d["step"]) if d["kind"] == L.GPU else child.run_cpu(d["step"])
    st = child.state()
    assert st["gens"]["0"]["a"]["served"] == "m-p-r1-a1" and st["gens"]["0"]["origin"] == "p:g1"
    steps = drive(child)
    assert child.state()["stop"] == "the parent chain released generation 2"
    assert len(child.state()["rounds"]) == 1 and "base:select" not in steps


def test_with_server_line_keeps_placeholders_for_the_wrapper():
    line = L.with_server({"a": {"served": "m", "weights": "/w"}, "b": {"served": "m", "weights": "/w"}},
                         ["python", "x", "--base-urls", '"$RCI_BASE_URLS"'])
    argv = shlex.split(line)
    assert argv[:4] == ["scripts/launch/with_server.sh", "--model", "m", "/w"] and argv[4] == "--"
    assert argv[5] == 'python x --base-urls "$RCI_BASE_URLS"'


def tensors(vals: dict, dtype="F32") -> list:
    out = []
    for name, v in vals.items():
        a = np.asarray(v, dtype=np.float64)
        out.append((name, dtype, list(a.shape), MG.encode(a.ravel(), dtype)))
    return out


def test_merge_interpolates_and_extrapolates(tmp_path):
    base, trained = tmp_path / "base", tmp_path / "trained"
    base.mkdir()
    trained.mkdir()
    MG.write_safetensors(base / "model.safetensors", tensors({"w": [[1.0, 2.0], [3.0, 4.0]]}) +
                         [("i", "I64", [2], np.array([5, 6], dtype="<i8").tobytes())])
    MG.write_safetensors(trained / "model.safetensors", tensors({"w": [[3.0, 2.0], [5.0, 0.0]]}) +
                         [("i", "I64", [2], np.array([5, 6], dtype="<i8").tobytes())])
    (trained / "config.json").write_text("{}")
    for alpha, want in (("1/2", [2.0, 2.0, 4.0, 2.0]), ("2", [5.0, 2.0, 7.0, -4.0]), ("1", [3.0, 2.0, 5.0, 0.0])):
        out = tmp_path / f"a{alpha.replace('/', '_')}"
        meta = MG.merge(trained, base, out, alpha)
        hd, off = MG.read_header(out / "model.safetensors")
        raw = (out / "model.safetensors").read_bytes()
        a, b = hd["w"]["data_offsets"]
        assert list(np.frombuffer(raw[off + a:off + b], dtype="<f4")) == want
        assert Fraction(meta["alpha"]) == Fraction(alpha) and (out / "config.json").exists()
    out = tmp_path / "bf16"
    MG.merge(trained, base, out, "1/2", "BF16")
    hd, off = MG.read_header(out / "model.safetensors")
    raw = (out / "model.safetensors").read_bytes()
    a, b = hd["w"]["data_offsets"]
    assert hd["w"]["dtype"] == "BF16" and list(MG.bf16_decode(raw[off + a:off + b])) == [2.0, 2.0, 4.0, 2.0]


def test_merge_refuses_a_changed_integer_tensor(tmp_path):
    base, trained = tmp_path / "base", tmp_path / "trained"
    base.mkdir()
    trained.mkdir()
    MG.write_safetensors(base / "m.safetensors", [("i", "I64", [1], np.array([1], dtype="<i8").tobytes())])
    MG.write_safetensors(trained / "m.safetensors", [("i", "I64", [1], np.array([2], dtype="<i8").tobytes())])
    with pytest.raises(MG.MergeError):
        MG.merge(trained, base, tmp_path / "out", "1/2")


def test_bf16_rounds_to_nearest_even():
    x = np.array([1.0, 1.00390625, 1.01171875, -2.5], dtype=np.float32)
    assert list(MG.bf16_decode(MG.bf16_encode(x))) == [1.0, 1.0, 1.015625, -2.5]

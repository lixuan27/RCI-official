"""Corpus manifest of the contradiction audit: documentation projects pinned to public commits."""
from __future__ import annotations

from dataclasses import dataclass

DOC_EXTS = (".md", ".markdown", ".rst")


@dataclass(frozen=True)
class Project:
    name: str
    split: str
    repo: str
    roots: tuple[str, ...]
    ref: str | None = None
    note: str = ""
    exts: tuple[str, ...] = DOC_EXTS

    @property
    def url(self) -> str:
        return f"https://github.com/{self.repo}.git"


EVAL = [
    Project("cpython", "eval", "python/cpython", ("Doc",)),
    Project("docker", "eval", "docker/docs", ("content",)),
    Project("godot", "eval", "godotengine/godot-docs", ("",)),
    Project("jax", "eval", "jax-ml/jax", ("docs",)),
    Project("kubernetes", "eval", "kubernetes/website", ("content/en/docs",)),
    Project("linux", "eval", "torvalds/linux", ("Documentation",)),
    Project("llvm", "eval", "llvm/llvm-project", tuple(f"{s}/docs" for s in (
        "llvm", "clang", "clang-tools-extra", "lld", "lldb", "mlir", "flang", "libc", "libcxx",
        "openmp", "bolt", "polly", "compiler-rt", "offload", "libunwind", "libsycl", "orc-rt",
        "llvm-libgcc", "utils"))),
    Project("numpy", "eval", "numpy/numpy", ("doc/source",)),
    Project("pandas", "eval", "pandas-dev/pandas", ("doc/source",)),
    Project("pytorch", "eval", "pytorch/pytorch", ("docs/source",)),
    Project("ray", "eval", "ray-project/ray", ("doc/source",)),
    Project("scipy", "eval", "scipy/scipy", ("doc/source",)),
    Project("sklearn", "eval", "scikit-learn/scikit-learn", ("doc",)),
    Project("sqlalchemy", "eval", "sqlalchemy/sqlalchemy", ("doc/build",)),
    Project("sympy", "eval", "sympy/sympy", ("doc/src",)),
    Project("transformers", "eval", "huggingface/transformers", ("docs/source/en",)),
]

TRAIN = [
    Project("django", "train", "django/django", ("docs",), exts=(".txt", ".rst"),
            note="reST sources with .txt extension"),
    Project("flask", "train", "pallets/flask", ("docs",)),
    Project("matplotlib", "train", "matplotlib/matplotlib", ("doc",)),
    Project("airflow", "train", "apache/airflow", ("airflow-core/docs",)),
    Project("spark", "train", "apache/spark", ("docs",)),
    Project("ansible", "train", "ansible/ansible-documentation", ("docs/docsite/rst",)),
    Project("rust_book", "train", "rust-lang/book", ("src",)),
    Project("rust_reference", "train", "rust-lang/reference", ("src",)),
    Project("cargo", "train", "rust-lang/cargo", ("doc/book/src",)),
    Project("nodejs", "train", "nodejs/node", ("doc",)),
    Project("vue", "train", "vuejs/docs", ("src",)),
    Project("tensorflow", "train", "tensorflow/docs", ("site/en",)),
    Project("pytest", "train", "pytest-dev/pytest", ("doc/en",)),
    Project("sphinx", "train", "sphinx-doc/sphinx", ("doc",)),
    Project("celery", "train", "celery/celery", ("docs",)),
    Project("xarray", "train", "pydata/xarray", ("doc",)),
    Project("dask", "train", "dask/dask", ("docs",)),
    Project("polars", "train", "pola-rs/polars", ("docs",)),
    Project("fastapi", "train", "fastapi/fastapi", ("docs/en/docs",)),
    Project("pydantic", "train", "pydantic/pydantic", ("docs",)),
    Project("networkx", "train", "networkx/networkx", ("doc",)),
    Project("istio", "train", "istio/istio.io", ("content/en/docs",)),
    Project("prometheus", "train", "prometheus/prometheus", ("docs",)),
    Project("grafana", "train", "grafana/grafana", ("docs/sources",)),
    Project("envoy", "train", "envoyproxy/envoy", ("docs/root",)),
    Project("qemu", "train", "qemu/qemu", ("docs",)),
    Project("zephyr", "train", "zephyrproject-rtos/zephyr", ("doc",)),
    Project("cmake", "train", "Kitware/CMake", ("Help",)),
    Project("ceph", "train", "ceph/ceph", ("doc",)),
    Project("ros2", "train", "ros2/ros2_documentation", ("source",)),
    Project("julia", "train", "JuliaLang/julia", ("doc/src",)),
    Project("electron", "train", "electron/electron", ("docs",)),
    Project("react", "train", "reactjs/react.dev", ("src/content",)),
    Project("typescript", "train", "microsoft/TypeScript-Website", ("packages/documentation/copy/en",)),
    Project("nova", "train", "openstack/nova", ("doc/source",)),
]

ALL = EVAL + TRAIN
BY_NAME = {p.name: p for p in ALL}


def get(names: list[str] | None = None, split: str | None = None) -> list[Project]:
    out = ALL if not names else [BY_NAME[n] for n in names]
    if split:
        out = [p for p in out if p.split == split]
    return out

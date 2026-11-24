"""Download base model weights from the Hugging Face Hub into $RCI_MODELS/<directory>."""
from __future__ import annotations

import argparse
import sys

from rci import paths

MODELS = {
    "qwen3.5-4b": ("Qwen/Qwen3.5-4B", "Qwen3.5-4B"),
    "qwen3.5-9b": ("Qwen/Qwen3.5-9B", "Qwen3.5-9B"),
    "qwen3.8-27b": ("Qwen/Qwen3.8-27B", "Qwen3.8-27B"),
    "qwen3.6-35b-a3b": ("Qwen/Qwen3.6-35B-A3B", "Qwen3.6-35B-A3B"),
    "gpt-oss-120b": ("openai/gpt-oss-120b", "gpt-oss-120b"),
    "glm-4.5-air-fp8": ("zai-org/GLM-4.5-Air-FP8", "GLM-4.5-Air-FP8"),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Download base model weights (served names as in rci.agents.families).")
    ap.add_argument("models", nargs="+", help=f"served names: {', '.join(MODELS)}")
    ap.add_argument("--revision", default=None, help="a Hub revision to pin (default: the main branch)")
    a = ap.parse_args(argv)
    from huggingface_hub import snapshot_download
    for m in a.models:
        if m not in MODELS:
            print(f"unknown model {m}", file=sys.stderr)
            return 2
        repo, directory = MODELS[m]
        dest = paths.models_dir(directory)
        snapshot_download(repo_id=repo, revision=a.revision, local_dir=str(dest))
        print(f"{m}: {repo} -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

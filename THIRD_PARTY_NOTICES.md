# Third-party notices

This repository adapts code, prompts and evaluation logic from the projects below. No third-party benchmark
data, corpora or model weights are redistributed: the scripts in `scripts/data/` download them from their
original sources, and their own licenses apply.

## Adapted code and prompts

### DeLM (`rci/protocols/delm.py`)

Portions of the shared-lessons selection and rendering logic are adapted from the DeLM reference
implementation (https://github.com/yuzhenmao/DeLM, branch `swe-bench`, `src/shared_lessons.py` and
`src/prompts/note_rules.py`). Modified: native tool calling on the shared harness, a claimable work queue with
dependencies, typed notes rendered as a live board, and members submit their own answers.

    MIT License
    Copyright (c) 2026-2027 Yuzhen Mao, Azalia Mirhoseini

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software
    and associated documentation files (the "Software"), to deal in the Software without
    restriction, including without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions: The above copyright notice
    and this permission notice shall be included in all copies or substantial portions of the
    Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
    INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE
    AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
    DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

### AOrchestra (`rci/protocols/aorchestra.py`)

Prompts adapted from https://github.com/FoundationAgents/AOrchestra (`aorchestra/prompts/gaia.py`,
`aorchestra/subagents/react_agent.py`, `aorchestra/tools/delegate.py`), licensed under the Apache License,
Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0). Modified: native tool calling instead of JSON
actions, no model-selection table, the shared harness's summarisation, and the trace summariser runs on the
same model as the agents.

### MMLongBench-Doc grading (`rci/tasks/mmlongdoc/eval_score.py`, `rci/tasks/mmlongdoc/grade.py`)

Rule-based scoring ported from `eval/eval_score.py` and the two-stage grading pipeline of
https://github.com/mayubo2333/MMLongBench-Doc (commit d73f0dc0be7e0a2ff6a403d5fe65fcd96461f384), code licensed
under the Apache License, Version 2.0 as declared in the upstream README; copyright the MMLongBench-Doc
authors. Modified: list literals are parsed with `ast.literal_eval` instead of `eval`; inputs on which upstream
raises return a score; debug prints removed. The answer-extraction prompt
(`eval/prompt_for_answer_extraction.md`) is not redistributed; `scripts/data/prepare_mmlongdoc.py` downloads it.

### GPQA prompt template (`rci/tasks/gpqa/direct.py`)

The single-model GPQA prompt follows the template of https://github.com/openai/simple-evals (MIT License; the
copyright notice is the one in the upstream LICENSE file).

### LongBench v2 prompt template (`rci/tasks/longbench2/direct.py`)

The single-model reference uses the zero-shot template `prompts/0shot.txt` and the answer extraction of
https://github.com/THUDM/LongBench (MIT License; the copyright notice is the one in the upstream LICENSE file).

### BrowseComp-Plus tools and grading (`rci/tasks/browsecomp/`)

The browser tools (search, open, find, citation format and front-matter handling) follow the OpenResearcher
browser setup for BrowseComp-Plus, and the LLM-judge grading prompt follows the public BrowseComp grading
template. Licenses as in the respective upstream repositories.

### PlanBench prompt and plan extraction (`rci/chains/planbench/text.py`, `rci/chains/planbench/domain.py`)

The Blocksworld domain description, the one-shot statement template, the object and action wording and the plan
extraction rule follow the PlanBench reference implementation, https://github.com/karthikv792/LLMs-Planning (commit
fc638a1aff7df3fe7a1a1d289fa2c04cc24dc284: `configs/blocksworld.yaml`, `prompt_generation/`, `utils/pddl_to_text.py`,
`utils/text_to_pddl.py`, the domain PDDL). Modified: the plan is validated by an exact STRIPS simulator instead of VAL,
an empty plan is never correct, and the response is also judged cut at its first `[PLAN END]`.
`rci/chains/planbench/official_iso_digests.json` holds only SHA-256 digests of the official problems' block-renaming
invariant keys, so that generated problems exclude them; no official problem is redistributed.

    MIT License
    Copyright (c) 2024 Valmeekam Karthik

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software
    and associated documentation files (the "Software"), to deal in the Software without
    restriction, including without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions: The above copyright notice
    and this permission notice shall be included in all copies or substantial portions of the
    Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
    INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE
    AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
    DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

### Hanabi rules (`rci/chains/hanabi/`)

The engine is an independent Python implementation of the rules and default parameters of the Hanabi Learning
Environment (https://github.com/google-deepmind/hanabi-learning-environment, Apache License 2.0) as described in
Bard et al., "The Hanabi Challenge" (Artificial Intelligence, 2020); no code of that environment is included.

## Data and environments that are downloaded, not shipped

- HiddenBench (MIT License): `benchmark.json` from https://github.com/Yassellee/HiddenBench_ICML (pinned commit), fetched by
  `scripts/data/prepare_hiddenbench.py`.
- GPQA-Diamond, LongBench v2, MMLongBench-Doc, BrowseComp-Plus (corpus, queries and judgements), SWE-bench
  Verified (instances and container images) and FATE-M (problems, Lean toolchain and Mathlib) are fetched by
  the corresponding `scripts/data/prepare_*` scripts under their own licenses and terms of use.
- The contradiction-audit corpora are the documentation trees of public open-source projects, fetched at the
  pinned commits listed in `rci/tasks/audit/build/projects.py`; each remains under its project's license.

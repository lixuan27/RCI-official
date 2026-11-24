"""Gated generation chains on public benchmarks (PlanBench, HiddenBench, Hanabi).

    util       hash orders, a deterministic random stream, JSON and JSONL files
    stats      paired gains on confirmation tasks and the release rules
    health     health lines anchored at the base model
    merge      candidates by weight interpolation theta_g + alpha (theta_trained - theta_g) on safetensors shards
    mix        a round's training file: the newest accepted rows first, older rounds as replay, call-skeleton twins
    loop       the round as a state machine: play, check, train, candidates, select, confirm, release or flat
    planbench, hiddenbench, hanabi   the benchmark adapters
"""

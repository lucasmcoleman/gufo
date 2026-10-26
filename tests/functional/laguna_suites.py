#!/usr/bin/env python3
"""Run the shared functional suites (run.py) against Laguna-S-2.1.

    python3 tests/functional/laguna_suites.py --binary build/release/gufo \
        --output /tmp/laguna-functional-1 --record-baseline

This is the Laguna opt-in to the shared harness: it fixes the sampling preset
(`laguna`: temperature 1.0, top_k 20, top_p 1.0, min_p 0), the server options a
68.6 GiB model needs, and the suites whose assertions hold for it. It starts
no model unless the real GGUF exists. Without the file it prints a SKIP line
and exits 77, the status CTest reports as skipped, so an absent model is
never mistaken for a pass.

Each suite runs under the server configuration impact.json assigns it, as
for the other families: `standard` (--context 32768 --sessions 2) and
`single-session` (--context 32768 --sessions 1; long-context, cache-depth,
cache-rotation). The selected suites are grouped by configuration, keeping
their order, and each group is one run.py invocation (one server) writing
OUTPUT/<configuration>/. --context/--sessions override every group, for a
deliberately different run. Two sessions at 32768 add 4.6 GiB to the
68.6 GiB of weights.

Suite groups (run.py names):
  core   discovery, token counting, responses, stops, conversation, structured
         output, sampling defaults and ranges, batching, long context, state
         edges, progress, stream start, metrics, reasoning separator/budget
  cache  RAM-snapshot cache behaviour: edits, growth, depth, rotation,
         concurrency, shared prefixes, prefill scheduling
  tools  tool-call suites, in Laguna's native <arg_key>/<arg_value> dialect
         (the bytes its template renders for calls in history).

Not selected: `cache` (it needs --cache-disk, which Laguna refuses: it has no
serialized snapshot format), image suites (no vision), and the cache suites
that are deep or long enough to need their own memory plan (cache-bridge,
cache-pressure, cache-stalls, cache-compaction, cache-transforms,
cache-messages-loop) - add them with --suite after a core run.

The server runs under whatever GPU window the caller holds, e.g.
    GPU_ADMISSION_LOCK=/server/state/gpu-admission/gpu.lock \
      /server/ai/strix-halo-club/scripts/gpu-lock-exec -w 600 \
      python3 tests/functional/laguna_suites.py ...
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKIP_EXIT = 77
DEFAULT_MODEL = ("/server/ai/models/Myric/Laguna-S-2.1-APEX-GGUF/"
                 "Laguna-S-2.1-APEX-i-quality-v2.gguf")

GROUPS = {
    "core": ["discovery", "messages-count-tokens", "responses", "stops",
             "conversation", "structured", "sampling-defaults", "sampling-ranges",
             "batch", "long-context", "state-edges", "progress", "stream-start",
             "metrics", "reasoning-separator", "reasoning-budget"],
    "cache": ["cache-edits", "cache-growth", "cache-depth", "cache-rotation",
              "cache-concurrency", "cache-shared-prefix", "prefill-scheduling"],
    "tools": ["tools", "tool-edges", "tool-history", "tool-choice-cache",
              "tool-untyped", "tool-schema-edges", "tool-streaming"],
}
GROUPS["all"] = GROUPS["core"] + GROUPS["cache"]

IMPACT = HERE / "impact.json"


def configurations(suites, manifest=None):
    """[(configuration, settings, suites)] in first-use order, from impact.json."""
    manifest = manifest or json.loads(IMPACT.read_text())
    groups = {}
    for suite in suites:
        name = manifest["suites"][suite]["configuration"]
        groups.setdefault(name, []).append(suite)
    return [(name, manifest["configurations"][name], members)
            for name, members in groups.items()]


def model_path(argument):
    return Path(argument or os.environ.get("GUFO_LAGUNA_GGUF") or DEFAULT_MODEL)


def run_command(args, model, extra, suites, settings=None, output=None,
                baseline=None):
    """The run.py invocation for these suites; pure, so it is unit tested."""
    settings = settings or {}
    sessions = args.sessions or settings.get("sessions", 2)
    context = args.context or settings.get("context", 32768)
    baseline = baseline if baseline is not None else args.baseline
    command = [sys.executable, str(HERE / "run.py"),
               "--output", str(output or args.output),
               "--sampling-preset", "laguna",
               "--expected-input-modalities", "text",
               "--startup-timeout", str(args.startup_timeout),
               "--suite-timeout", str(args.suite_timeout),
               "--request-timeout", str(args.request_timeout)]
    command += (["--baseline", str(baseline)] if baseline
                else ["--record-baseline"])
    if args.tier:
        command += ["--tier", args.tier]
    for suite in suites:
        command += ["--suite", suite]
    command += ["--", str(args.binary), "serve", "llm", "--model", str(model),
                "--sessions", str(sessions), "--context", str(context)]
    if settings.get("cache_ram_bytes"):
        command += ["--cache-ram-bytes", str(settings["cache_ram_bytes"])]
    command += extra
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, help="production gufo binary")
    parser.add_argument("--output", type=Path,
                        help="new report directory (run.py never overwrites)")
    parser.add_argument("--model", help=f"GGUF (default $GUFO_LAGUNA_GGUF or {DEFAULT_MODEL})")
    parser.add_argument("--group", choices=sorted(GROUPS), default="all",
                        help="suite group (default: core and cache)")
    parser.add_argument("--suite", action="append", default=[],
                        help="extra run.py suite names, appended to the group")
    parser.add_argument("--sessions", type=int,
                        help="override every group's impact.json sessions")
    parser.add_argument("--context", type=int,
                        help="override every group's impact.json context")
    parser.add_argument("--startup-timeout", type=float, default=1200,
                        help="seconds for the 68.6 GiB upload (default 1200)")
    parser.add_argument("--suite-timeout", type=float, default=3600)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--tier", choices=("pr", "full"))
    parser.add_argument("--baseline", type=Path,
                        help="matched previous laguna_suites.py output directory, "
                             "with one subdirectory per configuration "
                             "(default: record one)")
    parser.add_argument("--ctest", action="store_true",
                        help="CTest mode (laguna.functional): also needs "
                             "GUFO_LAGUNA_FUNCTIONAL=1, else it skips; the "
                             "output directory gets a timestamp suffix")
    parser.add_argument("--list", action="store_true", help="print the groups and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the run.py command and exit; starts nothing")
    parser.add_argument("extra", nargs=argparse.REMAINDER,
                        help="-- further `gufo serve llm` options")
    args = parser.parse_args(argv)
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra

    if args.list:
        for name, suites in GROUPS.items():
            print(f"{name}: {' '.join(suites)}")
        return 0
    model = model_path(args.model)
    if not model.is_file():
        print(f"SKIP: the Laguna-S-2.1 GGUF is not at {model} "
              "(set GUFO_LAGUNA_GGUF or pass --model); no functional suite "
              "ran and nothing was verified", flush=True)
        return SKIP_EXIT
    if args.ctest and os.environ.get("GUFO_LAGUNA_FUNCTIONAL") != "1":
        print("SKIP: the GGUF is present but laguna.functional starts a server "
              "that loads the 68.6 GiB model onto the GPU, so it also needs "
              "GUFO_LAGUNA_FUNCTIONAL=1 and a GPU window; no functional suite "
              "ran and nothing was verified", flush=True)
        return SKIP_EXIT
    if args.binary is None or args.output is None:
        parser.error("--binary and --output are required")
    if args.ctest:
        args.output = args.output.with_name(
            f"{args.output.name}-{time.strftime('%Y%m%d-%H%M%S')}")
    if not args.binary.is_file():
        parser.error(f"--binary {args.binary} does not exist")
    suites = list(dict.fromkeys(GROUPS[args.group] + args.suite))
    plan = configurations(suites)
    commands = []
    for name, settings, members in plan:
        output = args.output / name if len(plan) > 1 else args.output
        baseline = None
        if args.baseline:
            baseline = args.baseline / name if len(plan) > 1 else args.baseline
        commands.append((name, run_command(args, model, extra, members, settings,
                                           output, baseline)))
    if args.dry_run:
        for _, command in commands:
            print(" ".join(command))
        return 0
    if len(commands) > 1:
        args.output.mkdir(parents=True, exist_ok=False)
    status = 0
    for name, command in commands:
        print(f"laguna_suites: configuration {name}: {' '.join(command)}", flush=True)
        code = subprocess.run(command).returncode
        print(f"laguna_suites: configuration {name} exited {code}", flush=True)
        status = status or code
    return status


if __name__ == "__main__":
    sys.exit(main())

"""Checks for the Laguna opt-in to the shared functional harness (no GPU, no
model): the suite groups name real suites, the run.py command it builds, the
`laguna` sampling preset reaching run.py and openai_sdk.py, and the visible
SKIP (exit 77) when the real GGUF is absent."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/functional"))
import laguna_suites  # noqa: E402

spec = importlib.util.spec_from_file_location("functional", ROOT / "tests/functional/run.py")
functional = importlib.util.module_from_spec(spec)
spec.loader.exec_module(functional)


def run_main(argv, env=None):
    out, err = io.StringIO(), io.StringIO()
    old = dict(os.environ)
    try:
        os.environ.pop("GUFO_LAGUNA_GGUF", None)
        os.environ.update(env or {})
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = laguna_suites.main(argv)
            except SystemExit as exit_:
                code = exit_.code
    finally:
        os.environ.clear()
        os.environ.update(old)
    return code, out.getvalue(), err.getvalue()


class LagunaSuitesTest(unittest.TestCase):
    def test_every_group_names_real_suites_once(self):
        for group, suites in laguna_suites.GROUPS.items():
            self.assertEqual(len(suites), len(set(suites)), group)
            for suite in suites:
                self.assertIn(suite, functional.SUITES, f"{group}: {suite}")
        # The disk-backed suites and the image suites are not selected.
        everything = {s for suites in laguna_suites.GROUPS.values() for s in suites}
        self.assertFalse(everything & {"cache", "image-inputs", "image-count", "tool-images"})

    def test_missing_model_skips_visibly(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out, _ = run_main(["--binary", sys.executable,
                                     "--output", str(Path(tmp) / "out"),
                                     "--model", str(Path(tmp) / "absent.gguf")])
        self.assertEqual(code, 77)
        self.assertIn("SKIP", out)
        self.assertIn("nothing was verified", out)

    def test_ctest_mode_needs_the_opt_in_even_with_the_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"GGUF")
            argv = ["--ctest", "--binary", sys.executable, "--output",
                    str(Path(tmp) / "out"), "--model", str(model), "--dry-run"]
            code, out, _ = run_main(argv)
            self.assertEqual(code, 77)
            self.assertIn("GUFO_LAGUNA_FUNCTIONAL=1", out)
            self.assertNotIn("run.py", out)
            code, out, _ = run_main(argv, {"GUFO_LAGUNA_FUNCTIONAL": "1"})
            self.assertEqual(code, 0)
            self.assertRegex(out, r"--output \S+/out-\d{8}-\d{6}/standard ")

    def test_environment_selects_the_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out, _ = run_main(
                ["--binary", sys.executable, "--output", str(Path(tmp) / "o")],
                {"GUFO_LAGUNA_GGUF": str(Path(tmp) / "absent.gguf")})
        self.assertEqual(code, 77)
        self.assertIn("absent.gguf", out)

    def dry_run(self, *argv):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"GGUF")
            binary = Path(tmp) / "gufo"
            binary.write_bytes(b"")
            code, out, _ = run_main(
                ["--binary", str(binary), "--output", str(Path(tmp) / "out"),
                 "--model", str(model), "--dry-run", *argv])
        self.assertEqual(code, 0)
        rows = []
        for line in out.splitlines():
            words = line.split()
            tail = words[words.index("--") + 1:]
            rows.append({
                "words": words, "tail": tail,
                "output": Path(words[words.index("--output") + 1]).name,
                "suites": [words[i + 1] for i, w in enumerate(words) if w == "--suite"],
                "sessions": tail[tail.index("--sessions") + 1],
                "context": tail[tail.index("--context") + 1]})
        return rows

    def test_command_runs_run_py_with_the_laguna_preset(self):
        rows = self.dry_run("--group", "core", "--suite", "cache-bridge",
                            "--sessions", "2", "--context", "4096",
                            "--", "--think", "off")
        for row in rows:
            words, tail = row["words"], row["tail"]
            self.assertIn("--record-baseline", words)
            self.assertEqual(words[words.index("--sampling-preset") + 1], "laguna")
            self.assertEqual(words[words.index("--expected-input-modalities") + 1], "text")
            self.assertEqual(tail[1:4], ["serve", "llm", "--model"])
            # Explicit options override every configuration.
            self.assertEqual((row["sessions"], row["context"]), ("2", "4096"))
            self.assertEqual(tail[-2:], ["--think", "off"])
            self.assertNotIn("--cache-disk", words)
        selected = [suite for row in rows for suite in row["suites"]]
        self.assertEqual(sorted(selected), sorted(laguna_suites.GROUPS["core"] + ["cache-bridge"]))
        bridge = next(row for row in rows if "cache-bridge" in row["suites"])
        self.assertEqual(bridge["output"], "cache-pressure")
        self.assertIn("--cache-ram-bytes", bridge["tail"])

    def test_each_suite_gets_its_impact_configuration(self):
        manifest = json.loads((ROOT / "tests/functional/impact.json").read_text())
        rows = self.dry_run("--group", "all")
        self.assertEqual([row["output"] for row in rows], ["standard", "single-session"])
        for row in rows:
            settings = manifest["configurations"][row["output"]]
            self.assertEqual((int(row["sessions"]), int(row["context"])),
                             (settings["sessions"], settings["context"]))
            for suite in row["suites"]:
                self.assertEqual(manifest["suites"][suite]["configuration"], row["output"])
        single = next(row for row in rows if row["output"] == "single-session")
        self.assertEqual(single["suites"], ["long-context", "cache-depth", "cache-rotation"])
        # Order within a configuration is the group order.
        standard = rows[0]["suites"]
        self.assertEqual(standard, [s for s in laguna_suites.GROUPS["all"] if s in standard])
        # One configuration keeps the plain output directory.
        only = self.dry_run("--group", "tools")
        self.assertEqual(len(only), 1)
        self.assertEqual(only[0]["output"], "out")

    def test_baseline_replaces_the_recording(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "m.gguf"
            model.write_bytes(b"GGUF")
            binary = Path(tmp) / "gufo"
            binary.write_bytes(b"")
            code, out, _ = run_main(
                ["--binary", str(binary), "--output", str(Path(tmp) / "o"),
                 "--model", str(model), "--baseline", str(Path(tmp) / "base"),
                 "--dry-run"])
        self.assertEqual(code, 0)
        self.assertIn("--baseline", out)
        self.assertNotIn("--record-baseline", out)

    def test_run_py_and_the_sdk_accept_the_preset(self):
        # The parser accepts `laguna`; it then stops at the missing server
        # command, which proves the preset was not what it rejected.
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, str(ROOT / "tests/functional/run.py"),
                 "--output", str(Path(tmp) / "x"), "--sampling-preset", "laguna",
                 "--record-baseline", "--suite", "responses"],
                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("pass a gufo serve llm command", result.stderr)
        self.assertNotIn("invalid choice", result.stderr)
        sdk = subprocess.run([sys.executable, str(ROOT / "tests/functional/openai_sdk.py"),
                              "--help"], capture_output=True, text=True)
        if sdk.returncode == 0:  # needs the pinned client packages
            self.assertIn("laguna", sdk.stdout)

    def test_sampling_defaults_expect_laguna_values(self):
        source = (ROOT / "tests/functional/openai_sdk.py").read_text()
        self.assertIn('(1. if preset == "laguna" else .95)', source)
        self.assertIn('preset in ("qwen38", "laguna")', source)


FAKE_GUFO = r'''#!/usr/bin/env python3
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

argv = sys.argv
port = int(argv[argv.index("--port") + 1])
calls = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.reply({"status": "ok"})

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        calls.append(request)
        n = len(calls)
        assert request["model"] == "laguna" and request["temperature"] == 0
        if n == 1:
            assert request["tools"], request
            message = {"role": "assistant", "content": None, "reasoning_content": "hm",
                       "tool_calls": [{"id": "call_1", "type": "function", "function": {
                           "name": "get_weather", "arguments": "{\"location\": \"Tokyo\"}"}}]}
            finish = "tool_calls"
        else:
            roles = [m["role"] for m in request["messages"]]
            if n == 2:
                assert roles[-2:] == ["assistant", "tool"], roles
                assert request["messages"][-2]["tool_calls"][0]["id"] == "call_1"
            else:
                assert roles[-2:] == ["assistant", "user"], roles
            message = {"role": "assistant", "content": FAKE_ANSWERS[n]}
            finish = "stop"
        self.reply({"choices": [{"message": message, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 50, "cached_tokens": 0 if n == 1 else 40}})


FAKE_ANSWERS = {2: "It is 18 C and cloudy in Tokyo.", 3: "Mild."}
HTTPServer(("127.0.0.1", port), Handler).serve_forever()
'''


class LagunaSmokeTest(unittest.TestCase):
    """family_smoke.py laguna against a scripted stand-in for `gufo serve llm`,
    so the three-turn tool chat's logic is checked without the model."""

    def run_smoke(self, fake):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "gufo"
            binary.write_text(fake)
            binary.chmod(0o755)
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"GGUF")
            return subprocess.run(
                [sys.executable, str(ROOT / "tests/functional/family_smoke.py"), "laguna",
                 "--binary", str(binary), "--model", str(model),
                 "--startup-timeout", "30", "--log-dir", str(Path(tmp) / "logs")],
                capture_output=True, text=True, timeout=120)

    def test_three_turn_chat_with_one_tool_call_passes(self):
        result = self.run_smoke(FAKE_GUFO)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS laguna: 3 turns, one get_weather call", result.stdout)

    def test_a_missing_tool_call_fails(self):
        result = self.run_smoke(FAKE_GUFO.replace('"name": "get_weather"', '"name": "search_web"'))
        self.assertEqual(result.returncode, 1)
        self.assertIn("exactly one get_weather call", result.stderr)

    def test_no_cache_reuse_fails(self):
        result = self.run_smoke(FAKE_GUFO.replace('0 if n == 1 else 40', '0'))
        self.assertEqual(result.returncode, 1)
        self.assertIn("reused no cached prefix", result.stderr)


if __name__ == "__main__":
    unittest.main()

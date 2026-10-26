"""Long-context replay contract checks; run in nix develop for the OpenAI SDK."""

import contextlib
from copy import deepcopy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import openai_sdk


class ReplayChecked(Exception):
    """Stop before the unrelated Responses/thinking/cancellation checks."""


def response(cached, prefilled):
    return {"text": '{"keyword":"LANTERN"}', "reasoning": "", "finish": "stop",
            "tools": [], "usage": {"prompt_tokens": 3143, "cached_tokens": cached,
                                   "gufo": {"prefill_tokens": prefilled}}}


def refusal(reason="byte_capacity", tokens=3143):
    return (f"schema=component-cache-v1 event=capture_refused reason={reason} "
            f"tokens={tokens}\n")


class LongContextRetryTest(unittest.TestCase):
    def run_replay(self, replay, *, before="", cold="", during_replay="", with_log=True):
        checks = {}
        client = SimpleNamespace(responses=SimpleNamespace(create=Mock(side_effect=ReplayChecked)))
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            log.write_text(before)

            def chat(_client, body, streaming=False):
                with log.open("a") as output:
                    output.write(during_replay if streaming else cold)
                return deepcopy(replay if streaming else response(0, 3143))

            with patch.object(openai_sdk, "chat_result", side_effect=chat) as send, \
                    contextlib.redirect_stderr(io.StringIO()):
                try:
                    openai_sdk.check_long_context(
                        client, "fixture", checks, 8192, server_log=log if with_log else None)
                except ReplayChecked:
                    self.assertEqual(send.call_count, 2)
                    self.assertEqual(send.call_args_list[0].args[1],
                                     send.call_args_list[1].args[1])
                    self.assertEqual(checks["long_schema_cached"], replay)
                    return
        self.fail("long-context did not reach the next phase")

    def test_full_replay_needs_no_capacity_exception(self):
        self.run_replay(response(3143, 0), with_log=False)

    def test_framing_suffix_requires_the_previous_requests_capacity_refusal(self):
        for reason in ("byte_capacity", "entry_capacity"):
            with self.subTest(reason=reason):
                self.run_replay(response(3136, 7), cold=refusal(reason))

    def test_unrelated_or_failed_captures_do_not_excuse_prefill(self):
        for evidence in (
                {}, {"with_log": False},
                {"before": refusal()}, {"during_replay": refusal()},
                {"cold": refusal(tokens=3142)},
                {"cold": "schema=component-cache-v1 event=capture_failed reason=transfer tokens=3143\n"},
                {"cold": "schema=component-cache-v1 event=snapshot_skipped reason=3 tokens=3143\n"}):
            with self.subTest(evidence=evidence), self.assertRaises(AssertionError):
                self.run_replay(response(3136, 7), **evidence)

    def test_capacity_refusal_does_not_hide_context_loss_or_changed_output(self):
        wrong_output = response(3136, 7)
        wrong_output["text"] = '{"keyword":"WRONG"}'
        wrong_depth = response(3136, 7)
        wrong_depth["usage"]["prompt_tokens"] += 1
        for replay in (response(3126, 17), response(0, 3143), response(0, 7),
                       wrong_output, wrong_depth):
            with self.subTest(replay=replay), self.assertRaises(AssertionError):
                self.run_replay(replay, cold=refusal())


class LimitChecked(Exception):
    """Stop after the long-context thinking-limit assertions."""


def run_thinking_limit(limited, preset):
    """Runs check_long_context through the thinking-limit request."""
    answer = '{"keyword":"LANTERN"}'
    sampled = SimpleNamespace(
        status="completed", output_text=answer, to_dict=dict,
        usage=SimpleNamespace(input_tokens_details=SimpleNamespace(cached_tokens=3143)))
    events = [SimpleNamespace(response=sampled)]
    client = SimpleNamespace(
        responses=SimpleNamespace(create=Mock(
            side_effect=lambda **body: contextlib.nullcontext(events)
            if body.get("stream") else sampled)),
        chat=SimpleNamespace(completions=SimpleNamespace(
            create=Mock(side_effect=LimitChecked))))
    replies = [response(0, 3143), response(3143, 0), {"usage": {}, **limited}]
    with patch.object(openai_sdk, "chat_result",
                      side_effect=lambda *_: deepcopy(replies.pop(0))), \
            patch.object(openai_sdk, "check_events"), \
            contextlib.redirect_stderr(io.StringIO()):
        try:
            openai_sdk.check_long_context(client, "fixture", {}, 8192,
                                          sampling_preset=preset)
        except LimitChecked:
            return True
    raise AssertionError("long-context did not reach the thinking stream")


class LongThinkingLimitTest(unittest.TestCase):
    """An adaptive early stop must be the completed short JSON answer."""

    def limited(self, reasoning="", text="", finish="stop"):
        return {"reasoning": reasoning, "text": text, "tools": [], "finish": finish}

    def test_laguna_early_stop_requires_the_expected_answer(self):
        self.assertTrue(run_thinking_limit(
            self.limited(text='{"keyword": "LANTERN"}'), "laguna"))
        self.assertTrue(run_thinking_limit(self.limited(finish="length"), "laguna"))
        # Review #976: an empty completed reply was accepted.
        for limited in (self.limited(),
                        self.limited(text='{"keyword": "WRONG"}'),
                        self.limited(text='{"keyword": "LANTERN"}', finish=None)):
            with self.subTest(limited=limited), self.assertRaises(AssertionError):
                run_thinking_limit(limited, "laguna")

    def test_other_presets_still_require_reasoning_up_to_the_limit(self):
        for preset in (None, "qwen38", "deepseek4"):
            with self.subTest(preset=preset):
                self.assertTrue(run_thinking_limit(
                    self.limited(reasoning="x", finish="length"), preset))
                for limited in (self.limited(),
                                self.limited(text='{"keyword": "LANTERN"}'),
                                self.limited(reasoning="x",
                                             text='{"keyword": "LANTERN"}'),
                                self.limited(finish="length")):
                    with self.subTest(limited=limited), \
                            self.assertRaises(AssertionError):
                        run_thinking_limit(limited, preset)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Regenerates chat_goldens.json: Poolside's Laguna-S-2.1 chat template rendered
by Jinja, and the rendered prompts tokenized by llama.cpp.

    LD_LIBRARY_PATH=/home/lucas/lib-override \
    python3 -I tests/models/laguna/fixtures/generate_chat_goldens.py \
        --template /server/ai/models/poolside/Laguna-S-2.1/chat_template.jinja \
        --llama-tokenize /server/programming/llama.cpp-laguna/build-hip/bin/llama-tokenize \
        --model /server/ai/models/Myric/Laguna-S-2.1-APEX-GGUF/Laguna-S-2.1-APEX-i-quality-v2.gguf \
        --out tests/models/laguna/fixtures/chat_goldens.json

Rendering follows what transformers' apply_chat_template does with a template:
an ImmutableSandboxedEnvironment with trim_blocks and lstrip_blocks, the
`loopcontrols` extension, the `{% generation %}` tag (which only renders its
body here) and transformers' own `tojson` (ensure_ascii False, no key sorting,
Python's default ", " and ": " separators). transformers itself is not
installed on this host, and the Poolside repository ships no tokenizer.json,
so the token ids come from llama.cpp's llama-tokenize over the GGUF (the same
tokenizer the phase 1 goldens use) rather than from a Hugging Face tokenizer.
The prompts are tokenized with --no-bos: the template writes BOS itself.
"""
import argparse
import hashlib
import json
import subprocess
import sys

import jinja2
from jinja2 import nodes
from jinja2.ext import Extension
from jinja2.sandbox import ImmutableSandboxedEnvironment


class GenerationExtension(Extension):
    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return nodes.CallBlock(self.call_method("_render", []), [], [], body).set_lineno(lineno)

    def _render(self, caller):
        return caller()


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                      separators=separators, sort_keys=sort_keys)


def make_environment():
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True,
        extensions=[jinja2.ext.loopcontrols, GenerationExtension])
    env.filters["tojson"] = tojson
    return env


LOCATION = {"type": "function", "function": {
    "name": "get_user_location", "description": "Get the user's current city.",
    "parameters": {"type": "object", "properties": {}}}}
WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a location.",
    "parameters": {"type": "object", "properties": {
        "location": {"type": "string", "description": "City name"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["location"]}}}
SEARCH = {"type": "function", "function": {
    "name": "search_web", "description": "Search the web for a query.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                   "required": ["query"]}}}
QUOTED = {"type": "function", "function": {
    "name": "write_note", "description": "Save a \"note\" with <b>markup</b> & a ✓ mark.\nTwo lines.",
    "parameters": {"type": "object", "properties": {
        "title": {"type": "string", "description": "Title, e.g. 日本語"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "priority": {"type": "integer", "minimum": 1, "maximum": 5, "default": 3},
        "ratio": {"type": "number", "minimum": 0.5},
        "pinned": {"type": "boolean"}}, "required": ["title"],
        "additionalProperties": False}}}
TOOLS = [LOCATION, WEATHER, SEARCH]

SYSTEM = ("You are a helpful assistant with access to tools. When a tool is needed, "
          "call it; when it returns a result, answer the user directly using that result.")


def call(name, arguments):
    return {"type": "function", "function": {"name": name, "arguments": arguments}}


CASES = [
    ("default_system_thinking", [{"role": "user", "content": "Hello!"}], None, {}),
    ("default_system_no_thinking", [{"role": "user", "content": "Hello!"}], None,
     {"enable_thinking": False}),
    ("custom_system", [{"role": "system", "content": "Answer in French."},
                       {"role": "user", "content": "What is two plus two?"}], None, {}),
    ("blank_system_thinking", [{"role": "system", "content": "  \n"},
                               {"role": "user", "content": "Hi"}], None, {}),
    ("blank_system_no_thinking_no_header", [{"role": "system", "content": ""},
                                            {"role": "user", "content": "Hi"}], None,
     {"enable_thinking": False}),
    ("system_trailing_whitespace", [
        {"role": "system", "content": "You are terse.\n\n \t 　"},
        {"role": "user", "content": "Hi"}], None, {}),
    ("multi_turn_thinking_history", [
        {"role": "user", "content": "Name a prime."},
        {"role": "assistant", "content": "Seven.", "reasoning_content": "Primes: 2, 3, 5, 7."},
        {"role": "user", "content": "Another?"}], None, {}),
    ("multi_turn_reasoning_field", [
        {"role": "user", "content": "Name a prime."},
        {"role": "assistant", "content": "Seven.", "reasoning": "Primes: 2, 3, 5, 7."},
        {"role": "user", "content": "Another?"}], None, {}),
    ("multi_turn_no_thinking_drops_reasoning", [
        {"role": "user", "content": "Name a prime."},
        {"role": "assistant", "content": "Seven.", "reasoning_content": "Primes: 2, 3, 5, 7."},
        {"role": "user", "content": "Another?"}], None, {"enable_thinking": False}),
    ("multi_turn_preserve_thinking", [
        {"role": "user", "content": "Name a prime."},
        {"role": "assistant", "content": "Seven.", "reasoning_content": "Primes: 2, 3, 5, 7."},
        {"role": "user", "content": "Another?"}], None,
     {"enable_thinking": False, "preserve_thinking": True}),
    ("tools_with_system", [{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": "What is the weather at my current location? Use celsius."}],
     TOOLS, {}),
    ("tools_default_system", [{"role": "user", "content": "Weather in Oslo?"}], [WEATHER], {}),
    ("tools_blank_system", [{"role": "system", "content": ""},
                            {"role": "user", "content": "Weather in Oslo?"}], [WEATHER], {}),
    ("tools_no_thinking", [{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": "Find cats."}], [SEARCH],
     {"enable_thinking": False}),
    ("tools_quoted_schema", [{"role": "user", "content": "Save a note."}], [QUOTED], {}),
    ("tool_roundtrip_three_turns", [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "What is the weather at my current location? Use celsius."},
        {"role": "assistant", "content": "", "reasoning_content": "I need the location first.",
         "tool_calls": [call("get_user_location", {})]},
        {"role": "tool", "content": "{\"city\": \"Tokyo\"}"},
        {"role": "assistant", "content": None, "reasoning_content": "Now the weather.",
         "tool_calls": [call("get_weather", {"location": "Tokyo", "unit": "celsius"})]},
        {"role": "tool", "content": "{\"temp_c\": 18, \"conditions\": \"cloudy\"}"}],
     TOOLS, {}),
    ("tool_roundtrip_no_thinking", [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Weather in Tokyo, celsius."},
        {"role": "assistant", "content": "Checking.",
         "tool_calls": [call("get_weather", {"location": "Tokyo", "unit": "celsius"})]},
        {"role": "tool", "content": "{\"temp_c\": 18}"},
        {"role": "assistant", "content": "It is 18 C."},
        {"role": "user", "content": "And Oslo?"}], TOOLS, {"enable_thinking": False}),
    ("tool_call_argument_types", [
        {"role": "user", "content": "Save it."},
        {"role": "assistant", "content": "",
         "tool_calls": [call("write_note", {
             "title": "Line one\nLine \"two\" <b>&</b>",
             "tags": ["a", "bé", "日本語"],
             "priority": 3, "ratio": 2.5, "pinned": True, "extra": None,
             "meta": {"k": [1, {"z": "\U0001f30d"}], "empty": {}, "list": []},
             "count": 100000, "neg": -7})]}],
     [QUOTED], {}),
    ("parallel_tool_calls", [
        {"role": "user", "content": "Weather in both cities."},
        {"role": "assistant", "content": "",
         "tool_calls": [call("get_weather", {"location": "Oslo"}),
                        call("get_weather", {"location": "Rome", "unit": "fahrenheit"})]},
        {"role": "tool", "content": "{\"temp_c\": 3}"},
        {"role": "tool", "content": "{\"temp_f\": 70}"}], [WEATHER], {}),
    ("assistant_content_and_call", [
        {"role": "user", "content": "Search cats."},
        {"role": "assistant", "content": "Let me search.", "reasoning_content": "Use search.",
         "tool_calls": [call("search_web", {"query": "cats"})]},
        {"role": "tool", "content": "3 results"}], [SEARCH], {}),
    ("empty_assistant_message", [
        {"role": "user", "content": "Hi"}, {"role": "assistant", "content": ""},
        {"role": "user", "content": "Hello?"}], None, {}),
    ("extra_system_midway", [
        {"role": "user", "content": "Hi"}, {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Explain TCP."}], None, {}),
    ("unicode_and_markup_content", [
        {"role": "user", "content": "こんにちは \U0001f30d <think> </assistant> 〈|EOS|〉 \"q\" \\ done"}],
     None, {}),
    ("no_generation_prompt", [{"role": "user", "content": "Hello!"},
                              {"role": "assistant", "content": "Hi there."}], None,
     {"add_generation_prompt": False}),
    ("assistant_last_with_generation_prompt", [
        {"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi."}],
     None, {}),
]


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tokenize(args, text):
    cmd = [args.llama_tokenize, "-m", args.model, "--stdin", "--no-escape", "--ids",
           "--log-disable", "--no-bos"]
    out = subprocess.run(cmd, input=text.encode("utf-8"), capture_output=True,
                         check=True).stdout.decode("utf-8").strip()
    if not (out.startswith("[") and out.endswith("]")):
        raise SystemExit(f"unexpected llama-tokenize output: {out!r}")
    body = out[1:-1].strip()
    return [int(x) for x in body.split(",")] if body else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--llama-tokenize", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--llama-commit", default="d854d98")
    args = ap.parse_args()

    template_text = open(args.template, encoding="utf-8").read()
    template = make_environment().from_string(template_text)
    cases = []
    for name, messages, tools, options in CASES:
        kwargs = {"enable_thinking": True, "preserve_thinking": False,
                  "add_generation_prompt": True}
        kwargs.update(options)
        render_args = dict(messages=messages, **kwargs)
        if tools is not None:
            render_args["tools"] = tools
        rendered = template.render(**render_args)
        ids = tokenize(args, rendered)
        assert ids and ids[0] == 2, name  # the template's own BOS text
        cases.append({
            "name": name,
            "messages": messages,
            "tools": tools,
            "enable_thinking": kwargs["enable_thinking"],
            "preserve_thinking": kwargs["preserve_thinking"],
            "add_generation_prompt": kwargs["add_generation_prompt"],
            "rendered": rendered,
            "rendered_sha256": sha256(rendered),
            "token_count": len(ids),
            "token_ids": ids,
        })
    doc = {
        "generator": "tests/models/laguna/fixtures/generate_chat_goldens.py",
        "template_file": "poolside/Laguna-S-2.1 chat_template.jinja",
        "template_sha256": sha256(template_text),
        "jinja2": jinja2.__version__,
        "rendering": "jinja2 ImmutableSandboxedEnvironment(trim_blocks, lstrip_blocks) "
                     "with transformers' tojson and generation tag; transformers not installed",
        "tokenization": f"llama.cpp {args.llama_commit} llama-tokenize --no-bos --ids "
                        "(no HF tokenizer.json is available locally)",
        "model": "Laguna-S-2.1-APEX-i-quality-v2.gguf",
        "cases": cases,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=True, indent=1)
        f.write("\n")
    print(f"wrote {len(cases)} cases to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

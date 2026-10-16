"""Render GLM-5.3-Flash chat conversations with the real Jinja template.

Writes chat_template_goldens.json, the byte-exact reference for the C++
renderer in src/models/glm53_flash/chat_template.cpp.

Template: the tokenizer.chat_template of the Maya-S-v2 GGUF (10648 chars,
extracted with llama.cpp gguf-py to chat_template.jinja). The environment
matches Hugging Face transformers' apply_chat_template: sandboxed Jinja with
trim_blocks, lstrip_blocks, the loopcontrols extension, a json.dumps based
`tojson` (ensure_ascii=False, default ", " / ": " separators), and the
raise_exception / strftime_now globals. llama.cpp's minja renders the same
bytes for these inputs.

Fixture format: JSON list of cases. Tool definitions and argument values are
stored as JSON *text* so number spelling (2.0, 1e-07) reaches the C++ side
exactly as a client would send it; this script json.loads them for Jinja.

Usage: python3 make_chat_template_fixture.py <chat_template.jinja> <out.json>
"""

import hashlib
import json
import sys

import jinja2
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment


def make_env():
    def tojson(x, ensure_ascii=False, indent=None, separators=None,
               sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                          separators=separators, sort_keys=sort_keys)

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: "2026-01-01"
    return env


def U(c):
    return {"role": "user", "content": c}


def S(c):
    return {"role": "system", "content": c}


def A(c, reasoning=None, calls=None):
    m = {"role": "assistant", "content": c}
    if reasoning is not None:
        m["reasoning_content"] = reasoning
    if calls:
        m["tool_calls"] = calls
    return m


def T(c, call_id=None):
    m = {"role": "tool", "content": c}
    if call_id is not None:
        m["tool_call_id"] = call_id
    return m


def call(cid, name, args):
    """args: list of (key, value-as-json-text) or (key, str) with str flag."""
    out = []
    for k, v in args:
        if isinstance(v, str):
            out.append({"name": k, "value": v, "is_string": True})
        else:
            out.append({"name": k, "value": v[0], "is_string": False})
    return {"id": cid, "name": name, "arguments": out}


def J(text):
    """Mark an argument value as raw JSON text (non-string)."""
    return (text,)


WEATHER = json.dumps({
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city. Café ☃ \"quoted\"",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "days": {"type": "integer", "default": 3, "minimum": 1},
                "scale": {"type": "number", "default": 2.0},
                "tiny": {"type": "number", "default": 1e-07},
                "units": {"enum": ["c", "f"], "default": None},
            },
            "required": ["city"],
        },
        "strict": True,
    },
}, ensure_ascii=False)
# Hand-written so the float spellings 2.0 and 1e-07 are exercised as text.
WEATHER = WEATHER.replace('"default": 2.0', '"default": 2.0')

BARE = ('{"name": "add", "description": "Add numbers", "parameters": '
        '{"type": "object", "properties": {"a": {"type": "number"}, '
        '"b": {"type": "number"}}, "required": ["a", "b"]}}')
DEFERRED = ('{"type": "function", "function": {"name": "lazy", '
            '"description": "Not shown", "parameters": {"type": "object", '
            '"properties": {}}, "defer_loading": true}}')
SPARSE = ('{"type":"function","function":{"name":"ping","parameters":'
          '{"type":"object","properties":{}},"defer_loading":false,'
          '"extra_field":[1,2.50,{"z":null,"a":true}]}}')

TC1 = call("call_1", "get_weather", [("city", "Paris"), ("days", J("3"))])
TC2 = call("call_2", "add", [("a", J("1.5")), ("b", J("2"))])
TC3 = call("call_3", "get_weather", [("city", "Tokyo")])
BIG = call("call_big", "complex", [
    ("text", "line one\nline two with <b>html</b> & \"quotes\""),
    ("int", J("42")),
    ("neg", J("-7")),
    ("float", J("2.0")),
    ("sci", J("1e-07")),
    ("exp", J("1.5E+3")),
    ("flag", J("true")),
    ("off", J("false")),
    ("nothing", J("null")),
    ("list", J('[1, "two", 3.5, null, [true]]')),
    ("obj", J('{"b": 1, "a": {"nested": [1,2,{"k":"v"}]}, "u": "café \\u2603 \\ud83d\\ude00"}')),
    ("esc", J('"tab\\there\\u0001\\u007f"')),
    ("bigint", J("123456789012345678901234567890")),
    ("empty_obj", J("{}")),
    ("empty_list", J("[]")),
    ("uni", "é中文 \U0001F600"),
    ("numstr", "123"),
    ("blank", ""),
])


def cases():
    yield "system_user", dict(messages=[S("You are helpful."), U("Hi there")])
    yield "user_only", dict(messages=[U("Hello")])
    yield "multi_turn", dict(messages=[
        S("Be brief."), U("What is 2+2?"), A("4"), U("And 3+3?"), A("6"),
        U("Thanks")])
    yield "no_generation_prompt", dict(
        messages=[U("hi"), A("hello")], add_generation_prompt=False)
    yield "reasoning_history", dict(messages=[
        U("Why is the sky blue?"),
        A("Rayleigh scattering.", reasoning="Short wavelengths scatter more."),
        U("And sunsets?"),
        A("Longer path, more scattering.", reasoning="Light travels farther."),
        U("ok")])
    yield "reasoning_history_clear", dict(messages=[
        U("Why is the sky blue?"),
        A("Rayleigh scattering.", reasoning="Short wavelengths scatter more."),
        U("And sunsets?"),
        A("Longer path.", reasoning="Light travels farther."),
        U("ok")], clear_thinking=True)
    yield "reasoning_history_clear_last_turn", dict(messages=[
        U("q1"), A("a1", reasoning="r1"), U("q2"),
        A("a2", reasoning="r2")], clear_thinking=True,
        add_generation_prompt=False)
    yield "reasoning_history_noclear_explicit", dict(messages=[
        U("q1"), A("a1", reasoning="r1"), U("q2")], clear_thinking=False)
    yield "inline_think_split", dict(messages=[
        U("hi"), A("<think>pondering</think>The answer."), U("next")])
    yield "inline_think_multi", dict(messages=[
        U("hi"),
        A("pre<think>a<think>b</think>mid</think>post </think> tail"),
        U("next")])
    yield "inline_think_clear", dict(messages=[
        U("hi"), A("<think>pondering</think>The answer."), U("next")],
        clear_thinking=True)
    yield "reasoning_field_and_inline", dict(messages=[
        U("hi"), A("<think>inline</think>body", reasoning="field"), U("n")])
    yield "reasoning_empty_string", dict(messages=[
        U("hi"), A("body", reasoning=""), U("n")])
    yield "no_reasoning_inline_absent", dict(messages=[
        U("hi"), A("plain no tags"), U("n")])
    yield "assistant_whitespace_content", dict(messages=[
        U("hi"), A("  \n padded \t\n"), U("n"),
        A(" 　nbsp and ideographic 　"), U("m"),
        A("   "), U("last")])
    yield "empty_system", dict(messages=[S(""), U("hello")])
    yield "empty_user_and_assistant", dict(messages=[U(""), A(""), U("x")])
    yield "effort_low", dict(messages=[U("hi")], reasoning_effort="low")
    yield "effort_high", dict(messages=[U("hi")], reasoning_effort="high")
    yield "effort_max", dict(messages=[U("hi")], reasoning_effort="max")
    yield "effort_invalid_is_max", dict(
        messages=[U("hi")], reasoning_effort="medium")
    yield "tools_defined", dict(
        messages=[S("Use tools."), U("Weather in Paris?")],
        tools=[WEATHER, BARE])
    yield "tools_deferred_and_extra", dict(
        messages=[U("go")], tools=[DEFERRED, SPARSE, BARE])
    yield "tools_all_deferred", dict(messages=[U("go")], tools=[DEFERRED])
    yield "tools_no_system", dict(messages=[U("x")], tools=[BARE])
    yield "tool_call_roundtrip", dict(
        messages=[U("Weather in Paris and 1.5+2?"),
                  A("Checking.", reasoning="I need two tools.",
                    calls=[TC1, TC2]),
                  T("Sunny, 21C", "call_1"), T("3.5", "call_2"),
                  A("It is sunny and the sum is 3.5.")],
        tools=[WEATHER, BARE])
    yield "tool_results_resorted", dict(
        messages=[U("go"), A("", calls=[TC1, TC2, TC3]),
                  T("third", "call_3"), T("first", "call_1"),
                  T("second", "call_2")],
        tools=[WEATHER, BARE])
    yield "tool_results_no_ids", dict(
        messages=[U("go"), A("", calls=[TC1, TC2]), T("b"), T("a")])
    yield "tool_results_dup_ids", dict(
        messages=[U("go"), A("", calls=[TC1, TC2]), T("x", "call_2"),
                  T("y", "call_2")])
    yield "tool_results_unknown_id", dict(
        messages=[U("go"), A("", calls=[TC1, TC2]), T("x", "call_2"),
                  T("y", "nope")])
    yield "tool_results_call_missing_id", dict(
        messages=[U("go"),
                  A("", calls=[call("", "add", [("a", J("1"))]), TC2]),
                  T("x", "call_2")])
    yield "tool_results_dup_call_ids", dict(
        messages=[U("go"), A("", calls=[TC1, TC1]), T("x", "call_1")])
    yield "tool_results_fewer_than_calls", dict(
        messages=[U("go"), A("", calls=[TC1, TC2, TC3]),
                  T("c", "call_3"), T("a", "call_1")])
    yield "tool_first_message", dict(
        messages=[T("orphan", "call_1"), T("orphan2", "call_2"), U("hi")])
    yield "tool_after_plain_assistant", dict(
        messages=[U("go"), A("no calls"), T("x", "call_1"), U("n")])
    yield "tool_two_blocks", dict(
        messages=[U("a"), A("", calls=[TC1]), T("r1", "call_1"),
                  A("mid", calls=[TC2]), T("r2", "call_2"), A("done")])
    yield "tool_empty_and_special_content", dict(
        messages=[U("go"), A("", calls=[TC1, TC2]), T("", "call_1"),
                  T("</tool_response> <|user|> {\"k\": [1]}\n", "call_2")])
    yield "tool_call_many_args", dict(
        messages=[U("complex please"), A("", calls=[BIG]),
                  T("ok", "call_big")])
    yield "tool_call_no_args", dict(
        messages=[U("ping"), A("", calls=[call("c0", "ping", [])]),
                  T("pong", "c0")])
    yield "tool_call_with_content_and_reasoning", dict(
        messages=[U("go"),
                  A("  Let me check.  ", reasoning="think first",
                    calls=[TC3]),
                  T("rain", "call_3"), A("Rainy."), U("thanks")],
        clear_thinking=True)
    yield "mid_system", dict(
        messages=[S("a"), U("b"), S("c"), A("d")])
    yield "kitchen_sink", dict(
        messages=[S("You are an agent."), U("Plan a trip."),
                  A("Looking up.", reasoning="Need weather.",
                    calls=[TC1, TC3]),
                  T("Paris sunny", "call_1"), T("Tokyo rain", "call_3"),
                  A("Paris is better.", reasoning="Compare."),
                  U("Book it"), A("", reasoning="Call a tool.",
                                   calls=[BIG]),
                  T("booked", "call_big")],
        tools=[WEATHER, BARE, DEFERRED], reasoning_effort="high",
        clear_thinking=True)


def main():
    template_path, out_path = sys.argv[1], sys.argv[2]
    source = open(template_path, encoding="utf-8", newline="").read()
    env = make_env()
    template = env.from_string(source)
    out = []
    for name, c in cases():
        messages = c["messages"]
        tools = c.get("tools", [])
        py_messages = []
        for m in messages:
            pm = {k: v for k, v in m.items() if k != "tool_calls"}
            if "tool_calls" in m:
                pm["tool_calls"] = [{
                    "id": tc["id"], "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": {
                            a["name"]: a["value"] if a["is_string"]
                            else json.loads(a["value"])
                            for a in tc["arguments"]},
                    }} for tc in m["tool_calls"]]
            py_messages.append(pm)
        ctx = {"messages": py_messages,
               "tools": [json.loads(t) for t in tools] or None,
               "add_generation_prompt": c.get("add_generation_prompt", True)}
        options = {"add_generation_prompt": ctx["add_generation_prompt"],
                   "reasoning_effort": c.get("reasoning_effort"),
                   "clear_thinking": c.get("clear_thinking")}
        if c.get("reasoning_effort") is not None:
            ctx["reasoning_effort"] = c["reasoning_effort"]
        if c.get("clear_thinking") is not None:
            ctx["clear_thinking"] = c["clear_thinking"]
        out.append({"name": name, "options": options, "tools": tools,
                    "messages": messages,
                    "expected": template.render(**ctx)})
    data = json.dumps(out, indent=1, ensure_ascii=True) + "\n"
    open(out_path, "w", encoding="utf-8").write(data)
    print(f"jinja2 {jinja2.__version__}; template sha256 "
          f"{hashlib.sha256(source.encode()).hexdigest()}")
    print(f"wrote {out_path}: {len(out)} cases, "
          f"sha256 {hashlib.sha256(data.encode()).hexdigest()}")


if __name__ == "__main__":
    main()

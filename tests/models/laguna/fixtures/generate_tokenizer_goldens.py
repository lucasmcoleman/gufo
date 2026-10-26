#!/usr/bin/env python3
"""Regenerates tokenizer_goldens.json from llama.cpp's llama-tokenize.

    LD_LIBRARY_PATH=/home/lucas/lib-override \
    python3 -I tests/models/laguna/fixtures/generate_tokenizer_goldens.py \
        --llama-tokenize /server/programming/llama.cpp-laguna/build-hip/bin/llama-tokenize \
        --model /server/ai/models/Myric/Laguna-S-2.1-APEX-GGUF/Laguna-S-2.1-APEX-i-quality-v2.gguf \
        --out tests/models/laguna/fixtures/tokenizer_goldens.json

Each text goes to llama-tokenize on stdin with --no-escape, so the bytes are
exactly the JSON string. Three variants are recorded per text, so a comparison
never depends on an implicit BOS policy:
  ids_bos          llama-tokenize default (BOS 2 prepended, specials parsed)
  ids_no_bos       --no-bos
  ids_no_special   --no-bos --no-parse-special
"""
import argparse
import json
import subprocess
import sys

CORPUS = [
    # ASCII prose
    ("prose", "The quick brown fox jumps over the lazy dog."),
    ("prose", "Hello, world!"),
    ("prose", "Laguna is a sparse mixture-of-experts model with sliding-window attention."),
    ("prose", "It was the best of times, it was the worst of times; it was the age of wisdom."),
    ("prose", "A  B   C    D"),
    ("prose", "Short."),
    ("prose", "UPPERCASE lowercase MiXeD CaSe camelCaseIdentifier snake_case_identifier"),
    # contractions
    ("contraction", "I don't think they'll say we've already seen what he'd do; she's sure it's fine."),
    ("contraction", "DON'T SHOUT, WE'LL HEAR YOU. I'M HERE, THEY'RE THERE, YOU'VE GONE."),
    ("contraction", "rock 'n' roll, y'all, o'clock, 'tis the season, '90s kids, the dogs' bowls"),
    ("contraction", "can't won't shan't ain't"),
    # code
    ("code", "def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n"),
    ("code", "#include <stdio.h>\nint main(void) {\n\tprintf(\"hello, %s\\n\", \"world\");\n\treturn 0;\n}\n"),
    ("code", "fn main() {\n    let v: Vec<u32> = (0..10).map(|x| x * x).collect();\n    println!(\"{:?}\", v);\n}"),
    ("code", "const add = (a, b) => a + b;\nconsole.log(`sum=${add(1, 2)}`);"),
    ("code", "SELECT id, name FROM users WHERE age >= 18 AND country IN ('US', 'CA') ORDER BY name;"),
    ("code", "<div class=\"box\" id='main'><p>Hi &amp; bye</p></div>"),
    ("code", "{\"key\": [1, 2.5, null, true], \"nested\": {\"a\": \"b\"}}"),
    ("code", "#!/bin/sh\nset -eu\nfor f in *.txt; do echo \"$f\"; done"),
    ("code", "x = a if b else c  # trailing comment\n\n\n# another\n"),
    ("code", "package main\n\nimport \"fmt\"\n\nfunc main() { fmt.Println(\"hi\") }"),
    ("code", "a->b::c<d>(e)[f]{g};h?i:j|k&l^m~n!o%p*q/r-s+t"),
    # numbers
    ("numbers", "0123456789"),
    ("numbers", "The year 2026 had 365 days and 3.14159 is pi; 1,000,000 and 1e-9 and 0xDEADBEEF."),
    ("numbers", "12345678901234567890123456789012345678901234567890"),
    ("numbers", "-42 +17 3/4 5% $9.99 #1 No.7 v2.1.0 10:30 2026-10-26"),
    ("numbers", "\u0663\u0664\u0665 \u0967\u0968\u0969 \u2460\u2461 \u00bd \u00b2\u00b3"),
    # whitespace
    ("whitespace", "      "),
    ("whitespace", "a" + " " * 40 + "b"),
    ("whitespace", "trailing spaces   "),
    ("whitespace", "   leading spaces"),
    ("whitespace", " "),
    ("whitespace", "word \u00a0 nbsp \u2003 emspace \u3000 ideographic"),
    ("whitespace", "\u200b zero width \u200d joiner \ufeff bom"),
    # tabs and newlines
    ("tabs_newlines", "a\tb\t\tc\t\t\td"),
    ("tabs_newlines", "line1\nline2\n\nline4\n\n\n\nline8"),
    ("tabs_newlines", "\n"),
    ("tabs_newlines", "\n\n\n"),
    ("tabs_newlines", "ends with newline\n"),
    ("tabs_newlines", "\nstarts with newline"),
    ("tabs_newlines", "windows\r\nline\r\nendings\r\n"),
    ("tabs_newlines", "old mac\rlines\r\r"),
    ("tabs_newlines", "spaces before newline   \n   spaces after newline"),
    ("tabs_newlines", "\t\n\t \n \t\n"),
    ("tabs_newlines", "mixed \t \r\n \n\t tail"),
    ("tabs_newlines", "a \n b \n\n c \n\n\n d"),
    # Unicode: CJK
    ("unicode_cjk", "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8\u3092\u30c8\u30fc\u30af\u30f3\u5316\u3057\u307e\u3059\u3002"),
    ("unicode_cjk", "\u4eca\u5929\u5929\u6c14\u5f88\u597d\uff0c\u6211\u4eec\u53bb\u516c\u56ed\u73a9\u5427\uff01"),
    ("unicode_cjk", "\ud55c\uad6d\uc5b4 \ud14d\uc2a4\ud2b8 \ud1a0\ud070\ud654 \ud14c\uc2a4\ud2b8"),
    ("unicode_cjk", "mixed \u6f22\u5b57 and English \u30ab\u30bf\u30ab\u30ca123"),
    # Unicode: emoji
    ("unicode_emoji", "\U0001F600\U0001F601\U0001F602 party \U0001F389!"),
    ("unicode_emoji", "family: \U0001F468\u200D\U0001F469\u200D\U0001F467\u200D\U0001F466 flag: \U0001F1FA\U0001F1F8 thumbs: \U0001F44D\U0001F3FD"),
    ("unicode_emoji", "\u2764\ufe0f \u2728 \u26a0\ufe0f \u2705 \u274c"),
    # Unicode: accents and other scripts
    ("unicode_accents", "caf\u00e9 na\u00efve fa\u00e7ade \u00fcber se\u00f1or \u00c5ngstr\u00f6m"),
    ("unicode_accents", "cafe\u0301 nai\u0308ve a\u030a (decomposed)"),
    ("unicode_accents", "\u0414\u043e\u0431\u0440\u044b\u0439 \u0434\u0435\u043d\u044c, \u043c\u0438\u0440! \u0393\u03b5\u03b9\u03b1 \u03c3\u03bf\u03c5 \u03ba\u03cc\u03c3\u03bc\u03b5"),
    ("unicode_accents", "\u0939\u093f\u0928\u094d\u0926\u0940 \u092d\u093e\u0937\u093e \u0e20\u0e32\u0e29\u0e32\u0e44\u0e17\u0e22"),
    ("unicode_accents", "\u0130stanbul \u00df \u017f \u212a \ufb01"),
    # Unicode: RTL
    ("unicode_rtl", "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645! \u0647\u0630\u0627 \u0627\u062e\u062a\u0628\u0627\u0631."),
    ("unicode_rtl", "\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd, \u05d6\u05d4\u05d5 \u05de\u05d1\u05d7\u05df."),
    ("unicode_rtl", "mixed \u0645\u0631\u062d\u0628\u0627 and english \u05e9\u05dc\u05d5\u05dd 123 \u200f\u200e"),
    # punctuation clusters
    ("punctuation", "Wait... what?! No way!!! (really?) [yes] {no} <maybe> \"quoted\" 'single'"),
    ("punctuation", "!@#$%^&*()_+-=[]{}\\|;:'\",.<>/?`~"),
    ("punctuation", "...---===***___###"),
    ("punctuation", "end.\n\n-- next --\n* item\n* item"),
    ("punctuation", "\u201csmart quotes\u201d \u2018single\u2019 \u2014 em dash \u2013 en dash \u2026 ellipsis \u00ab guillemets \u00bb"),
    ("punctuation", "a,b;c:d.e!f?g"),
    ("punctuation", "\u00a9 \u00ae \u2122 \u00b0 \u00b1 \u00d7 \u00f7 \u221e \u2260 \u2264 \u2265 \u2192 \u2190 \u21d2"),
    # special-token text
    ("special", "\u3008|EOS|\u3009"),
    ("special", "text \u3008|CODE_START|\u3009 code \u3008|CODE_END|\u3009 more"),
    ("special", "<think>reasoning here</think> answer"),
    ("special", "<assistant>Hello</assistant>"),
    ("special", "<tool_call>{\"name\": \"f\", \"args\": {}}</tool_call>"),
    ("special", "\u3008|SPECIAL_1|\u3009\u3008|SPECIAL_2|\u3009\u3008|SPECIAL_50|\u3009"),
    ("special", "partial \u3008| and |\u3009 and \u3008|/ and /|\u3009 and |\u25ca| fragments"),
    ("special", "\u3008|EOS \u3008|NOPE|\u3009 EOS|\u3009"),
    ("special", "\u3008|FIM_START|\u3009prefix\u3008|FIM_SUFFIX|\u3009suffix\u3008|FIM_MIDDLE|\u3009"),
    ("special", "no special here: <think not closed, </assistan, \u3008|eos|\u3009 lowercase"),
    ("special", "<|im_end|> <|endoftext|> <s> </s> [INST] (other families' markers are plain text)"),
    ("special", "\u3008|EOS|\u3009\n\u3008|PAD|\u3009\t\u3008|SEP|\u3009  \u3008|MASK|\u3009"),
    # misc
    ("misc", ""),
    ("misc", "a"),
    ("misc", "https://example.com/path/to/page?query=1&other=two#fragment"),
    ("misc", "user@example.com, +1 (555) 010-9999, C:\\Users\\name\\file.txt, /usr/local/bin"),
    ("misc", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
    ("misc", "the the the the the the the the the the"),
    ("misc", "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. " * 6),
]


def run(args, text, extra):
    cmd = [args.llama_tokenize, "-m", args.model, "--stdin", "--no-escape",
           "--ids", "--log-disable"] + extra
    out = subprocess.run(cmd, input=text.encode("utf-8"), capture_output=True,
                         check=True).stdout.decode("utf-8").strip()
    if not (out.startswith("[") and out.endswith("]")):
        raise SystemExit(f"unexpected llama-tokenize output for {text!r}: {out!r}")
    body = out[1:-1].strip()
    return [int(x) for x in body.split(",")] if body else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-tokenize", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--llama-commit", default="d854d98")
    args = ap.parse_args()
    assert len(CORPUS) >= 40
    cases = []
    for i, (category, text) in enumerate(CORPUS):
        cases.append({
            "name": f"{category}_{i:03d}",
            "category": category,
            "text": text,
            "utf8_bytes": len(text.encode("utf-8")),
            "ids_bos": run(args, text, []),
            "ids_no_bos": run(args, text, ["--no-bos"]),
            "ids_no_special": run(args, text, ["--no-bos", "--no-parse-special"]),
        })
    doc = {
        "generator": "tests/models/laguna/fixtures/generate_tokenizer_goldens.py",
        "reference": f"llama.cpp {args.llama_commit} llama-tokenize (--stdin --no-escape --ids)",
        "model": "Laguna-S-2.1-APEX-i-quality-v2.gguf",
        "bos": "ids_bos is llama-tokenize's default (BOS id 2 first); "
               "ids_no_bos and ids_no_special pass --no-bos",
        "cases": cases,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=True, indent=1)
        f.write("\n")
    print(f"wrote {len(cases)} cases to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

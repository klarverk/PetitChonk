"""Mistral Small 4 behind the server (Petit Chonk P3): TK5, its template's date helper, and TK6, its thinking.

Mistral Small 4's chat template (llama.cpp architecture `mistral4`, pre-tokenizer `tekken`) calls `strftime_now`, the
helper Hugging Face's chat templates get, and takes `reasoning_effort` "none" or "high" only, raising on any other.
A reply that thinks opens with [THINK] and ends its reasoning with [/THINK].  Both changes are for Mistral only:
another model's template never sees the helper, and its call format, parser and efforts are unchanged.

    python -m unittest serve.test_mistral4
"""
import datetime
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from serve.frontend import ChatTemplate, OutputParser, effort_kwargs  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, load_chat_template  # noqa: E402

# the parts of Mistral Small 4's template the server has to meet: the date helper, the two efforts (any other
# raises) written into [MODEL_SETTINGS], and [INST] turns
TEMPLATE = """{%- set today = strftime_now("%Y-%m-%d") -%}
{{- bos_token -}}[SYSTEM_PROMPT]Today is {{ today }}.[/SYSTEM_PROMPT]
{%- set reasoning_effort = reasoning_effort if reasoning_effort is defined and reasoning_effort is not none else 'none' -%}
{%- if reasoning_effort not in ['none', 'high'] -%}
{{- raise_exception('reasoning_effort must be either "none" or "high"') -}}
{%- endif -%}
[MODEL_SETTINGS]{"reasoning_effort": "{{ reasoning_effort }}"}[/MODEL_SETTINGS]
{%- for m in messages -%}
{%- if m.role == "user" -%}[INST]{{ m.content }}[/INST]{%- else -%}{{ m.content }}{{ eos_token }}{%- endif -%}
{%- endfor -%}"""

# a template that uses the helper only when it is there, as some models' templates do
OPTIONAL = """{%- if strftime_now is defined -%}dated {{ strftime_now("%Y") }}{%- else -%}undated{%- endif -%}"""


def template(text):
    d = tempfile.mkdtemp()
    path = Path(d) / "chat_template.jinja"
    path.write_text(text, encoding="utf-8")
    return path


class TekkenTokens(ByteTokenizer):
    SPECIALS = ["<s>", "</s>", "[INST]", "[/INST]", "[THINK]", "[/THINK]", "[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]",
                "[MODEL_SETTINGS]", "[/MODEL_SETTINGS]"]
    pre = "tekken"
    special_ids = {"tokenizer.ggml.bos_token_id": 256, "tokenizer.ggml.eos_token_id": 257}


class PixtralTokens(TekkenTokens):
    pre = "pixtral"


class Lfm2Tokens(ByteTokenizer):
    SPECIALS = ["<|startoftext|>", "<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    pre = "lfm2"
    special_ids = {"tokenizer.ggml.bos_token_id": 256, "tokenizer.ggml.eos_token_id": 257}


class DateHelper(unittest.TestCase):
    """TK5"""

    def test_off_by_default(self):
        self.assertEqual(ChatTemplate(template(OPTIONAL)).render([{"role": "user", "content": "x"}]), "undated")

    def test_on_when_asked(self):
        year = datetime.datetime.now().strftime("%Y")
        self.assertEqual(ChatTemplate(template(OPTIONAL), date_helper=True).render([{"role": "user", "content": "x"}]),
                         f"dated {year}")

    def test_only_mistral_packs_get_it(self):
        path = template(OPTIONAL)
        user = [{"role": "user", "content": "x"}]
        year = datetime.datetime.now().strftime("%Y")
        self.assertEqual(load_chat_template(path, TekkenTokens()).render(user), f"dated {year}")
        self.assertEqual(load_chat_template(path, PixtralTokens()).render(user), f"dated {year}")
        self.assertEqual(load_chat_template(path, Lfm2Tokens()).render(user), "undated")
        self.assertEqual(load_chat_template(path, ByteTokenizer()).render(user), "undated")

    def test_mistral_template_renders_today(self):
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        prompt = load_chat_template(template(TEMPLATE), TekkenTokens()).render(
            [{"role": "user", "content": "hi"}], bos_token="<s>", eos_token="</s>")
        self.assertIn(f"Today is {today}.", prompt)


class Parser(unittest.TestCase):
    """TK6: [THINK]...[/THINK] is reasoning, the rest content, however the text arrives"""

    def run_parser(self, text, thinking, step):
        p = OutputParser(thinking=thinking, call_format="mistral")
        evs = []
        for i in range(0, len(text), step):
            evs += p.feed(text[i:i + step])
        evs += p.finish()
        return ("".join(e.text for e in evs if e.kind == "reasoning"),
                "".join(e.text for e in evs if e.kind == "content"), evs)

    def test_reasoning_then_answer(self):
        for thinking in (True, False):
            for step in (1, 2, 3, 7, 10_000):
                with self.subTest(thinking=thinking, step=step):
                    r, c, _ = self.run_parser("[THINK]Two plus two.[/THINK]It is 4.", thinking, step)
                    self.assertEqual((r, c), ("Two plus two.", "It is 4."))

    def test_answer_without_thinking(self):
        for step in (1, 5, 10_000):
            with self.subTest(step=step):
                self.assertEqual(self.run_parser("It is 4.", False, step)[:2], ("", "It is 4."))

    def test_no_partial_marker_reaches_the_client(self):
        _, _, evs = self.run_parser("[THINK]a[/THINK]b", True, 1)
        for e in evs:
            self.assertNotIn("[", e.text or "")
            self.assertNotIn("]", e.text or "")

    def test_other_formats_unchanged(self):
        # the generic format still reads <think>...</think> and not Mistral's markers
        p = OutputParser(thinking=True, call_format="function")
        evs = p.feed("a</think>b") + p.finish()
        self.assertEqual("".join(e.text for e in evs if e.kind == "content"), "b")


class ServiceForMistral(unittest.TestCase):
    """TK6: the call format, and the efforts the template takes"""

    def setUp(self):
        tok = TekkenTokens()
        self.svc = Service(MockEngine(tok, "hi"), tok, load_chat_template(template(TEMPLATE), tok))
        self.user = [{"role": "user", "content": "hi"}]

    def effort_of(self, given):
        prompt = self.svc.render_prompt(self.user, None, effort_kwargs(given))
        return prompt.split('"reasoning_effort": "')[1].split('"')[0]

    def test_call_format_and_turns(self):
        self.assertEqual(self.svc.call_format, "mistral")
        self.assertEqual(Service(MockEngine(PixtralTokens(), "hi"), PixtralTokens(),
                                 load_chat_template(template(TEMPLATE), PixtralTokens())).call_format, "mistral")
        # as before P3 (when a Mistral pack fell into the generic format): no turn openers
        self.assertEqual(self.svc.turn_ids, frozenset())

    def test_efforts_map_to_none_or_high(self):
        for given, expected in ((None, "none"), ("", "none"), ("none", "none"), ("off", "none"), (False, "none"),
                                ("minimal", "none"), ("low", "high"), ("medium", "high"), ("high", "high"),
                                ("xhigh", "high"), ("max", "high")):
            with self.subTest(given=given):
                self.assertEqual(self.effort_of(given), expected)

    def test_other_models_keep_their_efforts(self):
        tok = Lfm2Tokens()
        svc = Service(MockEngine(tok, "hi"), tok, load_chat_template(template(
            "{{ reasoning_effort if reasoning_effort is defined else 'unset' }}|{{ enable_thinking }}"), tok))
        self.assertEqual(svc.call_format, "lfm2")
        self.assertEqual(svc.render_prompt(self.user, None, effort_kwargs("high")), "xhigh|True")
        self.assertEqual(svc.render_prompt(self.user, None, effort_kwargs("none")), "unset|False")


if __name__ == "__main__":
    unittest.main()

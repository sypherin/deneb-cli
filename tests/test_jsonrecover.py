"""Pure JSON recovery for a local model's agent-step output."""
from __future__ import annotations

import json

import pytest

from deneb import jsonrecover as jr


def test_strip_think_removes_closed_and_unclosed_blocks():
    assert jr.strip_think("<think>hmm {x}</think>answer").strip() == "answer"
    # only a block the response STARTS with is reasoning; the same tags later are content
    assert jr.strip_think("pre <think>a</think> mid <think>b</think> post") == "pre <think>a</think> mid <think>b</think> post"
    assert jr.strip_think("  \n<think>a</think> then <think>b</think> post").strip() == "then <think>b</think> post"
    assert jr.strip_think("<think>never closed, no answer").strip() == ""
    # a bare </think> is stripped only before one whole JSON object; before prose it is content
    assert jr.strip_think('reasoning only</think>{"type":"final","answer":"x"}').strip() == '{"type":"final","answer":"x"}'
    assert jr.strip_think("reasoning only</think>the answer") == "reasoning only</think>the answer"


def test_plain_json_action():
    r = jr.parse_step('{"type":"action","action":"run","cmd":"ollama ps","thought":"look"}')
    assert r["type"] == "action" and r["action"] == "run" and r["cmd"] == "ollama ps"


def test_a_fenced_json_object_alone_is_an_action():
    text = '```json\n{"type":"action","action":"read_file","path":"/etc/os-release"}\n```'
    r = jr.parse_step(text)
    assert r["action"] == "read_file" and r["path"] == "/etc/os-release"
    r = jr.parse_step('<think>hm</think>\n' + text)
    assert r["action"] == "read_file"


def test_prose_around_a_fenced_object_is_prose_not_an_action():
    text = 'Sure, here you go:\n```json\n{"type":"action","action":"read_file","path":"/etc/os-release"}\n```\nthanks'
    r = jr.parse_step(text)
    assert r["type"] == "final" and "Sure, here you go" in r["answer"]


def test_balanced_object_with_braces_inside_strings():
    text = '{"type":"final","answer":"use {curly} and \\"quotes\\" } here"}'
    r = jr.parse_step(text)
    assert r["type"] == "final" and r["answer"] == 'use {curly} and "quotes" } here'


def test_think_block_with_json_inside_is_ignored():
    text = '<think>maybe {"type":"action","action":"run","cmd":"rm -rf /"}</think>{"type":"final","answer":"ok"}'
    r = jr.parse_step(text)
    assert r["type"] == "final" and r["answer"] == "ok"


def test_action_final_alias_becomes_final():
    r = jr.parse_step('{"action":"final","answer":"done"}')
    assert r["type"] == "final" and r["answer"] == "done"


def test_missing_type_with_known_action_is_an_action():
    r = jr.parse_step('{"action":"list_dir","path":"/tmp"}')
    assert r["type"] == "action" and r["action"] == "list_dir"


def test_apply_actions_keep_content_and_why():
    r = jr.parse_step('{"type":"action","action":"write_file","path":"/a","content":"x","why":"w"}')
    assert (r["action"], r["path"], r["content"], r["why"]) == ("write_file", "/a", "x", "w")


def test_unknown_action_is_never_passed_through():
    r = jr.parse_step('{"type":"action","action":"delete_everything","cmd":"x"}')
    assert r["type"] == "final"
    assert "delete_everything" not in r["answer"]


def test_prose_answer_is_used_as_the_final_answer():
    r = jr.parse_step("Your service is stopped. Run: systemctl --user start llama")
    assert r["type"] == "final" and "stopped" in r["answer"]
    assert r["escalated"] is False and r["sources"] == []


def test_malformed_json_never_dumps_raw_json_at_the_user():
    r = jr.parse_step('{"type":"action","action":"run","cmd":"ollama ps"')
    assert r["type"] == "final"
    assert "{" not in r["answer"] and '"cmd"' not in r["answer"]
    assert r["answer"].strip()


def test_empty_output_gets_a_plain_message():
    r = jr.parse_step("   ")
    assert r["type"] == "final" and r["answer"].strip()
    r = jr.parse_step("<think>only thinking</think>")
    assert r["type"] == "final" and r["answer"].strip()


def test_non_string_fields_are_coerced_not_crashed():
    r = jr.parse_step('{"type":"action","action":"run","cmd":["ollama","ps"]}')
    assert r["type"] == "action" and isinstance(r["cmd"], str)


def test_final_answer_missing_gets_message_not_none():
    r = jr.parse_step('{"type":"final"}')
    assert r["type"] == "final" and r["answer"].strip()


# ── S13: reasoning is stripped only at the START of the response ──────────────────────
def test_think_tags_inside_a_json_string_survive():
    r = jr.parse_step('{"answer":"the model writes <think> blocks first"}')
    assert r == {"type": "final", "answer": "the model writes <think> blocks first",
                 "escalated": False, "sources": []}
    r = jr.parse_step('{"type":"final","answer":"reasoning runs up to </think> in your client."}')
    assert r["answer"] == "reasoning runs up to </think> in your client."


def test_leading_think_block_then_json_still_parses():
    r = jr.parse_step('  \n<think>maybe {"action":"run"} </think>\n{"type":"final","answer":"ok </think> ok"}')
    assert r["answer"] == "ok </think> ok"


# ── S14: an action needs the WHOLE response to be one JSON object ──────────────────────
def test_prose_with_an_embedded_action_is_shown_as_prose():
    text = ('You could restart it with {"type":"action","action":"fix",'
            '"cmd":"systemctl --user restart llama"} but I would not.')
    r = jr.parse_step(text)
    assert r["type"] == "final" and r["answer"] == text


def test_truncated_final_with_steps_is_text_with_a_cut_off_line_and_runs_nothing():
    text = ('{"type":"final","answer":"I will look at the unit","steps":'
            '[{"type":"action","action":"run","cmd":"ls /"}')
    r = jr.parse_step(text)
    assert r["type"] == "final"
    assert "cut off" in r["answer"]
    assert '"action"' not in r["answer"] and "{" not in r["answer"]


def test_truncated_action_is_never_run_and_says_cut_off():
    r = jr.parse_step('{"type":"action","action":"fix","cmd":"systemctl --user restart x')
    assert r["type"] == "final" and "cut off" in r["answer"] and "systemctl" not in r["answer"]


def test_nested_actions_inside_a_complete_final_are_not_promoted():
    r = jr.parse_step('{"type":"final","answer":"done","steps":[{"action":"run","cmd":"ls"}]}')
    assert r["type"] == "final" and r["answer"] == "done"


def test_a_clean_action_still_parses():
    r = jr.parse_step('{"type":"action","action":"run","cmd":"ollama ps","thought":"look"}')
    assert (r["type"], r["action"], r["cmd"]) == ("action", "run", "ollama ps")


# ── N1 and N2 ────────────────────────────────────────────────────────────────────────
def test_list_content_is_joined_with_newlines():
    r = jr.parse_step('{"type":"action","action":"write_file","path":"/a","content":["[Service]","ExecStart=/usr/bin/x"]}')
    assert r["content"] == "[Service]\nExecStart=/usr/bin/x"


def test_list_cmd_is_one_command_line_not_newlines():
    r = jr.parse_step('{"type":"action","action":"run","cmd":["ollama","ps"]}')
    assert r["cmd"] == "ollama ps"


def test_answer_that_is_an_object_is_shown_as_readable_text():
    r = jr.parse_step('{"type":"final","answer":{"cause":"cpu fallback","fix":"rebuild"}}')
    assert "{" not in r["answer"] and "cause: cpu fallback" in r["answer"] and "fix: rebuild" in r["answer"]


def test_answer_aliases_are_honoured():
    for key in ("response", "message", "content", "output", "result", "reply"):
        r = jr.parse_step(json.dumps({"type": "final", key: "the text"}))
        assert r["answer"] == "the text", key


def test_a_top_level_json_string_is_shown_without_quotes():
    assert jr.parse_step('"just text"')["answer"] == "just text"


@pytest.mark.parametrize("text", [
    "{'type': 'final', 'answer': 'single quoted'}",
    '{"type":"final","answer":"trailing comma",}',
])
def test_single_quotes_and_trailing_commas_are_repaired(text):
    r = jr.parse_step(text)
    assert r["type"] == "final" and "{" not in r["answer"] and r["answer"] != jr.GARBLED
    assert r["answer"] in ("single quoted", "trailing comma")

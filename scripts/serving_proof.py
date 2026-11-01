"""Proof of the weights a model server is actually serving, not the name it answers to.

Nov 1 incident: from the 08:02 deploy, :8000 served run 1's directory (r3-voice-206506525c) under run 2's name
(r3-voice-730f8ba68e) for over two hours; /v1/models and .env both said run 2. Three independent lines, printed and
written to the ledger under the block's run id:
  1. path       the model directory in the listener's command line (ps), compared with --expect-dir when given
  2. name       what /v1/models answers; it only echoes --served-model-name, so it is never proof on its own
  3. behaviour  one scripted booking turn, greedy, straight at the server: Dana verified by the sandbox's own lookup, then
                "a mortgage consultation Tuesday morning at the Market Street branch". Run 1 re-asked the weekday here in
                every live call of the 09:59 block; whether run 2 does is measured, not assumed (recorded at each deploy).
Usage: uv run python scripts/serving_proof.py --run-id <block id> [--expect-dir <model dir>] [--port 8000] [--no-ledger]
Exit 1 when --expect-dir is given and the served directory differs."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEED = json.loads((ROOT / "eval/sandbox/seed.json").read_text())
SYSTEM_PROMPT = (ROOT / "schemas/system_prompt.md").read_text().replace("{today}", SEED["clock_spoken"])
TOOLS = [json.loads(p.read_text()) for p in sorted((ROOT / "schemas/tools").glob("*.json"))]
GREETING = "Thanks for calling Marlow Mutual Bank. Can I get the phone number on your account?"
BOOKING_LINE = "I'd like to book a mortgage consultation Tuesday morning at the Market Street branch."

_CMD_PATTERNS = (
    re.compile(r"(?:vllm_mlx_serve\.py|vllm-mlx|vllm_mlx)\s+serve\s+(\S+)"),
    re.compile(r"mlx_lm[._]server\b.*?--model\s+(\S+)"),
)
_REASK = re.compile(r"\b(which|what)\s+(day|date)\b", re.I)
_RAW_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def model_dir_from_cmdline(cmd: str) -> str | None:
    for pat in _CMD_PATTERNS:
        m = pat.search(cmd)
        if m:
            return m.group(1)
    return None


def expected_name(model_dir: str) -> str:
    """The served name a directory must carry: its basename, the base model as r3-voice-base (CONTRACT.md model ids)."""
    base = os.path.basename(os.path.normpath(model_dir))
    return "r3-voice-base" if base == "Qwen3-4B-8bit" else base


def reasks_weekday(text: str, tool_calls: list) -> bool:
    """The run-1 signature in this turn: no tool call, and a question for the day the caller has just given."""
    return not tool_calls and bool(_REASK.search(text or ""))


def fingerprint_messages() -> list[dict]:
    """The 09:59 live call up to the booking request, with the sandbox's real lookup result (the executor's own serialization)."""
    from services.tools.sandbox import Sandbox

    args = {"phone": "5550142219", "date_of_birth": "1988-03-12"}
    result = Sandbox(json.loads(json.dumps(SEED))).lookup_account(**args)
    call = {"id": "call_proof_1", "type": "function", "function": {"name": "lookup_account", "arguments": json.dumps(args)}}
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": GREETING},
        {"role": "user", "content": "Five five five. Zero one four two two one nine."},
        {"role": "assistant", "content": "Thanks. What's your date of birth?"},
        {"role": "user", "content": "March twelfth, nineteen eighty eight."},
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, separators=(",", ":"))},
        {"role": "assistant", "content": "Thanks, Dana, you're verified. How can I help you today?"},
        {"role": "user", "content": BOOKING_LINE},
    ]


def _args(a):
    if isinstance(a, str):
        try:
            return json.loads(a)
        except json.JSONDecodeError:
            return a
    return a


def parse_reply(resp: dict) -> tuple[str, list[dict]]:
    msg = resp["choices"][0]["message"]
    calls = [{"name": c["function"]["name"], "arguments": _args(c["function"].get("arguments"))} for c in (msg.get("tool_calls") or [])]
    text = msg.get("content") or ""
    for raw in _RAW_CALL.findall(text):
        try:
            d = json.loads(raw)
            calls.append({"name": d.get("name"), "arguments": _args(d.get("arguments"))})
        except json.JSONDecodeError:
            pass
    text = re.sub(r"<think>.*?</think>", "", _RAW_CALL.sub("", text), flags=re.S).strip()
    return text, calls


def listener_cmdline(port: int) -> str:
    pid = subprocess.run(["lsof", f"-ti:{port}", "-sTCP:LISTEN"], capture_output=True, text=True).stdout.split()
    if not pid:
        return ""
    return subprocess.run(["ps", "-o", "command=", "-p", pid[0]], capture_output=True, text=True).stdout.strip()


def main() -> int:
    import httpx
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--expect-dir")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-ledger", action="store_true")
    ap.add_argument("--expect-name-from-dir", action="store_true", help="fail unless /v1/models names the served directory (rehearsal blocks label rows by that name)")
    a = ap.parse_args()

    cmd = listener_cmdline(a.port)
    served_dir = model_dir_from_cmdline(cmd) or "?"
    match = a.expect_dir is None or os.path.realpath(served_dir) == os.path.realpath(a.expect_dir)
    base = f"http://127.0.0.1:{a.port}"
    try:
        ids = [m["id"] for m in httpx.get(f"{base}/v1/models", timeout=10).json().get("data", [])]
    except Exception as e:
        ids = [f"error: {e!r}"]
    served = ids[0] if ids and not ids[0].startswith("error") else "unknown"
    try:
        r = httpx.post(f"{base}/v1/chat/completions", timeout=120, json={
            "model": served, "messages": fingerprint_messages(), "tools": TOOLS, "temperature": 0, "max_tokens": 60, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}})
        r.raise_for_status()
        text, calls = parse_reply(r.json())
        reask = reasks_weekday(text, calls)
        said = json.dumps(calls[0]) if calls else text
        verdict = "re-asks the weekday (the run-1 signature)" if reask else "does not re-ask the weekday"
    except Exception as e:
        reask, said, verdict = None, f"error: {e!r}", "fingerprint request failed"
    print(f"proof path:      {served_dir}" + ("" if a.expect_dir is None else f"  (expected {a.expect_dir}: {'MATCH' if match else 'MISMATCH'})"))
    print(f"proof name:      /v1/models -> {', '.join(ids)}" + ("" if served == expected_name(served_dir) else f"  (the directory's name is {expected_name(served_dir)}: MISMATCH)"))
    print(f"proof behaviour: {verdict}: {said[:160]}")
    if not a.no_ledger:
        import psycopg

        with psycopg.connect(os.environ["DATABASE_URL"]) as con:
            rows = [
                ("proof.serving_path", 1 if match else 0, "check", f"path={served_dir}; expected={a.expect_dir or '-'}; cmd={cmd[:300]}"),
                ("proof.v1_models", int(served == expected_name(served_dir)), "check", f"ids={','.join(ids)}; directory name={expected_name(served_dir)}"),
                ("proof.fingerprint_reasks_weekday", -1 if reask is None else int(reask), "flag", f"{verdict}: {said[:300]}"),
            ]
            for item, qty, unit, note in rows:
                con.execute("insert into cost_ledger(run_id, model_id, item, qty, unit, measured, note) values (%s, %s, %s, %s, %s, true, %s)",
                            (a.run_id, served, item, qty, unit, note))
        print(f"ledger: proof rows written under {a.run_id}")
    if not match:
        print(f"SERVING PATH MISMATCH: {served_dir} is not {a.expect_dir}", file=sys.stderr)
        return 1
    if a.expect_name_from_dir and served != expected_name(served_dir):
        print(f"NAME/WEIGHTS MISMATCH: /v1/models says {served} but the process serves {served_dir} ({expected_name(served_dir)})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

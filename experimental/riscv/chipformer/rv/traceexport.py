"""Trace export (phase 3): console transcripts -> tune data.

Every agent session on the console is a training episode: the task,
each (situation -> reply) turn, and the final answer. The gaps log
and these traces are what the seat tunes train on (see
docs/agent-framework.md: the gap log doubles as training data).

    python3 -m chipformer.rv.traceexport [--out episodes.jsonl]

Pulls the live transcript over HTTP (read-only) and writes one JSON
episode per finished session: {"task": ..., "turns":
[{"situation": ... (prompt tail: TASK+RESULT, capped), "reply": ...}],
"answer": ..., "rounds": ...}. Prompts are stored tail-only: the SYS
prefix repeats every turn and teaches nothing.
"""

import argparse
import json
import os
import urllib.request

TAIL = 800  # chars of prompt tail kept per turn


def _new_ep(task, sub):
    return {"task": task, "turns": [], "answer": None, "rounds": 0,
            "sub": sub}


def parse_episodes(text):
    """Split a console transcript into finished agent episodes.

    A delegated child's stdout is captured, so its start line and its
    AGENT_ANSWER never reach the console - but its QWEN: turns do, and
    they stay inside the parent's episode in order (full causal
    history: the right training shape). An ask-delegated child is
    bracketed by ask.sh's own console lines instead. Only prompts
    carrying "TASK: " count as agent turns, so a terminal ask's QWENs
    never pollute an episode.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    episodes = []
    stack = []
    cur = None
    pending_prompt = None
    for line in text.split("\n"):
        if line.startswith("[agent] start: ") or \
                line.startswith("[agent] start (sub-agent): "):
            # top-level starts can't nest (the console lock); a start
            # while one is open means the old one died unfinished
            task = line.split(": ", 1)[1].rsplit(" (", 1)[0]
            cur = _new_ep(task, "sub-agent" in line)
            pending_prompt = None
        elif line.startswith("[ask] the model delegated this to "
                             "hermes-agent: "):
            if cur is not None:
                stack.append(cur)
            task = line.split(": ", 1)[1]
            cur = _new_ep(task, True)
            pending_prompt = None
        elif line.startswith("[ask] hermes-agent reported: "):
            if cur is not None and cur["sub"]:
                cur["answer"] = line.split(": ", 1)[1].strip() or \
                    "(empty report)"
                cur["rounds"] = len(cur["turns"])
                episodes.append(cur)
                cur = stack.pop() if stack else None
                pending_prompt = None
        elif cur is not None and line.startswith("QWEN:"):
            prompt = line[len("QWEN:"):]
            pending_prompt = prompt if ("TASK: " in prompt or
                                        prompt.startswith("FETCHURL ")) \
                else None
        elif cur is not None and line.startswith("QWENANS:"):
            if pending_prompt is not None:
                cur["turns"].append({
                    "situation": pending_prompt[-TAIL:],
                    "reply": line[len("QWENANS:"):]})
                pending_prompt = None
        elif cur is not None and line.startswith("AGENT_ANSWER:"):
            cur["answer"] = line.split("AGENT_ANSWER:", 1)[1].strip()
            cur["rounds"] = len(cur["turns"])
            episodes.append(cur)
            cur = stack.pop() if stack else None
            pending_prompt = None
    return [e for e in episodes if e["answer"]]


def export(base, outpath, max_bytes=2_000_000, infile=None):
    if infile:
        with open(infile, "rb") as f:
            data = f.read()[:max_bytes].decode("utf-8", "replace")
    else:
        with urllib.request.urlopen(
                "%s/api/out?since=0" % base, timeout=30) as r:
            data = json.loads(r.read())["data"][:max_bytes]
    episodes = parse_episodes(data)
    with open(outpath, "w") as f:
        for e in episodes:
            f.write(json.dumps(e) + "\n")
    turns = sum(len(e["turns"]) for e in episodes)
    return "ok: %d episodes, %d turns -> %s" % (
        len(episodes), turns, outpath)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8951")
    ap.add_argument("--out", default=os.path.expanduser(
        "~/qwenputer-test/episodes.jsonl"))
    ap.add_argument("--file", default=None,
                    help="parse a saved transcript instead of HTTP")
    a = ap.parse_args(argv)
    print(export(a.base, a.out, infile=a.file))


if __name__ == "__main__":
    main()

"""Host-side mind mirror (phase 3): the guest's mind survives reboots.

The agent's memory, jobs, plans, skills and apps live in guest RAM:
a cold reboot without a snapshot is amnesia. This tool snapshots
them to the host through the live console (read-only: one shell
command when the console is idle at its prompt) and drips them back
after a reboot.

    python3 -m chipformer.rv.mindmirror snapshot [--out DIR]
    python3 -m chipformer.rv.mindmirror restore [--in DIR] [--dry-run]

Snapshot only runs when the console is idle (tail sits at ``agent>``)
so it never interleaves a live session; otherwise it skips with a
message. Restore writes files with chunked ``printf`` octal lines
(every install line stays far under the ~1021-char tty limit).
"""

import argparse
import json
import os
import time
import urllib.request

BEGIN = "MINDMIR_BEGIN"
END = "MINDMIR_END"
RESTORED = "MINDMIR_RESTORED"
SNAPSHOT_CMD = (
    "echo %s; "
    "for f in /root/BRAIN.md /root/FAILURES.md /root/SOUL.md /root/jobs "
    "/root/jobs.log /root/history /root/HEARTBEAT.md /root/PLAN.md; do "
    'echo "===FILE $f"; cat "$f" 2>/dev/null; done; '
    "for f in /root/nb/* /root/skills/*.md /root/apps/*; do "
    '[ -f "$f" ] || continue; echo "===FILE $f"; cat "$f"; done; '
    "echo %s" % (BEGIN, END)
)
RESTORE_CAP = 8192  # bytes per file; beyond this restore truncates
CHUNK_BYTES = 60    # bytes per printf line (~240 chars, tty-safe)


def _get(base, since):
    with urllib.request.urlopen(
            "%s/api/out?since=%d" % (base, since), timeout=20) as r:
        return json.loads(r.read())


def _post(base, data):
    req = urllib.request.Request(
        base + "/api/in", data=json.dumps({"data": data}).encode(),
        headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=10).read()


def parse_snapshot(text):
    """Parse a snapshot transcript slice into {guest path: content}.

    Markers are whole lines: the echoed command line itself contains
    both marker words inline, so substring matching would stop at the
    echo and parse nothing.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    try:
        b = next(i for i, l in enumerate(lines) if l.strip() == BEGIN)
        e = next(i for i, l in enumerate(lines)
                 if l.strip() == END and i > b)
    except StopIteration:
        raise ValueError("snapshot markers missing")
    files = {}
    cur = None
    buf = []
    for line in lines[b + 1:e]:
        if line.startswith("===FILE "):
            if cur is not None:
                files[cur] = "\n".join(buf)
            cur = line[len("===FILE "):].strip()
            buf = []
        elif cur is not None:
            buf.append(line)
    if cur is not None:
        files[cur] = "\n".join(buf)
    # every section ends with the newline before the next header
    return {k: v + "\n" if v and not v.endswith("\n") else v
            for k, v in files.items() if k.startswith("/root/")}


def chunk_restore(path, content):
    """Shell lines that recreate a guest file (pure: no I/O)."""
    data = content.encode("utf-8", "replace")[:RESTORE_CAP]
    lines = ["mkdir -p $(dirname '%s'); : > '%s'" % (path, path)]
    for i in range(0, len(data), CHUNK_BYTES):
        octs = "".join("\\%03o" % b for b in data[i:i + CHUNK_BYTES])
        lines.append("printf '%s' >> '%s'" % (octs, path))
    return lines


def snapshot(base, outdir, idle_timeout=60):
    mark = _get(base, 0)["until"]
    data = _get(base, 0)["data"]
    if "ASK_READY" not in data or not data.rstrip().endswith("agent>"):
        return "skip: console is not idle at its prompt"
    _post(base, SNAPSHOT_CMD + "\n")
    deadline = time.time() + 120
    text = ""
    while time.time() < deadline:
        time.sleep(3)
        d = _get(base, mark)
        mark = d["until"]
        text += d["data"].replace("\r\n", "\n")
        if any(l.strip() == END for l in text.split("\n")):
            break
    else:
        return "fail: end marker never arrived"
    files = parse_snapshot(text)
    os.makedirs(outdir, exist_ok=True)
    for path, content in files.items():
        dest = os.path.join(outdir, path.lstrip("/"))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w") as f:
            f.write(content)
    with open(os.path.join(outdir, "SNAPSHOT.txt"), "w") as f:
        f.write("taken: %s\nfiles: %d\n%s\n" % (
            time.strftime("%F %T"), len(files),
            "\n".join(sorted(files))))
    return "ok: %d files -> %s" % (len(files), outdir)


def restore(base, indir, dry_run=False):
    todo = []
    for dirpath, _, names in os.walk(indir):
        for n in names:
            if n == "SNAPSHOT.txt":
                continue
            full = os.path.join(dirpath, n)
            guest = "/" + os.path.relpath(full, indir)
            with open(full) as f:
                todo.append((guest, f.read()))
    todo.sort()
    lines = []
    for path, content in todo:
        lines.extend(chunk_restore(path, content))
    lines.append("echo %s" % RESTORED)
    if dry_run:
        return lines
    mark = _get(base, 0)["until"]
    for line in lines:
        _post(base, line + "\n")
        time.sleep(0.4)
    deadline = time.time() + 180
    text = ""
    while time.time() < deadline:
        time.sleep(3)
        d = _get(base, mark)
        mark = d["until"]
        text += d["data"].replace("\r\n", "\n")
        if any(l.strip() == RESTORED for l in text.split("\n")):
            break
    else:
        return "fail: restore marker never arrived"
    return "ok: %d files restored" % len(todo)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("snapshot", "restore"))
    ap.add_argument("--base", default="http://127.0.0.1:8951")
    ap.add_argument("--out", default=os.path.expanduser(
        "~/qwenputer-test/mind"))
    ap.add_argument("--in", dest="indir", default=os.path.expanduser(
        "~/qwenputer-test/mind"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "snapshot":
        print(snapshot(a.base, a.out))
    else:
        r = restore(a.base, a.indir, dry_run=a.dry_run)
        if isinstance(r, list):
            print("\n".join(r))
        else:
            print(r)


if __name__ == "__main__":
    main()

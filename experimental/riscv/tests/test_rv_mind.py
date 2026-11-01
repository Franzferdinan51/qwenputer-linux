"""Phase 1-3 guards: new harness verbs/branches exist in the guest
scripts, the host fetch wrapper behaves (against a local HTTP
server, no internet), and the mirror/trace pure functions round-trip.
Behavioral sims live in /tmp/mind_sim.sh; the 10-task ground-truth
battery is chipformer.rv.battery (needs a guest image).

Web UI guards: the sysstat parser, the mind reader over a fixture
mirror, the new settings keys, and the page's element-id contract
(every id the script touches exists in the markup)."""

import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import chipformer.rv.liveterm as LT
from chipformer.rv import dashboard as DB
from chipformer.rv import qwenbackend as qb
from chipformer.rv.mindmirror import chunk_restore, parse_snapshot
from chipformer.rv.traceexport import parse_episodes

HERE = Path(__file__).parent.parent / "chipformer" / "rv"
AGENT = (HERE / "env" / "linux" / "agent.sh").read_text()
ASHELL = (HERE / "env" / "linux" / "agentshell.sh").read_text()


def test_agent_has_think_scratchpad():
    assert '"THINK:"*|"THINK "*)' in AGENT
    assert "maxthinks=6" in AGENT
    assert "round=$((round - 1))" in AGENT
    assert "SCRATCH (your own notes so far)" in AGENT


def test_agent_has_critique_revise_stages():
    assert "pending_stage=1" in AGENT
    assert "pending_stage=2" in AGENT
    assert "TASK: critique." in AGENT
    assert "TASK: revise." in AGENT


def test_agent_memory_is_structured_with_forget():
    assert '*) fact="[fact] $fact" ;;' in AGENT
    assert '"in:"*)' in AGENT
    assert "TOOL forget " in AGENT
    assert "/root/FAILURES.md" in AGENT
    assert "PITFALLS (do not repeat these)" in AGENT


def test_agent_has_delegate_plan_fetch():
    assert "TOOL delegate " in AGENT
    assert "AGENT_DEPTH=1 AGENTD=1" in AGENT
    assert "one level only" in AGENT
    assert "TOOL plan " in AGENT
    assert "Pending plan steps" in AGENT
    assert "TOOL fetch " in AGENT
    assert "FETCHURL $url" in AGENT


def test_agent_failure_logging_and_depth():
    assert "log_failure()" in AGENT
    assert "DEPTH=${AGENT_DEPTH:-0}" in AGENT
    assert "maxround=6" in AGENT  # children get the short budget


def test_agent_lines_stay_tty_safe():
    bad = [i + 1 for i, l in enumerate(AGENT.splitlines())
           if len(l) > 900]
    assert bad == [], bad


def test_agentshell_seeds_upkeep():
    assert "seed_upkeep" in ASHELL
    assert "MEMORY UPKEEP" in ASHELL
    assert "1440|0|never run" in ASHELL


def test_liveterm_installs_seed_skills_guarded():
    assert "SKILL_UPKEEP" in Path(LT.__file__).read_text()
    assert LT.SKILL_UPKEEP.startswith("# skill: upkeep\n")
    assert LT.SKILL_PLAN.startswith("# skill: plan-long-task\n")
    for s in (LT.SKILL_UPKEEP, LT.SKILL_PLAN):
        assert s.endswith("\n")
        assert all(len(l) < 700 for l in s.splitlines())


def test_fetch_wrapper_live_server():
    body = ("<html>marker-word " + "z" * 9000 + "</html>").encode()

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = "http://127.0.0.1:%d/t" % srv.server_port
        out = qb.fetch_page(url)
        assert out.startswith("<html>marker-word"), out[:40]
        assert len(out) <= 2000 and "\n" not in out
        assert qb.fetch_page("file:///etc/passwd").startswith(
            "[fetch: refused")
        assert qb.fetch_page("https://u:p@h/").startswith("[fetch: refused")
        assert qb.fetch_page("http://127.0.0.1:1/x").startswith(
            "[fetch: failed")
        w = qb.fetch_backend_wrapper(lambda p: "MODEL")
        assert w("FETCHURL " + url).startswith("<html>")
        assert w("plain question") == "MODEL"
    finally:
        srv.shutdown()


def test_mirror_parse_and_restore_roundtrip():
    fixture = ("agent> echo MINDMIR_BEGIN\r\nMINDMIR_BEGIN\r\n"
               "===FILE /root/BRAIN.md\r\n[fact] a\r\n"
               "===FILE /root/PLAN.md\r\n- [ ] s\r\n"
               "MINDMIR_END\r\nagent> \n")
    files = parse_snapshot(fixture)
    assert files == {"/root/BRAIN.md": "[fact] a\n",
                     "/root/PLAN.md": "- [ ] s\n"}
    try:
        parse_snapshot("nothing here")
    except ValueError:
        pass
    else:
        raise AssertionError("missing markers must raise")
    content = "a'b\"c$d `e` f\\g\n" + "0123456789\n" * 20
    lines = chunk_restore("/root/BRAIN.md", content)
    assert all(len(l) < 900 for l in lines)
    d = tempfile.mkdtemp()
    adj = [l.replace("/root/", d + "/") for l in lines]
    p = d + "/do.sh"
    open(p, "w").write("\n".join(adj) + "\n")
    subprocess.run(["sh", p], check=True)
    assert open(d + "/BRAIN.md").read() == content


def test_trace_parser_folds_delegate_turns_in_order():
    # the child's stdout is captured (no start/answer on the console),
    # so its turns belong to the parent's episode, in order
    t = ("[agent] start: p (15 rounds max)\n"
         "QWEN:SYS TASK: p\nQWENANS:TOOL delegate c\n"
         "QWEN:SYS TASK: c\nQWENANS:SAY: crep\n"
         "QWEN:SYS TASK: p RESULT: sub-agent report: crep "
         "Now reply with EXACTLY ONE line: x\nQWENANS:SAY: pdone\n"
         "AGENT_ANSWER: pdone\n"
         "QWEN:terminal question, no task marker\nQWENANS:ignored\n")
    eps = parse_episodes(t)
    assert len(eps) == 1
    assert eps[0]["answer"] == "pdone"
    assert [x["reply"] for x in eps[0]["turns"]] == [
        "TOOL delegate c", "SAY: crep", "SAY: pdone"]


def test_trace_parser_brackets_ask_children():
    t = ("[ask] the model delegated this to hermes-agent: do file work\n"
         "QWEN:SYS TASK: do file work\nQWENANS:SAY: filed\n"
         "[ask] hermes-agent reported: filed\n"
         "QWEN:orig HERMES REPORT: filed\nQWENANS:final\n")
    eps = parse_episodes(t)
    assert len(eps) == 1
    assert eps[0]["answer"] == "filed" and eps[0]["sub"] is True
    assert len(eps[0]["turns"]) == 1


def test_sysstat_parser_reads_live_shaped_reply():
    text = ("QWSYSB\r\nMemTotal:          62836 kB\r\n"
            "MemFree:           55512 kB\r\n"
            "MemAvailable:      54148 kB\r\n"
            "cpu  11604 0 99452 235862 0 0 1028 0 0 0\r\n"
            "0.57 0.28 0.24 1/17 337\r\nQWSYSE\r\n")
    p = DB._parse_sysstat(text)
    assert p["mem_total_kb"] == 62836
    assert p["mem_avail_kb"] == 54148
    assert p["cpu"] == (11604 + 99452 + 235862 + 1028, 235862)
    assert p["load1"] == 0.57
    try:
        DB._parse_sysstat("nothing here")
    except ValueError:
        pass
    else:
        raise AssertionError("missing markers must raise")


def test_harvest_last_fills_mem_not_cpu():
    DB._SYS.update({"at": 0, "mem_total_kb": 0, "cpu_pct": None,
                    "note": "no sample yet"})
    text = ("QWSYSB\nMemTotal: 62836 kB\nMemAvailable: 54148 kB\n"
            "cpu  1 0 2 100 0 0 0 0\n0.50 0.1 0.1 1/1 1\nQWSYSE\n")
    assert DB._harvest_last(text, "stale") is True
    assert DB._SYS["mem_total_kb"] == 62836
    assert DB._SYS["load1"] == 0.50
    assert DB._SYS["cpu_pct"] is None  # needs a fresh delta pair
    assert DB._SYS["note"] == "stale"
    assert DB._harvest_last("nothing here", "x") is False
    DB._SYS.update({"at": 0, "mem_total_kb": 0, "mem_avail_kb": 0,
                    "load1": None, "note": "no sample yet"})


def test_mind_reader_over_fixture_mirror(tmp_path):
    root = tmp_path / "root"
    (root / "nb").mkdir(parents=True)
    (root / "skills").mkdir()
    (root / "apps").mkdir()
    (tmp_path / "SNAPSHOT.txt").write_text("taken: 2026-11-01 12:00:00\n")
    (root / "jobs").write_text("9999999999|45|0|never run|probe job\n")
    (root / "BRAIN.md").write_text("[fact] a\n[lesson] b\n")
    (root / "FAILURES.md").write_text("f1\n")
    (root / "PLAN.md").write_text("- [x] done\n- [ ] todo\n")
    (root / "nb" / "2026-11-01.md").write_text("n\n")
    (root / "skills" / "s1.md").write_text("# skill: s1\nsaved: x\nbody one\n")
    (root / "apps" / "ap1").write_text("#!/bin/sh\n# built\n echo hi\n")
    (root / "agentd.off").write_text("")
    m = DB._read_mind(str(tmp_path))
    assert m["present"] and m["taken"] == "2026-11-01 12:00:00"
    assert len(m["jobs"]) == 1 and m["jobs"][0]["task"] == "probe job"
    assert m["jobs"][0]["due_in_s"] > 0
    assert m["brain_lines"] == 2 and m["failures_lines"] == 1
    assert m["nb_days"] == 1
    assert m["skills"] == [{"name": "s1", "preview": "body one"}]
    assert m["apps"] == [{"name": "ap1", "preview": "echo hi"}]
    assert (m["plan_total"], m["plan_pending"]) == (2, 1)
    assert m["upkeep_paused"] is True
    assert DB._read_mind(str(tmp_path / "nope"))["present"] is False


def test_settings_new_keys_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(DB, "SETTINGS_PATH", str(tmp_path / "s.json"))
    s = DB.save_settings({"guest_enabled": False, "guest_budget": 90,
                          "temperature": 0.2, "max_tokens": 100})
    assert s["guest_enabled"] is False and s["guest_budget"] == 90
    s = DB.save_settings({"guest_budget": 99999})
    assert s["guest_budget"] == 3600
    s = DB.save_settings({"guest_budget": 5})
    assert s["guest_budget"] == 60


def test_page_id_contract_and_tabs():
    import re
    need = set(re.findall(r"\$\('([^']+)'\)", DB.PAGE))
    ids = re.findall(r'id="([^"]+)"', DB.PAGE)
    assert len(ids) == len(set(ids)), "duplicate ids break getElementById"
    have = set(ids)
    assert need - have == set()
    for tab in ("overview", "activity", "chat", "terminal", "desktop",
                "apps", "system", "settings"):
        assert 'data-t="%s"' % tab in DB.PAGE
        assert 'id="%s"' % tab in DB.PAGE
    for wid in ("gauge-cpu", "gauge-ram", "spark", "builtapps",
                "skills", "jobs", "upstream", "viztip", "t-cpu", "t-ram",
                "s-guest", "s-budget", "actsnap", "acttrace", "actupkeep",
                "vizbig", "vizpause", "vizreset", "vizstats", "entryapps"):
        assert 'id="%s"' % wid in DB.PAGE, wid

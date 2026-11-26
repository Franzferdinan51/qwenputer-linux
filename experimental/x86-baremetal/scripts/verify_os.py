#!/usr/bin/env python3
"""
Phase 2 Bare-Metal OS Verification, Determinism, Benchmark, REPL & VGA Capture Suite.
"""
import argparse
import os
import re
import select
import socket
import subprocess
import sys
import time
from pathlib import Path

MODEL_DEFAULT = "models/qwen3-0.6b-q8_0.asmbin"
PHASE1_BIN = "./build/asm-llm"
OS_ELF = "./build/asm-llm-os.elf"


def run_phase1(model: str, prompt: str, max_tokens: int, think: bool = False, threads: int = 32):
    cmd = [
        PHASE1_BIN,
        "-m", model,
        "-p", prompt,
        "-n", str(max_tokens),
        "-t", str(threads),
        "--print-tokens",
    ]
    if think:
        cmd.append("--think")
    res = subprocess.run(cmd, capture_output=True, timeout=60, check=True)
    stdout_text = res.stdout.decode("utf-8", errors="replace").rstrip("\n")
    stderr_text = res.stderr.decode("utf-8", errors="replace")
    m_p = re.search(r"PROMPT_TOKENS:([0-9,]+)", stderr_text)
    m_g = re.search(r"GEN_TOKENS:([0-9,]+)", stderr_text)
    return {
        "text": stdout_text,
        "prompt_tokens": m_p.group(1) if m_p else "",
        "gen_tokens": m_g.group(1) if m_g else "",
        "stderr": stderr_text,
    }


def parse_os_output(raw_bytes: bytes):
    text = raw_bytes.decode("utf-8", errors="replace")
    # Extract markers
    m_smp = re.search(r"SMP\s*:\s*(\d+)/(\d+)\s+x86-64 cores online", text)
    m_boot = re.search(r"\[asm-llm-os\] boot\s*:\s*ready\s+([0-9.]+)\s+ms\s*\|\s*boot->1st-tok\s+([0-9.]+)\s+ms", text)
    m_pf = re.search(r"\[asm-llm-os\] prefill:\s*(\d+)\s+tok in\s+([0-9.]+)\s+ms\s*\(([0-9.]+)\s+tok/s\)", text)
    m_dec = re.search(r"\[asm-llm-os\] decode\s*:\s*(\d+)\s+tok in\s+([0-9.]+)\s+ms\s*\(([0-9.]+)\s+tok/s,\s*cores=(\d+)\)(?:\s*ctx=(\d+)/(\d+))?", text)
    m_p = re.search(r"PROMPT_TOKENS:([0-9,]+)", text)
    m_g = re.search(r"GEN_TOKENS:([0-9,]+)", text)

    # Extract generated text between PROMPT_TOKENS line (or Prompt: line) and GEN_TOKENS (or [asm-llm-os] prefill:)
    if m_p:
        start_idx = text.find("\n", m_p.end()) + 1
    else:
        m_pr = re.search(r"Prompt: .*\n", text)
        start_idx = m_pr.end() if m_pr else 0
    if m_g:
        end_idx = m_g.start()
    elif m_pf:
        end_idx = m_pf.start()
    else:
        end_idx = len(text)
    gen_text = text[start_idx:end_idx].rstrip("\n")

    return {
        "raw": text,
        "text": gen_text,
        "prompt_tokens": m_p.group(1) if m_p else "",
        "gen_tokens": m_g.group(1) if m_g else "",
        "smp_online": int(m_smp.group(1)) if m_smp else 0,
        "smp_total": int(m_smp.group(2)) if m_smp else 0,
        "boot_ready_ms": float(m_boot.group(1)) if m_boot else 0.0,
        "boot_ttft_ms": float(m_boot.group(2)) if m_boot else 0.0,
        "prefill_tok": int(m_pf.group(1)) if m_pf else 0,
        "prefill_ms": float(m_pf.group(2)) if m_pf else 0.0,
        "prefill_tps": float(m_pf.group(3)) if m_pf else 0.0,
        "decode_tok": int(m_dec.group(1)) if m_dec else 0,
        "decode_ms": float(m_dec.group(2)) if m_dec else 0.0,
        "decode_tps": float(m_dec.group(3)) if m_dec else 0.0,
        "cores": int(m_dec.group(4)) if m_dec else 0,
        "ctx_used": int(m_dec.group(5)) if (m_dec and m_dec.group(5)) else 0,
        "ctx_max": int(m_dec.group(6)) if (m_dec and m_dec.group(6)) else 0,
    }


def run_os_batch(model: str, prompt: str, max_tokens: int, smp: int = 16,
                 think: bool = False, print_tokens: bool = True, timeout_s: int = 90):
    esc_prompt = prompt.replace("\\", "\\\\").replace('"', '\\"')
    cmdline = f'prompt="{esc_prompt}" max_tokens={max_tokens} think={1 if think else 0} shutdown=1'
    if print_tokens:
        cmdline += " print_tokens=1"
    cmd = [
        "qemu-system-x86_64",
        "-enable-kvm", "-cpu", "host",
        "-m", "2G", "-smp", str(smp),
        "-kernel", OS_ELF,
        "-initrd", model,
        "-append", cmdline,
        "-nographic",
    ]
    t0 = time.monotonic()
    res = subprocess.run(cmd, capture_output=True, timeout=timeout_s, check=True)
    wall_ms = (time.monotonic() - t0) * 1000.0
    parsed = parse_os_output(res.stdout)
    parsed["wall_ms"] = wall_ms
    return parsed


def test_repl(model: str, smp: int = 16):
    cmd = [
        "qemu-system-x86_64",
        "-enable-kvm", "-cpu", "host",
        "-m", "2G", "-smp", str(smp),
        "-kernel", OS_ELF,
        "-initrd", model,
        "-append", "max_tokens=48 think=0",
        "-nographic",
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    buf = b""

    def read_until(marker: bytes, timeout: float = 30.0):
        nonlocal buf
        deadline = time.monotonic() + timeout
        while marker not in buf:
            rem = deadline - time.monotonic()
            if rem <= 0:
                proc.kill()
                raise TimeoutError(f"Timed out waiting for {marker!r}; got tail:\n{buf[-500:].decode('utf-8', 'replace')}")
            r, _, _ = select.select([proc.stdout], [], [], min(rem, 0.25))
            if r:
                chunk = os.read(proc.stdout.fileno(), 4096)
                if not chunk:
                    break
                buf += chunk

    try:
        read_until(b"asm-llm> ", timeout=30.0)
        # Question 1
        proc.stdin.write(b"What is the capital of France? Answer in one short sentence.\n")
        proc.stdin.flush()
        buf = buf.split(b"asm-llm> ", 1)[1]
        read_until(b"asm-llm> ", timeout=30.0)
        turn1 = buf.decode("utf-8", errors="replace")

        # Question 2
        proc.stdin.write(b"What is 7 * 8? Reply with just the number.\n")
        proc.stdin.flush()
        buf = buf.split(b"asm-llm> ", 1)[1]
        read_until(b"asm-llm> ", timeout=30.0)
        turn2 = buf.decode("utf-8", errors="replace")

        # Clean shutdown via /halt
        proc.stdin.write(b"/halt\n")
        proc.stdin.flush()
        rest, _ = proc.communicate(timeout=15.0)
        turn3 = rest.decode("utf-8", errors="replace")
        return {
            "returncode": proc.returncode,
            "turn1": turn1,
            "turn2": turn2,
            "shutdown_msg": turn3,
        }
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_multiturn_os(model: str, smp: int = 16):
    cmd = [
        "qemu-system-x86_64",
        "-enable-kvm", "-cpu", "host",
        "-m", "2G", "-smp", str(smp),
        "-kernel", OS_ELF,
        "-initrd", model,
        "-append", "max_tokens=64 think=0",
        "-nographic",
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    buf = b""

    def read_until(marker: bytes, timeout: float = 30.0):
        nonlocal buf
        deadline = time.monotonic() + timeout
        while marker not in buf:
            rem = deadline - time.monotonic()
            if rem <= 0:
                proc.kill()
                raise TimeoutError(
                    f"Timed out waiting for {marker!r}; got tail:\n{buf[-500:].decode('utf-8', 'replace')}"
                )
            r, _, _ = select.select([proc.stdout], [], [], min(rem, 0.25))
            if r:
                chunk = os.read(proc.stdout.fileno(), 4096)
                if not chunk:
                    break
                buf += chunk

    def send_and_read(line: str) -> str:
        nonlocal buf
        proc.stdin.write((line + "\n").encode("utf-8"))
        proc.stdin.flush()
        read_until(b"asm-llm> ", timeout=30.0)
        part, buf = buf.split(b"asm-llm> ", 1)
        return part.decode("utf-8", errors="replace").strip()

    try:
        read_until(b"asm-llm> ", timeout=30.0)
        buf = buf.split(b"asm-llm> ", 1)[1]

        t1 = send_and_read("My name is Alex and my favorite number is 42. Reply with just OK.")
        t2 = send_and_read("What is my name and what is my favorite number?")
        t3 = send_and_read("/clear")
        t4 = send_and_read("What is my name?")

        proc.stdin.write(b"/halt\n")
        proc.stdin.flush()
        rest, _ = proc.communicate(timeout=15.0)
        shutdown_msg = (buf + rest).decode("utf-8", errors="replace").strip()

        c1 = re.search(r"ctx=(\d+)/(\d+)", t1)
        c2 = re.search(r"ctx=(\d+)/(\d+)", t2)
        c4 = re.search(r"ctx=(\d+)/(\d+)", t4)
        ctx1 = int(c1.group(1)) if c1 else 0
        ctx2 = int(c2.group(1)) if c2 else 0
        ctx4 = int(c4.group(1)) if c4 else 0

        ok = (
            proc.returncode == 0
            and "Alex" in t2
            and "42" in t2
            and "[chat] history cleared" in t3
            and "Alex" not in t4
            and ctx1 > 0
            and ctx2 > ctx1
            and ctx4 > 0
            and ctx4 < ctx2
            and "Powering off" in shutdown_msg
        )
        return {
            "ok": ok,
            "returncode": proc.returncode,
            "turn1": t1,
            "turn2": t2,
            "turn3_clear": t3,
            "turn4_after_clear": t4,
            "shutdown_msg": shutdown_msg,
            "ctx": (ctx1, ctx2, ctx4),
        }
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_multiturn_phase1(model: str, threads: int = 16):
    cmd = [
        PHASE1_BIN,
        "-m", model,
        "--chat",
        "-n", "64",
        "-t", str(threads),
    ]
    script_in = (
        "My name is Alex and my favorite number is 42. Reply with just OK.\n"
        "What is my name and what is my favorite number?\n"
        "/clear\n"
        "What is my name?\n"
        "/quit\n"
    ).encode("utf-8")
    res = subprocess.run(
        cmd,
        input=script_in,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
        check=True,
    )
    transcript = res.stdout.decode("utf-8", errors="replace")
    parts = transcript.split("asm-llm> ")
    # parts[0] = banner, parts[1] = turn1, parts[2] = turn2, parts[3] = /clear, parts[4] = turn4, parts[5] = /quit
    t1 = parts[1].strip() if len(parts) > 1 else ""
    t2 = parts[2].strip() if len(parts) > 2 else ""
    t3 = parts[3].strip() if len(parts) > 3 else ""
    t4 = parts[4].strip() if len(parts) > 4 else ""

    c1 = re.search(r"ctx=(\d+)/(\d+)", t1)
    c2 = re.search(r"ctx=(\d+)/(\d+)", t2)
    c4 = re.search(r"ctx=(\d+)/(\d+)", t4)
    ctx1 = int(c1.group(1)) if c1 else 0
    ctx2 = int(c2.group(1)) if c2 else 0
    ctx4 = int(c4.group(1)) if c4 else 0

    ok = (
        res.returncode == 0
        and "Alex" in t2
        and "42" in t2
        and "[chat] history cleared" in t3
        and "Alex" not in t4
        and ctx1 > 0
        and ctx2 > ctx1
        and ctx4 > 0
        and ctx4 < ctx2
    )
    return {
        "ok": ok,
        "returncode": res.returncode,
        "turn1": t1,
        "turn2": t2,
        "turn3_clear": t3,
        "turn4_after_clear": t4,
        "ctx": (ctx1, ctx2, ctx4),
        "raw": transcript,
    }


def capture_vga_screenshot(model: str, out_dir: Path, smp: int = 16):
    from PIL import Image

    sock_path = str(out_dir / "qemu-mon.sock")
    ser_path = out_dir / "vga-serial.log"
    ppm_path = out_dir / "vga-screenshot.ppm"
    png_path = out_dir / "vga-screenshot.png"
    for p in (sock_path, str(ser_path), str(ppm_path)):
        if os.path.exists(p):
            os.unlink(p)

    prompt = "Explain how a CPU cache works in 3 concise sentences."
    cmdline = f'prompt="{prompt}" max_tokens=96 think=0 shutdown=0'
    cmd = [
        "qemu-system-x86_64",
        "-enable-kvm", "-cpu", "host",
        "-m", "2G", "-smp", str(smp),
        "-kernel", OS_ELF,
        "-initrd", model,
        "-append", cmdline,
        "-vga", "std",
        "-display", "none",
        "-serial", f"file:{ser_path}",
        "-monitor", f"unix:{sock_path},server,nowait",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        # Wait until model finishes generating and prints REPL prompt
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if ser_path.exists():
                txt = ser_path.read_text(errors="replace")
                if "asm-llm> " in txt:
                    break
            time.sleep(0.1)
        else:
            raise TimeoutError("Timed out waiting for VGA boot & inference completion")

        time.sleep(0.3)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(sock_path)
            s.settimeout(3.0)
            def read_prompt():
                data = b""
                end = time.monotonic() + 3.0
                while time.monotonic() < end and b"(qemu)" not in data:
                    try:
                        chunk = s.recv(4096)
                        if not chunk:
                            break
                        data += chunk
                    except socket.timeout:
                        break
                return data
            read_prompt()
            s.sendall(f"screendump {ppm_path}\r\n".encode())
            read_prompt()
            # Wait for PPM file to be non-empty on disk
            for _ in range(30):
                if ppm_path.exists() and ppm_path.stat().st_size > 100:
                    break
                time.sleep(0.1)
            s.sendall(b"quit\r\n")
            time.sleep(0.3)
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=5.0)

        img = Image.open(ppm_path)
        img.save(png_path, optimize=True)
        os.unlink(ppm_path)
        if os.path.exists(sock_path):
            os.unlink(sock_path)
        return png_path, img.size
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if os.path.exists(sock_path):
            os.unlink(sock_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--only-vga", action="store_true")
    ap.add_argument("--only-chat", action="store_true")
    ap.add_argument("--max-threads", type=int, default=32)
    args = ap.parse_args()

    out_dir = Path(args.run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.only_vga:
        png_path, size = capture_vga_screenshot(args.model, out_dir, smp=min(16, args.max_threads))
        print(f"[PASS] Saved VGA screenshot: {png_path} ({size[0]}x{size[1]}, {png_path.stat().st_size} bytes)")
        return
    if args.only_chat:
        smp = min(16, args.max_threads)
        mt_os = test_multiturn_os(args.model, smp=smp)
        mt_p1 = test_multiturn_phase1(args.model, threads=smp)
        print(f"[{'PASS' if mt_os['ok'] else 'FAIL'}] Phase 2 Bare-Metal Multi-Turn Chat (smp={smp}, ctx={mt_os['ctx']})")
        print("--- OS Turn 1 ---\n" + mt_os["turn1"])
        print("--- OS Turn 2 (Memory Recall) ---\n" + mt_os["turn2"])
        print("--- OS /clear ---\n" + mt_os["turn3_clear"])
        print("--- OS Turn 4 (After /clear) ---\n" + mt_os["turn4_after_clear"])
        print(f"\n[{'PASS' if mt_p1['ok'] else 'FAIL'}] Phase 1 Linux --chat Multi-Turn (threads={smp}, ctx={mt_p1['ctx']})")
        print("--- P1 Turn 1 ---\n" + mt_p1["turn1"])
        print("--- P1 Turn 2 (Memory Recall) ---\n" + mt_p1["turn2"])
        print("--- P1 /clear ---\n" + mt_p1["turn3_clear"])
        print("--- P1 Turn 4 (After /clear) ---\n" + mt_p1["turn4_after_clear"])
        if not (mt_os["ok"] and mt_p1["ok"]):
            sys.exit(1)
        return
    log_path = out_dir / "report.log"
    lines = []

    def log(msg=""):
        print(msg, flush=True)
        lines.append(msg)

    log("=" * 78)
    log("asm-llm Phase 2 — Bare-Metal Bootable LLM OS Verification & Benchmarks")
    log("=" * 78)

    # 1. Exact Parity vs Phase 1 across 5 diverse prompts (EN, RU UTF-8, Math, Think)
    log("\n=== 1. EXACT PARITY VS PHASE 1 (LINUX build/asm-llm vs BARE-METAL KVM 16 vCPU) ===")
    prompts = [
        ("Phase1-Cache-64", "Explain the difference between L1, L2, and L3 CPU caches and why cache locality matters.", 64, False),
        ("EN-Concise-64", "Explain how a CPU cache works in 3 concise sentences.", 64, False),
        ("RU-UTF8-64", "Объясни в двух предложениях, как работает стек вызовов в процессоре.", 64, False),
        ("Logic-Math-96", "If Alice has 3 apples and Bob gives her 4 more, then she eats 2, how many apples does Alice have?", 96, False),
        ("Think-Mode-192", "What is 17 * 23?", 192, True),
    ]
    all_parity_ok = True
    p1_threads = min(32, args.max_threads)
    os_smp = min(16, args.max_threads)
    for name, prompt, max_tok, think in prompts:
        p1 = run_phase1(args.model, prompt, max_tok, think=think, threads=p1_threads)
        os_res = run_os_batch(args.model, prompt, max_tok, smp=os_smp, think=think, print_tokens=True)
        ptok_ok = (p1["prompt_tokens"] == os_res["prompt_tokens"]) and bool(p1["prompt_tokens"])
        gtok_ok = (p1["gen_tokens"] == os_res["gen_tokens"]) and bool(p1["gen_tokens"])
        text_ok = (p1["text"] == os_res["text"])
        ok = ptok_ok and gtok_ok and text_ok
        all_parity_ok = all_parity_ok and ok
        n_ptok = len(p1["prompt_tokens"].split(",")) if p1["prompt_tokens"] else 0
        n_gtok = len(p1["gen_tokens"].split(",")) if p1["gen_tokens"] else 0
        status = "PASS (100% EXACT)" if ok else "FAIL"
        log(f"[{status}] {name:16s} | prompt_tok={n_ptok:3d} gen_tok={n_gtok:3d} | "
            f"prefill={os_res['prefill_tps']:6.2f} tok/s decode={os_res['decode_tps']:6.2f} tok/s")
        if not ok:
            log(f"  Phase1 ptok={p1['prompt_tokens'][:80]} | OS ptok={os_res['prompt_tokens'][:80]}")
            log(f"  Phase1 gtok={p1['gen_tokens'][:80]} | OS gtok={os_res['gen_tokens'][:80]}")
            log(f"  Phase1 text={p1['text']!r}")
            log(f"  OS     text={os_res['text']!r}")
        else:
            preview = os_res["text"].replace("\n", " / ")[:110]
            log(f"    Output: {preview}")

    # 2. SMP Determinism & KVM Scaling Benchmark (1, 4, 8, 16, 24, 32 vCPUs)
    log("\n=== 2. BARE-METAL SMP SCALING & DETERMINISM BENCHMARK (QEMU KVM, AMD EPYC 7642) ===")
    bench_prompt = "Explain the difference between L1, L2, and L3 CPU caches and why cache locality matters."
    ref_p1 = run_phase1(args.model, bench_prompt, 64, think=False, threads=p1_threads)
    log(f"Prompt: {bench_prompt!r} (34 prompt tok, 64 gen tok, greedy)")
    log(f"{'vCPUs':>5s} | {'Online':>6s} | {'Boot(ms)':>8s} | {'Boot->1st':>9s} | {'Prefill ms':>10s} | {'Prefill t/s':>11s} | {'Decode ms':>9s} | {'Decode t/s':>10s} | {'QEMU Wall':>9s} | Determinism")
    log("-" * 116)
    smp_all_ok = True
    for smp in [s for s in (1, 4, 8, 16, 24, 32) if s <= args.max_threads]:
        r = run_os_batch(args.model, bench_prompt, 64, smp=smp, think=False, print_tokens=True, timeout_s=120)
        online_ok = (r["smp_online"] == smp)
        det_ok = (r["gen_tokens"] == ref_p1["gen_tokens"]) and (r["text"] == ref_p1["text"])
        smp_all_ok = smp_all_ok and online_ok and det_ok
        det_str = "EXACT (64/64)" if det_ok else "MISMATCH"
        log(f"{smp:5d} | {r['smp_online']:2d}/{r['smp_total']:<3d} | {r['boot_ready_ms']:8.2f} | {r['boot_ttft_ms']:9.2f} | "
            f"{r['prefill_ms']:10.2f} | {r['prefill_tps']:11.2f} | {r['decode_ms']:9.2f} | {r['decode_tps']:10.2f} | "
            f"{r['wall_ms']:9.1f} | {det_str}")

    # 3. Interactive REPL Test (2 consecutive questions + /halt)
    log("\n=== 3. INTERACTIVE BARE-METAL REPL TEST (COM1 SERIAL, 2 TURNS + /halt) ===")
    repl = test_repl(args.model, smp=os_smp)
    repl_ok = (repl["returncode"] == 0 and "Paris" in repl["turn1"] and "56" in repl["turn2"] and "Powering off" in repl["shutdown_msg"])
    log(f"[{'PASS' if repl_ok else 'FAIL'}] REPL exit_code={repl['returncode']}")
    log("--- Turn 1 ---\n" + repl["turn1"].strip())
    log("--- Turn 2 ---\n" + repl["turn2"].strip())
    log("--- Shutdown ---\n" + repl["shutdown_msg"].strip())

    # 4. Multi-Turn KV-Cache Memory & /clear Tests (Phase 2 OS + Phase 1 --chat)
    log("\n=== 4. MULTI-TURN KV-CACHE MEMORY & /clear TESTS (PHASE 2 OS + PHASE 1 --chat) ===")
    mt_os = test_multiturn_os(args.model, smp=os_smp)
    log(f"[{'PASS' if mt_os['ok'] else 'FAIL'}] Phase 2 Bare-Metal Multi-Turn (smp={os_smp}, ctx={mt_os['ctx']})")
    log("--- OS Turn 1 ---\n" + mt_os["turn1"])
    log("--- OS Turn 2 (Memory Recall) ---\n" + mt_os["turn2"])
    log("--- OS /clear ---\n" + mt_os["turn3_clear"])
    log("--- OS Turn 4 (After /clear) ---\n" + mt_os["turn4_after_clear"])

    mt_p1 = test_multiturn_phase1(args.model, threads=os_smp)
    log(f"[{'PASS' if mt_p1['ok'] else 'FAIL'}] Phase 1 Linux --chat Multi-Turn (threads={os_smp}, ctx={mt_p1['ctx']})")
    log("--- P1 Turn 1 ---\n" + mt_p1["turn1"])
    log("--- P1 Turn 2 (Memory Recall) ---\n" + mt_p1["turn2"])
    log("--- P1 /clear ---\n" + mt_p1["turn3_clear"])
    log("--- P1 Turn 4 (After /clear) ---\n" + mt_p1["turn4_after_clear"])

    # 5. VGA Screenshot Capture
    log("\n=== 5. VGA TEXT MODE (0xB8000) SCREENSHOT CAPTURE ===")
    png_path, size = capture_vga_screenshot(args.model, out_dir, smp=os_smp)
    log(f"[PASS] Saved VGA screenshot: {png_path} ({size[0]}x{size[1]}, {png_path.stat().st_size} bytes)")

    log("\n=== SUMMARY ===")
    log(f"Parity vs Phase 1 (5 prompts): {'PASS' if all_parity_ok else 'FAIL'}")
    log(f"SMP Determinism (<= {args.max_threads} vCPU) : {'PASS' if smp_all_ok else 'FAIL'}")
    log(f"Interactive REPL (2 turns)   : {'PASS' if repl_ok else 'FAIL'}")
    log(f"Multi-Turn KV-Cache (OS)     : {'PASS' if mt_os['ok'] else 'FAIL'}")
    log(f"Multi-Turn KV-Cache (Phase 1): {'PASS' if mt_p1['ok'] else 'FAIL'}")
    log(f"VGA Screenshot               : PASS ({png_path})")

    log_path.write_text("\n".join(lines) + "\n")
    if not (all_parity_ok and smp_all_ok and repl_ok and mt_os["ok"] and mt_p1["ok"]):
        sys.exit(1)


if __name__ == "__main__":
    main()

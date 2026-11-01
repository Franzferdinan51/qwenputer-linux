"""Seat resolution (builtin brain, incl. fused/GGUF ids), the dashboard's
guest FIFO queue, and regression guards that the activity view keeps
showing everything it showed (brain, live computer, circuit)."""

import io
import json
import threading
import time

import chipformer.rv.dashboard as D
import chipformer.rv.qwenbackend as qb


# --- pick_builtin_model: pure seat resolution ---

def test_pick_default_family_and_pristine_wins():
    ids = ["qwen3.5-4b-3d-coder", "llama-8b", "qwen3.5-4b-mlx"]
    # pristine/base build wins over the tagged fine-tune
    assert qb.pick_builtin_model(ids) == "qwen3.5-4b-mlx"


def test_pick_gguf_carrying_family_matches_by_default():
    # a fused build converted to GGUF keeps working with zero config
    # as long as its LM Studio id still carries the family substring
    ids = ["divagr1925/qwen3.5-4b-qwenputer"]
    assert qb.pick_builtin_model(ids) == ids[0]


def test_pick_pin_wins_even_outside_family():
    ids = ["qwen3.5-4b-mlx", "qwenputer-fused-q8"]
    assert qb.pick_builtin_model(ids, pin="qwenputer-fused-q8") == \
        "qwenputer-fused-q8"


def test_pick_pin_auto_falls_through_to_family():
    ids = ["qwen3.5-4b-mlx"]
    assert qb.pick_builtin_model(ids, pin="auto") == "qwen3.5-4b-mlx"


def test_pick_missing_pin_raises_never_silent_fallback():
    try:
        qb.pick_builtin_model(["qwen3.5-4b-mlx"], pin="qwenputer-fused-q8")
    except RuntimeError as e:
        assert "qwenputer-fused-q8" in str(e)
    else:
        raise AssertionError("missing pin must raise, not fall back")


def test_pick_no_family_match_raises_with_recovery_hint():
    try:
        qb.pick_builtin_model(["llama-8b"])
    except RuntimeError as e:
        assert "QWEN_MODEL" in str(e) and "QWEN_FAMILY" in str(e)
    else:
        raise AssertionError("missing family must raise, not fall back")


def test_pick_custom_family_seats_renamed_fused_build():
    ids = ["llama-8b", "qwenputer-fused-q8"]
    assert qb.pick_builtin_model(ids, family="qwenputer") == \
        "qwenputer-fused-q8"


# --- builtin_backend: env wiring (HTTP stubbed) ---

class _FakeResp:
    def __init__(self, payload):
        self._b = io.BytesIO(json.dumps(payload).encode())

    def read(self):
        return self._b.read()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _list_models(monkeypatch, module, ids):
    def fake_open(req, timeout=None):
        return _FakeResp({"data": [{"id": m} for m in ids]})

    class FakeOpener:
        def open(self, req, timeout=None):
            return fake_open(req, timeout)

    monkeypatch.setattr(module.urllib.request, "build_opener",
                        lambda *a, **k: FakeOpener())


def test_builtin_backend_uses_env_pin_and_family(monkeypatch):
    _list_models(monkeypatch, qb, ["qwen3.5-4b-mlx", "qwenputer-fused-q8"])
    seen = []

    def fake_lm(base, proxy, model):
        seen.append(model)
        return lambda p: "ok"

    monkeypatch.setattr(qb, "lmstudio_backend", fake_lm)
    monkeypatch.delenv("QWEN_MODEL", raising=False)
    monkeypatch.delenv("QWEN_FAMILY", raising=False)
    qb.builtin_backend("http://127.0.0.1:1")
    monkeypatch.setenv("QWEN_MODEL", "qwenputer-fused-q8")
    qb.builtin_backend("http://127.0.0.1:1")
    monkeypatch.delenv("QWEN_MODEL", raising=False)
    monkeypatch.setenv("QWEN_FAMILY", "qwenputer")
    qb.builtin_backend("http://127.0.0.1:1")
    assert seen == ["qwen3.5-4b-mlx", "qwenputer-fused-q8",
                    "qwenputer-fused-q8"]


def test_dashboard_resolve_model_honors_family_env(monkeypatch):
    def fake_urlopen(url, timeout=None):
        return _FakeResp({"data": [{"id": "qwenputer-fused-q8"}]})
    monkeypatch.setattr(D.urllib.request, "urlopen", fake_urlopen)
    s = {"backend_url": "http://127.0.0.1:1", "model": "auto"}
    monkeypatch.delenv("QWEN_FAMILY", raising=False)
    model, err = D.resolve_model(s)
    assert model is None and "qwen3.5-4b" in err
    monkeypatch.setenv("QWEN_FAMILY", "qwenputer")
    model, err = D.resolve_model(s)
    assert model == "qwenputer-fused-q8" and err is None


# --- guest FIFO queue ---

def _reset_guest():
    with D._GUEST_COND:
        D._GUEST_QUEUE.clear()
        D._GUEST_BUSY[0] = False
        D._GUEST_TASK[0] = ""


def test_guest_queue_is_strict_fifo():
    _reset_guest()
    try:
        assert D._guest_acquire(lambda m: None)  # main thread owns it
        order = []
        t1_sent, t2_sent = threading.Event(), threading.Event()
        t1_got, t2_got = threading.Event(), threading.Event()
        msgs2 = []

        def waiter(tag, got, sent, record):
            def send(m):
                record.append(m.get("status", ""))
                sent.set()
            if D._guest_acquire(send, budget=30):
                order.append(tag)
                got.set()
                time.sleep(0.2)
                D._guest_release()

        t1 = threading.Thread(target=waiter,
                              args=("t1", t1_got, t1_sent, []))
        t1.start()
        assert t1_sent.wait(10)
        t2 = threading.Thread(target=waiter,
                              args=("t2", t2_got, t2_sent, msgs2))
        t2.start()
        assert t2_sent.wait(10)
        D._guest_release()  # main hands over; t1 then t2, in order
        assert t1_got.wait(10)
        assert t2_got.wait(10)
        t1.join(10)
        t2.join(10)
        assert order == ["t1", "t2"]
        assert any("queued #2" in m for m in msgs2)
        with D._GUEST_COND:
            assert not D._GUEST_BUSY[0] and not D._GUEST_QUEUE
    finally:
        _reset_guest()


def test_guest_acquire_budget_expiry_drops_ticket():
    _reset_guest()
    try:
        assert D._guest_acquire(lambda m: None)
        assert D._guest_acquire(lambda m: None, budget=0.3) is False
        with D._GUEST_COND:
            assert len(D._GUEST_QUEUE) == 0 and D._GUEST_BUSY[0] is True
        D._guest_release()
        assert D._guest_acquire(lambda m: None)
        D._guest_release()
    finally:
        _reset_guest()


def test_activity_reports_guest_counters():
    _reset_guest()
    try:
        a = D.get_activity()
        for k in ("guest_hands", "guest_dones", "guest_active",
                  "guest_queue", "guest_task"):
            assert k in a, k
        assert a["guest_active"] is False and a["guest_queue"] == 0
    finally:
        _reset_guest()


# --- the activity view keeps everything it had ---

def test_dashboard_page_keeps_all_four_rows_and_counters():
    for row in ("now-chat", "now-computer", "now-circuit", "now-guest"):
        assert ('id="%s"' % row) in D.PAGE, row
    assert "COMPILED CIRCUIT (EXPERIMENT)" in D.PAGE
    assert "guest_hands" in D.PAGE and "guest_dones" in D.PAGE
    assert "vizPulse" in D.PAGE


# --- decision model (opt-in sidecar seat, LM Studio by default) ---

def test_decider_defaults_off():
    assert D.DEFAULTS["decider_enabled"] is False
    assert D.DEFAULTS["decider_url"].endswith(":1234")
    assert D.DEFAULTS["decider_model"] == ""


def test_decider_off_probes_nothing(monkeypatch):
    def boom(url, timeout=None):
        raise AssertionError("must not probe when off")
    monkeypatch.setattr(D.urllib.request, "urlopen", boom)
    model, err = D.resolve_decider({"decider_enabled": False})
    assert model is None and err == "off"


def test_decider_enabled_needs_exact_loaded_id(monkeypatch):
    def fake_urlopen(url, timeout=None):
        return _FakeResp({"data": [{"id": "liquidai/d1-omni-600m"}]})
    monkeypatch.setattr(D.urllib.request, "urlopen", fake_urlopen)
    s = {"decider_enabled": True, "decider_url": "http://127.0.0.1:1234",
         "decider_model": "liquidai/d1-omni-600m"}
    model, err = D.resolve_decider(s)
    assert model == "liquidai/d1-omni-600m" and err is None
    s["decider_model"] = "something-else"
    model, err = D.resolve_decider(s)
    assert model is None and "not loaded" in err


def test_decider_enabled_blank_id_is_an_error():
    model, err = D.resolve_decider(
        {"decider_enabled": True, "decider_url": "http://127.0.0.1:1",
         "decider_model": "  "})
    assert model is None and "no model id" in err


def test_decider_unreachable_is_an_error(monkeypatch):
    def fake_urlopen(url, timeout=None):
        raise OSError("refused")
    monkeypatch.setattr(D.urllib.request, "urlopen", fake_urlopen)
    model, err = D.resolve_decider(
        {"decider_enabled": True, "decider_url": "http://127.0.0.1:1",
         "decider_model": "liquidai/d1-omni-600m"})
    assert model is None and "refused" in err


def test_save_settings_coerces_decider_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "SETTINGS_PATH", str(tmp_path / "s.json"))
    s = D.save_settings({"decider_enabled": 1})
    assert s["decider_enabled"] is True
    assert D.load_settings()["decider_enabled"] is True


def test_dashboard_page_has_decider_controls():
    for cid in ("s-decider", "s-deciderurl", "s-decidermodel",
                "testdecider", "deciderstat", "s-deciderask",
                "askdecider", "deciderout"):
        assert ('id="%s"' % cid) in D.PAGE, cid


def _decider_settings(tmp_path, monkeypatch, **over):
    monkeypatch.setattr(D, "SETTINGS_PATH", str(tmp_path / "s.json"))
    s = dict(D.DEFAULTS)
    s.update(decider_enabled=True,
             decider_url="http://127.0.0.1:1",
             decider_model="liquidai/d1-omni-600m")
    s.update(over)
    with open(D.SETTINGS_PATH, "w") as f:
        json.dump(s, f)


def _chat_urlopen(payload=None, exc=None):
    def fake_urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else req
        if url.endswith("/v1/models"):
            return _FakeResp({"data": [{"id": "liquidai/d1-omni-600m"}]})
        if exc is not None:
            raise exc
        fake_urlopen.bodies.append(json.loads(req.data))
        return _FakeResp(payload)
    fake_urlopen.bodies = []
    return fake_urlopen


def test_decider_chat_off_fails_closed_no_probe(tmp_path, monkeypatch):
    _decider_settings(tmp_path, monkeypatch, decider_enabled=False)

    def boom(req, timeout=None):
        raise AssertionError("must not probe when off")
    monkeypatch.setattr(D.urllib.request, "urlopen", boom)
    r = D.decider_chat([{"role": "user", "content": "go?"}])
    assert r["ok"] is False and "off" in r["error"]


def test_decider_chat_empty_messages_never_probes(tmp_path, monkeypatch):
    _decider_settings(tmp_path, monkeypatch)

    def boom(req, timeout=None):
        raise AssertionError("must not probe with no messages")
    monkeypatch.setattr(D.urllib.request, "urlopen", boom)
    r = D.decider_chat([])
    assert r["ok"] is False and "no messages" in r["error"]


def test_decider_chat_pins_model_and_zero_temp(tmp_path, monkeypatch):
    _decider_settings(tmp_path, monkeypatch)
    fake = _chat_urlopen({"choices": [{"message": {"content": " yes "}}]})
    monkeypatch.setattr(D.urllib.request, "urlopen", fake)
    r = D.decider_chat([{"role": "user", "content": "go?"}])
    assert r == {"ok": True, "text": "yes",
                 "model": "liquidai/d1-omni-600m"}
    assert fake.bodies[0]["model"] == "liquidai/d1-omni-600m"
    assert fake.bodies[0]["temperature"] == 0


def test_decider_chat_unreachable_fails_closed(tmp_path, monkeypatch):
    _decider_settings(tmp_path, monkeypatch)
    monkeypatch.setattr(
        D.urllib.request, "urlopen",
        _chat_urlopen(exc=OSError("refused")))
    r = D.decider_chat([{"role": "user", "content": "go?"}])
    assert r["ok"] is False and "refused" in r["error"]

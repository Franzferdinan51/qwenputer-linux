"""Tests for chipformer.rv.liveterm's served terminal page (no guest needed)."""

from chipformer.rv.liveterm import PAGE


def test_page_folds_long_qwen_protocol_lines():
    # the full ~2KB QWEN: prompt repeats every agent round; the page
    # must fold those lines or the terminal drowns in prompt text
    assert "function fold(line)" in PAGE
    assert "line.startsWith('QWEN:')" in PAGE
    assert "chars]" in PAGE


def test_page_scroll_follows_only_at_bottom():
    # readers scrolled up into history must not get yanked to the bottom
    assert "term.scrollHeight - term.scrollTop - term.clientHeight" in PAGE

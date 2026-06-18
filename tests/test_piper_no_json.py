"""Regression test for piper TTS engine — guards against reintroducing
the --json flag (which caused TTS to fall back to a demo phrase)."""
import subprocess
import sys
from pathlib import Path

ENGINE_FILE = Path(__file__).resolve().parents[1] / "stt2tts_mcp/engines/tts/piper.py"


def test_no_json_flag_in_piper_cmd():
    """The speak() method must NOT pass --json to the piper CLI.
    Plain-text input was being misinterpreted as JSONL, causing fallback."""
    src = ENGINE_FILE.read_text(encoding="utf-8")
    assert '"--json"' not in src, (
        "piper.py must not pass --json — it makes plain-text input "
        "fail JSON parsing and fall back to a demo phrase."
    )


def test_piper_cmd_includes_model_and_output():
    """Sanity: the cmd list must still set --model and --output_file."""
    src = ENGINE_FILE.read_text(encoding="utf-8")
    assert '"--model"' in src
    assert '"--output_file"' in src


if __name__ == "__main__":
    try:
        test_no_json_flag_in_piper_cmd()
        test_piper_cmd_includes_model_and_output()
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
    print("OK: piper regression tests passed")

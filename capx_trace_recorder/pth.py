"""Small entry point executed by Python's site machinery."""

import os
import sys


def start() -> None:
    if not os.environ.get("CAPX_TRACE_DIR", "").strip():
        return
    try:
        from capx_trace_recorder.bootstrap import activate

        activate()
    except Exception as exc:
        try:
            print(f"capx_trace_recorder: activation failed: {type(exc).__name__}", file=sys.stderr)
        except Exception:
            pass


start()

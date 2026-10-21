"""Loopback HTTP fakes independent of CaP-X."""
from __future__ import annotations
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

DECISION_MARKER = "Please respond with EXACTLY ONE of the following"
BLOCK_1 = "print('hello')"
BLOCK_2 = "print('second')"
BLOCK_3 = "print('third')"

class ScriptedLLM:
    """Replies chosen from the request text; records every request body."""

    def __init__(self, decisions: list[str] | None = None, fail_first: int = 0,
                 initial_code: str = BLOCK_1) -> None:
        self.decisions = list(decisions if decisions is not None else [
            f"REGENERATE\n```python\n{BLOCK_2}```",
            f"REGENERATE\n```python\n{BLOCK_3}```",
            "FINISH",
        ])
        self.fail_first = fail_first
        self.initial_code = initial_code
        self.bodies: list[bytes] = []
        self.paths: list[str] = []
        self.lock = threading.Lock()
        self.n_decisions = 0

    def reply(self, body: dict[str, Any]) -> str:
        messages = body.get("messages") or []
        system = ""
        texts = []
        for m in messages:
            content = m.get("content")
            if m.get("role") == "system" and isinstance(content, str):
                system = content
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                texts.extend(p.get("text", "") for p in content if isinstance(p, dict))
        text = "\n".join(texts)
        if len(messages) == 1 and texts and texts[0].startswith("Point at "):
            return '<points coords="1 1 500 400">object</points>'
        if "single word OK" in text:
            return "OK"
        if "describes the initial state" in system:
            return "The red cube is to the left of the green cube."
        if "describes the difference" in system:
            return "The red cube did not move; the gripper moved up."
        if "robot execution videos" in system:
            return "The video shows the arm moving above the cubes."
        if DECISION_MARKER in text:
            with self.lock:
                i = self.n_decisions
                self.n_decisions += 1
            return self.decisions[min(i, len(self.decisions) - 1)]
        return f"Here is the program.\n```python\n{self.initial_code}```"


def make_handler(llm: ScriptedLLM):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: Any) -> None:  # silence http.server
            return

        def _send(self, status: int, payload: bytes, ctype: str = "application/json") -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            with llm.lock:
                llm.bodies.append(raw)
                llm.paths.append(self.path)
                n = len(llm.bodies)
            if self.path.endswith("/ik"):
                self._send(200, json.dumps({"joint_positions": [0.1] * 7}).encode())
                return
            if n <= llm.fail_first:
                self._send(503, b'{"error": "busy", "api_key": "sk-should-not-leak-123456789012345"}')
                return
            body = json.loads(raw)
            text = llm.reply(body)
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                parts = [text[: len(text) // 2], text[len(text) // 2:]]
                chunks = [{"choices": [{"delta": {"reasoning": "thinking briefly"}}]}]
                chunks += [{"choices": [{"delta": {"content": p}}]} for p in parts]
                for chunk in chunks:
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                self.close_connection = True
                return
            reply = {
                "id": f"cmpl-{n}",
                "model": body.get("model"),
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": text,
                                         "reasoning": "short reasoning"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
            self._send(200, json.dumps(reply).encode())

    return Handler


class LoopbackServer:
    def __init__(self, llm: ScriptedLLM) -> None:
        self.llm = llm
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(llm))
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True,
                                       name="fake-llm-server")

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> "LoopbackServer":
        self.thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()



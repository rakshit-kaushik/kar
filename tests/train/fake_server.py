"""A stand-in OpenAI-compatible server for eval-hook tests: /health, /v1/models, /v1/chat/completions."""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok"})
        elif self.path.startswith("/v1/models"):
            self._send(200, {"object": "list", "data": [{"id": MODEL, "object": "model"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/v1/chat/completions":
            self._send(200, {"id": "fake", "object": "chat.completion", "model": req.get("model", MODEL),
                             "choices": [{"index": 0, "finish_reason": "stop",
                                          "message": {"role": "assistant", "content": "Thanks for calling. What's the phone number on your account?"}}],
                             "usage": {"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22}})
        else:
            self._send(404, {"error": "not found"})

    def log_message(self, *a):  # quiet
        pass


if __name__ == "__main__":
    port = int(sys.argv[1])
    MODEL = sys.argv[2] if len(sys.argv) > 2 else "fake"
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()

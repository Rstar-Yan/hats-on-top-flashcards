#!/usr/bin/env python3
"""
Hats On Top 2 学习记录同步服务
- GET  /api/progress      读取整份记录
- POST /api/progress      覆盖写入整份记录
数据存到 /opt/hot2/progress.json
只用 Python 标准库，端口 8080，由 nginx 反代 /api/ 过来。
"""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


DATA = os.environ.get("HOT2_DATA", "/opt/hot2/progress.json")
_lock = threading.Lock()


def load():
    try:
        with open(DATA, encoding="utf-8") as f:
            d = json.load(f)
            return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def save(d):
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    tmp = DATA + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)
    os.replace(tmp, DATA)


class Handler(BaseHTTPRequestHandler):
    server_version = "hot2sync/1.0"

    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/progress" or path == "/api/progress/":
            with _lock:
                self._send(200, {"ok": True, "data": load()})
        else:
            self._send(404, {"ok": False, "err": "not found"})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path != "/api/progress" and path != "/api/progress/":
            return self._send(404, {"ok": False, "err": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
        except ValueError:
            n = 0
        raw = self.rfile.read(n) if n > 0 else b""
        try:
            d = json.loads(raw or b"{}")
        except Exception:
            return self._send(400, {"ok": False, "err": "bad json"})
        if not isinstance(d, dict):
            return self._send(400, {"ok": False, "err": "expect object"})
        with _lock:
            if isinstance(d.get("set"), dict) or isinstance(d.get("del"), list):
                # 增量：只合并/删除指定的词，不影响其它
                cur = load()
                if isinstance(d.get("set"), dict):
                    cur.update(d["set"])
                for k in (d.get("del") or []):
                    cur.pop(k, None)
                save(cur)
                return self._send(200, {"ok": True, "n": len(cur)})
            save(d)   # 全量覆盖（兼容旧调用）
            self._send(200, {"ok": True, "n": len(d)})

    def do_OPTIONS(self):
        self._send(204, {})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(os.environ.get("HOT2_PORT", "8080"))
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()

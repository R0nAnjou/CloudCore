"""HTTP 服务: 5 秒 deadline 守卫 + 全局异常兜底。

判题器约定:
- 建立连接 >10s 或响应 >5s 记一次异常; 累计 5 次淘汰。
- 兜底永远返回合法 JSON(roleCommandMap 至少为空表)。
"""
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from . import brain

LOGGER = logging.getLogger(__name__)

DEADLINE_SECONDS = 4.0  # 留 1 秒余量给网络


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        start = time.monotonic()
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            LOGGER.exception("bad request json")
            self._send({"roleCommandMap": {}, "prompt": "", "executeCmd": ""})
            return

        result_holder: dict[str, Any] = {}
        done = threading.Event()
        state_lock = threading.Lock()
        state = {"expired": False}

        def fallback() -> dict[str, Any]:
            return {
                "roleCommandMap": {},
                "prompt": "",
                "executeCmd": "",
            }

        def publish(value: dict[str, Any]) -> bool:
            """超时判定与 MEMORY 提交共用同一发布关口。"""
            with state_lock:
                if state["expired"]:
                    return False
                result_holder["value"] = value
                done.set()
                return True

        def worker() -> None:
            try:
                brain.decide_transactional(payload, publish)
            except Exception:
                LOGGER.exception("decide failed, fallback")
                publish(fallback())

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        if not done.wait(timeout=DEADLINE_SECONDS):
            with state_lock:
                if not done.is_set():
                    state["expired"] = True
                    LOGGER.error(
                        "decision timeout (round %s), fallback", payload.get("roundNo")
                    )
                    result_holder["value"] = fallback()
        self._send(result_holder.get("value") or {"roleCommandMap": {}})
        LOGGER.info(
            "round %s handled in %.3fs", payload.get("roundNo"), time.monotonic() - start
        )

    def _send(self, value: dict[str, Any]) -> None:
        try:
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        except Exception:
            LOGGER.exception("serialize failed")
            body = b'{"roleCommandMap":{}}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return


def serve(port: int) -> None:
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    server.serve_forever()

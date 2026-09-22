#!/usr/bin/env python3
"""《未来战争》参赛程序入口。

用法: python main.py <port>
判题器以 HTTP POST 推送回合状态 JSON, 本程序 5 秒内返回指令 JSON。
"""
import logging
import os
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python main.py <port>")
    port = int(sys.argv[1])
    root = Path(__file__).resolve().parent
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
    )

    from agent.server import serve

    logging.info("OurBot listening on 0.0.0.0:%d", port)
    serve(port)


if __name__ == "__main__":
    main()

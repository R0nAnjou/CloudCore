#!/usr/bin/env bash
# 《未来战争》参赛程序启动脚本 — 判题器通过 bash run.sh <port> 拉起本程序
PORT="${1:?Usage: bash run.sh <port>}"
cd "$(dirname "$0")"
exec python3 main3.py "$PORT"

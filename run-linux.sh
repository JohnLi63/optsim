#!/bin/bash
# optsim 源码方式启动（Linux）。第一次运行会在 ~/.local/share/optsim/venv 建 Python 环境（需联网，约 5 分钟）。
# 如果提示缺 venv：sudo apt install python3-venv
cd "$(dirname "$0")"
VENV="$HOME/.local/share/optsim/venv"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV" || { echo "请先运行: sudo apt install python3-venv"; read -p "按回车退出"; exit 1; }
  "$VENV/bin/pip" install -r requirements.txt || { read -p "安装失败，按回车退出"; exit 1; }
fi
exec "$VENV/bin/python" -m optsim_gui

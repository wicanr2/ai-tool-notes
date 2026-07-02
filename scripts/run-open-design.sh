#!/usr/bin/env bash
# Open Design AppImage 啟動器
#   - 過濾掉 dbus/systemd/zygote 等無害噪音,只保留真正的錯誤
#   - 首次啟動會解壓約 200MB 到 /tmp,視窗約 10-30 秒後出現,請耐心等
#
# 用法:
#   ./run-open-design.sh              # 前景啟動,看得到過濾後的 log
#   ./run-open-design.sh --quiet      # 背景啟動,完全靜音
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="$(ls -1 "$HERE"/Open-Design-*.AppImage 2>/dev/null | head -1)"

if [[ -z "${APP:-}" || ! -x "$APP" ]]; then
  echo "找不到可執行的 Open-Design-*.AppImage(應與本 script 同目錄)" >&2
  exit 1
fi

# 已知無害噪音:systemd scope 註冊失敗、chromium zygote、自動更新不支援 Linux、
# GPU/vulkan fallback、以及解壓時的檔名列表。過濾這些,其餘照常顯示。
NOISE='StartTransientUnit|org.freedesktop.systemd1|dbus/object_proxy|zygote|GetTerminationStatus|unsupported-platform|Failed to send GetTerminationStatus|File exists and file size matches|^/tmp/appimage_extracted_'

if [[ "${1:-}" == "--quiet" ]]; then
  nohup "$APP" --appimage-extract-and-run >/dev/null 2>&1 &
  echo "Open Design 啟動中(背景)。首次啟動需解壓,視窗約 10-30 秒後出現。PID=$!"
  exit 0
fi

echo "Open Design 啟動中… 首次啟動需解壓約 200MB,視窗約 10-30 秒後出現,請稍候。"
echo "（dbus / systemd / zygote 等訊息為無害噪音,已過濾。）"
echo "----------------------------------------------------------------------"
# stdbuf 讓過濾即時輸出;grep -vE 濾噪音;--line-buffered 避免緩衝卡住。
exec stdbuf -oL -eL "$APP" --appimage-extract-and-run 2>&1 | grep --line-buffered -avE "$NOISE"

# ai-tool-notes

AI 工具與相關工程實務的筆記彙整。

## 筆記

- [Subagent × 多模型分工:Claude Code 的 token 成本，與串接更便宜的模型](notes/subagent-multi-model-cost.md) — 用一次真實開發 session 的 subagent token 數據（旗艦協調 + haiku/sonnet 實作 ≈ 80 萬 token），算成本分工的實際效益；並附把 grunt 工作接到 DeepSeek-V4 / GLM-5.2 的具體做法（換 `ANTHROPIC_BASE_URL`、opencode per-agent 路由、claude-code-router）與定價對照。
- [把 Open Design 編成 Linux AppImage](notes/open-design-linux-appimage.md) — Open Design 官方只發 mac/win，記錄用容器建置出可執行 Linux AppImage 的七個排錯關卡、修正 patch，以及在目標機器執行的注意事項。
  - patch：[`patches/open-design-linux-appimage.patch`](patches/open-design-linux-appimage.patch)
  - 啟動器：[`scripts/run-open-design.sh`](scripts/run-open-design.sh)

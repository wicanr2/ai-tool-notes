# ai-tool-notes

AI 工具與相關工程實務的筆記彙整。

## 筆記

- [AI 說「做完了」其實沒做完:單元綠 ≠ 完成,以及怎麼防](notes/ai-said-done-but-wasnt.md) — 有 ground truth 的長專案(port/remake/遷移)最貴的失敗:AI 因為「能編譯、測試綠」反覆宣稱完成,卻沒達成目標,token 一路燒。剖析三根因(單元綠當完成、沒真的跑一遍、深挖方向偏)+ 可抄的對策清單(reference 實機 > AI 宣稱、驗收=終端行為、視覺核驗、先確定 spec 才動手)。
- [Subagent × 多模型分工:Claude Code 的 token 成本，與串接更便宜的模型](notes/subagent-multi-model-cost.md) — 用一次真實開發 session 的 subagent token 數據（旗艦協調 + haiku/sonnet 實作 ≈ 80 萬 token），算成本分工的實際效益；並附把 grunt 工作接到 DeepSeek-V4 / GLM-5.2 的具體做法（換 `ANTHROPIC_BASE_URL`、opencode per-agent 路由、claude-code-router）與定價對照。
- [把 Open Design 編成 Linux AppImage](notes/open-design-linux-appimage.md) — Open Design 官方只發 mac/win，記錄用容器建置出可執行 Linux AppImage 的七個排錯關卡、修正 patch，以及在目標機器執行的注意事項。
  - patch：[`patches/open-design-linux-appimage.patch`](patches/open-design-linux-appimage.patch)
  - 啟動器：[`scripts/run-open-design.sh`](scripts/run-open-design.sh)

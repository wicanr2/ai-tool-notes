# ai-tool-notes

AI 工具與相關工程實務的筆記彙整。

## 筆記

- [從學歷、算力到量產：拆解一篇 AI 人才論戰，與可驗收的數學學習路線](notes/ai-credentials-production-math-roadmap.md) — 分開檢驗 QS 學歷、GPU 規模、頂會、量產與薪資等能力訊號，指出原文合理的工程洞見與推論跳躍；再整理一條 28 週 AI 數學主線，涵蓋線代、微積分、機率統計、最佳化、數值計算、資訊理論，以及 LLM、視覺、Physical AI 三條可驗收專項。
- [AI 說「做完了」其實沒做完:單元綠 ≠ 完成,以及怎麼防](notes/ai-said-done-but-wasnt.md) — 有 ground truth 的長專案(port/remake/遷移)最貴的失敗:AI 因為「能編譯、測試綠」反覆宣稱完成,卻沒達成目標,token 一路燒。剖析三根因(單元綠當完成、沒真的跑一遍、深挖方向偏)+ 可抄的對策清單(reference 實機 > AI 宣稱、驗收=終端行為、視覺核驗、先確定 spec 才動手)。
- [Subagent × 多模型分工:Claude Code 的 token 成本，與串接更便宜的模型](notes/subagent-multi-model-cost.md) — 用一次真實開發 session 的 subagent token 數據（旗艦協調 + haiku/sonnet 實作 ≈ 80 萬 token），算成本分工的實際效益；並附把 grunt 工作接到 DeepSeek-V4 / GLM-5.2 的具體做法（換 `ANTHROPIC_BASE_URL`、opencode per-agent 路由、claude-code-router）與定價對照。
- [把 Open Design 編成 Linux AppImage](notes/open-design-linux-appimage.md) — Open Design 官方只發 mac/win，記錄用容器建置出可執行 Linux AppImage 的七個排錯關卡、修正 patch，以及在目標機器執行的注意事項。
  - patch：[`patches/open-design-linux-appimage.patch`](patches/open-design-linux-appimage.patch)
  - 啟動器：[`scripts/run-open-design.sh`](scripts/run-open-design.sh)

### Gemma-4 / chat_template 系列

- [chat_template 是什麼：從「LLM 只吃 token 序列」講起](notes/gemma4-chat-template-first-principles.md) — 第一性原理拆解 chat_template 存在的理由(多輪對話/工具呼叫必須序列化成單一字串)，用 `google/gemma-4-E4B-it` 的真實 Jinja2 樣板實際渲染兩組 messages(純對話、system+tool calling)，並整理 transformers `apply_chat_template` / vLLM serving 何時套用樣板、樣板過舊會出現的症狀。
- [HuggingFace 模型 repo 檔案逐一解說：以 google/gemma-4-E4B-it 為例](notes/gemma4-hf-repo-files.md) — 把一個 HF 模型 repo 的九個檔案(`config.json`、`tokenizer.json`/`tokenizer_config.json`、`chat_template.jinja`、`processor_config.json`、`model.safetensors` 等)逐一拆解:是什麼、誰讀它、改壞了會怎樣。
- [gemma-4 chat_template 變更事件：2026 年 7 月前後發生了什麼](notes/gemma4-chat-template-change-incident.md) — 查證後釐清:沒有 commit 精確落在 2026-07-05,實際是兩起事件——07-02~07-07 vLLM 封裝缺檔 + template/parser 版本錯位，與 07-15 HF 官方 `chat_template.jinja` 大改(null 處理/reasoning 保留/turn 標籤配平/輸入驗證)。附逐段 diff 分析、vLLM issue 症狀對照表、通用排查 SOP。

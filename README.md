# ai-tool-notes

AI 工具與相關工程實務的筆記彙整。

## 筆記

- [從學歷、算力到量產：拆解一篇 AI 人才論戰，與可驗收的數學學習路線](notes/ai-credentials-production-math-roadmap.md) — 分開檢驗 QS 學歷、GPU 規模、頂會、量產與薪資等能力訊號，指出原文合理的工程洞見與推論跳躍；再整理一條 28 週 AI 數學主線，涵蓋線代、微積分、機率統計、最佳化、數值計算、資訊理論，以及 LLM、視覺、Physical AI 三條可驗收專項。
- [AI 說「做完了」其實沒做完:單元綠 ≠ 完成,以及怎麼防](notes/ai-said-done-but-wasnt.md) — 有 ground truth 的長專案(port/remake/遷移)最貴的失敗:AI 因為「能編譯、測試綠」反覆宣稱完成,卻沒達成目標,token 一路燒。剖析三根因(單元綠當完成、沒真的跑一遍、深挖方向偏)+ 可抄的對策清單(reference 實機 > AI 宣稱、驗收=終端行為、視覺核驗、先確定 spec 才動手)。
- [Subagent × 多模型分工:Claude Code 的 token 成本，與串接更便宜的模型](notes/subagent-multi-model-cost.md) — 用一次真實開發 session 的 subagent token 數據（旗艦協調 + haiku/sonnet 實作 ≈ 80 萬 token），算成本分工的實際效益；並附把 grunt 工作接到 DeepSeek-V4 / GLM-5.2 的具體做法（換 `ANTHROPIC_BASE_URL`、opencode per-agent 路由、claude-code-router）與定價對照。
- [把 Open Design 編成 Linux AppImage](notes/open-design-linux-appimage.md) — Open Design 官方只發 mac/win，記錄用容器建置出可執行 Linux AppImage 的七個排錯關卡、修正 patch，以及在目標機器執行的注意事項。
  - patch：[`patches/open-design-linux-appimage.patch`](patches/open-design-linux-appimage.patch)
  - 啟動器：[`scripts/run-open-design.sh`](scripts/run-open-design.sh)

- [DeepSeek-V4 × J-Space 報告評讀，以及單卡 L40S 能不能自架](notes/deepseek-v4-jspace-report-review.md) — 評讀 2026-08-17 發布的 J-Space 能力釋放報告：摘述「能力實現損失」「Minimal 介面過擬合」「思維鏈二極體」三個主張，並逐項評估證據強度（單次執行無信賴區間、跨廠商欄位非同一 harness、效率表用未公開的統一縮放係數、作者即受測對象、但主動列出六項可證偽條件）。後半用 HF API 的實際張量 dtype 分佈算出 V4-Flash 需 292.5 GB、V4-Pro 需 1,601.9 GB，對上 g6e.xlarge 的 48.3 GB 單卡分別差 6.1 倍與 33.2 倍；並指出 J-Space 是 2.3 MB 的提示詞協定套件，不吃 GPU。

- [推論速度怎麼估：decode 是頻寬問題，不是算力問題](notes/llm-decode-throughput-formula.md) — 為什麼單一請求的生成速度只由記憶體頻寬決定；建立 `tok/s = 頻寬 × MBU / 每 token 讀取位元組` 這條公式，逐層算出 gemma-4-E4B 在 L40S 上每 token 讀 9.05 GB、預測 57–76 tok/s；擴充到 offload 情境（瓶頸是 DRAM 頻寬而非 PCIe），並附 DeepSeek-V4-Flash 在 RTX 5090 + 系統記憶體的估算表與驗證方法。
  - 互動計算器：[LLM Decode 速度估算器](tools/decode-throughput-calculator.html)（填參數即估速度、判斷顯存夠不夠）
- [量化格式：FP8、NVFP4，以及模型怎麼搭配](notes/quantization-fp8-nvfp4.md) — 從三份真實設定檔（gemma-4-26B FP8 per-channel、DeepSeek-V4 官方 FP8 block 128×128、NVIDIA NVFP4 group 16）拆解縮放粒度為何比位元數更關鍵、scale 開銷讓 FP4 實際是 0.56 bytes/param、weight-only 與 W8A8/W4A4 的差別、`ignore` 清單揭示哪些層碰不得（router / attention / head）、硬體原生支援如何決定量化划不划算，以及 GGUF K-quant 實測體積表與 LoRA adapter 的精度綁定問題。

- [2026 年主流開源 LLM 架構圖鑑](notes/llm-architecture-map-2026.md) — 把 gemma-4（E4B / 26B-A4B）、Qwen 3.6 / 3.8、DeepSeek-V4（Flash / Pro）、Kimi K3、MiniMax M2.7 的 config 攤開對照：一覽表（層數/hidden/總參數/激活參數/注意力組成/專家配置/context）、四種長 context 策略的層堆疊圖（滑動視窗交錯、線性注意力混合、MLA 壓縮 KV、不做處理）、MoE 從 128 專家 6.3% 激活走到 896 專家 1.8% 激活的演化，以及這些差異對 KV cache 成本與部署的實際影響。
- [量化與 offload：速度算得出來，品質算不出來](notes/quantization-quality-and-offload.md) — 前半用頻寬模型估 offload 速度並對照實測（ktransformers + SGLang 在單張 RTX 5090 上跑 DeepSeek-V4-Flash 達 20+ tok/s，高於公式估的 14–16，差距來自熱點專家動態調度）；後半說明品質為何無法從第一性原理推導、能量的三種代理指標、五個讓退化加劇的因素，以及唯一算得出來的部分——多步任務把每步 1% 的損失在 50 步後放大成 41%。
- [DRAM offload 專案地圖：把大模型塞進小顯卡](notes/dram-offload-projects.md) — 為什麼 MoE 讓 offload 從不可行變可行、兩種基本策略（搬權重到 GPU 算 vs 留在 CPU 算）的瓶頸差異，以及 llama.cpp / ktransformers / vLLM / MoE-Infinity / PowerInfer / AirLLM 等十個專案的定位與現況（含 star 數與最後更新），最後給選型建議與硬體配置優先序——記憶體通道數比顯卡重要。

### Gemma-4 / chat_template 系列

- [vLLM：怎麼用，以及 KV cache 為什麼是它的核心](notes/vllm-serving-and-architecture.md) — 從啟動參數與 OpenAI 相容端點講起，拆解 APIServer/EngineCore 分工、連續批次、prefill 與 decode 的性質差異、CUDA graph 與冷啟動時間組成；再從「自迴歸為何需要 KV cache」推到 PagedAttention，用同一張 L40S 上兩個模型的實測數字（12.75 GiB / 321,600 tokens / 併發 9.81x 對上 22.01 GiB / 1,033,831 tokens / 63.10x）說明 max-model-len 與併發的取捨，並以 config 與 vLLM 原始碼佐證滑動視窗與跨層 KV 共享如何把每 token 成本壓到 22.3 KiB。
- [PEFT 與 LoRA adapter：為什麼十幾 MB 能改變一個 15 GB 的模型](notes/peft-lora-adapter.md) — 先算清楚全參數微調的顯存帳單（權重+梯度+優化器狀態 ≈ 12P～16P bytes）指出 PEFT 要省的是哪一項；比較 adapter layers / prefix tuning / IA³ / LoRA 的推論延遲與可合併性；推導 LoRA 的低秩分解、B 初始化為零的理由、參數量計算；整理 r / alpha / target_modules / 學習率的選法，以及 merge vs 動態掛載、單 base 多 adapter 的部署經濟性、QLoRA/rsLoRA/DoRA 變體與五個常見誤解。
- [在 gemma-4-E4B-it 上掛載 LoRA adapter](notes/gemma4-lora-adapter-guide.md) — 從「base model 與 adapter 各是什麼」講起，逐檔拆解 PEFT 交付物的 `adapter_config.json` / `adapter_model.safetensors` / `native_lora_config.json`，給出 transformers+PEFT、vLLM serving、merge 三條載入路線的可執行指令；附一次在 L40S 上的實測紀錄（張量對應、PEFT 掛載、vLLM 載入器）與五個不會報錯的靜默失效點（模型類別選錯、請求填錯 model id、chat template 版本不一致、量化 base、rank 上限）。
- [chat_template 是什麼：從「LLM 只吃 token 序列」講起](notes/gemma4-chat-template-first-principles.md) — 第一性原理拆解 chat_template 存在的理由(多輪對話/工具呼叫必須序列化成單一字串)，用 `google/gemma-4-E4B-it` 的真實 Jinja2 樣板實際渲染兩組 messages(純對話、system+tool calling)，並整理 transformers `apply_chat_template` / vLLM serving 何時套用樣板、樣板過舊會出現的症狀。
- [HuggingFace 模型 repo 檔案逐一解說：以 google/gemma-4-E4B-it 為例](notes/gemma4-hf-repo-files.md) — 把一個 HF 模型 repo 的九個檔案(`config.json`、`tokenizer.json`/`tokenizer_config.json`、`chat_template.jinja`、`processor_config.json`、`model.safetensors` 等)逐一拆解:是什麼、誰讀它、改壞了會怎樣。
- [gemma-4 chat_template 變更事件：2026 年 7 月前後發生了什麼](notes/gemma4-chat-template-change-incident.md) — 查證後釐清:沒有 commit 精確落在 2026-07-05,實際是兩起事件——07-02~07-07 vLLM 封裝缺檔 + template/parser 版本錯位，與 07-15 HF 官方 `chat_template.jinja` 大改(null 處理/reasoning 保留/turn 標籤配平/輸入驗證)。附逐段 diff 分析、vLLM issue 症狀對照表、通用排查 SOP。

## 線上閱讀

https://wicanr2.github.io/ai-tool-notes/

# Subagent × 多模型分工:Claude Code 的 token 成本,與串接更便宜的模型

在高階（貴）模型的 session 裡做工程,最容易浪費的方式是拿旗艦模型做「不吃判斷力的機械活」——寫實作碼、跑測試、grep 比對、逐檔盤點。這些工作便宜模型做得一樣好。這篇記錄兩件事:(1) 用 Claude Code 的 subagent 把實作交給便宜模型、旗艦只做編排與核實時,**實際的 token 分工數據**;(2) 想更省,把 grunt 工作接到 DeepSeek / GLM 這類更便宜模型的**具體接法**(opencode、claude-code-router 等)。

- 資料來源:一次真實的多階段開發 session（把一個 C 遊戲 remake 移植成 Go/Ebiten + Android 觸控 UI + 打包 APK),主迴圈跑 Claude Opus,實作全部下放 subagent。
- 適用:任何用高階模型當「協調者」、想把實作成本壓下來的人。

## 一句話結論

**旗艦模型的價值在判斷,不在打字。** 把實作、測試、資料抽取交給 subagent（Claude Code 內用 haiku/sonnet,或外接 DeepSeek/GLM),旗艦只做「拆解任務 → 寫規格 → 獨立核實 → commit」,總實作 token 量的絕大部分就跑在便宜的價格層上,品質不變、成本降一個量級。關鍵紀律:**subagent 的回報一律不採信,由協調者用 `git diff` + 實跑測試獨立驗證後才 commit**——省成本不能省在驗證上。

## Part 1 — 一次真實 session 的 token 分工數據

同一個 session,主迴圈是 Claude Opus(負責:任務拆解、寫 subagent 規格、**獨立跑 docker 測試/核對 git diff/抽查幾何**、commit/push、成本分工決策)。所有「寫程式 + 寫測試」下放 subagent,依工作性質選模型:機械性 baked 資料抽取給 **haiku**,遊戲邏輯/UI 接線給 **sonnet**。

| 委派任務 | 模型 | subagent token | tool 呼叫 | 牆鐘 |
|---|---|--:|--:|--:|
| 復原 boss sprite 128/129（抽 8KB baked 陣列 + 接回退解碼) | haiku | 97,361 | 58 | 8m02s |
| 消耗品劇情道具位置閘（遊戲邏輯移植) | sonnet | 70,898 | 25 | 5m31s |
| 設定選單 UI + 自建字形（跨 gaudio/battle/UI 整合) | sonnet | 155,288 | 79 | 12m25s |
| 端到端主線整合測試（串全系統驗銜接) | sonnet | 137,013 | 54 | 13m03s |
| Android 觸控 P1 情境鍵 | sonnet | 81,736 | 32 | 4m21s |
| Android 觸控 P2 選單直接點選（六類選單 hit-test) | sonnet | 174,017 | 77 | 11m47s |
| Android 觸控 P3 打磨（壓感回饋 + 安全內縮) | sonnet | 84,503 | 23 | 5m35s |
| **小計（下放實作)** | haiku+sonnet | **≈ 800,816** | 348 | — |

> 這 ≈ 80 萬 token 的實作量,**沒有一個** token 跑在 Opus 價格層上。Opus 主迴圈的 token 花在拆任務、寫規格、跑驗證、寫 commit ——都是判斷密集、量相對小的工作。

### 分工原則(從這次實踐歸納)
- **依工作性質切模型**:判斷 / 取捨 / 架構理解 / 核實 → 高階模型;機械核實 / 資料抽取 / 樣板接線 → 便宜模型。
- **不要預設 fan-out**:subagent 預設繼承主迴圈模型,一次派四個等於燒四倍旗艦。要派就明確指定 `model: haiku` / `model: sonnet`。
- **協調者只做協調 + 核實**:每個 subagent 交回後,主迴圈**自己重跑測試 + 看 diff 範圍 + 抽查關鍵邏輯**才 commit。這次就靠這條抓到一個 subagent 寫的 flaky 測試(用了非決定性的隨機遭遇),subagent 回報「全綠」但實際時綠時紅——**綠燈可信度是判斷,必須留在協調者手上**。
- **subagent 誠實回報缺口**:規格裡要求「串接發現真 bug 不要繞過、如實回報」;這次 subagent 主動標出兩處「只能靠 debug 觸發、正常玩家路徑碰不到」的缺口,沒假裝完成。

## Part 2 — 成本換算:token × 各模型定價

> ⚠ 定價變動快。以下為 **2026-07-03** 查證的數字,每百萬 token(1M),input / output 分列。實際以各家官方頁為準(來源見文末)。

| 模型 | Input $/1M | Output $/1M |
|---|--:|--:|
| Claude Opus 4.8 | 5.00 | 25.00 |
| Claude Sonnet 5 | 3.00（促銷 2.00,至 2026-08-31) | 15.00（促銷 10.00) |
| Claude Haiku 4.5 | 1.00 | 5.00 |

上表 subagent 回報的 `subagent_tokens` 是 **input+output 合計**,無細分。下面用「(in+out)/2 粗略混合率」估**數量級**——重點在**層級比**,不在絕對值(coding agent 通常 input 重、output 輕,真實成本會更靠近較低的 input 價)。

把本 session ≈ 80 萬下放實作 token(實際混搭:haiku 97k + sonnet 703k)套價:

| 若這 0.8M 全用… | 混合率 $/1M | 估算成本 |
|---|--:|--:|
| Claude Opus 4.8 | 15.0 | ≈ $12.0 |
| Claude Sonnet 5 | 9.0（促銷 6.0) | ≈ $7.2（促銷 4.8) |
| Claude Haiku 4.5 | 3.0 | ≈ $2.4 |
| **本 session 實際混搭**(haiku+sonnet) | — | **≈ $6.6** |

**對照**:同樣的實作量,若全塞給 Opus 主迴圈 ≈ $12;實際把它下放 haiku/sonnet ≈ $6.6,**省約 45%**——而且旗艦 Opus 省下來的算力全用在判斷(拆任務、寫規格、獨立核實),不是打字。這還沒動到外接模型;下一節再砍一個數量級。

## Part 3 — 更省一階:把 grunt 工作接到 DeepSeek / GLM

Claude Code 主迴圈綁 Anthropic **協議**(不是綁 Anthropic 伺服器)。只要對方端點講 Anthropic Messages API,換個 base URL 就能把實作 grunt 導到便宜家。

### 3.1 DeepSeek / GLM 定價與相容性(對照 Claude)

| 模型 | Input $/1M | Output $/1M | Context | Anthropic 相容端點 |
|---|--:|--:|--:|---|
| Claude Opus 4.8 | 5.00 | 25.00 | 1M | 官方 |
| Claude Sonnet 5 | 3.00 | 15.00 | 1M | 官方 |
| DeepSeek-V4-flash | 0.14 | 0.28 | 1M | `https://api.deepseek.com/anthropic` |
| DeepSeek-V4-pro | 0.435 | 0.87 | 1M | 同上 |
| GLM-5.2（按量) | ~1.4 | ~4.4 | 未證實 | `https://api.z.ai/api/anthropic` |
| GLM Coding Plan（訂閱) | $10/月起（Lite),額度內不另計 token | — | — | 同上 |

- **DeepSeek 已從 V3.2/R1 整併為 V4**(`deepseek-v4-flash` / `deepseek-v4-pro`);舊 `deepseek-chat` / `deepseek-reasoner` 於 **2026-07-24** 停用。cache-hit input 便宜到 $0.0028/1M。
- **GLM 5.2 確有其事**,是 2026-07 當下 Z.ai 旗艦。**GLM Coding Plan** 是訂閱制($10/月起),額度內經 Anthropic 端點呼叫不再按 token 計——對「大量下放實作」特別划算。
- 數量級感:DeepSeek-V4-flash 的 output $0.28/1M vs Sonnet $15/1M ≈ **1/54**;vs Opus $25 ≈ **1/89**。同一批 0.8M 實作 token,DeepSeek-flash 混合率估 ≈ **$0.17**(對照 Opus 的 $12)。

### 3.2 讓 Claude Code 直接走便宜模型(換 base URL,不換工具)

最簡單、零額外工具:編 `~/.claude/settings.json`(或 export 同名環境變數)。以 GLM 為例:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic",
    "ANTHROPIC_AUTH_TOKEN": "<your_zai_api_key>",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "glm-5.2",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "glm-4.7-flash"
  }
}
```

DeepSeek 同理,`ANTHROPIC_BASE_URL` 換成 `https://api.deepseek.com/anthropic`。啟動後 Claude Code 照常運作,底層模型換掉。

要**依請求類型分流**(如背景/長任務走 DeepSeek、一般對話走 GLM)才需要 proxy:**claude-code-router**(`musistudio/claude-code-router`)—— 起一個 local proxy(`127.0.0.1:3456`),`ANTHROPIC_BASE_URL` 指到它,再依規則轉發到 DeepSeek / GLM / Qwen / 本地模型等。
> 註:新版 router 設定改走桌面 UI + SQLite(`~/.claude-code-router/config.sqlite`);純 CLI/JSON headless 設定範例**未證實**仍在官方文件,需查該 repo 的 examples/。

### 3.3 用 opencode 做 per-agent 多模型路由

opencode(`opencode.ai`)原生支援每個 agent / subagent 綁不同 provider+model,另有 `small_model` 專跑輕量任務。接第三方 OpenAI 相容 API 用 `@ai-sdk/openai-compatible`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "deepseek": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "DeepSeek",
      "options": {
        "baseURL": "https://api.deepseek.com/v1",
        "apiKey": "{env:DEEPSEEK_API_KEY}"
      },
      "models": { "deepseek-v4-pro": { "name": "DeepSeek V4 Pro" } }
    }
  }
}
```

有 GLM Coding Plan 訂閱者更簡單:opencode 內選 provider「Z.AI Coding Plan」→ `/models` 選 GLM-5.2 / GLM-4.7 即可,不必自建 provider block。可做到「coder 用 Opus、researcher 用便宜模型、debugger 用另一家」三 agent 三 provider。

## Part 4 — 落地決策表

| 工作性質 | 交給 | 為什麼 |
|---|---|---|
| 拆任務 / 寫規格 / 架構取捨 / **核實綠燈可信度** | 旗艦(Opus / Sonnet) | 判斷密集、量小、錯了代價高 |
| 遊戲邏輯 / UI 接線 / 中等複雜移植 | Sonnet 或 GLM-5.2 / DeepSeek-V4-pro | 有結構的實作,便宜一階仍穩 |
| baked 資料抽取 / 樣板碼 / 大量機械改寫 | Haiku 或 DeepSeek-V4-flash | 純機械,便宜兩階照樣做對 |
| web 研究 / 資料蒐集 / 逐斷言查證 | 便宜模型 + web 工具 | token 重(本篇研究就燒 358k)、判斷輕 |

### 混搭架構的鐵則(這次踩過才寫的)
1. **協調者自己驗,不採信 subagent 回報**:每個交回的 diff,主迴圈**重跑測試 + 看改動範圍 + 抽查關鍵邏輯**才 commit。這次靠這條抓到 subagent 交的 flaky 測試(非決定性隨機遭遇,時綠時紅,回報卻說全綠)。**模型越便宜,這條越不能省。**
2. **規格要求誠實回報缺口**:明講「串接發現真 bug 不要繞過、如實說」;好的 subagent 會主動標「這裡只能 debug 觸發、正常路徑碰不到」。
3. **不要預設 fan-out**:subagent 繼承主迴圈模型,一次派四個 = 燒四倍旗艦。要派就顯式指定便宜 model。
4. **成本分工本身是架構決策**:「全交旗艦」和「全省著用」都是懶惰預設;依工作性質切模型才對。

---

## 來源(查證日期 2026-07-03)

- Claude 定價:Anthropic 官方定價表(platform.claude.com/docs/pricing,快取 2026-06-24;絕對即時數字建議另 fetch)。
- DeepSeek:[Models & Pricing | DeepSeek API Docs](https://api-docs.deepseek.com/quick_start/pricing)、[Integrate with Claude Code | DeepSeek](https://api-docs.deepseek.com/quick_start/agent_integrations/claude_code)、[Integrate with OpenCode | DeepSeek](https://api-docs.deepseek.com/quick_start/agent_integrations/opencode)。
- GLM / Z.ai:[Pricing — Z.AI Docs](https://docs.z.ai/guides/overview/pricing)、[GLM Coding Plan](https://z.ai/subscribe)、[Claude Code — Z.AI Docs](https://docs.z.ai/devpack/tool/claude)。
- opencode:[Providers](https://opencode.ai/docs/providers/)、[Agents](https://opencode.ai/docs/agents/)、[Models](https://opencode.ai/docs/models/)。
- claude-code-router:[musistudio/claude-code-router](https://github.com/musistudio/claude-code-router)。
- 多模型分工模式:Aider architect/editor、Roo Code 多 mode、Kilo Code Orchestrator(社群比較文,見各自官方文件為準)。

> 標「未證實」處(GLM context window、claude-code-router 純 CLI 設定、Cline orchestrator 機制)為查詢時無官方明確資料,引用前請自行覆核。定價與模型版本變動快,本篇為 2026-07-03 快照。

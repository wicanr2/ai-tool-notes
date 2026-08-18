# gemma-4 chat_template 變更事件：2026 年 7 月前後發生了什麼

任務起點是「查 2026-07-05 gemma-4 chat_template 變更事件」。逐一核對 HuggingFace commit 記錄與 vLLM issue/PR 的**實際時間戳**後，結論是：**沒有任何一次 chat_template 相關變更精確落在 2026-07-05 這一天**。但那一週前後，確實有兩條獨立又互相牽連的事件線，合理對應到「gemma-4 chat_template 出問題」這個印象的來源。這篇把兩條線分開講清楚，附完整查證依據。

- 系列前兩篇：[chat_template 是什麼](gemma4-chat-template-first-principles.md)、[HuggingFace 模型 repo 檔案逐一解說](gemma4-hf-repo-files.md)。

## 一句話結論

**日期對不上,但機制對得上。** 真正發生的是兩件事：(1) 2026-07-02～07-07，vLLM 這邊發現 `pip install vllm` 裝出來的套件**沒把** `tool_chat_template_gemma4.jinja` 這個範例樣板一起打包，導致官方推薦指令直接啟動失敗；討論中還牽出「vLLM 0.24.0 的 parser 修正後，HF Hub 官方預設的 chat_template 反而解析不動，只有 vLLM 自帶的範例樣板正常」這個版本錯位問題。(2) 2026-07-15，Google 才真的對 `google/gemma-4-E4B-it` 的 `chat_template.jinja` 做了一次大改（PR #36：null 處理、reasoning 保留、turn 標籤配平、輸入驗證）。7/16 有 vLLM 社群成員在前述討論串裡回報「官方 template 昨天更新了」，把兩條線接在一起。**如果聽到的版本是「7 月初 gemma-4 chat_template 出包」，講的極可能是 vLLM 封裝/相容性問題，不是 HF 樣板內容本身在那天被改。**

## 查證方法與時間軸

用 HuggingFace 官方 API（非網頁摘要，避免小模型幻覺）直接取 commit 記錄：

```bash
curl -sL "https://huggingface.co/api/models/google/gemma-4-E4B-it/commits/main" | jq .
```

`chat_template.jinja` 相關的完整 commit 歷史（`google/gemma-4-E4B-it` 全 repo 只有 11 個 commit，這裡列出跟樣板有關的）：

| 日期（UTC） | commit | 內容 |
|---|---|---|
| 2026-04-02 | `292a7e2` | 初版釋出（"Preparing for release!"） |
| 2026-04-10 | `ab4d202` | #14 prefix-preserving tool calls + dialog compliance |
| 2026-04-28 | `c53e9d3` | #28 update SI and tool call handling |
| 2026-05-18 | `d6436b3` | #30 multimodal placeholders in tool response content-parts |
| **2026-07-15** | **`6ea8ce9`** | **#36 null handling、reasoning preservation、turn-tag balance、input validation（本篇分析對象）** |

同段期間，vLLM repo（`vllm-project/vllm`）用 `gh api search/issues` 核實到的相關 issue/PR（時間精確到秒，用 `gh api repos/vllm-project/vllm/issues/<n>` 逐筆核對，非網頁摘要）：

| 日期（UTC） | # | 標題 | 狀態 |
|---|---|---|---|
| 2026-06-27 | #46921 | Package example Jinja chat templates in wheels | 有 merge conflict，未即時合併 |
| 2026-07-02 | #47401 | `vllm/vllm-openai-cpu:v0.24.0` image 缺 `tool_chat_template_gemma4.jinja` | closed |
| 2026-07-02 | #47447 | CPU release image 沒打包 `examples/` | closed |
| **2026-07-04** | **#47600** | **`ValueError: ... appears path-like` — pip 裝的 vLLM 缺這個 jinja 檔** | open |
| 2026-07-06 | #47678 | 補上封裝路徑下的 `tool_chat_template_gemma4.jinja`（fix #47600） | open，討論串延續到 07-16 |
| 2026-07-15 | #48678 | Gemma4 parser：字串參數在內部 `<\|"\|>` token 處被截斷 | — |

## 事件一（2026-07-02 ~ 07-07）：vLLM 封裝缺檔 + 版本錯位

`recipes.vllm.ai` 官方文件教的啟動指令長這樣：

```bash
vllm serve google/gemma-4-31B-it \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --chat-template examples/tool_chat_template_gemma4.jinja \
  --reasoning-parser gemma4
```

`#47600`（07-04）回報：只要不是從 git checkout 跑（例如單純 `pip install vllm` 或用官方 wheel build 的 Docker image），這個指令會直接炸掉：

```
ValueError: The supplied chat template string (examples/tool_chat_template_gemma4.jinja)
appears path-like, but doesn't exist! Tried: examples/tool_chat_template_gemma4.jinja
and .../site-packages/vllm/transformers_utils/chat_templates/examples/tool_chat_template_gemma4.jinja
```

`#47678` 的修復說明講得很清楚：vLLM 的 `validate_chat_template()` 本來就有「找不到相對路徑，就去套件內建的 `chat_templates/examples/` 目錄再找一次」的 fallback 邏輯，這段邏輯本身沒壞——**只是那個 fallback 目錄裡，一直沒人把 `tool_chat_template_gemma4.jinja` 這份檔案放進去**（`#46921`，06-27，本來想順手修，卡在 merge conflict 沒進主線）。是打包遺漏，不是樣板內容錯誤。

比缺檔更值得記錄的，是 `#47678` 討論串裡 07-06～07-07 的這段對話：vLLM maintainer DarkLight1337 問「HF Hub 現在自己就附了 chat_template，vLLM 是不是不用再自帶一份範例了？」，貢獻者 nikhilesh-csa 回覆（07-06）：

> "I just tested the HF chat template - it works fine (tested tool use, reasoning and basic chat) for v0.23.0 and before but since the parser fixes on 0.24.0, the HF chat template misbehaves but the template from vllm/examples works fine."

也就是說：**vLLM 升級到 0.24.0 之後，自己的 tool-call/reasoning parser 改了，反而跟 HF 當時官方預設的 chat_template 對不上——要靠 vLLM 自己維護的另一份 `examples/tool_chat_template_gemma4.jinja` 才正常。** 這正是[第一篇](gemma4-chat-template-first-principles.md)提到的「template 語法跟 parser 期待的格式要對齊」問題的真實案例：template 沒變，但下游 parser 版本變了，一樣會出現「工具呼叫解析失敗」的症狀；反過來，template 變了但 parser 沒跟上，也是同一種錯位。

## 事件二（2026-07-15）：HF 官方 chat_template.jinja 真正的內容變更

抓變更前（`d6436b3`，05-18）與變更後（`6ea8ce9`，07-15）兩個 revision 的 `chat_template.jinja` 直接 diff，四個主題對應 PR 標題「null handling, reasoning preservation, turn-tag balance, input validation」逐一验证：

**1. Null handling（null 處理）**
{% raw %}- `format_argument` 巨集原本沒有處理 `argument is none` 的分支，None 值會落到最後的 `{{- argument -}}`，被 Jinja2 渲染成字面 `None`（不是合法的 `null`）；新版明確加了 `{%- if argument is none -%}{{- 'null' -}}` 分支。{% endraw %}
- 大量 `message['tool_calls']`／`message['content']` 的直接索引改成 `message.get('tool_calls')`／`message.get('content')`，避免訊息缺少該欄位時直接 `KeyError` 整個渲染中斷。
- 系統訊息區塊加了 `messages and messages[0]['role'] in [...]` 的空列表防呆——舊版 `messages` 是空列表時會直接對 `messages[0]` 索引越界。

**2. Reasoning preservation（thinking 內容保留）**
- 舊版：`thinking_text and loop.index0 > ns_turn.last_user_idx and message.get('tool_calls')`——只有「這一則訊息剛好也有 tool_calls」時，才會把 reasoning/thinking 內容渲染進 `<|channel>thought` 區塊。這代表工具呼叫後的**純文字**回覆訊息，就算帶了 reasoning，也會被樣板整段吃掉。
- 新版拆成 `thinking_gate = (loop.index0 > ns_turn.last_user_idx) or (preserve_thinking and message.get('tool_calls'))`，把「這輪在最後一次使用者發言之後」跟「要不要保留工具呼叫輪的 thinking」拆成兩個獨立條件，不再綁死必須同時有 `tool_calls`。
- `add_generation_prompt` 區塊新增：若上一則是 tool_response 且 `enable_thinking` 為真，補上 `<|channel>thought\n`，讓模型在工具回應後、開始下一輪之前，能重新進入 thinking 通道，而不是直接被推去輸出最終答案。

**3. Turn-tag balance（turn 標籤配平）**
- 新增一段「forward-scan」邏輯（`next_nt`）去看**下一則非 tool 訊息的角色**，多算出一個 `continues_into_next` 條件：若目前這輪是 model、下一則非 tool 訊息也是 assistant、且（這輪沒有 tool_calls 或工具回應已經處理過），就**不要**提前補上收尾的 `<turn|>`——留給下一輪接續處理。這是在修「assistant 有 content 又有 tool_calls，接續下一輪時多冒出一個不該有的 `<turn|>`」這類配對錯誤（對應 commit list 裡 "prevent extra `<turn\|>` when assistant has content + tool_calls + continuation"、"Fix chat template turn closure after tool-call-only turns"）。
- continuation（連續兩則 assistant 要不要合併成一個 turn）的判斷，從**每次迭代往回掃描整個歷史**（`for j in range(loop.index0 - 1, -1, -1)`，O(n²) 全域）改成用 `ns.prev_non_tool_role` 狀態變數即時追蹤（O(1)）——這既是效能修正，也順手讓邏輯更不容易因為掃描範圍算錯而出現配對失準。

**4. Input validation（輸入驗證）**
- 舊版工具呼叫參數若是字串（`function['arguments'] is string`），會**原樣輸出**這段字串；新版明確擋掉這個分支，改丟 `raise_exception`，要求呼叫端必須先把 JSON 字串反序列化成 mapping 才能進樣板：

{% raw %}
  ```jinja
  {%- else -%}
      {{- raise_exception(
          "chat_template: tool_calls[].function.arguments must be a "
          "JSON object (mapping), not a string. Deserialize arguments "
          "before passing to the template."
      ) -}}
  {%- endif -%}
  ```
{% endraw %}

  這是刻意的「早失敗」設計：與其讓格式不對的參數悄悄混進 prompt，變成訓練分佈外輸入（見第一篇），不如直接在樣板渲染階段就報錯擋下來。

## vLLM 為什麼可能「沒吃到」新樣板：機制而非單一事件

即使 HF 官方在 07-15 把樣板修好，serving 端不一定會馬上生效，原因是機制性的，跟這次事件本身無關：

- **huggingface_hub 本地快取以 commit hash 為單位存快照**（`~/.cache/huggingface/hub/models--<org>--<name>/snapshots/<commit_hash>/`），`refs/main` 記著目前 `main` 分支對應到哪個 commit。若程式碼是用固定 `revision=<某舊 commit hash>` 或部署時就已經下載過快照、之後沒有重新觸發 `snapshot_download()`，就算 HF 上的 `main` 已經更新，本地永遠讀的是那個舊 snapshot，不會自動升級。
- **vLLM 啟動時載入 template 只發生一次**：`vllm serve` 啟動階段解析 `--chat-template`（明確路徑覆蓋）或 fallback 到 tokenizer 自帶樣板；serving 過程中不會每次請求都重新檢查 HF 上是否有新版本。要吃到新樣板必須重啟 server，且重啟前要先讓本地快取更新到新 revision。
- **`--chat-template` 顯式覆蓋會蓋掉 HF 官方樣板**，如同事件一裡的做法——這是把雙面刃：好處是繞開 HF 樣板本身的 bug（or 版本沒對齊 parser），代價是 HF 官方之後修正了樣板，你也不會自動吃到，除非手動同步覆蓋檔案。

## 症狀對照表（僅列有查證依據的 vLLM issue，日期均為 `gh api` 核實的實際建立時間）

| 症狀 | 對應 issue（已查證） | 日期 | 根因類別 |
|---|---|---|---|
| 啟動直接失敗：`ValueError...appears path-like` | [#47600](https://github.com/vllm-project/vllm/issues/47600) | 2026-07-04 | 封裝遺漏（非 template 內容錯誤） |
| 工具呼叫字串參數在內部 `<\|"\|>` 處被截斷、後續參數變亂碼 key | [#48678](https://github.com/vllm-project/vllm/pull/48678) | 2026-07-15 | parser 邊界掃描邏輯錯誤 |
| 工具回應後 reasoning 內容洩漏 | [#45834](https://github.com/vllm-project/vllm/issues/45834) | 2026-06-16 | thinking gate 條件不完整（同類問題，07-15 HF 版本已修正邏輯本體） |
| `enable_thinking=false` 時解析錯誤 | [#45832](https://github.com/vllm-project/vllm/pull/45832) | 2026-06-16 | parser 未同步 template 的分支邏輯 |
| 停止 token 判斷失效：工具呼叫後生成停不下來 | [#45591](https://github.com/vllm-project/vllm/issues/45591) | 2026-06-14 | `--override-generation-config` 誤蓋掉 `generation_config.json` 的 `eos_token_id` |
| streaming／非 streaming 解析結果不一致 | [#48217](https://github.com/vllm-project/vllm/issues/48217) | 2026-07-10 | parser 對「無 channel 標記輸出」的分類邏輯不一致 |

> 標「推測」的部分：本文未找到任何一則 issue／commit 明確寫「因為快取 snapshot 沒更新所以吃到舊版」這句話——**快取機制導致沒吃到新版**是根據 huggingface_hub 官方快取文件（見上一節）推出的**通用機制性風險**，不是這次事件裡被直接證實發生過的具體案例，讀者部署時仍應自行按下方 SOP 驗證，不要預設一定是這個根因。

## 通用處置 SOP

**1. 確認 serving 端實際用的是哪個版本**

```bash
# 看本地快取實際落在哪個 commit
hf cache ls --revisions | grep gemma-4-E4B-it

# 直接對 serving 端點送一則帶 tools 的測試請求，
# 把回傳的 raw prompt（或用 vLLM debug log）跟本地重新渲染的結果比對
```

**2. 若懷疑是 template 過舊：強制重新拉取 main 的最新 revision**

```bash
hf download google/gemma-4-E4B-it --revision main --force-download \
  --include "chat_template.jinja" "tokenizer_config.json"
```

**3. 若懷疑是 template／parser 版本不對齊（如事件一）：顯式覆蓋，別依賴預設 fallback**

```bash
vllm serve google/gemma-4-E4B-it \
  --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --chat-template /path/to/known-good/chat_template.jinja
```

覆蓋檔案版本要跟 `--tool-call-parser`／`--reasoning-parser` 的 vLLM 版本一起記錄（例如寫進部署 repo 的 README），因為如事件一所示，**template 沒動、parser 版本升級一樣可能讓兩者對不上**——這是雙向風險，不是只有「template 太舊」一種情境。

**4. 更新後務必回歸測試工具呼叫、多輪對話、thinking 三類請求**，而不是只跑一次基本 chat——本篇兩個事件的症狀（截斷、洩漏、提前/延遲停止）都只在特定訊息結構下才會觸發，單純「你好」式測試看不出來。

## 來源（查證日期 2026-07-20）

- HF commit 記錄（API 直查，非網頁摘要）：`https://huggingface.co/api/models/google/gemma-4-E4B-it/commits/main`。
- `chat_template.jinja` 變更前後版本：[revision d6436b3（2026-05-18）](https://huggingface.co/google/gemma-4-E4B-it/blob/d6436b3d62967e1af08bbb046c6300b2a9ae8e85/chat_template.jinja)、[revision 6ea8ce9（2026-07-15，PR #36）](https://huggingface.co/google/gemma-4-E4B-it/blob/6ea8ce9dfc868b201e9c6eba6d85d2d78a752f83/chat_template.jinja)。
- vLLM issue/PR（`gh api repos/vllm-project/vllm/issues/<n>` 逐筆核實建立時間）：[#46921](https://github.com/vllm-project/vllm/pull/46921)、[#47401](https://github.com/vllm-project/vllm/issues/47401)、[#47447](https://github.com/vllm-project/vllm/pull/47447)、[#47600](https://github.com/vllm-project/vllm/issues/47600)、[#47678](https://github.com/vllm-project/vllm/pull/47678)、[#48678](https://github.com/vllm-project/vllm/pull/48678)、[#45834](https://github.com/vllm-project/vllm/issues/45834)、[#45832](https://github.com/vllm-project/vllm/pull/45832)、[#45591](https://github.com/vllm-project/vllm/issues/45591)、[#48217](https://github.com/vllm-project/vllm/issues/48217)。
- huggingface_hub 快取機制：[Understand caching — huggingface_hub docs](https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache)。
- 補充背景（社群整理文，非官方一手來源，交叉比對用）：[Gemma 4 Update: FA4, Tool Calling, Vision (July 2026) — explainx.ai](https://explainx.ai/blog/gemma-4-updates-flash-attention-tool-calling-july-2026)（確認 07-15 為官方社群更新日）、[Gemma 4 GGUF Chat Template Fixed in Updated Builds — aiproductivity.ai](https://aiproductivity.ai/news/gemma-4-gguf-chat-template-fix/)（另一起 2026-05-04、範圍限定 GGUF 量化版的獨立事件，跟本篇兩起官方 safetensors／vLLM 事件無關，特此排除避免混淆）。

## 關聯

- [chat_template 是什麼](gemma4-chat-template-first-principles.md)——本篇「turn-tag balance」「reasoning preservation」兩節的 diff，對照第一篇「template 錯誤或過舊會怎樣」的症狀分類表更容易理解。
- [HuggingFace 模型 repo 檔案逐一解說](gemma4-hf-repo-files.md)——`chat_template.jinja` 在 repo 裡的角色與其他檔案的關係。

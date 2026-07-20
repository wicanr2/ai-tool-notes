# chat_template 是什麼：從「LLM 只吃 token 序列」講起

LLM 服務端常見的詭異症狀——工具呼叫格式跑掉、多輪對話角色錯亂、模型不停生成——有相當比例根因不在模型本身，而在一個常被忽略的檔案：`chat_template.jinja`。這篇從第一性原理拆解它到底在解決什麼問題，再用 Google `google/gemma-4-E4B-it` 的實際 template 跑一個可重現的渲染範例。

- 適用：串接 transformers `apply_chat_template`、或用 vLLM／TGI 等框架 serving 對話模型的工程師。
- 系列另兩篇：[HuggingFace 模型 repo 檔案逐一解說](gemma4-hf-repo-files.md)、[2026-07 gemma-4 chat_template 變更事件](gemma4-chat-template-change-incident.md)。

## 一句話結論

**LLM 的輸入輸出只有一種東西：token 序列。** 「多輪對話」「系統提示」「工具呼叫」都是人類加上去的抽象概念，模型完全不認得 `role: user` 這種結構化欄位——這些欄位在送進模型前，必須先被拉平（序列化）成一串夾雜特殊 token 的純文字。**chat_template（一段 Jinja2 樣板）就是這道序列化規則本身**，不是「附加說明文件」。樣板錯了或版本過舊，模型看到的輸入就已經是錯的，之後任何調校都救不回來。

## 為什麼需要 chat_template：從根本問題推導

### 問題 1：模型只認得一維 token 流

一個 causal LM（自迴歸語言模型）不管有沒有經過 chat 微調，運作方式永遠只有一種：吃一段 token 序列，預測下一個 token。訓練時餵進去的，最終也只是一長串扁平的文字。所以「這是使用者說的」「這是系統設定」「這是上一輪模型的回答」這些角色資訊，必須靠**插在文字裡的特殊 token** 來標記，模型才能在訓練階段學會辨認結構——例如「看到這個 token 之後的內容都當使用者輸入」。

### 問題 2：每個模型家族的標記規則互不相容

不同團隊各自發明了自己的一套特殊 token：Gemma 系列早期用 `<start_of_turn>user` / `<end_of_turn>`，OpenAI 系相容格式常見 `<|im_start|>` / `<|im_end|>`，本篇要看的 gemma-4-E4B-it 又換了一套（見下）。這些 token 是模型在微調階段**實際訓練過**的信號；用錯家族的標記，等於餵給模型一串它從沒見過結構意義的雜訊——不會報錯，但輸出品質會悄悄變差，因為模型是在「盡力理解一段格式陌生的文字」，而不是照訓練分佈做結構化推理。

### 問題 3：組裝規則需要邏輯，不只是字串拼接

多輪對話、系統提示、工具呼叫（tool calls）、工具回傳結果、reasoning/thinking 內容——這些不是簡單串接，而是有條件邏輯的：第一則系統訊息要不要獨立成一個 turn？連續兩則 assistant 訊息要不要合併，還是各自開新 turn？工具回應要接在呼叫後面、還是等使用者下一輪？這正是為什麼序列化規則要用一個**模板語言**表達，而不是寫死在推論程式碼裡——Jinja2 提供的迴圈、條件、巨集（macro）剛好夠用來表達這些組裝邏輯，同時保持「規則和推論引擎解耦」：推論框架（transformers、vLLM）不需要為每個模型家族各寫一套組裝程式碼，只要照著模型 repo 附的 `.jinja` 樣板去渲染即可。

## 實際跑一次：拿 gemma-4-E4B-it 的真實 template 渲染

以下直接抓 `google/gemma-4-E4B-it`（2026-07 發布，2026-07-15 更新過 chat_template；事件細節見[本系列第三篇](gemma4-chat-template-change-incident.md)）repo 裡的 `chat_template.jinja` 原始檔，用 Jinja2 渲染兩組 messages。**不用系統 Python**，容器內裝 `jinja2` 即可（渲染邏輯本身不需要載入模型權重，`apply_chat_template` 在 transformers 內部也是純 Jinja2 沙箱渲染，見下一節）：

```bash
docker run --rm -v "$PWD:/work" -w /work python:3.12-slim \
  bash -c "pip install -q jinja2 && python render_demo.py"
```

`render_demo.py` 的核心只有幾行：用 `jinja2.Environment` 載入抓下來的 `chat_template.jinja`，並註冊一個 `raise_exception` 全域函式（transformers 的渲染環境本來就會注入這個函式，樣板內用它來擋不合法輸入，見下方範例二的模板原始碼片段）：

```python
from jinja2 import Environment, FileSystemLoader

def raise_exception(msg):
    raise ValueError(msg)

env = Environment(loader=FileSystemLoader("/work"), trim_blocks=True, lstrip_blocks=True)
env.globals["raise_exception"] = raise_exception
tmpl = env.get_template("chat_template_main.jinja")

out = tmpl.render(messages=messages, bos_token="<bos>", add_generation_prompt=True)
```

### 範例一：純文字多輪對話

渲染前（Python 端的 messages list，結構化資料）：

```json
[
  {"role": "user", "content": "台北明天會下雨嗎？"},
  {"role": "assistant", "content": "我沒有即時氣象資料，建議查中央氣象署。"},
  {"role": "user", "content": "那你可以幫我查嗎？"}
]
```

渲染後（實際送進模型的字串，`add_generation_prompt=True`）：

```
<bos><|turn>user
台北明天會下雨嗎？<turn|>
<|turn>model
我沒有即時氣象資料，建議查中央氣象署。<turn|>
<|turn>user
那你可以幫我查嗎？<turn|>
<|turn>model
```

可以看到 gemma-4 這一版**不是**舊 Gemma 熟悉的 `<start_of_turn>` / `<end_of_turn>`，而是改用 `<|turn>role` … `<turn|>` 包住每一輪，角色名稱直接寫在 turn 開頭（`user`、`model`）。結尾補的 `<|turn>model\n`（沒有內容、沒有收尾 `<turn|>`）就是 `add_generation_prompt=True` 的作用——告訴模型「輪到你接著寫」，讓推論引擎知道要在這之後開始生成，而不是模型自己去猜這一步該不該接話。

### 範例二：system prompt + 工具呼叫（function calling）

渲染前：

```json
[
  {"role": "system", "content": "你是一個會使用工具查天氣的助理。"},
  {"role": "user", "content": "台北明天會下雨嗎？"},
  {"role": "assistant", "content": null, "tool_calls": [
    {"id": "call_1", "function": {"name": "get_weather", "arguments": {"city": "台北"}}}
  ]},
  {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "明天台北降雨機率 70%"},
  {"role": "assistant", "content": "明天台北降雨機率約 70%，建議帶傘。"}
]
```

渲染後：

```
<bos><|turn>system
你是一個會使用工具查天氣的助理。<|tool>declaration:get_weather{description:<|"|>查詢指定城市的天氣<|"|>,parameters:{properties:{city:{description:<|"|>城市名稱<|"|>,type:<|"|>STRING<|"|>}},required:[<|"|>city<|"|>],type:<|"|>OBJECT<|"|>}}<tool|><turn|>
<|turn>user
台北明天會下雨嗎？<turn|>
<|turn>model
<|tool_call>call:get_weather{city:<|"|>台北<|"|>}<tool_call|><|tool_response>response:get_weather{value:<|"|>明天台北降雨機率 70%<|"|>}<tool_response|>明天台北降雨機率約 70%，建議帶傘。<turn|>
<|turn>model
```

這段完整示範第四節要講的重點：`tools` 參數（Python 端的 JSON schema）被 template 的 `format_function_declaration` 巨集轉譯成一種**自訂的、非 JSON 的緊湊語法**（`declaration:get_weather{description:<|"|>...<|"|>,parameters:{...}}`），而不是直接塞一段 JSON 字串進去。這是模型訓練時實際看過的格式；用真的 JSON（帶標準引號、逗號空格習慣）去問它，等於又是一次「格式陌生」的分佈外輸入。

## 誰在什麼時候套用 template

| 場景 | 套用時機 | 關鍵機制 |
|---|---|---|
| transformers 本地推論 | 呼叫 `tokenizer.apply_chat_template(messages, ...)` 時 | 讀 `tokenizer_config.json` 的 `chat_template` 欄位，或模型 repo 根目錄獨立的 `chat_template.jinja`（新版 transformers 優先找獨立檔案）；用內建 Jinja2 沙箱環境渲染，並自動注入 `raise_exception`、`strftime_now` 等輔助函式 |
| vLLM OpenAI-compatible `/v1/chat/completions` | server 啟動時解析 messages → 呼叫模板渲染再丟給 tokenizer 編碼成 token id | vLLM 原始碼裡 `ChatTemplateConfig`／`load_chat_template()` 負責載入樣板：可用 `--chat-template <path或內建模板名>` 顯式覆蓋；不給的話 fallback 用 tokenizer 自帶的樣板（來自下載下來的 `tokenizer_config.json` / `chat_template.jinja`） |
| 訓練 / SFT 資料前處理 | 把對話資料集轉成訓練用扁平文字時 | 同樣呼叫 `apply_chat_template`，但通常 `add_generation_prompt=False`（訓練不需要「請模型接著寫」的提示） |

`--chat-template` 覆蓋是重要的救火手段：如果 serving 端載入的樣板版本跟訓練用的對不上（見第三篇的實際案例），可以直接指定一份已知正確的 `.jinja` 檔案路徑，不必等模型 repo 更新或重新拉取快取。

### template 錯誤或過舊會怎樣

不是「編譯錯誤」那種會立刻炸掉的失敗模式，而是**悄悄的分佈外輸入**，症狀分幾類：

- **格式 token 錯位／缺漏**：模型看到訓練時沒見過的 turn 分界，輸出品質下降、答非所問機率上升，但不會有任何 exception。
- **工具呼叫格式跑掉**：template 若還在用舊語法序列化 `tools`，模型生成的呼叫格式會對不上新版樣板期待的 parser 正則，framework 端解析失敗或直接把工具呼叫原始文字洩漏到使用者可見的 `content` 裡。
- **停止 token 判斷失效**：`add_generation_prompt`、turn 收尾邏輯若跟訓練不一致，模型可能抓不到「這裡該停」的訊號，導致不斷生成到 `max_tokens` 上限，或提前在使用者輪次自問自答。
- **多輪角色錯亂**：continuation 判斷（同一個 assistant 連續兩則要不要合併成一個 turn）算錯，會讓模型看到錯誤的輪替節奏，這在長對話/agent loop 場景特別致命。

## 工具呼叫與 template 的關係

從範例二可以看到，`tools` 不是「額外資訊」，而是被渲染進 system turn 的一段結構化文字（`<|tool>declaration:...<tool|>`）。這代表：

1. **工具定義的序列化規則也是 template 的一部分**——同一個模型家族，不同版本的 template 可能用不同語法描述工具（JSON schema 直出、或像 gemma-4 這樣轉成自訂緊湊格式）。framework 端若沒有對應更新 tool-call parser，即使 template 正確渲染，回應也解析不出來。
2. **模型輸出的工具呼叫格式，是 template 「教」出來的**——不是模型自己決定要怎麼寫函式呼叫語法，而是訓練資料本身就是照這個 template 序列化的；template 描述工具的語法跟模型訓練時看到的語法不一致，是最常見的「function calling 突然不準」根因之一。
3. **工具回應（tool response）怎麼接回對話，也是條件邏輯**——範例二可以看到 `<|tool_response>...<tool_response|>` 緊接在 `<|tool_call>` 之後、同一個 model turn 內，而不是開新的一輪；這正是 gemma-4 template 裡 `continue_same_model_turn`／`ns_tr_out.flag` 這類狀態追蹤在處理的問題（見第三篇對這段邏輯改動的分析）。

## 來源（查證日期 2026-07-20）

- HuggingFace 官方文件：[Chat templates — transformers docs](https://huggingface.co/docs/transformers/en/chat_templating)。
- 實際渲染範例使用的樣板：`google/gemma-4-E4B-it` repo 的 [`chat_template.jinja`](https://huggingface.co/google/gemma-4-E4B-it/blob/main/chat_template.jinja)（抓取 commit `6ea8ce9dfc868b201e9c6eba6d85d2d78a752f83`，2026-07-15）。
- vLLM 樣板載入相關原始碼：[`vllm/entrypoints/chat_utils.py`](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/chat_utils.py)（`ChatTemplateConfig`、`_load_chat_template`、`validate_chat_template`）。
- vLLM 官方範例樣板：[`examples/tool_chat_template_gemma4.jinja`](https://github.com/vllm-project/vllm/blob/main/examples/tool_chat_template_gemma4.jinja)。

> 標「待查證」：vLLM 內部「CLI `--chat-template` vs tokenizer 內建樣板」的完整優先序判斷邏輯，因原始碼檔案較大、WebFetch 摘要工具只回傳了 `chat_utils.py` 的部分函式，未能逐行確認優先序程式碼路徑；本文的載入時機描述綜合官方文件與該檔案可見部分寫成，實際版本行為請以當下 vLLM 版本原始碼或 `vllm serve --help` 為準。

## 關聯

- 下一篇：[HuggingFace 模型 repo 檔案逐一解說](gemma4-hf-repo-files.md)——`chat_template.jinja` 只是 repo 裡九個檔案之一，這篇把整個 repo 攤開講。
- 實戰案例：[2026-07 gemma-4 chat_template 變更事件](gemma4-chat-template-change-incident.md)——本篇範例二用的 template，就是那次變更之後的版本；文中附了變更前後的 diff。

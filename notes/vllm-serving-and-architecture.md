# vLLM：怎麼用，以及 KV cache 為什麼是它的核心

一張 46 GB 的 L40S，放進一個 26 GB 的模型之後，剩下的空間要拿來服務多少個同時進來的對話？這個問題的答案幾乎完全由 KV cache 的配置方式決定，而不是由模型本身決定。實測那台機器的數字是：模型權重 25.8 GiB、CUDA graph 0.66 GiB、剩下 **12.75 GiB 給 KV cache，換算成 321,600 個 token**，在 32K context 的設定下等於 9.81 個滿載請求的容量。

這篇從「怎麼把服務跑起來」開始，接著拆解一個請求進到 vLLM 之後經過什麼，最後把 KV cache 的機制與實測數字對起來。所有數字都取自 vLLM 0.25.1 的實際啟動 log 與模型設定檔。

- 相關：[在 gemma-4-E4B-it 上掛載 LoRA adapter](gemma4-lora-adapter-guide.md)、[PEFT 與 LoRA adapter](peft-lora-adapter.md)、[chat_template 是什麼](gemma4-chat-template-first-principles.md)

---

## 一、把服務跑起來

vLLM 提供一個與 OpenAI API 相容的 HTTP 服務，最短的啟動方式是一行：

```bash
vllm serve google/gemma-4-E4B-it
```

實務上會用容器跑，因為它對 CUDA 版本敏感：

```bash
docker run -d --name gemma4-e4b \
  --gpus all --rm \
  --log-opt max-size=10m --log-opt max-file=3 \
  -p 8001:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model google/gemma-4-E4B-it \
  --served-model-name gemma-4-e4b-it \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768
```

掛 HuggingFace 快取目錄是必要的，否則每次重建容器都要重新下載十幾 GB 的權重。

### 常用參數

| 參數 | 作用 | 沒設好會怎樣 |
|---|---|---|
| `--model` | 模型 repo id 或本機路徑 | — |
| `--served-model-name` | API 對外顯示的名稱 | 客戶端要填一長串 repo id |
| `--gpu-memory-utilization` | 這個實例可以用掉整張卡的幾成（預設 0.9） | 設太高會與同卡的其他服務相撞；設太低則 KV cache 變小、併發下降 |
| `--max-model-len` | 單一請求的最大 context 長度 | 設超過模型能力會啟動失敗；設太大會壓縮併發（見第三節） |
| `--tensor-parallel-size` | 把模型切到幾張卡 | 單卡放不下時必須設 |
| `--api-key` 或環境變數 `VLLM_API_KEY` | 啟用金鑰驗證 | 不設等於服務全開 |
| `--enable-lora` / `--lora-modules 名稱=路徑` / `--max-lora-rank` | 掛載 LoRA adapter | 見 [LoRA adapter 教學](gemma4-lora-adapter-guide.md) |
| `--enable-auto-tool-choice` / `--tool-call-parser` | 工具呼叫：讓模型自行決定要不要呼叫，並把輸出解析成 `tool_calls` | 不設的話工具呼叫會以純文字回傳，客戶端拿不到結構化欄位 |
| `--reasoning-parser` | 把思考內容拆到 `reasoning_content` 欄位 | 思考內容會混在正式回覆裡 |
| `--chat-template` | 指定訊息序列化樣板 | 見下方 |

`--chat-template` 值得單獨講。它決定 messages 怎麼攤平成模型真正看到的字串，而**同一個模型往往有多份樣板在流通**——模型 repo 裡一份，vLLM 容器的 `examples/` 裡一份，兩者不見得相同。實測過 gemma-4 的兩份差 24 行，其中一處差在模型開始生成的位置，換一份就足以讓同一個微調模型的行為改變（細節見 [LoRA adapter 教學](gemma4-lora-adapter-guide.md) 第六節）。**寫死路徑，不要依賴預設值**，因為預設會隨引擎版本改變。

### 呼叫

```bash
curl http://localhost:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-e4b-it","temperature":0,
       "messages":[{"role":"user","content":"你好"}]}'
```

常用端點：`/v1/models` 列出可用的 model id、`/v1/chat/completions` 對話、`/v1/completions` 純續寫、`/health` 健康檢查（就緒後才回 200，是等待啟動完成的判斷點）、`/metrics` Prometheus 指標。

啟用了金鑰時，未帶 `Authorization` 的請求回 401——這也是判斷「服務活著且驗證生效」的最快方式。

### 啟動 log 該看哪幾行

冷啟動要好幾分鐘，期間唯一的資訊來源是 log。以下是一次真實啟動的關鍵行：

```
[gpu_model_runner.py:5306] Model loading took 25.8 GiB memory and 166.9 seconds
[gpu_model_runner.py:6639] Estimated CUDA graph memory: 0.66 GiB total
[gpu_worker.py:538]        Available KV cache memory: 12.75 GiB
[kv_cache_utils.py:2146]   GPU KV cache size: 321,600 tokens
[kv_cache_utils.py:2147]   Maximum concurrency for 32,768 tokens per request: 9.81x
[core.py:337]              init engine (profile, create kv cache, warmup model) took 140.65 s
```

這六行把整張卡的帳算清楚了：權重 25.8 GiB、CUDA graph 0.66 GiB、KV cache 12.75 GiB。最後一行說明冷啟動時間花在哪——**權重載入只佔 167 秒，引擎初始化又花了 141 秒**（其中編譯 63 秒），所以「模型載完了」不等於「可以接請求了」。等服務就緒要打 `/health`，不能只看 log 裡的 loading 完成。

### 啟動失敗的常見原因

| 症狀 | 成因 |
|---|---|
| `CUDA out of memory` 或啟動到一半被殺 | 同一張卡上有其他服務。`--gpu-memory-utilization` 是「整張卡的比例」，不是「剩餘空間的比例」 |
| `The model's max seq len is larger than the maximum number of tokens that can be stored in KV cache` | `--max-model-len` 超過 KV cache 能容納的單一序列長度，要降 context 或提高記憶體配額 |
| `appears path-like, but doesn't exist` | `--chat-template` 給了相對路徑，而該檔案不在套件裡 |
| 服務起來了但工具呼叫回傳純文字 | 少了 `--enable-auto-tool-choice` 與 `--tool-call-parser` |

---

## 二、架構：一個請求進來會發生什麼

vLLM 0.25.1（V1 引擎）把服務拆成兩個行程，在 log 裡直接看得到：

```
(APIServer  pid=1)  INFO ... Supported tasks: ['generate']
(EngineCore pid=81) INFO ... Initializing a V1 LLM engine (v0.25.1) ...
```

- **APIServer**：HTTP 端點、請求驗證、套用 chat template、tokenize、把工具呼叫從模型輸出解析成結構化欄位。
- **EngineCore**：排程與 GPU 執行。它不認得 messages 或 JSON，只認得 token 序列與 KV cache 區塊。

拆開的理由是前端的 Python 工作（JSON 解析、Jinja 渲染、tokenize）會與 GPU 排程搶 GIL。分成兩個行程之後，GPU 那一側可以維持穩定的批次節奏。

### 連續批次（continuous batching）

傳統推論服務是「湊滿一批 → 一起跑 → 一起回」。這在生成任務上很浪費：同一批裡有的請求生成 10 個 token 就結束，有的要生成 500 個，先結束的位置只能空等。

vLLM 的排程單位是**一個 decode step**，不是一個請求。每一步都重新決定這一步要跑哪些序列，結束的序列立刻離開、等待中的立刻補進來。log 裡的這行就是每一步的實況：

```
Engine 000: Avg prompt throughput: 423.7 tokens/s, Avg generation throughput: 16.0 tokens/s,
            Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: ...
```

`Running` 是這一刻在 GPU 上的序列數，`Waiting` 是排隊中的。**`Waiting` 持續大於 0 而 `Running` 上不去，通常不是算力不夠，是 KV cache 滿了**——新序列沒有區塊可配。

### prefill 與 decode

生成分成兩個性質完全不同的階段：

| 階段 | 做什麼 | 瓶頸 |
|---|---|---|
| **prefill** | 把整段 prompt 一次算完，填好 KV cache | 算力（一次算幾千個 token，矩陣夠大，GPU 吃得飽） |
| **decode** | 一次算一個 token | 記憶體頻寬（每步只算一個 token，卻要把全部權重讀一遍） |

上面 log 中 prompt throughput 423.7 tokens/s 而 generation throughput 只有 16.0 tokens/s，差了 26 倍，就是這個結構性差異。

兩者混在同一批會互相拖累：一個 8000 token 的 prefill 進來，同批的 decode 全部被卡住，使用者看到的是輸出突然停頓。vLLM 的 **chunked prefill**（本文的實例預設開啟）把長 prefill 切成小塊，跟 decode 交錯排程，用一點總吞吐換取穩定的逐字輸出。

### CUDA graph

decode 階段每一步都要發射幾百個 GPU kernel，而每個 kernel 的發射開銷在只算一個 token 時佔比很高。CUDA graph 把這串固定的 kernel 序列錄下來，之後整包重放，省掉逐次發射的 CPU 開銷。

代價是要為每種 batch size 各錄一份，佔顯存也佔啟動時間：

```
Profiling CUDA graph memory: PIECEWISE=51 (largest=512), FULL=35 (largest=256)
Estimated CUDA graph memory: 0.66 GiB total
Graph capturing finished in 16 secs, took 0.65 GiB
```

估計 0.66 GiB、實際 0.65 GiB，估得很準。要省這段時間可以加 `--enforce-eager` 關掉，但 decode 吞吐會下降——只適合除錯，不適合上線。

### 冷啟動時間的組成

實測一次完整啟動約 5–7 分鐘，拆開來看：

| 階段 | 26B FP8 | E4B bf16 |
|---|---|---|
| 權重載入 | 167 秒 | 約 190 秒 |
| 引擎初始化（profiling + KV cache + warmup） | 141 秒（含編譯 63 秒） | 121 秒（含編譯 72 秒） |

這個量級決定了運維節奏：**重啟一次 vLLM 不是「幾秒鐘的事」**。在共用 GPU 的機器上要暫停別人的服務來跑自己的東西，得把這 5–7 分鐘算進中斷時間裡。

---

## 三、KV cache：為什麼它決定併發

### 為什麼需要它

自迴歸生成每次只吐一個 token，然後把它接回輸入再算下一個。沒有快取的話，生成第 n 個 token 就要把前面 n−1 個 token 的注意力全部重算一次，總成本是 O(n²)。

但注意力的結構讓這件事有捷徑：每個 token 的 Key 與 Value 一旦算出來就不會再變（因果注意力只往前看）。把它們存起來，生成第 n 個 token 時只需要算新 token 的 Q、K、V，再跟快取中的 n−1 組 K/V 做注意力。成本從 O(n²) 降到 O(n)。

代價是記憶體。每個 token 每一層都要存一組 K 和一組 V：

```
每 token 每層的 KV 位元組數 = 2（K 與 V） × num_key_value_heads × head_dim × 精度位元組數
```

以 `gemma-4-E4B-it` 為例，`num_key_value_heads = 2`、`head_dim = 256`、bf16 為 2 bytes：

```
2 × 2 × 256 × 2 = 2,048 bytes/層/token
```

42 層若每層都要存，就是 86 KB/token。一個 32K 的 context 光 KV 就要 2.8 GB——一個請求。

### 樸素做法的問題

早期的推論框架把 KV cache 配成一段連續記憶體，而且**按 `max_model_len` 預留**：一個請求進來就先切 32K 的空間，不管它實際會不會用到。

這造成兩種浪費同時發生：一個只用 200 token 的請求佔著 32K 的坑（內部碎片），而釋放出來的坑大小不一，新請求要連續空間卻拼不出來（外部碎片）。實測研究指出這種配置方式下，實際被用到的顯存往往不到四成。

### PagedAttention：把作業系統的分頁搬過來

vLLM 的核心設計是把 KV cache 切成固定大小的**區塊（block）**，每個區塊存固定數量的 token（例如 16 個）。一個序列的 KV 不需要連續存放，而是由一張**區塊表**把邏輯位置對應到實際的物理區塊——這正是虛擬記憶體分頁的做法。

由此得到三件事：

1. **按需配置**：序列長多少就配多少區塊，浪費上限是一個區塊（最多 15 個 token 的空間），不再是預留整個 max_model_len。
2. **不需要連續空間**：外部碎片消失，釋放出來的區塊任何序列都能用。
3. **可共享**：多個序列若有相同前綴（同一段 system prompt、同一組工具宣告），可以指向同一批物理區塊，只有在要寫入時才複製。這就是 prefix caching——本文實例的 log 裡 `enable_prefix_caching=True` 是預設開啟的。

第 3 點對工具呼叫場景特別有價值：每個請求都帶著同一份長長的工具宣告，前綴共享讓這段只需要算一次、存一份。

### 實測：兩個模型的 KV cache 帳

同一張 L40S（46 GB），兩個模型的實際配置：

| | 26B A4B FP8 | E4B bf16 |
|---|---|---|
| `--gpu-memory-utilization` | 0.92 | 0.85 |
| 權重 | 25.8 GiB | 約 25.8 GiB（含 profiling 峰值） |
| CUDA graph | 0.66 GiB | 0.65 GiB |
| **可用 KV cache** | **12.75 GiB** | **22.01 GiB** |
| **KV cache 容量** | **321,600 tokens** | **1,033,831 tokens** |
| `--max-model-len` | 32,768 | 16,384 |
| **最大併發** | **9.81x** | **63.10x** |

「最大併發 9.81x」的算法是 `KV cache 容量 ÷ max_model_len`：321,600 ÷ 32,768 = 9.81。它的意思是**如果每個請求都用滿 32K context，同時只能服務 9.8 個**。實際併發通常遠高於此，因為多數請求用不到滿長度——但這個數字是容量的硬上限，也是調 `--max-model-len` 時真正在權衡的東西：context 砍半，併發加倍。

把記憶體除以 token 數，得到每個 token 的實際成本：

| | 每 token 的 KV |
|---|---|
| 26B A4B FP8 | 12.75 GiB ÷ 321,600 ≈ **41.6 KiB** |
| E4B bf16 | 22.01 GiB ÷ 1,033,831 ≈ **22.3 KiB** |

E4B 的 22.3 KiB 遠低於前面算出的 86 KB 理論值。差距來自兩個架構層級的機制。

### 機制一：滑動視窗注意力

`gemma-4-E4B-it` 的 `config.json` 裡：

```json
"layer_types": ["sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention", ...],
"sliding_window": 512
```

42 層裡以「5 層滑動 + 1 層全域」的節奏交替，滑動層的注意力只看最近 512 個 token。這種層不需要保留整個 context 的 KV，只需要滾動保留一個視窗——不論 context 是 4K 還是 32K，它的 KV 開銷都固定。

只有那幾層全域注意力層的開銷才隨 context 線性成長。這是近年長 context 模型的標準做法：把「記得很久以前的事」的成本集中在少數幾層。

### 機制二：跨層 KV 共享

同一份 config 裡還有一行：

```json
"num_kv_shared_layers": 18
```

`42 − 18 = 24`，代表第 24 層之後的 18 層**不維護自己的 KV cache**，直接讀前面某一層算好的。vLLM 的模型實作把這件事寫得很清楚：

```python
first_kv_shared_layer_idx = config.num_hidden_layers - num_kv_shared_layers
if layer_idx >= first_kv_shared_layer_idx:
    self.is_kv_shared_layer = True
    # 找出前面最後一個「同樣注意力型別」的非共享層當來源
    prev_layers = config.layer_types[:first_kv_shared_layer_idx]
    kv_shared_layer_index = len(prev_layers) - 1 - prev_layers[::-1].index(current_layer_type)
```

前向傳播時，共享層只對 Q 做 RoPE，K/V 既不做 norm 也不套 RoPE，算出來就丟掉：

```python
if not self.is_kv_shared_layer:
    k = self.k_norm(k); q, k = self.rotary_emb(positions, q, k)
    v = self.v_norm(v)
else:
    q = self.rotary_emb(positions, q, k)[0]   # 只有 Q
```

效果是 KV cache 的層數從 42 降到 24，直接省掉四成三。

這個機制有一個容易忽略的下游影響：**對這 18 層的 `k_proj` / `v_proj` 做微調不會改變任何輸出**，因為它們的結果根本不進注意力。實際交付的 LoRA adapter 正是這樣分佈的——第 0～23 層掛 q/k/v/o，第 24～41 層只掛 q/o。這不是訓練時隨手的選擇，是架構決定的（見 [LoRA adapter 教學](gemma4-lora-adapter-guide.md)）。

把兩個機制疊起來就解釋了 22.3 KiB 這個數字的量級。至於 vLLM 混合型 KV cache 對「滑動層」與「全域層」分組配置的確切公式，本文沒有逐行追到，因此上面只用「量級相符」而不是精確推導來說明。

### 監看：三個要盯的指標

```
Engine 000: Avg prompt throughput: X tokens/s, Avg generation throughput: Y tokens/s,
            Running: N reqs, Waiting: M reqs, GPU KV cache usage: Z%
```

| 指標 | 怎麼讀 |
|---|---|
| `GPU KV cache usage` | 長期貼近 100% 表示容量吃緊，新請求會排隊或被搶佔 |
| `Waiting` | 持續 > 0 而 `Running` 上不去，多半是 KV cache 滿了，不是算力不足 |
| `Avg generation throughput` | decode 速度。它比 prompt throughput 低一到兩個數量級是正常的 |

KV cache 滿了的時候 vLLM 會**搶佔**（preemption）：把某個序列的區塊釋放掉，等有空間再重算。這不會出錯，但那個請求的延遲會突然變長。log 裡出現 preemption 訊息，就是該降 `--max-model-len` 或加卡的訊號。

---

## 四、實務上怎麼調

以「一張卡、要服務多少人」為出發點：

1. **先量 KV cache 的實際容量**：啟動一次，把 `Available KV cache memory` 與 `GPU KV cache size` 記下來。這兩個數字是所有容量規劃的基礎，不用猜的。
2. **用 `KV cache 容量 ÷ 目標 context` 估併發上限**，跟預期同時線上人數比對。
3. **不夠就先砍 context，不要先加卡**：`--max-model-len` 從 32K 降到 16K，併發直接加倍，而多數應用根本用不到 32K。
4. **`--gpu-memory-utilization` 只在獨占整張卡時才往上調**。這台機器上同時跑著別的服務，把它設到 0.92 等於宣告整張卡都是自己的。
5. **工具呼叫類的應用要確認 prefix caching 有開**：每個請求都帶同一份工具宣告，共享前綴省下的是實打實的 prefill 算力。

最後一點與其說是調參，不如說是紀律：**這些數字每次改設定都要重新量**。`--max-model-len`、`--gpu-memory-utilization`、量化格式、掛不掛 LoRA，任何一項變動都會讓上面那張表整個重算。

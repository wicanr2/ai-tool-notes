# Attention 的四個軸：FlashAttention 與 Sparse Attention 不是同一類東西

vLLM 的注意力後端目錄裡有二十幾個檔案，名字全都掛著 `attn`：`flash_attn`、`flashinfer`、`triton_attn`、`gdn_attn`、`linear_attn`、`mamba2_attn`、`short_conv_attn`、`turboquant_attn`……名字長得像同一類東西，很容易以為是「二十幾種可以互換的注意力演算法」。

但目錄裡有一個檔名把真相講清楚了：

```
vllm/v1/attention/backends/mla/flashattn_mla_sparse.py
```

**FlashAttention、MLA、Sparse 三個名詞同時出現在同一個檔名裡**——因為它們不是三個選項，是三個**互相垂直的軸**。這個檔案是「用 FlashAttention 的算法、算 MLA 壓縮過的 KV、只在稀疏選中的 token 上算」。

這篇把那些名詞歸位：哪些改變輸出、哪些不改變，哪些是部署時能選的、哪些是模型訓練時就定死的。

實例取自 vLLM 0.25.1 與各模型的實際設定檔，查證日 **2026-08-18**。

- **每個名詞的機制細節**：[Attention 名詞解釋](attention-terms-explained.md)（GQA、MLA、線上 softmax、SSM、DeltaNet、TurboQuant、硬體世代與 kernel 可攜性）
- 相關：[讀懂模型設定檔](model-config-fields-reference.md)、[架構圖鑑](llm-architecture-map-2026.md)、[vLLM 架構與 KV cache](vllm-serving-and-architecture.md)

---

## 一、先看標準注意力貴在哪

```
attn = softmax(Q Kᵀ / √d) · V
```

序列長度 `S`、頭維度 `d`。這一行有三筆成本，而且它們的來源不同：

| 成本 | 量級 | 誰在付 |
|---|---|---|
| **計算量** | `O(S² · d)` | prefill 階段的算力 |
| **中間矩陣** | `O(S²)` | 存 `Q Kᵀ` 的結果，用完就丟 |
| **KV cache** | `O(S · d)` | 顯存，而且要一直留著 |

中間矩陣那一項最容易被低估。context 32768、fp16：

```
32768 × 32768 × 2 bytes = 2.15 GB     ← 單一個頭、單一層
```

八個頭就是 17 GB，而這還只是一層、還只是「算完就丟」的暫存。**這個數字就是 FlashAttention 要解決的問題。**

KV cache 則是另一回事：它不隨 `S²` 成長，但它不能丟，而且每個請求各一份。這是併發上限的來源（見 [vLLM 架構與 KV cache](vllm-serving-and-architecture.md)）。

---

## 二、四個軸

| 軸 | 改的是什麼 | 輸出會變嗎 | 誰決定 | 代表 |
|---|---|---|---|---|
| **一、怎麼算** | 計算的排程與記憶體搬移 | **不變**（數學等價） | 部署時可選 | FlashAttention、FlashInfer、Triton |
| **二、存什麼** | KV 的表示形式 | 變 | 模型訓練時定 | GQA、MQA、MLA、KV 共享 |
| **三、看哪些 token** | 注意力的作用範圍 | 變 | 模型訓練時定 | 滑動視窗、NSA |
| **四、換掉機制** | 不用注意力了 | 變 | 模型訓練時定 | 線性注意力、SSM、GDN、KDA |

**軸一與其他三軸的性質完全不同。** 軸一不改變模型，同一份權重換個後端跑，輸出（在浮點誤差內）一模一樣；軸二三四改的是模型本身，權重必須照那個設計訓練出來，換不掉。

這也回答了「FlashAttention 和 Sparse Attention 哪個好」——這個問題本身不成立，它們解的不是同一件事，而且會同時用。

---

## 三、軸一：怎麼算（FlashAttention）

### 問題

標準實作照著公式一步步做：

```
S = Q @ K.T        寫進 HBM   ← S×S 的矩陣
P = softmax(S)     讀回、寫回 HBM
O = P @ V          讀回
```

那個 `S×S` 矩陣在 GPU 的高頻寬記憶體（HBM）裡進出好幾趟。而 GPU 的算力遠遠快過它讀寫 HBM 的速度——**時間全花在搬資料，不是在算**。

### 做法

FlashAttention 的核心是兩件事：

**分塊（tiling）**：把 Q、K、V 切成小塊，一次只把塊搬進晶片上的 SRAM（比 HBM 快一到兩個數量級但很小），算完該塊的貢獻就丟掉，**完整的 `S×S` 矩陣從頭到尾沒有被完整寫出來過**。

**線上 softmax（online softmax）**：softmax 需要整列的最大值與總和才能正規化，看起來不能分塊算。解法是邊算邊維護「目前為止的最大值」與「目前為止的總和」，遇到更大的值就把先前累積的結果按比例重新縮放。這是數值上等價的重排，不是近似。

結果是 HBM 的讀寫量從 `O(S²)` 降到約 `O(S²d/M)`（`M` 是 SRAM 大小），而**輸出與標準實作在數學上完全相同**。

### 世代差異

| 版本 | 主要改進 |
|---|---|
| FlashAttention-1 | 提出 tiling + online softmax，解決 HBM 進出 |
| FlashAttention-2 | 改善工作分配與平行度，讓 GPU 的計算單元不閒置 |
| FlashAttention-3 | 針對 Hopper 的非同步搬運與低精度（FP8）最佳化 |

### 它不能解決什麼

**FlashAttention 不會讓 KV cache 變小。** 它省的是中間矩陣與 HBM 流量，KV 該存多少還是多少。把 context 開到 128K 而顯存爆掉時，換 FlashAttention 沒有用——那是軸二和軸三的問題。

### 同一軸上的其他東西

vLLM 的後端目錄裡多數檔案都在這一軸：

| 後端 | 定位 |
|---|---|
| `flash_attn` | FlashAttention 系列的核心實作 |
| `flashinfer` | 另一套 kernel 庫，對某些形狀與量化組合更快 |
| `triton_attn` | 用 Triton 寫的版本，可攜性好、方便客製 |
| `flex_attention` | PyTorch 的可組合注意力，用來表達自訂 mask 樣式 |
| `cpu_attn` / `rocm_*` / `xpu_*` | 不同硬體平台的對應實作 |
| `turboquant_attn` | **例外**：它壓的是 KV cache 本身（Lloyd-Max 量化），會改變輸出，見下方說明 |

**PagedAttention 也在這一軸，但更偏系統層**：它管的是 KV cache 在顯存裡怎麼分頁配置，不改注意力的數學。

### 一個不合分類的例外：執行期 KV 量化

`turboquant_attn` 與 `--kv-cache-dtype fp8` 這類選項落在四軸的縫隙裡：它們是**部署時的選擇**（模型不用重訓，這點像軸一），但**會改變輸出**（量化有損，這點像軸二）。

TurboQuant 的做法是用 Lloyd-Max 最佳量化器求出碼本，把 KV cache 壓縮存放。實測的版面是 head_dim 256 時「K 壓到 100 bytes + V 維持 fp16 512 bytes」，相較全 fp16 的 1024 bytes 省四成。

把它們獨立看待比硬塞進某一軸有用：**這是唯一一類「你可以自己開、但要付品質代價」的注意力相關選項**，該不該開要實測，不能只看容量數字。機制細節見 [名詞解釋](attention-terms-explained.md)。

---

## 四、軸二：存什麼（KV 的表示）

這一軸的目標只有一個：**讓每個 token 要留下來的 K/V 更小。**

### MHA → GQA → MQA

```
MHA  Multi-Head Attention      每個 Q 頭配一組 K/V       KV 最大
GQA  Grouped-Query Attention   幾個 Q 頭共用一組 K/V     可調
MQA  Multi-Query Attention     所有 Q 頭共用一組 K/V     KV 最小
```

在 config 裡就是 `num_attention_heads` 與 `num_key_value_heads` 的比值：

| 模型 | Q 頭 | KV 頭 | 比值 | 類型 |
|---|---:|---:|---|---|
| Muse-Glimmer-30B | 32 | 2 | 16:1 | GQA |
| gemma-4-31B | 32 | 16 | 2:1 | GQA |
| gemma-4-E4B | 8 | 2 | 4:1 | GQA |
| DeepSeek-V4-Flash | 64 | 1 | 64:1 | 接近 MQA |

比值越大 KV 越小，但表達力也越少——所有模型都在這條線上選一個點。

### MLA：換一種表示

**MLA**（Multi-head Latent Attention）不是「共用」，是**壓縮**：把 K/V 投影到一個低維潛在空間，KV cache 存的是那個潛在向量，用的時候再展開。

相關欄位是 `kv_lora_rank`（潛在維度）、`q_lora_rank`、`qk_rope_head_dim` / `qk_nope_head_dim`（帶不帶位置編碼的維度分開處理）。

代價是實作複雜——這就是為什麼 vLLM 要為它開一個獨立的 `mla/` 目錄，裡面有十幾個針對不同硬體與稀疏組合的變體。

### KV 共享：乾脆不存

gemma-4 的 E 系列還多做一件事：`num_kv_shared_layers`。最後那幾層**完全不算自己的 K/V**，直接讀前面同型別層算好的。E4B 是 42 層裡有 18 層如此，等於省掉四成三的 KV。

副作用是這些層的 `k_proj` / `v_proj` 對輸出沒有影響——微調時對它們掛 LoRA 是白做的（見 [LoRA adapter 教學](gemma4-lora-adapter-guide.md)）。

---

## 五、軸三：看哪些 token（稀疏化）

前面兩軸都在「每個 token 都要看到所有 token」的前提下省錢。這一軸直接動那個前提。

依據是一個經驗觀察：注意力的分數矩陣**非常稀疏**——大部分 query 的注意力集中在少數幾個 token 上，其餘接近零。既然如此，那些接近零的部分不算也罷。

### 滑動視窗（SWA，Sliding Window Attention）

最簡單的稀疏樣式：只看最近 `sliding_window` 個 token。

好處是 KV **不隨 context 成長**——視窗多大就存多少。代價是這一層完全看不到更早的內容，所以沒有模型全部用滑動視窗，都是交錯：

| 模型 | 視窗 | 滑動 : 全域 |
|---|---:|---|
| gemma-4-E4B | 512 | 35 : 7（5:1） |
| gemma-4-31B | 1024 | 50 : 10（5:1） |
| Muse-Glimmer-30B | 2048 | 39 : 13（3:1） |
| DeepSeek-V4 | 128 | 局部分支 |

**少數幾層全域負責長距離，其餘層用滑動視窗負責局部。** 這是目前最主流的做法。

### NSA：學一個索引器來挑

**NSA**（Native Sparse Attention，原生稀疏注意力）比固定樣式更進一步：**用一個輕量的索引器動態決定要看哪些 token。**

DeepSeek-V4-Flash 的相關欄位：

```
index_n_heads   = 64      索引器的頭數
index_head_dim  = 128     索引器的頭維度
index_topk      = 512     每個 query 只注意 512 個被選中的 token
compress_ratios = [...]   逐層的壓縮倍率
sliding_window  = 128     局部分支的視窗
```

`index_topk = 512` 的意思是：**不管 context 是 4K 還是 1M，每個 query 都只在 512 個 token 上算注意力。** 注意力的成本從 `O(S²)` 變成 `O(S · k)`，`k` 固定。

「Native」指的是這套稀疏結構**在預訓練階段就存在**，模型是照著它學的——不是訓練完之後再套上去的近似。事後硬加稀疏會掉品質，原生訓練的不會。

vLLM 為此準備了一整組檔案：`mla/indexer.py`、`mla/sparse_swa.py`、`mla/flashmla_sparse.py`、`mla/flashinfer_mla_sparse.py`。

### 與 FlashAttention 的關係

這裡是最容易混淆的地方，講白一點：

> **FlashAttention 算的是「全部 S×S，但算得省」。**
> **Sparse Attention 算的是「只算其中一部分」。**
> **兩者疊起來就是「只算其中一部分，而且算得省」——那就是 `flashattn_mla_sparse.py`。**

FlashAttention 的輸出與標準注意力**完全相同**；Sparse Attention 的輸出**不同**，它是一個近似，只是模型被訓練成在這個近似下工作。

---

## 六、軸四：換掉機制

前三軸都還在做注意力。這一軸不做了。

### 線性注意力與 SSM

標準注意力對每個新 token 都要回頭掃描全部歷史，成本隨長度成長。**SSM**（State Space Model，狀態空間模型，Mamba 家族）改成維護一個**固定大小的狀態向量**：每來一個 token 就更新狀態，要用的時候只讀狀態。

```
注意力：  輸出 = f(當前 token, 全部歷史 token)     成本隨長度成長
SSM：     狀態 = g(狀態, 當前 token)              成本固定
          輸出 = h(狀態, 當前 token)
```

記憶體完全不隨序列長度成長——這是它最大的優勢，也是它最大的限制：**固定大小的狀態裝不下任意多的資訊**，需要精確回憶很久以前某個特定 token 時會失手。

Qwen 3.5+ 的相關欄位（GGUF 那側標得更清楚）：

```
ssm.state_size      = 128    狀態向量維度
ssm.conv_kernel     = 4      短卷積核大小
ssm.inner_size      = 4096   擴展後的內部維度
ssm.time_step_rank  = 32     Δ（時間步長）投影的秩
ssm.group_count     = 16     狀態分組數
```

### 各家的變體

| 名稱 | 全名 | 用在 |
|---|---|---|
| GDN | Gated DeltaNet | vLLM 的 `gdn_attn` 後端 |
| KDA | Kimi Delta Attention | Kimi K3（93 層裡 69 層） |
| Mamba / Mamba2 | — | vLLM 的 `mamba*_attn` 後端 |

它們都是「用固定狀態取代 KV cache」這個想法的不同實作。

### 一定要混

沒有模型全部用線性層，都是交錯：

| 模型 | 線性 : 全域 |
|---|---|
| Qwen3.6-27B | 48 : 16（3:1） |
| Qwen3.8-2.4T-A95B | 69 : 23（3:1） |
| Kimi-K3 | 69 KDA : 24 Gated MLA（約 3:1） |

**少數幾層真正的注意力補足「精確回憶」的能力，其餘層用線性層扛長度。** 跟軸三的滑動視窗交錯是同一種思路，只是替代品不同。

---

## 七、它們怎麼疊起來

四個軸互相垂直，所以會同時出現。把幾個模型攤開：

| 模型 | 軸二（存什麼） | 軸三（看哪些） | 軸四（換機制） | 軸一（怎麼算） |
|---|---|---|---|---|
| gemma-4-E4B | GQA 4:1 + KV 共享 18 層 | 滑動視窗 512，5:1 交錯 | — | FlashAttention |
| gemma-4-31B | GQA 2:1 | 滑動視窗 1024，5:1 交錯 | — | FlashAttention |
| Muse-Glimmer-30B | GQA 16:1 | 滑動視窗 2048，3:1 交錯 | — | FlashAttention |
| DeepSeek-V4-Flash | **MLA** | **NSA top-512** + 滑動 128 | — | FlashAttention（`flashattn_mla_sparse`） |
| Qwen3.8-2.4T | GQA | — | **SSM**，3:1 交錯 | FlashAttention（全域層）+ `gdn_attn`（線性層） |
| Kimi-K3 | **MLA**（Gated） | — | **KDA**，約 3:1 | 同上 |
| MiniMax-M2.7 | GQA 6:1 | — | — | FlashAttention |

`MiniMax-M2.7` 是唯一只用軸二的：**它不做任何注意力稀疏化，全部 62 層都是完整的全域注意力**，代價是 context 只到 200K，明顯短於同期其他模型。這反過來證明了軸三和軸四確實是為了長 context 而存在的。

---

## 八、實務上你能決定什麼

**軸一是你能選的，而且只有這一軸。** `--attention-backend flashinfer` 這種參數換的是實作，模型不變、輸出不變，只影響速度與硬體相容性。選錯了會慢，不會錯。

**軸二三四是模型作者在訓練時定死的。** 你唯一能做的是**確認推論引擎支援**——這也是[相容性檢查](model-config-fields-reference.md)那段腳本在做的事：

```
· MLA 低秩注意力 → 引擎需專門支援
· NSA 稀疏注意力 (top-512) → 引擎需專門支援
· 含 SSM/線性注意力層 → 引擎需專門支援
```

看到這些標記，代表這個模型需要引擎有對應的 kernel。**新機制的支援度往往落後模型發布好幾個月**，而且常常綁特定硬體世代——DeepSeek-V4 的 NSA 在非 Hopper 卡上要走 tilelang 的 fallback 路徑，就是這個現象。

選型時的順序：先確認引擎版本吃不吃得下這個 `model_type`，再談速度與品質。跑不起來的模型，benchmark 分數再漂亮都沒有意義。

---

## 附錄：一句話分辨

| 你看到 | 它在改 | 輸出會變嗎 |
|---|---|---|
| FlashAttention / FlashInfer / Triton / PagedAttention | 怎麼算、怎麼搬記憶體 | **不變** |
| TurboQuant / `--kv-cache-dtype fp8` | 壓縮 KV cache（部署時可選） | 變（有損量化） |
| MHA / GQA / MQA / MLA / KV 共享 | KV 存成什麼樣子 | 變（模型層面） |
| 滑動視窗 / NSA / 各種稀疏樣式 | 注意力看哪些 token | 變（模型層面） |
| 線性注意力 / SSM / Mamba / GDN / KDA | 不用注意力了 | 變（模型層面） |

**名字裡有 Flash 的通常在軸一，有 Sparse 的在軸三，有 Linear / SSM / Delta 的在軸四，有 Query / KV 的在軸二。** 這個粗略的規則能擋掉大部分的混淆。

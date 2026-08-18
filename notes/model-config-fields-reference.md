# 讀懂模型設定檔：從欄位到部署決策

拿到一個沒看過的模型，在**不下載任何權重**的前提下，要回答三個問題：

1. 我的卡塞不塞得下？
2. 生成會有多快？
3. 我手上的推論引擎撐不撐得住？

三個答案全都藏在幾個 JSON 檔的欄位裡。這篇把那些欄位攤開——每個都給英文全名、實際用途、以及「這個數字變了會怎樣」——然後用三段可以直接複製執行的程式碼，把它們變成上面三個問題的答案。

實例值取自實際模型設定檔，查證日 **2026-08-18**。

- 相關：[HuggingFace 模型 repo 檔案逐一解說](gemma4-hf-repo-files.md)（每個**檔案**是什麼）、[Attention 的四個軸](attention-mechanisms-taxonomy.md)（欄位背後的機制分類）、[架構圖鑑](llm-architecture-map-2026.md)、[推論速度怎麼估](llm-decode-throughput-formula.md)

---

## 第 0 步：資料在哪三個地方

同一個模型的資訊分散在三處，命名不同、涵蓋範圍也不同。

| 來源 | 是什麼 | 只有它有 | 怎麼取 |
|---|---|---|---|
| repo 的 `config.json` | 模型結構定義 | `layer_types` 逐層清單 | `curl .../raw/main/config.json` |
| HF API metadata | Hub 對 repo 的索引與統計 | 按 dtype 分組的參數量、`gated`、commit sha | `curl .../api/models/<id>` |
| GGUF metadata | 量化檔內嵌的結構描述 | 全域層與滑動層分開的頭維度、SSM 欄位 | `curl <ollama>/api/show` |

```bash
M=google/gemma-4-E4B-it

# 一：結構定義
curl -s "https://huggingface.co/$M/raw/main/config.json" | jq .

# 二：Hub 統計（含實際參數量，不必下載權重）
curl -s "https://huggingface.co/api/models/$M" | jq '{safetensors, gated, pipeline_tag, sha, transformersInfo}'

# 三：GGUF（若模型已在 Ollama 上）
curl -s http://<host>:11434/api/show -d '{"model":"gemma4:e4b"}' | jq .model_info
```

**多模態模型的結構欄位在 `text_config` 底下**，頂層只有 `architectures`、`model_type` 與各模態的子設定。忘了這件事，讀什麼都是 `null`。

---

## 第 1 步：先把注意力的計算寫出來

大部分欄位的意義，在看到它出現在哪個矩陣的哪一維時就清楚了。

```
輸入 x                 [seq_len, hidden_size]

Q = x · Wq             Wq: [hidden_size, num_attention_heads × head_dim]
K = x · Wk             Wk: [hidden_size, num_key_value_heads × head_dim]
V = x · Wv             Wv: [hidden_size, num_key_value_heads × head_dim]

attn = softmax(Q Kᵀ / √head_dim) · V
out  = attn · Wo       Wo: [num_attention_heads × head_dim, hidden_size]

FFN(out) = down( act(gate(out)) ⊙ up(out) )
                       gate/up: [hidden_size, intermediate_size]
                       down:    [intermediate_size, hidden_size]
```

三件事立刻可以推出來：

- **KV cache 的大小只跟 `num_key_value_heads × head_dim` 有關**，跟 `num_attention_heads` 無關。
- **`num_attention_heads` 大於 `num_key_value_heads` 就是 GQA**，多個 Q 頭共用一組 K/V，純粹為了壓 KV cache。
- **FFN 通常佔全模型參數的三分之二**（`3 × hidden × intermediate` vs 注意力的約 `2 × hidden × heads × head_dim`）。這就是為什麼 MoE 動的是 FFN 而不是注意力。

---

## 第 2 步：欄位逐項

### 骨架

| 欄位 | 英文全名 | 用途 | 這個數字變了會怎樣 |
|---|---|---|---|
| `num_hidden_layers` | number of hidden layers | transformer 區塊重複幾次 | 線性影響參數量、顯存、每 token 計算量 |
| `hidden_size` | hidden state dimension（論文常寫 d_model） | 殘差流的寬度，所有矩陣共用的一邊 | 平方影響參數量（矩陣兩邊都跟它有關） |
| `vocab_size` | vocabulary size | 詞表大小 | 影響 embedding 與輸出頭；大詞表在小模型上佔比驚人 |
| `tie_word_embeddings` | tied word embeddings（權重綁定） | 輸入 embedding 與輸出頭是否共用同一份權重 | `true` 省一份詞表矩陣。gemma-4 是 262144 × 2560 ≈ 6.7 億參數 |
| `dtype` / `torch_dtype` | data type | 權重的原生精度 | 決定「每參數幾個位元組」的基準值 |
| `model_type` | model type | 架構識別碼 | **推論引擎靠它決定用哪份實作**，不認得就跑不起來 |
| `hidden_act` | hidden activation function | FFN 的啟動函數 | `silu`、`gelu_pytorch_tanh`；影響數值行為不影響大小 |
| `rms_norm_eps` | Root Mean Square normalization epsilon | 正規化的數值下限，防除以零 | 幾乎不用動；改了可能造成數值不穩 |
| `initializer_range` | initializer range | 訓練時的權重初始化標準差 | **推論無關**，是訓練殘留 |

### 注意力

| 欄位 | 英文全名 | 用途 | 這個數字變了會怎樣 |
|---|---|---|---|
| `num_attention_heads` | number of attention heads | Q 分成幾個頭 | 影響 Q/O 投影的參數量，**不影響 KV cache** |
| `num_key_value_heads` | number of key-value heads | K/V 分成幾個頭 | **KV cache 與它成正比**。等於 Q 頭數是 MHA、少於是 GQA、等於 1 是 MQA |
| `head_dim` | attention head dimension | 每個頭的維度 | 同時影響參數量與 KV cache。**不一定等於 `hidden_size / num_attention_heads`** |
| `global_head_dim` | global-attention head dimension | 全域注意力層專用的頭維度（gemma-4） | 隨 context 成長的那些層用這個值。gemma-4 是滑動層的兩倍 |
| `sliding_window` | sliding window size（SWA，Sliding Window Attention） | 局部注意力只看最近幾個 token | 視窗層的 KV 固定在這個長度，**不隨 context 成長** |
| `layer_types` | layer types | 逐層標注注意力型別 | 數出 `full_attention` 有幾層——那才是會隨 context 爆的部分 |
| `full_attention_interval` | full-attention interval | 每隔幾層放一層全域（Qwen 用） | 等價於 `layer_types` 的壓縮寫法 |
| `num_kv_shared_layers` | number of KV-shared layers | 最後幾層直接讀前面層算好的 K/V | 這些層**完全不維護自己的 KV**，也不需要對它們的 k/v 投影做微調 |
| `attention_bias` | attention bias | QKVO 投影是否帶偏置項 | 參數量差異很小，但實作必須對得上否則載入失敗 |
| `attention_dropout` | attention dropout | 訓練期的隨機失活比例 | **推論時不生效** |
| `max_position_embeddings` | maximum position embeddings | 位置編碼支援的最大長度 | context 的硬上限，超過就要靠 `rope_scaling` 外推 |
| `use_cache` | use KV cache | 是否啟用 KV 快取 | 推論一律 `true`；`false` 只在特殊除錯用 |

### 前饋網路與混合專家

| 欄位 | 英文全名 | 用途 | 這個數字變了會怎樣 |
|---|---|---|---|
| `intermediate_size` | FFN intermediate dimension | dense FFN 的中間層寬度 | 通常是 `hidden_size` 的 2.7–4 倍；佔全模型參數大宗 |
| `num_experts` / `n_routed_experts` | number of routed experts | 路由專家總數（MoE，Mixture of Experts） | **決定總參數量**，但不影響每 token 的計算量 |
| `num_experts_per_tok` / `top_k_experts` | number of experts per token（top-k） | 每個 token 實際啟用幾個專家 | **決定激活參數量**，也就是速度 |
| `moe_intermediate_size` | MoE expert intermediate dimension | **單一專家**的中間層寬度 | 算 MoE 參數量要用這個，不是 `intermediate_size` |
| `n_shared_experts` / `shared_expert_intermediate_size` | shared experts | 每個 token 都會經過的共享專家 | 承載通用能力，讓路由專家專注分化 |
| `norm_topk_prob` | normalize top-k probabilities | 選中專家的權重是否重新正規化 | 影響專家輸出的加權方式 |
| `topk_method` | top-k selection method | 路由的選擇演算法（如 `noaux_tc`） | 決定負載平衡怎麼做；`noaux` 系列不靠輔助損失 |
| `scoring_func` | router scoring function | 路由分數的計算方式 | `softmax`、`sigmoid`、`sqrtsoftplus` |
| `routed_scaling_factor` | routed output scaling factor | 專家輸出的整體縮放 | 平衡共享與路由專家的貢獻 |
| `router_aux_loss_coef` | router auxiliary loss coefficient | 訓練期的負載平衡損失權重 | **推論無關** |

### 低秩壓縮與稀疏注意力（DeepSeek-V4 系）

| 欄位 | 英文全名 | 用途 |
|---|---|---|
| `q_lora_rank` | query low-rank projection rank | Q 先降到這個維度再升回去，省 Q 投影的參數 |
| `kv_lora_rank` | key-value low-rank projection rank | K/V 壓到的潛在維度。**MLA 的 KV cache 存的就是這個壓縮表示** |
| `o_lora_rank` | output low-rank projection rank | 輸出投影也走低秩 |
| `qk_rope_head_dim` | query-key RoPE head dimension | 帶位置編碼的那部分維度 |
| `qk_nope_head_dim` | query-key non-positional head dimension | 不帶位置編碼的那部分維度 |
| `v_head_dim` | value head dimension | V 的頭維度，可與 K 不同 |
| `index_n_heads` / `index_head_dim` | indexer heads / head dimension | 輕量索引器，用來挑選要注意哪些 token |
| `index_topk` | indexer top-k | **每個 query 只注意這麼多個被選中的 token** |
| `compress_ratios` | per-layer compression ratios | 逐層的壓縮倍率 |

**MLA**（Multi-head Latent Attention，多頭潛在注意力）把 K/V 投影到低維潛在空間再存，取代傳統的「每個頭各存一份 K/V」。

`index_*` 與 `compress_*` 這組欄位屬於 **NSA**（Native Sparse Attention，原生稀疏注意力）：一個輕量索引器先挑出最相關的 `index_topk` 個 token，注意力只在那些 token 上算。DeepSeek-V4-Flash 的 `index_topk = 512` 意味著**不管 context 是 4K 還是 1M，每個 query 都只注意 512 個 token**——這才是它能撐到 1M context 的主因，MLA 只是把每個 token 的 KV 存得更小。ktransformers 的部署文件直接把這個路徑稱為「NSA sparse MLA」，可以互相佐證。

### 線性注意力 / 狀態空間模型（Qwen 3.5+）

| 欄位 | 英文全名 | 用途 |
|---|---|---|
| `linear_num_key_heads` / `linear_key_head_dim` | linear-attention key heads / dimension | 線性層 K 側的頭數與維度 |
| `linear_num_value_heads` / `linear_value_head_dim` | linear-attention value heads / dimension | 同上，V 側 |
| `linear_conv_kernel_dim` | short convolution kernel size | 線性層前的短卷積核大小 |
| `mamba_ssm_dtype` | Mamba SSM computation dtype | 狀態計算的精度（通常維持 fp32 以免發散） |
| `attn_output_gate` | attention output gate | 注意力輸出是否過閘門 |

GGUF 那一側標得更直白：`ssm.state_size`、`ssm.conv_kernel`、`ssm.inner_size`、`ssm.time_step_rank`、`ssm.group_count`。

**SSM**（State Space Model，狀態空間模型，Mamba 家族）維護一個**固定大小的狀態**，而不是逐 token 累積的 KV cache。所以它的記憶體完全不隨序列長度成長——但也因此需要推論引擎專門支援，狀態管理與 KV cache 是兩套不同的機制。

`time_step_rank` 是 SSM 裡 Δ（時間步長）投影的秩，`state_size` 是狀態向量的維度，`inner_size` 是擴展後的內部維度。

### 位置編碼

| 欄位 | 英文全名 | 用途 | 這個數字變了會怎樣 |
|---|---|---|---|
| `rope_theta` / `rope.freq_base` | RoPE base frequency（Rotary Position Embedding） | 旋轉位置編碼的頻率基數 | **越大越適合長 context**。改了會破壞既有位置行為，不能隨意調 |
| `rope_parameters` | RoPE parameters（per layer type） | 分注意力型別各給一組 | gemma-4：全域層 1e6、滑動層 1e4 |
| `partial_rotary_factor` | partial rotary factor | 只對一部分維度套 RoPE | 0.25 代表只有四分之一的維度帶位置資訊 |
| `rope_scaling` | RoPE scaling | 外推到訓練長度以外的縮放方法 | 用來把 context 拉長，代價是長距離精度下降 |

### 輸出與推測解碼

| 欄位 | 英文全名 | 用途 |
|---|---|---|
| `final_logit_softcapping` | final logit soft capping | 用 tanh 把輸出 logits 軟性壓在上限內，避免極端值 |
| `logit_scale` | logit scale | 輸出前的縮放係數 |
| `num_nextn_predict_layers` / `mtp_num_hidden_layers` | multi-token prediction layers（MTP） | **內建的推測解碼草稿頭層數** |

`num_nextn_predict_layers` 不是 0，代表模型自帶推測解碼模組，可直接開 MTP / EAGLE——通常是一到兩倍的速度，而且不用另外準備草稿模型。這個欄位很容易被漏掉。

---

## 第 3 步：實戰一 — 算顯存

不下載權重，只用 API 與 config 算出「這個模型要多少顯存」。

```python
import json, urllib.request, urllib.error

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    return json.load(urllib.request.urlopen(req, timeout=30))

def vram(repo, ctx=8192, batch=1, kv_bytes=2):
    api = get(f"https://huggingface.co/api/models/{repo}")
    cfg = get(f"https://huggingface.co/{repo}/raw/main/config.json")
    tc  = cfg.get("text_config", cfg)

    # 權重：按 dtype 分別算，不能用「參數量 × 2」
    W = {"BF16":2, "F16":2, "F32":4, "F64":8, "I64":8, "I32":4,
         "I16":2, "I8":1, "U8":1, "F8_E4M3":1, "F8_E5M2":1, "F8_E8M0":1}
    dist = (api.get("safetensors") or {}).get("parameters", {})
    weight = sum(n * W.get(dt, 2) for dt, n in dist.items())

    # KV：只有全域注意力層會隨 context 成長
    types  = tc.get("layer_types") or []
    note   = ""
    if types:
        shared = tc.get("num_kv_shared_layers", 0) or 0
        owned  = types[: len(types) - shared] if shared else types
        n_glob = owned.count("full_attention")   # 共享層不維護自己的 KV
    else:
        n_glob = tc["num_hidden_layers"]         # 沒有逐層清單 → 全部當全域，這是上界
        note   = "（無 layer_types，以全部為全域層估算，實際會更低）"
    if tc.get("index_topk"):                     # NSA：注意力只落在 top-k 個 token
        note = f"（NSA top-{tc['index_topk']}，KV 實際不隨 ctx 線性成長，下值為高估）"
    hd_glob = tc.get("global_head_dim") or tc.get("head_dim")
    kvh     = tc.get("num_key_value_heads") or tc["num_attention_heads"]
    kv      = 2 * n_glob * kvh * hd_glob * kv_bytes * ctx * batch

    print(f"{repo}")
    print(f"  參數量      {sum(dist.values())/1e9:>9.1f} B   {dist}")
    print(f"  權重        {weight/1e9:>9.1f} GB")
    print(f"  隨 ctx 成長的層 {n_glob} 層 × {kvh} kv_heads × {hd_glob} head_dim")
    print(f"  KV @ {ctx}   {kv/1e9:>9.2f} GB  (batch {batch}) {note}")
    print(f"  合計        {(weight+kv)/1e9:>9.1f} GB")
    return weight, kv

vram("google/gemma-4-E4B-it", ctx=32768)
vram("google/gemma-4-31B-it", ctx=32768)
vram("deepseek-ai/DeepSeek-V4-Flash", ctx=32768)
```

實際跑出來：

```
google/gemma-4-E4B-it
  參數量            8.0 B
  權重             16.0 GB
  隨 ctx 成長的層 4 層 × 2 kv_heads × 512 head_dim
  KV @ 32768        0.54 GB  (batch 1)
  合計             16.5 GB

google/gemma-4-31B-it
  參數量           31.3 B
  權重             62.5 GB
  隨 ctx 成長的層 10 層 × 16 kv_heads × 512 head_dim
  KV @ 32768       10.74 GB  (batch 1)
  合計             73.3 GB

deepseek-ai/DeepSeek-V4-Flash
  權重            292.5 GB
  KV @ 32768        2.89 GB （NSA top-512，KV 實際不隨 ctx 線性成長，下值為高估）
```

E4B 與 31B 的對比很說明問題：31B 的權重是 E4B 的 3.9 倍，但 **KV 是 20 倍**——因為它有 10 層全域（E4B 只有 4 層）、16 個 KV head（E4B 只有 2 個）。在長 context、高併發的場景，決定容量的往往是 KV 而不是權重。

三個要點：

- **權重必須按 dtype 分別算。** 混合精度與量化模型用「參數量 × 2」會錯得離譜。
- **KV 只算全域層。** 滑動視窗層的 KV 固定在視窗大小，線性/SSM 層根本沒有 KV。
- **全域層要用 `global_head_dim`。** 用 `head_dim` 會低估一半（gemma-4 全系列適用）。
- **沒有 `layer_types` 的模型只能取上界。** 這時腳本把所有層都當全域層，得到的是保守的高估值。
  用了 NSA 或 SSM 的模型（DeepSeek-V4、Qwen 3.5+）實際 KV 遠低於此——它們的注意力範圍本來就有上限，
  不隨 context 線性成長。要精確估這類模型，得看引擎啟動時報的 `Available KV cache memory`
  （方法見 [vLLM 架構與 KV cache](vllm-serving-and-architecture.md)）。

## 第 4 步：實戰二 — 算速度

decode 階段受記憶體頻寬限制，速度 ≈ 頻寬 ÷ 每 token 讀取的位元組。

```python
def speed(repo, gpu_bw_gbs=864, mbu=0.7):
    """gpu_bw_gbs：L40S 864、RTX 5090 1792、H100 SXM 3350"""
    cfg = get(f"https://huggingface.co/{repo}/raw/main/config.json")
    tc  = cfg.get("text_config", cfg)
    L, H = tc["num_hidden_layers"], tc["hidden_size"]
    hd   = tc.get("head_dim") or H // tc["num_attention_heads"]
    kvh  = tc.get("num_key_value_heads") or tc["num_attention_heads"]

    attn = L * (H*tc["num_attention_heads"]*hd + 2*H*kvh*hd + tc["num_attention_heads"]*hd*H)

    n_exp = tc.get("num_experts") or tc.get("n_routed_experts")
    if n_exp:                                    # MoE：只讀被選中的專家
        k  = tc.get("num_experts_per_tok") or tc.get("top_k_experts")
        mi = tc["moe_intermediate_size"]
        sh = tc.get("n_shared_experts", 0) or 0
        ffn = L * (k + sh) * 3 * H * mi
    else:
        ffn = L * 3 * H * tc["intermediate_size"]

    act = attn + ffn + tc["vocab_size"] * H      # 加上輸出頭
    per_token = act * 2 / 1e9                    # bf16
    print(f"{repo}: 激活 {act/1e9:.1f} B → 每 token 讀 {per_token:.2f} GB "
          f"→ {gpu_bw_gbs*mbu/per_token:.0f} tok/s")

speed("google/gemma-4-E4B-it")
speed("Qwen/Qwen3.6-35B-A3B")
speed("google/gemma-4-26B-A4B-it")
```

實際跑出來（L40S，MBU 0.7）：

```
google/gemma-4-E4B-it:     激活 4.5 B → 每 token 讀 9.05 GB → 67 tok/s
Qwen/Qwen3.6-35B-A3B:      激活 2.3 B → 每 token 讀 4.54 GB → 133 tok/s
google/gemma-4-26B-A4B-it: 激活 3.2 B → 每 token 讀 6.41 GB → 94 tok/s
```

`Qwen3.6-35B-A3B` 的總參數是 `gemma-4-E4B` 的 4.5 倍，生成卻快一倍——因為它是 MoE，每個 token 只讀 2.3 B 的激活參數。

MoE 的關鍵在那個 `if`：**顯存要裝下全部專家，但頻寬只付被選中那幾個的錢。** 這就是 MoE 模型「很大卻不慢」的原因。

完整推導與 offload 情境見 [推論速度怎麼估](llm-decode-throughput-formula.md)，或直接用[估算器](../tools/decode-throughput-calculator.html)。

## 第 5 步：實戰三 — 判斷相容性

在花時間下載幾百 GB 之前，先確認這個模型跑不跑得起來。

```python
def compat(repo):
    api = get(f"https://huggingface.co/api/models/{repo}")
    cfg = get(f"https://huggingface.co/{repo}/raw/main/config.json")
    tc  = cfg.get("text_config", cfg)
    flags = []

    if api.get("gated"):        flags.append("需要接受條款 → 準備 HF_TOKEN")
    if api.get("pipeline_tag") not in ("text-generation","image-text-to-text","any-to-any"):
        flags.append(f"pipeline_tag = {api.get('pipeline_tag')} → 可能不是語言模型")

    q = cfg.get("quantization_config") or tc.get("quantization_config")
    if q: flags.append(f"量化 {q.get('quant_method')} → 確認硬體有原生支援")

    if any(k.startswith("linear_") for k in tc) or "mamba_ssm_dtype" in tc:
        flags.append("含 SSM/線性注意力層 → 引擎需專門支援")
    if "kv_lora_rank" in tc or "q_lora_rank" in tc:
        flags.append("MLA 低秩注意力 → 引擎需專門支援")
    if tc.get("index_topk"):
        flags.append(f"NSA 稀疏注意力 (top-{tc['index_topk']}) → 引擎需專門支援")
    if tc.get("num_nextn_predict_layers") or tc.get("mtp_num_hidden_layers"):
        flags.append("自帶推測解碼草稿頭 → 記得開，通常 1.2–2 倍速")

    am = (api.get("transformersInfo") or {}).get("auto_model")
    print(f"{repo}\n  model_type = {cfg.get('model_type')}  |  auto_model = {am}")
    print(f"  sha = {api.get('sha','')[:12]}  ← 釘版本用")
    for f in flags: print("  ·", f)

for r in ["deepseek-ai/DeepSeek-V4-Flash", "google/gemma-4-E4B-it",
          "Qwen/Qwen3.6-35B-A3B", "MiniMaxAI/MiniMax-H3"]:
    try:
        compat(r)
    except urllib.error.HTTPError as e:
        print(f"{r}\n  · config.json 取不到（{e.code}）→ 不是標準 transformers 模型，"
              f"先看 pipeline_tag 與 repo 結構")
```

實際跑出來：

```
deepseek-ai/DeepSeek-V4-Flash
  model_type = deepseek_v4  |  auto_model = AutoModelForCausalLM
  sha = 60d8d70770c6  ← 釘版本用
  · 量化 fp8 → 確認硬體有原生支援
  · MLA 低秩注意力 → 引擎需專門支援
  · NSA 稀疏注意力 (top-512) → 引擎需專門支援
  · 自帶推測解碼草稿頭 → 記得開，通常 1.2–2 倍速

google/gemma-4-E4B-it
  model_type = gemma4  |  auto_model = AutoModelForMultimodalLM
  sha = ee0ef6023621  ← 釘版本用

Qwen/Qwen3.6-35B-A3B
  model_type = qwen3_5_moe  |  auto_model = AutoModelForMultimodalLM
  sha = 995ad96eacd9  ← 釘版本用
  · 含 SSM/線性注意力層 → 引擎需專門支援
  · 自帶推測解碼草稿頭 → 記得開，通常 1.2–2 倍速

MiniMaxAI/MiniMax-H3
  · config.json 取不到（404）→ 不是標準 transformers 模型，先看 pipeline_tag 與 repo 結構
```

最後那筆是真實案例：`MiniMax-H3` 的名字看起來像語言模型，實際上是影片生成模型（`diffusers` 架構，設定檔分散在各子目錄）。**在下載 33 GB 之前先跑這段，三秒就知道。**

DeepSeek-V4-Flash 那筆則一次列出四個部署前必須確認的事——三個「引擎需專門支援」加一個免費的加速機會。

`transformersInfo.auto_model` 特別值得看——載 LoRA adapter 時模型類別選錯，PEFT 會**靜默掛上 0 個模組**，模型照常回答但完全沒調整過（見 [LoRA adapter 教學](gemma4-lora-adapter-guide.md)）。

`sha` 是 commit hash。chat template 這類會被更新的檔案，要把版本釘住才有可重現性——gemma-4 的樣板在 2026-07 就被改過一次。

---

## 第 6 步：三套來源的欄位對照

| 概念 | HF `config.json` | GGUF metadata |
|---|---|---|
| 層數 | `num_hidden_layers` | `<arch>.block_count` |
| 殘差流寬度 | `hidden_size` | `<arch>.embedding_length` |
| Q 頭數 | `num_attention_heads` | `<arch>.attention.head_count` |
| KV 頭數 | `num_key_value_heads` | `<arch>.attention.head_count_kv`（`None` = 與 Q 相同） |
| 全域層頭維度 | `global_head_dim` | `<arch>.attention.key_length` |
| 滑動層頭維度 | `head_dim` | `<arch>.attention.key_length_swa` |
| FFN 寬度 | `intermediate_size` | `<arch>.feed_forward_length`（MoE 模型為 0） |
| 專家數 | `num_experts` / `n_routed_experts` | `<arch>.expert_count` |
| top-k | `num_experts_per_tok` / `top_k_experts` | `<arch>.expert_used_count` |
| 單專家寬度 | `moe_intermediate_size` | `<arch>.expert_feed_forward_length` |
| KV 共享層數 | `num_kv_shared_layers` | `<arch>.attention.shared_kv_layers` |
| 滑動視窗 | `sliding_window` | `<arch>.attention.sliding_window` |
| context 上限 | `max_position_embeddings` | `<arch>.context_length` |
| RoPE 基數 | `rope_theta` | `<arch>.rope.freq_base`（另有 `freq_base_swa`） |

GGUF 的命名規則是 `<architecture>.<group>.<field>`，架構名取自 `general.architecture`。

**兩邊互補，對照著看最保險。** GGUF 用 `key_length` / `key_length_swa` 兩個欄位，把「全域層與滑動層頭維度不同」寫得毫無歧義；HF 那邊要注意到 `head_dim` 之外還有 `global_head_dim`。反過來 `layer_types` 的逐層清單只有 HF 有。

---

## 附錄 A：縮寫全表

| 縮寫 | 全名 | 一句話 |
|---|---|---|
| MHA | Multi-Head Attention | 標準多頭注意力，Q 與 KV 頭數相同 |
| GQA | Grouped-Query Attention | 多個 Q 頭共用一組 KV，壓 KV cache |
| MQA | Multi-Query Attention | GQA 的極端，所有 Q 頭共用一組 KV |
| MLA | Multi-head Latent Attention | 把 KV 壓進低維潛在空間再存 |
| NSA | Native Sparse Attention | 用索引器挑出 top-k 個 token 才算注意力 |
| SWA | Sliding Window Attention | 只看最近 N 個 token |
| SSM | State Space Model | 維護固定大小狀態，記憶體不隨長度成長 |
| KDA | Kimi Delta Attention | Kimi K3 的線性注意力變體 |
| MoE | Mixture of Experts | FFN 拆成多個專家，每個 token 只用少數幾個 |
| MTP | Multi-Token Prediction | 一次預測多個 token，用於推測解碼 |
| RoPE | Rotary Position Embedding | 用旋轉矩陣編碼位置的方法 |
| PLE | Per-Layer Embedding | 每層各有一份輸入嵌入（gemma-4 E 系列） |
| FFN | Feed-Forward Network | transformer 區塊裡的前饋層 |
| MBU | Model Bandwidth Utilization | 實際達到的記憶體頻寬佔峰值比例 |

## 附錄 B：六個容易踩的地方

**一、`head_dim` 不等於 `hidden_size / num_attention_heads`。**

| 模型 | hidden | heads × head_dim | 相等？ |
|---|---:|---:|---|
| gemma-4-E4B | 2560 | 8 × 256 = 2048 | 否 |
| Qwen3.6-35B-A3B | 2048 | 16 × 256 = 4096 | 否 |
| MiniMax-M2.7 | 3072 | 48 × 128 = 6144 | 否 |

Q/O 投影本來就允許非方陣。**永遠讀欄位，不要自己除。**

**二、gemma-4 的全域層頭維度是滑動層的兩倍。** `head_dim` 256、`global_head_dim` 512。隨 context 成長的是全域層，用 256 算會低估一半。

**三、多模態模型的結構欄位在 `text_config` 底下。** 讀頂層只會拿到 `None`。

**四、`num_key_value_heads` 為 `None` 不代表沒有。** GGUF 裡這個值缺席時，意思是「與 Q 頭數相同」，也就是標準 MHA。

**五、參數量不能乘 2。** 要用 `safetensors.parameters` 按 dtype 分別算。DeepSeek-V4-Flash 的 2909 億參數實際是 292.5 GB，不是 582 GB。

**六、`feed_forward_length = 0` 不是資料缺失。** 代表這是純 MoE 模型沒有 dense FFN，該看 `expert_feed_forward_length`。

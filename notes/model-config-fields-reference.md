# 模型設定檔的每個數字在算什麼

打開一個模型的 `config.json`，裡面三四十個欄位，多數人只看得懂 `num_hidden_layers` 和 `hidden_size`。但要回答「這個模型能不能塞進我的卡」「生成會有多快」「這個量化版本能不能用」，需要的正好是那些看起來像實作細節的欄位。

這篇把設定檔裡的數字接回它們在計算裡的位置，並整理 HuggingFace `config.json`、HuggingFace API metadata、GGUF metadata 三套來源的欄位對照——它們描述同一件事，命名卻各不相同。

所有實例值取自實際的模型設定檔，查證日 **2026-08-18**。

- 相關：[HuggingFace 模型 repo 檔案逐一解說](gemma4-hf-repo-files.md)（每個**檔案**是什麼）、[架構圖鑑](llm-architecture-map-2026.md)（這些數字組合出的架構）、[推論速度怎麼估](llm-decode-throughput-formula.md)

---

## 一、先把注意力的計算寫出來

大部分欄位的意義，在看到它們出現在哪個矩陣的哪一維時就清楚了。

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

於是：

| 欄位 | 在哪 | 直接決定 |
|---|---|---|
| `hidden_size` | 每個 token 的向量寬度 | 所有矩陣的一邊 |
| `num_attention_heads` | Q 有幾個頭 | Q 與 O 矩陣的大小 |
| `num_key_value_heads` | K/V 有幾個頭 | **KV cache 大小** |
| `head_dim` | 每個頭的維度 | 上面三者的另一邊 |
| `intermediate_size` | FFN 中間層寬度 | FFN 佔的參數（通常是全模型最大宗） |
| `num_hidden_layers` | 上述整組重複幾次 | 全部乘以層數 |

**`num_attention_heads` 與 `num_key_value_heads` 不相等就是 GQA**（Grouped Query Attention）：多個 Q 頭共用一組 K/V，用來壓 KV cache。比值越大壓得越兇——`gemma-4-E4B` 是 8:2、`DeepSeek-V4-Flash` 是 64:1。

---

## 二、欄位速查

### 骨架

| 欄位 | 意義 | 實例 |
|---|---|---|
| `num_hidden_layers` | transformer 層數 | E4B 42、Kimi-K3 93 |
| `hidden_size` | 殘差流寬度 | E4B 2560、Kimi-K3 7168 |
| `vocab_size` | 詞表大小 | gemma-4 262144、DeepSeek 129280 |
| `tie_word_embeddings` | 輸入 embedding 與輸出 head 是否共用權重 | gemma-4 `true`、Qwen `false` |
| `dtype` / `torch_dtype` | 權重原生精度 | `bfloat16` |
| `model_type` | 架構識別碼，推論引擎靠它決定用哪份實作 | `gemma4_text`、`deepseek_v4` |

`tie_word_embeddings` 為 `true` 時，詞表那一大塊參數只存一份——對 262144 × 2560 的 gemma-4 來說是省下 6.7 億參數。

### 注意力

| 欄位 | 意義 | 為什麼要看 |
|---|---|---|
| `num_key_value_heads` | K/V 的頭數 | **KV cache 大小與它成正比** |
| `head_dim` | 每個頭的維度 | 同上；**不一定等於 `hidden_size / num_attention_heads`**（見陷阱一） |
| `global_head_dim` | 全域注意力層的頭維度（gemma-4 專有） | 真正隨 context 成長的那些層用的是這個值 |
| `sliding_window` | 滑動視窗大小 | 視窗層的 KV 固定在這個長度，不隨 context 成長 |
| `layer_types` | 逐層標注注意力型別 | 數出有幾層是 `full_attention`，那才是會爆的部分 |
| `num_kv_shared_layers` | 最後幾層共用前面的 KV | 這些層完全不維護自己的 KV |
| `full_attention_interval` | 每隔幾層一次全域（Qwen 用） | 等價於 `layer_types` 的壓縮寫法 |
| `attention_bias` | QKVO 投影是否帶 bias | 影響參數量（很小），但實作要對得上 |
| `max_position_embeddings` | 位置編碼支援的最大長度 | context 上限的硬邊界 |

### FFN 與 MoE

| 欄位 | 意義 | 別名 |
|---|---|---|
| `intermediate_size` | dense FFN 的中間層寬度 | GGUF: `feed_forward_length` |
| `hidden_act` / `hidden_activation` | 啟動函數 | `silu`、`gelu_pytorch_tanh` |
| `num_experts` / `n_routed_experts` | 路由專家總數 | GGUF: `expert_count` |
| `num_experts_per_tok` / `top_k_experts` | 每個 token 選幾個 | GGUF: `expert_used_count` |
| `moe_intermediate_size` | **單一專家**的中間層寬度 | GGUF: `expert_feed_forward_length` |
| `n_shared_experts` / `shared_expert_intermediate_size` | 每個 token 都會經過的共享專家 | GGUF: `expert_shared_feed_forward_length` |
| `topk_method` / `scoring_func` / `norm_topk_prob` | 路由器怎麼選專家、分數怎麼正規化 | — |
| `routed_scaling_factor` | 專家輸出的縮放 | — |

**MoE 的參數量要用 `moe_intermediate_size` 算，不是 `intermediate_size`。** `Qwen3.6-35B-A3B` 的 `moe_intermediate_size` 只有 512——單一專家很小，靠 256 個專家堆出總容量。GGUF 裡 dense 的 `feed_forward_length` 會是 0，那不是資料缺失，是「這個模型沒有 dense FFN」。

### 低秩壓縮（MLA，DeepSeek 系）

| 欄位 | 意義 |
|---|---|
| `q_lora_rank` | Q 先降到這個維度再升回去，省 Q 投影的參數 |
| `kv_lora_rank` | K/V 壓到的潛在維度，**KV cache 存的是這個壓縮後的表示** |
| `qk_rope_head_dim` | 帶位置編碼的那部分維度 |
| `qk_nope_head_dim` | 不帶位置編碼的那部分維度 |
| `v_head_dim` | V 的頭維度（可與 K 不同） |
| `o_lora_rank` | 輸出投影也走低秩 |

MLA 把「多頭注意力的 KV」換成「一份低維潛在向量」，這是 DeepSeek 能在 1M context 下維持可用 KV 大小的核心。代價是實作複雜，推論引擎要專門支援。

### 線性注意力 / SSM（Qwen 3.5+）

| 欄位 | 意義 |
|---|---|
| `linear_num_key_heads` / `linear_key_head_dim` | 線性層的 K 頭數與維度 |
| `linear_num_value_heads` / `linear_value_head_dim` | 同上，V 側 |
| `linear_conv_kernel_dim` | 線性層前的短卷積核大小 |
| `mamba_ssm_dtype` | SSM 狀態的計算精度 |
| `attn_output_gate` | 注意力輸出是否過閘門 |

GGUF 那一側把它標得更直白——`qwen35moe` 的 metadata 裡有 `ssm.state_size`、`ssm.conv_kernel`、`ssm.inner_size`、`ssm.time_step_rank`、`ssm.group_count`。**Qwen 3.5 系的「線性注意力」實際上是 SSM（狀態空間模型，Mamba 家族）**：它維護一個固定大小的狀態而不是逐 token 累積的 KV，所以記憶體不隨序列長度成長。

這也解釋了為什麼這類模型的長 context 成本低，以及為什麼推論引擎要特別支援——SSM 的狀態管理與 KV cache 是兩套完全不同的機制。

### 位置編碼

| 欄位 | 意義 |
|---|---|
| `rope_theta` / `rope.freq_base` | RoPE 的頻率基數，**越大越適合長 context** |
| `rope_parameters` | 分注意力型別各給一組（gemma-4：全域 1e6、滑動 1e4） |
| `partial_rotary_factor` | 只對一部分維度套 RoPE |
| `rope_scaling` | 外推到訓練長度以外的縮放方法 |

gemma-4 對全域層與滑動層用**不同的 RoPE 基數**：滑動層只看 512 個 token，用小基數；全域層要看到 128K，用大基數。這是「同一個模型裡兩種注意力各自最佳化」的例子。

### 數值穩定與其他

| 欄位 | 意義 |
|---|---|
| `rms_norm_eps` | LayerNorm 的數值下限 |
| `final_logit_softcapping` | 輸出 logits 的軟性上限（gemma-4 為 30、Muse-Glimmer 為 20） |
| `logit_scale` | 輸出前的縮放係數 |
| `num_nextn_predict_layers` / `mtp_num_hidden_layers` | 內建的推測解碼草稿頭層數 |
| `quantization_config` | 量化方案（見[量化格式](quantization-fp8-nvfp4.md)） |
| `expert_dtype` | 專家單獨用不同精度（DeepSeek-V4 為 `fp4`） |

`num_nextn_predict_layers` 不是 0 代表模型自帶推測解碼模組，可以直接開 MTP/EAGLE——這是免費的一到兩倍速度，很容易被漏掉。

---

## 三、三套來源的對照

同一件事，三個地方講法不同：

| 概念 | HF `config.json` | GGUF metadata | 備註 |
|---|---|---|---|
| 層數 | `num_hidden_layers` | `<arch>.block_count` | |
| 殘差流寬度 | `hidden_size` | `<arch>.embedding_length` | |
| Q 頭數 | `num_attention_heads` | `<arch>.attention.head_count` | |
| KV 頭數 | `num_key_value_heads` | `<arch>.attention.head_count_kv` | GGUF 可能為 `None`，代表與 Q 相同 |
| 頭維度 | `head_dim` | `<arch>.attention.key_length` / `value_length` | **gemma-4 另有 `key_length_swa`** |
| 全域層頭維度 | `global_head_dim` | `<arch>.attention.key_length` | GGUF 的 `key_length` 對應全域層 |
| 滑動層頭維度 | `head_dim` | `<arch>.attention.key_length_swa` | |
| FFN 寬度 | `intermediate_size` | `<arch>.feed_forward_length` | MoE 模型此值為 0 |
| 專家數 | `num_experts` / `n_routed_experts` | `<arch>.expert_count` | |
| 每 token 選幾個 | `num_experts_per_tok` / `top_k_experts` | `<arch>.expert_used_count` | |
| 單專家寬度 | `moe_intermediate_size` | `<arch>.expert_feed_forward_length` | |
| KV 共享層數 | `num_kv_shared_layers` | `<arch>.attention.shared_kv_layers` | |
| 滑動視窗 | `sliding_window` | `<arch>.attention.sliding_window` | |
| context 上限 | `max_position_embeddings` | `<arch>.context_length` | |
| RoPE 基數 | `rope_theta` | `<arch>.rope.freq_base` | gemma-4 另有 `rope.freq_base_swa` |

GGUF 的命名規則是 `<architecture>.<group>.<field>`，架構名取自 `general.architecture`。

**GGUF 有時比 HF config 更明確。** `key_length` 與 `key_length_swa` 兩個欄位把「全域層與滑動層用不同頭維度」寫得清清楚楚，而 HF 那邊要注意到 `head_dim` 之外還有一個 `global_head_dim` 才看得出來。反過來 HF config 有 `layer_types` 逐層清單，GGUF 只給 `sliding_window_pattern` 或間隔數。**兩邊對照著看最保險。**

查 GGUF metadata 的方式：

```bash
# Ollama 服務上的模型
curl -s http://<host>:11434/api/show -d '{"model":"gemma4:e4b"}' | jq .model_info

# 本地 .gguf 檔
python -c "from gguf import GGUFReader; r=GGUFReader('x.gguf'); [print(f.name, f.contents()) for f in r.fields.values()]"
```

---

## 四、HuggingFace API 的 metadata JSON

`https://huggingface.co/api/models/<repo_id>` 回傳的 JSON 跟 repo 裡的檔案是兩回事——它是 Hub 對這個 repo 的**索引與統計**，有些資訊只有這裡有。

| 欄位 | 內容 | 用途 |
|---|---|---|
| `safetensors.total` | 權重張量的**實際參數總數** | 比模型卡上寫的「30B」可靠 |
| `safetensors.parameters` | 依 dtype 分組的參數量 | **算實際體積要用這個**（見下） |
| `gated` | 是否需要接受條款才能下載 | 決定要不要準備 token |
| `siblings` | repo 裡有哪些檔案 | 判斷是否為分片、有無 GGUF |
| `config` | `architectures`、`model_type`、`tokenizer_config` 摘要 | 不下載 config.json 就能先看架構 |
| `cardData` | 模型卡的 YAML front matter（`pipeline_tag`、`license`、`base_model`） | **判斷這是不是語言模型** |
| `pipeline_tag` | 任務類型 | `text-generation` / `image-text-to-text` / `text-to-video` |
| `transformersInfo` | `auto_model` 應該用哪個類別 | 決定要用 `AutoModelForCausalLM` 還是 `AutoModelForImageTextToText` |
| `usedStorage` | repo 佔用位元組 | 估下載量與磁碟需求 |
| `lastModified` / `sha` | 最後更新與 commit | **釘版本用**，chat template 之類會變動的東西要記這個 |

**`safetensors.parameters` 是算模型體積最可靠的方法**，因為它按 dtype 分開列。混合精度的模型不能用「參數量 × 2」估：

```python
W = {"BF16":2, "F16":2, "F32":4, "F8_E4M3":1, "I8":1, "I64":8}
bytes_ = sum(n * W[dtype] for dtype, n in meta["safetensors"]["parameters"].items())
```

`DeepSeek-V4-Flash` 的 2909 億參數裡，2835 億是 I8、60 億是 FP8、14 億是 BF16——實際體積 292.5 GB，而不是「291 B × 2 = 582 GB」。

`transformersInfo.auto_model` 這個欄位能省掉一類很難查的錯誤：載入 LoRA adapter 時模型類別選錯，PEFT 會靜默掛上 0 個模組（見 [LoRA adapter 教學](gemma4-lora-adapter-guide.md)）。

---

## 五、拿這些欄位做三件事

**算顯存。**

```
權重  = 依 safetensors.parameters 逐 dtype 加總
KV    = 2 × 全域層數 × num_key_value_heads × global_head_dim × 精度 × context × 併發
        + 滑動層數 × 同式 × sliding_window（不隨 context 成長）
```

全域層數 = `layer_types` 裡 `full_attention` 的數量，再扣掉落在 `num_kv_shared_layers` 範圍內的。

**算速度。** decode 受記憶體頻寬限制，分母是每 token 讀取的位元組：dense 是全部權重，MoE 是 `num_experts_per_tok × 單專家參數 × 層數 + attention + shared`。推導見 [推論速度怎麼估](llm-decode-throughput-formula.md)，或直接用[估算器](../tools/decode-throughput-calculator.html)。

**判斷相容性。** `model_type` 決定推論引擎有沒有對應實作；`quantization_config.quant_method` 決定目標硬體有沒有原生支援；有 SSM 或 MLA 欄位代表需要引擎專門支援，不是所有版本都跟得上。

---

## 六、六個容易踩的地方

**一、`head_dim` 不等於 `hidden_size / num_attention_heads`。** 這個習慣性的假設在新模型上經常不成立：

| 模型 | hidden | heads | heads × head_dim | 相等？ |
|---|---:|---:|---:|---|
| gemma-4-E4B | 2560 | 8 × 256 | 2048 | 否 |
| Qwen3.6-35B-A3B | 2048 | 16 × 256 | 4096 | 否 |
| MiniMax-M2.7 | 3072 | 48 × 128 | 6144 | 否 |

Q/O 投影本來就可以是非方陣。**永遠讀 `head_dim` 欄位，不要自己除。**

**二、gemma-4 的全域層與滑動層頭維度不同。** `head_dim` 256 但 `global_head_dim` 512。隨 context 成長的是全域層，用 256 去算會低估一半。

**三、多模態模型的欄位藏在 `text_config` 裡。** `config.json` 的頂層只有 `architectures`、`model_type` 和各模態的子設定；`num_hidden_layers` 這些要進 `text_config` 找。忘記這件事會拿到 `None`。

**四、`num_key_value_heads` 為 `None` 不代表沒有。** GGUF 裡這個值缺席時，意思是「與 Q 頭數相同」（標準 MHA，沒有 GQA）。

**五、參數量不能乘 2。** 混合精度與量化模型要按 dtype 分別算，見第四節。

**六、`feed_forward_length = 0` 不是壞掉。** GGUF 裡看到 0，代表這是純 MoE 模型，沒有 dense FFN，該看 `expert_feed_forward_length`。

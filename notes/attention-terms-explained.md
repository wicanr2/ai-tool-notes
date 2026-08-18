# Attention 名詞解釋：GQA、MLA、線上 softmax、SSM、DeltaNet、TurboQuant，與硬體世代

[Attention 的四個軸](attention-mechanisms-taxonomy.md)把名詞歸了位，這篇把每個名詞拆開講機制——它要解決什麼、怎麼做到、以及在實際模型或原始碼裡長什麼樣。

證據取自 vLLM 0.25.1 的原始碼與各模型設定檔，查證日 **2026-08-18**。

---

## GQA：Grouped-Query Attention

**要解決的問題**：KV cache 太大。

標準的多頭注意力（MHA，Multi-Head Attention）每個 Q 頭都配一組自己的 K/V，於是每個 token 每層要存：

```
2（K 和 V）× num_attention_heads × head_dim
```

`num_attention_heads` 通常是幾十個，這個乘法很快就吃光顯存。

**機制**：讓**多個 Q 頭共用一組 K/V**。Q 的頭數不變（表達力主要來自 Q 的多樣性），K/V 的頭數減少。

```
MHA   8 個 Q 頭 → 8 組 K/V     每 token 存 8 × head_dim × 2
GQA   8 個 Q 頭 → 2 組 K/V     每 token 存 2 × head_dim × 2   ← 省 4 倍
MQA   8 個 Q 頭 → 1 組 K/V     每 token 存 1 × head_dim × 2   ← 省 8 倍
```

MQA（Multi-Query Attention）是 GQA 的極端情況，所有 Q 頭共用同一組。

在 config 裡就是兩個欄位的比值：

| 模型 | `num_attention_heads` | `num_key_value_heads` | 比值 |
|---|---:|---:|---|
| gemma-4-E4B | 8 | 2 | 4:1 |
| gemma-4-31B | 32 | 16 | 2:1 |
| Muse-Glimmer-30B | 32 | 2 | 16:1 |
| MiniMax-M2.7 | 48 | 8 | 6:1 |

**代價**：K/V 的表達力下降。比值越大省越多，但模型能區分的「鍵值模式」越少。這是個純粹的取捨曲線，各家在上面選不同的點。

**注意**：GQA 是訓練時就決定的架構。不能拿一個 MHA 模型在部署時「開 GQA」——權重根本不是那個形狀。

---

## MLA：Multi-head Latent Attention

**要解決的問題**：同上，但 GQA 的省法有上限——降到 MQA 就到底了，再省就沒有 K/V 可以共用。

**機制**：不是「共用」，是**壓縮**。把 K/V 投影到一個低維的潛在向量，**KV cache 存的是那個潛在向量**，要用的時候再投影展開。

```
存的時候：  c = W_down · x          c 的維度 = kv_lora_rank（很小）
用的時候：  K = W_up_k · c
           V = W_up_v · c
```

關鍵在於 `W_up_k` 與 `W_up_v` 是**固定的模型權重**，不需要跟著每個 token 存。所以每個 token 只留下一個小小的 `c`。

以 DeepSeek-V4-Flash 為例，它把 `num_key_value_heads` 壓到 1、`head_dim` 開到 512：

```
若用 MHA：  2 × 64 頭 × 512 = 65,536 個數 / token / 層
實際 MLA：  2 ×  1 頭 × 512 =  1,024 個數 / token / 層     ← 64 倍差距
```

Kimi-K3 用的是同一族的做法，`kv_lora_rank = 512`。

相關欄位還有 `q_lora_rank`（Q 也走低秩，省的是參數不是 cache）、`qk_rope_head_dim` 與 `qk_nope_head_dim`（把帶位置編碼與不帶位置編碼的維度分開處理——因為 RoPE 是位置相關的旋轉，不能直接壓縮後還原）。

**代價**：實作複雜。展開的動作要融進 kernel 裡才不會變慢，所以每種硬體都要重寫一份。vLLM 為此開了一個 `mla/` 目錄，裡面十幾個檔案——這就是下面「kernel 可攜性」那節要講的問題。

---

## 線上 softmax（online softmax）

**要解決的問題**：softmax 看起來不能分塊算，而不分塊就得把 `S×S` 的矩陣整個攤在記憶體裡。

標準 softmax 為了數值穩定會先減去最大值：

```
softmax(xᵢ) = exp(xᵢ − max(x)) / Σⱼ exp(xⱼ − max(x))
```

問題是 `max(x)` 和分母的 `Σ` **都需要看過整列**才知道。所以直覺上必須先算完整列，才能做正規化。

**機制**：邊算邊修正。維護兩個running 的量——目前為止的最大值 `m` 與目前為止的分母 `ℓ`。來了新的一塊，先算這塊的局部最大值 `m_blk`，然後：

```
m'  = max(m, m_blk)                              新的全域最大值
ℓ'  = ℓ · exp(m − m')  +  ℓ_blk · exp(m_blk − m')
O'  = O · exp(m − m')  +  O_blk · exp(m_blk − m')
```

那個 `exp(m − m')` 是**修正因子**：先前累積的結果是用舊的最大值算的，現在最大值變了，就把舊結果按比例縮放回來。

**這是數學上完全等價的重排，不是近似。** 最後一塊算完時，`O / ℓ` 就是標準 softmax 注意力的精確結果。

**為什麼重要**：有了它，Q/K/V 就能切成小塊、一塊一塊搬進晶片上的 SRAM 算完就丟，那個 `S×S` 的中間矩陣**從頭到尾不需要被完整寫出來**。這正是 FlashAttention 的地基。

具體省了多少：context 32768、fp16，單一個頭、單一層的 `S×S` 矩陣是

```
32768 × 32768 × 2 bytes = 2.15 GB
```

八個頭就是 17 GB，而這還只是暫存。

---

## SSM：State Space Model

**要解決的問題**：注意力的成本隨序列長度成長，因為每個新 token 都要回頭看全部歷史。

**機制**：改成維護一個**固定大小的狀態**。

```
注意力：  輸出ₜ = f(輸入ₜ, 全部歷史 token)      要保留全部歷史 → KV cache 隨長度成長
SSM：     狀態ₜ = A · 狀態ₜ₋₁ + B · 輸入ₜ       只保留一個狀態 → 大小固定
          輸出ₜ = C · 狀態ₜ + D · 輸入ₜ
```

這個形式就是傳統的 RNN。SSM（狀態空間模型）與早期 RNN 的差別在於：`A`、`B`、`C` 這幾個矩陣的結構經過設計，讓整段序列可以用**平行掃描**（parallel scan）一次算完，而不是像 RNN 那樣一步一步跑——訓練時能吃滿 GPU，推論時又保有「固定狀態」的好處。

**Mamba** 讓 `A`、`B`、`C` 隨輸入變化（selective state space），大幅提升表達力，是這一族目前最有影響力的實作。

Qwen 3.5+ 的 GGUF metadata 直接標出這些欄位：

```
ssm.state_size      = 128    狀態向量的維度
ssm.conv_kernel     = 4      狀態更新前的短卷積核大小
ssm.inner_size      = 4096   擴展後的內部維度
ssm.time_step_rank  = 32     Δ（時間步長）投影的秩
ssm.group_count     = 16     狀態分組數
```

`time_step_rank` 對應的是 Mamba 裡控制「這一步要記多少、忘多少」的 Δ 參數，用低秩投影從輸入算出來。

**代價**：固定大小的狀態**裝不下任意多的資訊**。需要精確回憶很久以前某個特定 token 時（例如「文件第 3 頁那個編號是多少」），SSM 會失手，而注意力不會——因為注意力真的把每個 token 都留著。

這就是為什麼**沒有模型全部用 SSM**，都是交錯：Qwen3.8-2.4T 是 69 層線性配 23 層全域注意力，少數幾層真注意力負責精確回憶。

---

## DeltaNet 與 Gated DeltaNet

**DeltaNet** 是線性注意力家族裡的一支，差別在**狀態怎麼更新**。

一般的線性注意力用純累加：新的鍵值對直接加進狀態裡。問題是狀態會越加越滿，舊資訊沒有被清掉的機制。

**delta rule**（差分規則）改成「先看看目前狀態對這個鍵已經記了什麼，只補上差額」：

```
純累加：    狀態 ← 狀態 + k ⊗ v
delta rule： 狀態 ← 狀態 + β · k ⊗ (v − 目前狀態對 k 的預測)
```

那個 `β` 控制修正的力道。這是經典聯想記憶的更新規則——**寫入一個鍵時會先把該鍵的舊值抹掉**，而不是疊上去。

**Gated DeltaNet（GDN）** 再加一層閘門：用 sigmoid 算出一個衰減係數，讓狀態能主動遺忘。

vLLM 的實作可以直接對照：`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` 裡呼叫的是

```python
chunk_gated_delta_rule          # 分塊版本，訓練與 prefill 用
fused_recurrent_gated_delta_rule_packed_decode   # 逐步遞迴版本，decode 用
fused_sigmoid_gating_delta_rule_update           # sigmoid 閘門 + delta 更新
```

參數裡有 `beta`，kernel 來自 **FLA**（Flash Linear Attention）這套函式庫。

注意檔案路徑在 `mamba/` 底下、KV cache 介面用的是 `MambaSpec` 而不是 `AttentionSpec`——**引擎把它當成狀態而非 KV 來管理**，這也印證了它屬於「換掉機制」那一軸。

**Kimi K3 的 KDA**（Kimi Delta Attention）是同一族的另一個變體。

---

## TurboQuant

這個名字裡有 `attn`，但它其實不改注意力的算法，**它壓的是 KV cache 本身**。

vLLM 的實作說明寫得很清楚：

```
Prefill: 用未壓縮的 K/V 做標準注意力，算完再把 K 量化，K+V 一起存進 cache
Decode:  直接從壓縮過的 cache 算注意力分數，解包後做 softmax 與加權和
```

**機制**：用 **Lloyd-Max 最佳量化器**求出一組質心（centroid）當碼本。原始碼裡是

```python
def get_centroids(d: int, bits: int):
    centroids, _ = solve_lloyd_max(d, bits)   # 2^bits 個最小化 MSE 的質心
```

Lloyd-Max 是經典的純量量化演算法：給定資料分布與位元預算，解出讓量化誤差平方和最小的那組代表值。比「均勻切成 2^bits 段」準得多。

實際的儲存版面（`turboquant_k3v4_nc`，head_dim 256）：

```
[ key_packed 100 bytes | value_fp16 512 bytes ] = 612 bytes / head / token
```

K 被壓到 100 bytes（256 維，約 3 bit/維），V 在這個變體裡維持 fp16。相較全 fp16 的 `2 × 256 × 2 = 1024 bytes`，省了四成。

**它在分類上是個特例。** 我在[四個軸](attention-mechanisms-taxonomy.md)裡把「部署時可選」與「輸出不變」綁在一起，但 TurboQuant 兩者都佔一半：它是**部署時的選擇**（模型不用重訓），但它**會改變輸出**（量化是有損的）。

同一格裡的還有 vLLM 的 `--kv-cache-dtype fp8`——同樣是拿品質換 KV 容量的部署選項。這類選項該不該開，要用[量化品質的方法](quantization-quality-and-offload.md)實測，不能只看容量數字。

---

## Hopper 與 NVIDIA 的硬體世代

**Hopper** 是 NVIDIA 的一個 GPU 架構世代的代號，不是某張卡。各世代對照：

| 世代 | Compute Capability | 代表卡 | 這一代帶來的關鍵能力 |
|---|---|---|---|
| Ampere | `sm_80` / `sm_86` | A100 / RTX 3090 | BF16、稀疏化 tensor core |
| Ada Lovelace | `sm_89` | **L40S** / RTX 4090 | **FP8 tensor core** |
| Hopper | `sm_90` | H100 / H200 | FP8、TMA（非同步搬運）、分散式共享記憶體、`wgmma` |
| Blackwell | `sm_100` / `sm_120` | B200 / **RTX 5090** | **FP4 tensor core** |

`sm_XX` 是 CUDA 的 compute capability 代號，kernel 編譯時要指定目標。同一份 CUDA 程式碼在不同 `sm` 上不一定能跑，也不一定跑得快。

**為什麼很多最佳化只針對 Hopper**：它引入了 TMA（Tensor Memory Accelerator，把資料搬運交給專用硬體非同步進行）與 `wgmma`（warp-group 級的非同步矩陣乘法）。FlashAttention-3、FlashMLA 這類極致最佳化的 kernel 大量依賴這兩樣東西，換到沒有它們的世代就得整個重寫。

注意 **Blackwell 分成兩個 compute capability**：資料中心的 B200 是 `sm_100`，消費級的 RTX 5090 是 `sm_120`。**它們不是同一個目標**，為 `sm_100` 寫的 kernel 在 5090 上不一定有。這在 vLLM 的檔案清單裡看得到——有一個檔案專門叫 `flashinfer_mla_sparse_sm120.py`。

---

## tilelang 與 kernel 可攜性

**tilelang** 是一套寫 GPU kernel 的領域專用語言（DSL），源自 TVM 生態。你用比較高階的方式描述「怎麼分塊、怎麼搬資料、怎麼算」，由它編譯出各平台的 kernel。

**它為什麼會以「fallback」的身分出現**：一個新的注意力機制要在 N 種硬體上跑，理論上要 N 份手工最佳化 kernel。手寫 kernel 又慢又貴，所以實務上的做法是：**最主流的硬體給手寫版本，其餘的用 tilelang 之類的工具生成。**

ktransformers 部署 DeepSeek-V4 的文件把這件事寫得很直白：

> tilelang（需手動安裝——非 Hopper GPU 上的 NSA sparse-MLA indexer 路徑需要它）

也就是說：**Hopper 上有手工最佳化的 NSA indexer，其他卡上得靠 tilelang 現場編一個。** 能跑，但通常較慢，而且多一層安裝依賴（那份文件光是相關的版本坑就列了五條）。

vLLM 的目錄結構把這個成本攤開來看最清楚。**一個機制（sparse MLA），五份實作**：

```
mla/flashmla_sparse.py                  DeepSeek 官方 kernel（Hopper 導向）
mla/flashattn_mla_sparse.py             FlashAttention 路線
mla/flashinfer_mla_sparse.py            FlashInfer 路線
mla/flashinfer_mla_sparse_sm120.py      消費級 Blackwell 專用
mla/rocm_aiter_mla_sparse.py            AMD
mla/xpu_mla_sparse.py                   Intel
```

**這就是「新機制的支援度落後模型發布好幾個月」的實體原因。** 模型作者發布權重只要上傳檔案；讓它在六種硬體上都跑得動，是六批人各自寫 kernel。

選型時的實務意義：看到一個模型用了 MLA、NSA、SSM 這類新機制，**要確認的不只是「引擎支援嗎」，而是「引擎在我這張卡上支援嗎」**。這兩件事的答案經常不一樣。

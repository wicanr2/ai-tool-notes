# 推論速度怎麼估：decode 是頻寬問題，不是算力問題

「這張卡跑這個模型大概多快？」這個問題有一條相當準的估算公式，而且它用不到任何算力規格。一張 L40S 有 362 TFLOPS 的 bf16 算力，但單一請求生成 token 的速度只由另一個數字決定：864 GB/s 的記憶體頻寬。

理由是自迴歸生成的結構：每產生一個 token，GPU 必須把該讀的權重完整讀過一遍，而算術量只有一次矩陣**向量**乘。讀 9 GB 的權重去做一次向量乘，時間全部花在讀，算的部分幾乎不佔。

這篇建立那條公式，用它估 gemma-4-E4B 在 L40S 上的速度，再擴充到「模型放不進顯存、要靠系統記憶體」的情況。

- **互動計算器**：[LLM Decode 速度估算器](../tools/decode-throughput-calculator.html)——填入模型與硬體參數即可估算，含 offload 情境與顯存容量判斷
- 相關：[vLLM 架構與 KV cache](vllm-serving-and-architecture.md)、[量化格式：FP8、NVFP4](quantization-fp8-nvfp4.md)

---

## 一、為什麼 decode 卡在頻寬

生成分兩階段，性質完全相反：

| | prefill | decode |
|---|---|---|
| 一次處理 | 整段 prompt（幾百到幾千 token） | 一個 token |
| 矩陣運算 | 矩陣 × 矩陣 | 矩陣 × 向量 |
| 每讀一次權重能做幾次乘加 | 幾千次 | **一次** |
| 瓶頸 | 算力 | **記憶體頻寬** |

「每讀一個位元組的權重能做幾次浮點運算」這個比值叫**算術強度**（arithmetic intensity）。GPU 的平衡點通常在每 byte 幾百次運算；decode 的算術強度接近 1，遠低於平衡點，所以完全落在頻寬受限的那一側。

實務上的證據隨手可得。先前那台 L40S 的 vLLM log：

```
Avg prompt throughput: 423.7 tokens/s, Avg generation throughput: 16.0 tokens/s
```

同一個模型、同一張卡，prefill 比 decode 快 26 倍。差的不是算力，是每個 token 攤到多少權重讀取。

這也解釋了一個常見的困惑：**換一張算力翻倍但頻寬沒變的卡，單一請求的生成速度不會變快。**

---

## 二、公式

```
tokens/sec ≈ (記憶體頻寬 × MBU) / 每 token 必須讀取的位元組數
```

**MBU**（Model Bandwidth Utilization）是實際達到的頻寬佔理論峰值的比例。它涵蓋了 kernel 效率、記憶體存取樣式、以及非矩陣運算（norm、softmax、取樣）的時間。實務值落在 **0.6 – 0.8**，估算時取 0.7 是合理的起點。

分母怎麼算，dense 與 MoE 不同：

```
dense：  每 token 位元組 = 所有權重的位元組數
MoE：    每 token 位元組 = (共享部分 + top-k 被選中的專家) 的位元組數
```

MoE 的關鍵性質就在這裡：**顯存要裝下全部專家，但頻寬只付激活那幾個的錢。** 一個 291B 的 MoE 模型每 token 可能只讀 13B 參數，速度接近一個 13B 的 dense 模型，容量需求卻是 291B 的。

另外還要加上 KV cache 的讀取。每生成一個 token，注意力要讀完整段 context 的 K 與 V：

```
KV 位元組 ≈ 2 × 擁有 KV 的層數 × kv_heads × head_dim × 精度位元組 × 目前 context 長度
```

短 context 時這項相對權重讀取可以忽略；長 context 時它會逐漸變成主角，這也是「同一個模型，對話越長生成越慢」的根本原因。

---

## 三、實例：gemma-4-E4B 在 L40S

從 `config.json` 逐層算每 token 要讀多少參數（`hidden_size` 2560、`intermediate_size` 10240、42 層、8 個 attention head、2 個 KV head、`head_dim` 256）：

| 項目 | 計算 | 參數量 |
|---|---|---:|
| 每層 attention | q(2560×2048) + k(2560×512) + v(2560×512) + o(2048×2560) | 13.1 M |
| 每層 MLP | gate + up + down，各 2560×10240 | 78.6 M |
| 每層小計 | | 91.7 M |
| × 42 層 | | 3.85 B |
| lm_head | 262144 × 2560 | 0.67 B |
| **合計** | | **4.52 B** |

bf16 為 2 bytes：**每 token 讀取 9.05 GB**。

（`per-layer embedding` 那部分參數雖然佔了模型檔案的一大塊，但每個 token 只取其中幾列，頻寬上可以忽略。）

L40S 頻寬 864 GB/s：

| MBU | 預測 tok/s（單一請求） |
|---|---:|
| 100%（理論上限） | 95.5 |
| 80% | 76.4 |
| **70%（合理預期）** | **66.8** |
| 60% | 57.3 |

所以答案是**約 57–76 tok/s**，單一請求、短 context。

同樣的計算換一張卡只要改一個數字。H100 SXM 的 3350 GB/s 在同一個模型上會是 259 tok/s（MBU 0.7）——快 3.9 倍，而這個倍數正是頻寬的倍數。

### 這個模型的兩個架構特點會影響分母嗎

`gemma-4-E4B-it` 有 18 層共享前面層的 KV，且五分之四的層是滑動視窗注意力。這兩個機制大幅降低了 **KV cache** 的容量與頻寬成本（細節見 [vLLM 架構與 KV cache](vllm-serving-and-architecture.md)），但**不影響權重讀取**——共享層仍然要讀自己的 `qkv_proj` 與 MLP 權重。所以上面的分母不用修正，只有 KV 那一項受益。

---

## 四、模型放不進顯存時

模型比顯存大時，權重必須放在系統記憶體，這時候公式要分層寫：

```
tok/s ≈ 1 / ( GPU端位元組/(GPU頻寬 × MBU) + CPU端位元組/(DRAM頻寬 × MBU) )
```

這裡有個關鍵的實作選擇，會決定分母裡放的是哪個頻寬：

| 做法 | 資料流 | 瓶頸 |
|---|---|---|
| **搬到 GPU 算** | 每 token 把需要的權重經 PCIe 送進顯存 | PCIe 5.0 x16 只有 63 GB/s |
| **留在 CPU 算**（ktransformers、llama.cpp `-ot` / `--n-cpu-moe`） | 權重不動，CPU 就地算完把結果送給 GPU | 系統記憶體頻寬 |

主流做法是後者，因為 PCIe 比 DRAM 慢。所以**決定速度的是你的記憶體通道數，不是那條 PCIe，也不是那張顯卡**。

| 記憶體配置 | 頻寬 |
|---|---:|
| DDR5-6000 雙通道（一般桌機） | 96 GB/s |
| DDR5-6400 八通道（工作站/伺服器） | 410 GB/s |
| PCIe 5.0 x16（對照） | 63 GB/s |
| RTX 5090 顯存（對照） | 1792 GB/s |

差距是一個數量級：把權重從顯存移到雙通道 DDR5，同一份權重的讀取時間變成 18.7 倍。

### 實例：DeepSeek-V4-Flash 在 RTX 5090 上

`deepseek-ai/DeepSeek-V4-Flash` 是 290.9 B 總參數、約 13 B 激活的 MoE，原生 FP8/INT8 權重 292.5 GB。RTX 5090 只有 32 GB 顯存——**最小的 1-bit 量化（`UD-IQ1_S`，82.5 GB）也還是顯存的 2.6 倍**，完全放不進去。

但 MoE 的結構讓 offload 可行。假設 attention 與 shared expert（約佔每 token 讀取的 30%）留在顯存，routed experts（約 70%）放 DRAM 由 CPU 算：

| 量化 | 檔案體積 | 每 token 讀取 | DDR5 雙通道 | DDR5 八通道 |
|---|---:|---:|---:|---:|
| 原生 FP8/INT8 | 292.5 GB | 13.0 GB | 7.2 tok/s | 28.7 tok/s |
| `UD-Q4_K_XL` | 155.1 GB | 6.9 GB | 13.6 tok/s | 54.2 tok/s |
| `UD-IQ2_M` | 90.9 GB | 4.0 GB | 23.3 tok/s | 92.7 tok/s |

（量化體積取自 HuggingFace 上 `unsloth/DeepSeek-V4-Flash-0731-GGUF` 的實際檔案大小，查詢日 2026-08-18。）

實務判讀：

- **一般桌機（雙通道 + 128 GB RAM）**：Q4 約 13 tok/s。能跑，但比人閱讀速度慢，適合離線批次而非互動。
- **工作站（八通道 + 192 GB RAM）**：Q4 約 54 tok/s，互動可用。
- 系統記憶體容量是硬門檻：跑 Q4 至少要 160 GB RAM，跑 IQ2 至少 96 GB。

這些都是**上限**。實測通常再打七到八折，因為 MoE 每個 token 選中的專家不同，記憶體存取是隨機而非循序的，實際達到的 DRAM 頻寬遠低於循序讀取的規格值。

---

## 五、公式沒涵蓋的東西

上面算的是**單一請求**的速度。實際服務還有幾個效應會讓它偏離：

**批次會攤提權重讀取。** 同時處理 32 個請求時，那 9 GB 權重讀一次可以服務 32 個 token。總吞吐幾乎線性上升，但**每個使用者感受到的速度不變**（甚至略降）。所以「總吞吐」與「單使用者速度」是兩個不同的數字，談效能時要講清楚在講哪一個。

**長 context 會讓 KV 讀取成為主角。** 公式裡的 KV 項隨 context 線性成長。同一個模型在 32K context 下的生成速度會明顯低於 1K，這不是實作問題，是結構決定的。

**推測解碼（speculative decoding）打破一 token 一次讀取的前提。** 用小模型草擬多個 token、大模型一次驗證，可以在頻寬不變的情況下把有效速度提高一到三倍。命中率決定實際收益。

**Prefix caching 只省 prefill，不省 decode。** 它讓重複的 system prompt 不用重算，對 decode 速度沒有幫助。

**MoE 的專家命中分佈會影響 offload 效果。** 如果某些專家被頻繁選中，把它們釘在顯存裡（hot-expert pinning）可以顯著改善；反之若路由很分散，offload 的效果會接近最差情況。

---

## 六、怎麼驗證公式

估算再漂亮也要對得上實機。要量到有意義的數字，量測必須符合公式的前提：

1. **單一請求**，`Running: 1 reqs`，不要有其他流量。
2. **固定生成長度**，例如強制生成 256 個 token，用 `max_tokens` 且避免提早遇到 stop。
3. **扣掉 prefill 時間**：量 TTFT（首個 token 時間）與總時間，`(總時間 − TTFT) / (生成 token 數 − 1)` 才是純 decode 的每 token 時間。
4. **短 prompt**，讓 KV 讀取那一項可以忽略。
5. **多跑幾次取中位數**，第一次會受 CUDA graph 與快取暖身影響。

反過來說，**服務 log 裡那行 `Avg generation throughput` 不能拿來驗證公式**——它是固定時間視窗的平均，含 prefill、含閒置、含批次效應。把它當成單流速度會低估好幾倍。

量到的值除以公式的理論上限就是實際 MBU。若明顯低於 0.6，值得檢查的是：有沒有開 CUDA graph（`--enforce-eager` 會讓 MBU 掉很多）、量化格式有沒有硬體原生支援（沒有的話每次都要 dequant）、以及批次大小是不是真的等於 1。

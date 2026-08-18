# DRAM offload 專案地圖：把大模型塞進小顯卡

`DeepSeek-V4-Flash` 的權重是 292 GB，一張 RTX 5090 有 32 GB。中間差的九倍不是靠壓縮補起來的——即使壓到 1-bit 也還差 2.6 倍。真正讓這件事成立的是另一條路：**把絕大多數權重放在系統記憶體，並想辦法讓它不要變成瓶頸**。

這條路在 2023 年還是研究題目，2026 年已經有官方教學可以照抄（單張 5090 跑 V4-Flash，20+ tok/s）。這篇整理目前這個生態裡有哪些專案、各自解決什麼問題、以及怎麼選。

- 相關：[量化與 offload：速度算得出來，品質算不出來](quantization-quality-and-offload.md)、[推論速度怎麼估](llm-decode-throughput-formula.md)
- GitHub 數據查詢日 **2026-08-18**

---

## 一、為什麼 MoE 讓 offload 從不可行變成可行

Offload 這個想法很早就有，但在 dense 模型上幾乎沒有意義。原因是 dense 模型**每產生一個 token 都要把全部權重讀一遍**——把權重放在系統記憶體，等於每個 token 都要搬運整個模型，速度會掉到不可用。

MoE 改變了這個算式。以 `DeepSeek-V4-Flash` 為例：

| | 數量 |
|---|---:|
| 總參數 | 290.9 B |
| 每 token 激活 | 約 13 B |
| 比例 | **4.5%** |

每個 token 只需要 4.5% 的權重。剩下 95.5% 雖然佔著記憶體，但這一步用不到——它們**可以放在慢的地方**。

第二個關鍵性質是**專家使用不均勻**。路由不是隨機的，某些專家被選中的頻率遠高於其他。這讓「把熱門專家釘在顯存、冷門的放系統記憶體」變得有效——一個 14% 的顯存快取，實際命中率可以遠高於 14%。

這兩點合起來，就是所有 MoE offload 專案的共同基礎。

---

## 二、兩種基本策略

所有專案都在回答同一個問題：**權重在 host，運算在哪裡做？**

| 策略 | 資料流 | 瓶頸 | 代表 |
|---|---|---|---|
| **搬到 GPU 算** | 每 token 把需要的權重經 PCIe 送進顯存 | PCIe 5.0 x16 約 50 GB/s | vLLM UVA / prefetch、MoE-Infinity |
| **留在 CPU 算** | 權重不動，CPU 就地算完把結果交給 GPU | 系統記憶體頻寬（雙通道約 96、八通道約 410 GB/s） | llama.cpp、ktransformers |

第二種通常較快，因為 DRAM 頻寬高於 PCIe，而且省掉了搬運。代價是要有夠強的 CPU——CPU 側的矩陣運算需要 AVX2 以上，AVX512 或 AMX 差別很大。

第一種的優勢在於能用 GPU 上的專用格式（例如 Blackwell 的 FP4 tensor core），CPU 沒有對應的指令。

實務上最好的實作是**混合**：熱門專家常駐 GPU 用 GPU 算，冷門專家留在 CPU 就地算，並在執行期動態調整哪些算熱門。

---

## 三、專案地圖

| 專案 | Stars | 最後更新 | 定位 |
|---|---:|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | 124,446 | 2026-08-18 | 通用單機推論，GGUF 生態的基礎 |
| [vLLM](https://github.com/vllm-project/vllm) | 89,318 | 2026-08-18 | GPU 服務端，offload 是附加能力 |
| [AirLLM](https://github.com/lyogavin/airllm) | 31,507 | 2026-08-18 | 極端逐層載入，4 GB 顯卡跑 70B |
| [SGLang](https://github.com/sgl-project/sglang) | 31,994 | 2026-08-18 | 服務框架，與 KT-Kernel 整合 |
| [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) | 42,953 | 2026-08-17 | ZeRO-Inference，偏訓練生態 |
| [ktransformers](https://github.com/kvcache-ai/ktransformers) | 19,252 | 2026-08-17 | **異質 CPU-GPU 推論，MoE offload 的標竿** |
| [PowerInfer](https://github.com/Tiiny-AI/PowerInfer) | 9,717 | 2026-05-11 | 激活局部性（hot/cold neuron），已轉向端側 |
| [MoE-Infinity](https://github.com/EfficientMoE/MoE-Infinity) | 343 | 2026-08-17 | MoE 專用，含 SSD 層 |
| [Fiddler](https://github.com/efeslab/fiddler) | 268 | 2024-11-18 | ICLR'25 研究原型，已停更 |
| [FlexLLMGen](https://github.com/FMInference/FlexLLMGen) | 9,355 | 2024-10-28 | **已封存**，這條路的起點 |

### ktransformers / KT-Kernel

目前這個領域最完整的實作，也是唯一有 `DeepSeek-V4-Flash` 官方教學的。核心設計是**異質專家放置**：熱門專家放 GPU、冷門專家放 CPU，並在執行期動態調整。

實際的啟動參數把設計思路暴露得很清楚：

```bash
--kt-method MXFP4 \
--kt-num-gpu-experts 10 \        # 256 個專家裡只有 10 個常駐 GPU
--kt-cpuinfer 60 \               # CPU 側 60 個推論執行緒
--kt-enable-dynamic-expert-update # 熱門專家動態遷移
```

其他值得注意的能力：CPU 側支援 INT4/INT8 量化權重（配合 AVX512/AMX）、三層 GPU-CPU-Disk 的 prefix cache、以及 Ascend NPU 與 AMD ROCm 支援。它同時做推論與 SFT，宣稱在 MoE 微調上比 ZeRO-Offload 快 6–12 倍。

推論路徑現在是透過 SGLang 整合（`sglang-kt` fork），不是獨立的 server。

**適合**：要在消費級硬體上跑超大 MoE，且願意處理較複雜的安裝（教學裡光是依賴版本的坑就列了五條）。

### llama.cpp

最廣泛使用的單機推論引擎，offload 能力是逐步長出來的。相關參數：

| 參數 | 作用 |
|---|---|
| `--n-gpu-layers` | 最早的做法：前 N 層放 GPU，其餘留 CPU |
| `--override-tensor` / `-ot` | 用正則表達式指定哪些張量放哪個裝置，MoE 專家 offload 的通用手段 |
| `--cpu-moe` / `--n-cpu-moe` | 針對 MoE 的捷徑：把（前 N 層的）專家放 CPU |
| `--no-kv-offload` | KV cache 不放 GPU |
| `--mlock` / `--no-mmap` | 控制記憶體鎖定與映射行為 |

`--n-cpu-moe` 的出現很能說明這條路線的成熟：早期要手寫 `-ot "\.ffn_(up|down|gate)_exps\.=CPU"` 這種正則，現在變成一個數字。

**適合**：想快速試、硬體多樣（含 Apple Silicon、AMD）、或已經在用 GGUF 生態（ollama、LM Studio 都建在它上面）。

### vLLM

Offload 是附加能力而非設計重心，但 0.25.1 已經有兩套後端：

```bash
--cpu-offload-gb 60          # UVA：zero-copy，每次 forward 經 PCIe 讀
--offload-group-size 8 \     # prefetch：分組非同步預取，藏傳輸延遲
--offload-num-in-group 2
```

還有 `cpu_offload_params` 可以只針對特定參數名稱 offload（填 `experts` 就只搬專家）。兩種後端的運算都留在 GPU，所以瓶頸是 PCIe。

**適合**：已經在用 vLLM 做服務，只是差一點顯存。不適合當成「用消費級硬體跑超大模型」的主力方案。

### MoE-Infinity

MoE 專用，機制描述得很清楚：把專家權重 offload 到 host memory **與 SSD**，需要時取回；用**激活感知快取**把熱門專家留在 GPU；用追蹤與預取把傳輸成本藏起來。上層提供 HuggingFace 相容的 `MoE` 類別與 OpenAI 相容的服務端（含連續批次、分頁 KV cache、串流）。

多 GPU 是專家參數 round-robin 分散、每張卡各自快取；跨機器的分散式推論還不支援。開源版本與論文版本不同，論文版本更偏極致效能。

**適合**：想要 MoE offload 又要 HuggingFace/OpenAI 介面，且 SSD 那一層有價值（記憶體真的不夠時）。

### PowerInfer

出發點不是 MoE，而是 **dense 模型的激活稀疏性**：ReLU 類啟動函數讓每個 token 實際只用到一小部分神經元，而且哪些會被用到有局部性。把「熱神經元」放 GPU、「冷神經元」放 CPU，就能在 dense 模型上做出類似 MoE 的效果。

早期展示是單張 RTX 4090 跑 Falcon(ReLU)-40B-FP16，比 llama.cpp 快 11 倍。後續發展出 PowerInfer-2（手機端）、TurboSparse（把 Mistral/Mixtral 稀疏化到約 90%）、SmallThinker 系列模型。現在的主體已經轉向端側裝置（Tiiny AI Pocket Lab，宣稱本地跑 GPT-OSS-120B int4 達 20 tok/s）。

**適合**：概念參考，或端側部署。它的價值在於證明了「激活稀疏性」這個角度，而不是當作通用 offload 工具。

### AirLLM

走極端路線：**逐層載入**。一次只把一層的權重放進顯存，算完就換下一層。這讓 4 GB 顯卡也能跑 70B 模型——但每個 token 都要把整個模型讀一遍，速度極慢。

**適合**：一次性的批次推論、或純粹要證明「跑得動」。不適合互動。

### Fiddler 與 FlexLLMGen

兩個有歷史意義但已經不活躍的專案。FlexLLMGen（**已封存**）是這條路的起點之一，提出用 GPU-CPU-Disk 三層記憶體做吞吐導向的單卡推論；Fiddler（ICLR'25）專門處理 MoE 的 CPU-GPU 編排，但 2024 年 11 月後停止更新。

想理解這條路怎麼演化過來的可以讀它們的論文，實務部署不要用。

---

## 四、幾個共通的關鍵技術

**熱冷專家調度**是效能差距的最大來源。靜態分配（固定把前 N 個專家放 GPU）與動態調度（依實際使用頻率遷移）的差別，可能是一倍以上。[前一篇](quantization-quality-and-offload.md)裡公式估 14–16 tok/s 而實測 20+ tok/s，主要差距就在這裡——公式假設路由均勻，現實不是。

**CPU 側量化**。權重在 CPU 也要壓縮，但 CPU 沒有 FP4/FP8 tensor core，所以走的是 INT4/INT8 配合 AVX512 或 AMX 的路線。這是 CPU 選型的重點：核心數固然重要，指令集支援與記憶體通道數更重要。

**多層儲存**。GPU 顯存 → 系統記憶體 → SSD 三層。ktransformers 的 prefix cache 與 MoE-Infinity 的專家儲存都做到 SSD 這一層。SSD 慢得多，但對「記憶體再多也裝不下」的情況是唯一解。

**推測解碼**。它打破「一個 token 一次權重讀取」的前提，在 offload 場景收益特別大（因為每次讀取的成本被放大了）。`DeepSeek-V4-Flash` 內建 NextN draft head，官方數字是約 1.2 倍。

**pinned memory 的容量門檻**。vLLM 的 UVA 路線要求 host 記憶體是 page-locked 的。這不是「有 200 GB RAM 就好」，而是要能鎖住其中大部分——這個限制在規劃時很容易被忽略。

---

## 五、怎麼選

| 情境 | 建議 |
|---|---|
| 消費級硬體跑超大 MoE，願意折騰 | **ktransformers + SGLang** |
| 快速試、硬體雜、要 Apple Silicon | **llama.cpp**（或 ollama / LM Studio） |
| 已經在跑 vLLM，只差一點顯存 | **vLLM 的 `--cpu-offload-gb` 或 prefetch** |
| 要 MoE offload + HF/OpenAI 介面 + SSD 層 | **MoE-Infinity** |
| 端側 / 手機 | PowerInfer 系列 |
| 只要證明跑得動，速度無所謂 | AirLLM |

---

## 六、硬體怎麼配

Offload 場景的硬體優先序與純 GPU 推論完全相反：

1. **系統記憶體容量是第一硬門檻。** `DeepSeek-V4-Flash` 的官方驗證配置要求 **≥200 GB**。這一項不滿足，後面都不用談。
2. **記憶體通道數決定速度，不是顯卡。** 雙通道 DDR5 約 96 GB/s、八通道約 410 GB/s，差 4.3 倍——而這個倍數會直接反映在 tok/s 上。同樣的預算，加記憶體通道比升級顯卡有效得多。
3. **CPU 要有 AVX512，最好有 AMX。** ktransformers 的最低要求是 AVX2 + FMA，但明說 AVX512/AMX 會顯著改善吞吐。
4. **顯存決定能快取多少熱門專家**，不決定能不能跑。32 GB 與 24 GB 的差別是命中率，不是可行性——RTX 3090 也在驗證清單上。
5. **儲存要夠且要快。** V4-Flash 的權重約 340 GB，首次載入的時間與 SSD 速度直接相關。

一個具體的對照：拿買一張 RTX 5090（32 GB）的錢去升級到 256 GB 八通道記憶體，在 offload 場景多半更划算——**因為瓶頸從來不在那張卡上**。

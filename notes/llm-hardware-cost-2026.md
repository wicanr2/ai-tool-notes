# 要跑這些模型最少需要多少硬體，在台灣要花多少錢

一年前回答這個問題，重點會放在顯卡。2026 年 8 月不是——**記憶體在過去 12 個月漲了約 500%**，一套 256 GB 的 DDR5 現在的價格已經逼近一張 RTX 5090。對於靠系統記憶體撐起來的 offload 路線，成本重心整個移位了。

這篇建立一組從模型參數推到新台幣報價的公式，用今天的市場價跑出五個檔次的配置，並算出自建與租用的回本點。

- **互動估算器**：[本地 LLM 建置成本估算器](../tools/build-cost-calculator.html)——可即時抓匯率，零件單價可自行更新並記住
- 相關：[推論速度怎麼估](llm-decode-throughput-formula.md)、[DRAM offload 專案地圖](dram-offload-projects.md)、[架構圖鑑](llm-architecture-map-2026.md)
- **價格基準日 2026-08-18，匯率 1 USD = 31.91 TWD**

---

## 一、硬體需求怎麼推

### 顯存

```
顯存需求 = 權重位元組 + KV cache
權重位元組 = 總參數量 × 每參數位元組
KV cache  = 2 × 全域注意力層數 × (kv_heads × head_dim) × KV精度 × context × 併發數
GPU 張數  = ceil( 需放進顯存的部分 ÷ 單卡顯存 )
```

KV 那一項有個常被算錯的地方：**只有全域注意力層的 KV 會隨 context 成長**。滑動視窗層的 KV 固定在視窗大小，線性注意力層根本沒有傳統意義的 KV。`gemma-4-E4B` 有 42 層，但真正隨 context 成長的只有 4 層（見[架構圖鑑](llm-architecture-map-2026.md)）——照 42 層算會高估十倍。

### Offload 時的系統記憶體

```
系統記憶體 = 放不進顯存的權重 × 1.25
```

那個 1.25 涵蓋作業系統、頁面對齊、以及 pinned memory 的額外開銷。實務上再往上取到 32 GB 的級距。

### 各模型的最低門檻

| 模型 | 權重（該精度） | 最低可行配置 |
|---|---:|---|
| gemma-4-E4B（bf16） | 16 GB | 單張 24 GB 卡 |
| gemma-4-26B-A4B（FP8） | 27 GB | 單張 32 GB 卡，或兩張 24 GB |
| Qwen3.6-27B（FP8） | 28 GB | 同上 |
| MiniMax-M2.7（FP8） | 229 GB | 5 張 48 GB，或 offload + 256 GB RAM |
| DeepSeek-V4-Flash（FP8） | 292 GB | 7 張 48 GB，或 offload + 200 GB RAM |
| DeepSeek-V4-Flash（MXFP4） | 154 GB | offload + 192 GB RAM（官方驗證：單張 5090） |
| Kimi-K3（MXFP4 原生） | 1,473 GB | 多節點，或 offload + 1.7 TB RAM |
| DeepSeek-V4-Pro（FP8） | 1,602 GB | 多節點 |

---

## 二、價格基準

| 項目 | 2026-08-18 市價（USD） | 來源 |
|---|---|---|
| RTX 5090 32GB（新） | $4,300 – 4,900（中位約 4,700） | [videocardprices](https://videocardprices.com/card/nvidia-rtx-5090/)、[TechPowerUp](https://www.techpowerup.com/351508/nvidia-rtx-50-series-median-prices-jump-up-to-41-in-august-with-msrp-now-a-distant-dream) |
| RTX 4090 24GB（二手） | $2,000+ | [BestValueGPU](https://bestvaluegpu.com/history/new-and-used-rtx-5090-price-history-and-specs/) |
| RTX 3090 24GB（二手） | $600 – 1,050 | [BestValueGPU](https://bestvaluegpu.com/history/new-and-used-rtx-3090-price-history-and-specs/)、[XDA](https://www.xda-developers.com/used-rtx-3090-still-best-for-local-ai-in-value/) |
| NVIDIA L40S 48GB | $7,500 – 10,000（MSRP 8,000） | [Thunder Compute](https://www.thundercompute.com/blog/nvidia-l40-pricing)、[GPU Cost](https://gpucost.org/gpu/l40s) |
| DDR5 消費級 | 約 $13.5 / GB（192GB 套裝約 $2,600） | [Tom's Hardware 價格追蹤](https://www.tomshardware.com/pc-components/ram/ram-price-index-2026-lowest-price-on-ddr5-and-ddr4-memory-of-all-capacities) |
| DDR5 RDIMM（伺服器） | 約 $13.6 / GB（64GB 約 $873） | [Tom's Hardware](https://www.tomshardware.com/pc-components/ram/memory-prices-climb-500-percent-in-12-months-up-to-10x-the-lowest-ever-tracked-prices-128gb-of-ddr5-now-usd3-399) |
| AWS g6e.xlarge（1×L40S） | $1.861 / 小時 | [Vantage](https://instances.vantage.sh/aws/ec2/g6e.xlarge) |
| AWS g6e.48xlarge（8×L40S） | $30.13 / 小時 | [Vantage](https://instances.vantage.sh/aws/ec2/g6e.48xlarge) |
| USD / TWD | 31.91 | 估算器可即時抓取 |

**兩個價格背景值得單獨標出來：**

RTX 5090 的 MSRP 是 $1,999，現在的街價是它的 2.2 到 2.5 倍。原因不是需求炒作，是它用的 GDDR7 是目前供給最緊的記憶體類型——**顯卡漲價與記憶體漲價是同一件事**。

DDR5 一年漲約 500%，32 GB 套裝從 2025 年 9 月的約 $100 漲到 2026 年 8 月的 $380–589。TrendForce 預測 Q3 2026 契約價還會再漲 13–18%。這對「靠大量系統記憶體跑大模型」這條路線是直接打擊。

---

## 三、五個檔次

以下為新台幣估價，**不含**電費、機櫃、作業系統授權、以及台灣通路的加價（見第六節）。

### Tier 0 — 跑 8B 級模型　　`NT$ 88,000`

| 項目 | USD | TWD |
|---|---:|---:|
| 二手 RTX 3090 24GB | 800 | 25,528 |
| DDR5 64GB | 864 | 27,570 |
| 平台（消費級 CPU/主板/電源/機殼） | 900 | 28,719 |
| NVMe 2TB | 200 | 6,382 |
| **合計** | **2,764** | **88,199** |

跑 `gemma-4-E4B`、8B 以下模型、或 LoRA 微調實驗。二手 3090 在這個級距仍是性價比最高的選擇。

### Tier 1 — 跑 26B 級 FP8

**省錢版（雙卡）`NT$ 123,000`**

| 項目 | USD | TWD |
|---|---:|---:|
| 二手 RTX 3090 24GB × 2 = 48GB | 1,600 | 51,056 |
| DDR5 64GB | 864 | 27,570 |
| 平台（需支援雙卡） | 1,200 | 38,292 |
| NVMe 2TB | 200 | 6,382 |
| **合計** | **3,864** | **123,300** |

**單卡版　`NT$ 213,000`**

把兩張 3090 換成一張 RTX 5090（$4,700），總價跳到 NT$ 212,648。多花 NT$ 89,000 換來的是省一個 PCIe 插槽、少一半功耗、不用處理 tensor parallel。

跑 `gemma-4-26B-A4B` FP8、`Qwen3.6-27B` FP8。

### Tier 2 — DeepSeek-V4-Flash offload　`NT$ 386,000`

這是 ktransformers 官方驗證過的配置（單張 5090，20+ tok/s）。

| 項目 | USD | TWD |
|---|---:|---:|
| RTX 5090 32GB | 4,700 | 149,977 |
| **DDR5 RDIMM 256GB** | **3,481** | **111,079** |
| HEDT 平台（多核 + AVX512 + 8 通道） | 3,500 | 111,685 |
| NVMe 4TB | 400 | 12,764 |
| **合計** | **12,081** | **385,505** |

把顯卡換成二手 3090 可以降到 **NT$ 261,000**，速度會下降但仍可用——因為這條路線的瓶頸在記憶體不在顯卡。

### Tier 3 — DeepSeek-V4-Flash 全部進顯存　`NT$ 3,024,000`

| 項目 | USD | TWD |
|---|---:|---:|
| NVIDIA L40S 48GB × 8 | 72,000 | 2,297,520 |
| 雙路伺服器平台 | 15,000 | 478,650 |
| DDR5 RDIMM 512GB | 6,963 | 222,189 |
| NVMe 8TB | 800 | 25,528 |
| **合計** | **94,763** | **3,023,887** |

速度會遠高於 Tier 2（估計 300+ tok/s，見[速度估算](llm-decode-throughput-formula.md)），但價格是它的 7.8 倍。

### Tier 4 — DeepSeek-V4-Pro / Kimi-K3

V4-Pro 的 FP8 權重是 1,602 GB，Kimi-K3 的 MXFP4 是 1,473 GB。前者超出單機範圍需要跨節點；後者理論上可以用 offload 硬撐，但需要約 **1.7 TB 系統記憶體**——以今天的 RAM 價格，光記憶體就要 NT$ 740,000 以上。

**這個級距的結論是租，不是買。**

---

## 四、成本結構已經翻轉

把 Tier 2 的組成畫出來：

```
RTX 5090          $4,700   ████████████████████  39%
DDR5 RDIMM 256GB  $3,481   ███████████████       29%
HEDT 平台         $3,500   ███████████████       29%
NVMe 4TB            $400   ██                     3%
```

**記憶體的花費已經接近顯卡。** 一年前同樣的 256 GB 大約是 $600–700，佔比不到 10%。

這改變了兩件事的判斷：

**一、「加記憶體比換顯卡划算」現在只是勉強成立。** 效能上這句話仍然對——offload 的速度由記憶體通道數決定，不由顯卡決定。但價差已經從「一張顯卡換四倍記憶體」縮到「$4,700 對 $3,481」。

**二、二手顯卡的相對價值上升。** 在 offload 配置裡，把 5090 換成二手 3090 省下 NT$ 125,000，而速度損失有限（顯存少 8 GB 只影響能快取多少熱門專家）。Tier 2-B 那個 NT$ 261,000 的配置，性價比明顯優於 Tier 2-A。

---

## 五、自建 vs 租用

```
回本月數 = 自建總價 ÷ (雲端每小時 × 730 × 利用率)
```

| 規模 | 自建 | AWS 每月（24/7） | 利用率 100% | 50% | 20% |
|---|---:|---:|---:|---:|---:|
| 1×L40S 等級 | NT$ 454,000 | NT$ 43,000 | 10.5 個月 | 21.0 | 52.4 |
| 8×L40S 等級 | NT$ 3,024,000 | NT$ 702,000 | 4.3 個月 | 8.6 | 21.5 |

**利用率是這張表唯一重要的變數。** 一台每天真正跑 5 小時（20%）的機器，回本要四年多——那期間硬體已經折舊兩代。

判斷準則：

- **持續高負載（>50% 利用率）且需求穩定** → 自建划算，尤其大規模時（8 卡的回本點比單卡快 2.4 倍，因為雲端大機型的溢價更高）。
- **開發、實驗、間歇性使用** → 租。這也是目前那台 `g6e.xlarge` 的正確用法。
- **資料不能出境** → 沒得選，只能自建，成本不是唯一考量。

另一個常被忽略的選項：L40S 的雲端市場median 是 $1.27/GPU/小時，最低到 $0.47——遠低於 AWS 的 $1.861。跨供應商比價可以把租用成本再砍一半以上。

---

## 六、台灣的實際價格會更高

上面全部是「美金市價 × 匯率」，實際在台灣採購還要加上：

| 項目 | 影響 |
|---|---|
| 進口關稅 | 顯示卡與零組件多數為 0%，但仍有報關費用 |
| 營業稅 | 5% |
| 通路加價 | 消費級零組件通常 5–15%，伺服器級與缺貨品項更高 |
| 缺貨溢價 | RTX 5090 與大容量 DDR5 在缺貨期的實際成交價可能再高一截 |

**經驗值是把美金換算後再乘 1.1 到 1.2。** 以 Tier 2 為例，NT$ 386,000 的估價，實際落地大約 NT$ 420,000 – 460,000。

二手市場（露天、蝦皮、PTT 相關板）是另一條路，3090/4090 的流通量還可以，但要自行承擔沒有保固的風險——礦卡與長期高負載卡在這個價位帶不少。

---

## 七、給不同情境的建議

| 情境 | 建議 | 預算 |
|---|---|---|
| 學習、跑 8B 級模型、LoRA 實驗 | 二手 3090 單卡 | NT$ 90,000 |
| 內部工具、26B 級 FP8、少量併發 | 二手 3090 ×2 | NT$ 123,000 |
| 想在本地跑最強的開源模型 | 二手 3090 + 256GB RAM offload | NT$ 261,000 |
| 同上但要 20+ tok/s 與更好的 context | RTX 5090 + 256GB RAM | NT$ 386,000 |
| 生產服務、高併發、持續負載 | 8×L40S 或直接租 | NT$ 3,024,000 起 |
| 間歇使用、開發測試 | **租**，並跨供應商比價 | 每月 NT$ 43,000 起 |

最後一句提醒：**這些數字的半衰期很短。** 記憶體與顯卡都在異常的價格週期裡，季度層級的變動可以達到雙位數百分比。[估算器](../tools/build-cost-calculator.html)的每個單價都可以自己改，改完會記住——用之前先花兩分鐘點開來源核對，比引用這篇的絕對數字可靠得多。

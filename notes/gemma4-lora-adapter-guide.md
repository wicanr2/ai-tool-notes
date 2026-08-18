# 在 gemma-4-E4B-it 上掛載 LoRA adapter

拿到一份微調交付物時，最常見的困惑是：收到的只有一個十幾 MB 的資料夾，裡面沒有模型權重，卻宣稱能讓一個 15 GB 的模型行為改變。這份資料夾要怎麼跟原本的模型接起來、接錯了會怎樣、哪些數字可以改哪些不能改——這篇把這幾件事拆開講，並附上一次在 NVIDIA L40S 主機上的實際驗證紀錄。

以下用一個工具呼叫任務的 adapter 當範例，base model 為 `google/gemma-4-E4B-it`。原理與指令適用於任何 PEFT 格式的 LoRA adapter。

- 相關：[chat_template 是什麼](gemma4-chat-template-first-principles.md)、[HuggingFace 模型 repo 檔案逐一解說](gemma4-hf-repo-files.md)、[2026-07 gemma-4 chat_template 變更事件](gemma4-chat-template-change-incident.md)。

---

## 一、名詞解釋

### base model（基礎模型）

已經訓練完成、可以獨立運作的完整模型，本身就是一組完整的權重檔。`google/gemma-4-E4B-it` 就是一個 base model：42 層 transformer、hidden size 2560、詞彙表 262144 個 token，權重檔 `model.safetensors` 約 15 GB。

「base」是相對關係，不是等級。它指的是「後續調整以誰為基準」，而不是「比較弱的版本」。`-it` 後綴代表 instruction-tuned，也就是 Google 已經在預訓練之上做過一輪指令微調；再往上疊 LoRA，它仍然叫 base model。

一個關鍵性質：**base model 的權重完全不會被 adapter 改動**。同一份 15 GB 權重可以同時被十幾個不同 adapter 共用。

### adapter（轉接層）

一小組額外的權重，本身不能單獨運作，必須疊在某個特定 base model 上才有意義。它記錄的不是「模型是什麼」，而是「相對於 base model 要調整多少」。

用一個比喻：base model 是一台出廠的機器，adapter 是一份校正參數表。表本身不會動，機器也沒被拆開改造，但機器讀進這份表之後行為就變了。換一台不同型號的機器，同一份表就對不上——這也是為什麼 adapter 一定要標明它的 base model 是誰。

本文範例的 adapter 只有 18 MB，是 base model 的 0.12%。它包含 4,276,224 個參數，相對於 base model 的數十億參數。

### LoRA（Low-Rank Adaptation）

adapter 的一種具體做法，也是目前最通用的一種。

全參數微調要更新模型裡每一個權重矩陣 `W`，成本等同重訓一次。LoRA 的做法是把 `W` 凍結不動，另外學兩個很小的矩陣 `A` 和 `B`，推論時計算：

```
輸出 = W · x  +  (alpha / r) · B · A · x
        ↑基礎模型（凍結）    ↑adapter 的貢獻
```

若 `W` 的形狀是 `2560 × 2048`，那 `A` 是 `8 × 2560`、`B` 是 `2048 × 8`。原本 524 萬個參數要訓，現在只要訓 3.7 萬個，少了 99.3%。

之所以行得通，是因為微調帶來的權重變化通常集中在一個很低維的子空間——調整模型「在什麼情境下呼叫哪個工具」，並不需要動用整個矩陣的自由度。

### rank（`r`）

上面那個 8，也就是 `A` 和 `B` 之間那條「腰」有多寬。`r` 越大，adapter 能表達的調整越複雜，參數量與過擬合風險也越高。工具選擇這類任務，`r` 落在 8～16 是常見選擇；要讓模型學會一整個新領域的知識，才需要 64 以上。

`r` 一旦訓練完成就固定在權重的形狀裡了，載入時不能改。

### alpha 與 scaling

`lora_alpha` 搭配 `r` 決定 adapter 的強度：實際縮放係數是 `alpha / r`。本例 `alpha=16`、`r=8`，縮放係數為 `2.0`。

分開成兩個參數的理由是解耦：調 `r`（容量）時不希望整體強度跟著跑掉，所以用 `alpha` 補回來。這是唯一一個在部署階段還能有意義調整的旋鈕，詳見第五節。

### target_modules（目標模組）

指定 LoRA 要掛在模型的哪些層、哪些矩陣上。transformer 的每一層 self-attention 有四個投影矩陣：

| 模組 | 全名 | 作用 |
|---|---|---|
| `q_proj` | query projection | 產生「我在找什麼」 |
| `k_proj` | key projection | 產生「我是什麼」 |
| `v_proj` | value projection | 產生「我能提供什麼內容」 |
| `o_proj` | output projection | 把多頭注意力的結果合回去 |

MLP 部分則有 `gate_proj` / `up_proj` / `down_proj`。只掛 attention 通常足以改變「模型怎麼在情境間做選擇」；要改變模型記得什麼知識，才需要動到 MLP。

### PEFT

Hugging Face 的 Parameter-Efficient Fine-Tuning 套件，同時也是 adapter 的**事實標準檔案格式**。當交付物說「PEFT 格式 LoRA adapter」，意思是這個資料夾裡有 `adapter_config.json` 和 `adapter_model.safetensors` 兩個檔案，且命名規則符合 PEFT 的約定——vLLM、TGI、SGLang 等推論引擎都認得這個格式。

### merge（合併）與動態掛載

adapter 有兩種上線方式：

**動態掛載**保持 base 與 adapter 分離，推論時即時相加。好處是同一份 base 權重可服務多個 adapter、可隨時切換或關閉；代價是每次前向傳播多一次小矩陣運算。

**merge** 把 `(alpha/r)·B·A` 直接加進 `W`，輸出一份新的完整權重。好處是推論時零額外開銷、部署對象是一個普通模型；代價是產生一份 15 GB 的新檔案、失去切換能力，且合併後無法還原成原本的 adapter。

工具呼叫這類需要 A/B 對照的場景，動態掛載幾乎總是比較好的選擇。

### chat template

一段 Jinja2 樣板，負責把 `[{"role": "user", "content": "..."}]` 這種結構化訊息，攤平成模型真正吃進去的那一串文字。gemma-4 用的標記是 `<|turn>user` … `<turn|>`，工具宣告則被轉成 `<|tool>declaration:get_inventory{...}<tool|>` 這種自訂緊湊語法，而不是 JSON。

它對 adapter 的意義在第六節會展開，簡短版是：**LoRA 學到的是「在某種特定文字排列之後該接什麼」，樣板換了，模型看到的輸入就不是它學過的那一種了。**

### safetensors

一種權重檔格式，取代早期用 Python `pickle` 序列化的 `.pt` / `.bin`。差別在 pickle 反序列化時會執行任意程式碼，而 safetensors 只是「一段 JSON 標頭 + 一塊連續的數值」，載入時不執行任何邏輯。附帶好處是標頭可以單獨讀出來，不必載入 15 GB 就能查出裡面有哪些張量、形狀多少——第四節的驗證就是靠這個做的。

### checkpoint 與 step

訓練途中定期存下的中間狀態。`checkpoint-step-720` 表示這是第 720 次參數更新後存的。挑 checkpoint 是微調流程的一部分：訓練越久不代表越好，過了某個點模型會開始過擬合，評測分數反而下滑。

---

## 二、交付物裡的每個檔案

一份完整的 PEFT adapter 交付物長這樣：

```
my-tool-lora/
├── adapter_config.json          ← 必要：怎麼掛
├── adapter_model.safetensors    ← 必要：掛什麼
├── native_lora_config.json      ← 選用：訓練端的原始設定
├── conversion_info.json         ← 選用：格式轉換來源紀錄
└── README.md                    ← 選用
```

只有前兩個檔案是推論引擎會讀的。其餘是給人看的來源紀錄，刪掉不影響載入。

### adapter_config.json

PEFT 和 vLLM 唯一會讀的設定檔，決定 adapter 怎麼被掛上去。

| 欄位 | 本例的值 | 意義 |
|---|---|---|
| `peft_type` | `LORA` | adapter 種類（另有 `IA3`、`PREFIX_TUNING` 等） |
| `base_model_name_or_path` | `google/gemma-4-E4B-it` | 訓練時的 base model。**只是紀錄，載入時不會自動下載也不會驗證**，掛錯 base 不會有人提醒 |
| `r` | `8` | rank |
| `lora_alpha` | `16` | 縮放分子，實際係數 `16/8 = 2.0` |
| `lora_dropout` | `0.0` | 訓練期的正則化，推論時無作用 |
| `target_modules` | 132 個完整路徑 | 掛在哪些矩陣上 |
| `bias` | `none` | 不訓練 bias 項 |
| `modules_to_save` | `null` | 沒有任何模組被整個複製下來訓練。這一欄若非 null（常見於 `embed_tokens`、`lm_head`），交付物會大很多，且對 base model 版本更敏感 |
| `task_type` | `CAUSAL_LM` | PEFT 用來決定包裝哪一種模型類別 |
| `inference_mode` | `true` | 載入後預設凍結，不接受梯度 |
| `use_rslora` | `false` | 未啟用 rank-stabilized 縮放（啟用時係數改為 `alpha/√r`） |
| `use_dora` | `false` | 未啟用 DoRA（把權重拆成方向與大小分開調整的變體） |

`target_modules` 這一欄有兩種寫法。常見的是後綴形式：

```json
"target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
```

意思是「所有名字以這些結尾的模組都掛」。本例用的是另一種——完整路徑，一條一條列出來：

```json
"target_modules": [
  "model.language_model.layers.0.self_attn.q_proj",
  "model.language_model.layers.0.self_attn.k_proj",
  ...
  "model.language_model.layers.41.self_attn.o_proj"
]
```

兩種寫法 PEFT 都接受。完整路徑的好處是精確——本例的 132 個模組並非均勻分佈：

- 第 0～23 層：`q` / `k` / `v` / `o` 四個都掛（24 × 4 = 96）
- 第 24～41 層：只掛 `q` 和 `o`（18 × 2 = 36）

這個分界不是任意的。`google/gemma-4-E4B-it` 的 `config.json` 裡 `num_kv_shared_layers = 18`，而 `42 - 18 = 24`——第 24 層之後全部是 **KV 共享層**：它們不維護自己的 KV cache，注意力讀的是前面最後一個同類型層算出來的 K/V。vLLM 的模型實作在這些層只對 Q 套用 RoPE，K/V 既不做 norm 也不套 RoPE，算出來就丟掉：

```python
if not self.is_kv_shared_layer:
    k = self.k_norm(k); q, k = self.rotary_emb(positions, q, k)
    v = self.v_norm(v)
else:
    q = self.rotary_emb(positions, q, k)[0]   # 只有 Q
```

checkpoint 裡這 18 層仍有 `k_proj` / `v_proj` 權重，但它們的輸出不參與注意力計算。**在這些層對 k/v 掛 LoRA 不會改變任何輸出**，訓練端跳過它們是正確的。這個機制與 KV cache 的關係見 [vLLM 架構與 KV cache](vllm-serving-and-architecture.md)。

### adapter_model.safetensors

權重本體，264 個張量（132 個模組 × `lora_A` 與 `lora_B`），FP32 儲存，18.2 MB。

張量的命名規則決定了它能不能對上 base model：

```
base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight
└──── PEFT 包裝前綴 ────┘└──────── base model 裡的實際路徑 ────────┘└─ 側 ─┘
```

PEFT 在 base model 外面包一層 `base_model.model.`，後面接的是模組在 base model 裡的原始路徑。這代表**載入時 base model 的模組路徑必須逐字對上**，第三節的第一個陷阱就出在這裡。

形狀規則：

| 側 | 形狀 | 本例（q_proj） |
|---|---|---|
| `lora_A` | `[r, 輸入維度]` | `[8, 2560]` |
| `lora_B` | `[輸出維度, r]` | `[2048, 8]` |

儲存成 FP32 而推論多半用 bfloat16，這是正常的——adapter 很小，用高精度存不佔空間，載入時再轉換。

### native_lora_config.json

訓練框架自己的設定格式，PEFT 不會讀。它記錄了原始訓練意圖，在追查問題時比 `adapter_config.json` 更有資訊：

```json
{
  "format": "native_lora_v1",
  "base_model_id": "google/gemma-4-E4B-it",
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
  "lora_rank": 8,
  "lora_alpha": 16,
  "lora_include_prefixes": ["model.language_model"],
  "lora_exclude_prefixes": ["model.vision_tower"]
}
```

`lora_exclude_prefixes` 明確排除了視覺塔，代表這次微調只動語言部分——對一個純文字的工具呼叫任務來說是合理的設計。

### conversion_info.json

格式轉換的來源紀錄，說明這份 PEFT adapter 是從訓練框架的原生 `.pt` 檔轉出來的，並標明轉換時參照的 PEFT 版本（0.19.1）與模組數。載入無關，但在「為什麼結果跟訓練端對不起來」的排查中，它是唯一能指回原始 checkpoint 的線索。

### 交付物之外必須確認的兩個檔案

adapter 資料夾裡**沒有** tokenizer、也**沒有** chat template。這兩樣東西來自 base model 的 repo：

| 檔案 | 來源 | 為什麼要管 |
|---|---|---|
| `tokenizer.json` / `tokenizer_config.json` | base model repo | 決定文字怎麼切成 token。與訓練端不同版本會讓同一句話變成不同 token 序列 |
| `chat_template.jinja` | base model repo（或推論引擎自帶的範例） | 決定訊息怎麼攤平成文字。這是最容易被忽略、後果最嚴重的一項 |

**交付一份 adapter 時，應該連同「訓練時使用的 chat template 檔案本身或其 commit hash」一起交付。** 少了它，接手方無法確定自己渲染出來的 prompt 跟訓練時是否一致，而這個差異不會報錯。

---

## 三、載入方式

### 路線 A：transformers + PEFT

適合離線驗證、逐題比對、跑評測。

```python
from transformers import AutoTokenizer, AutoModelForImageTextToText
from peft import PeftModel
import torch

BASE    = "google/gemma-4-E4B-it"
ADAPTER = "/path/to/my-tool-lora"

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForImageTextToText.from_pretrained(
    BASE, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER, adapter_name="tool")
model.eval()
```

**模型類別要選對。** `google/gemma-4-E4B-it` 的 `config.json` 裡 `architectures` 是 `Gemma4ForConditionalGeneration`——一個含視覺與音訊分支的多模態外殼，語言部分的路徑是 `model.language_model.layers.N.…`。adapter 的張量名稱正是照這個路徑寫的。

若改用 `AutoModelForCausalLM` 載出純語言外殼，路徑會變成 `model.layers.N.…`，於是 PEFT 一個模組都對不上。它可能拋 `Target modules not found`，也可能靜默掛上 0 個模組——後者最麻煩，因為模型照常回答，只是回答的是完全沒調過的 base model。

所以載入後第一件事是數：

```python
n = sum(1 for name, _ in model.named_modules() if name.endswith("lora_A.tool"))
print(f"LoRA modules attached: {n}")   # 必須是 132
```

確認 adapter 真的在影響輸出，比對開關兩種狀態的 logits：

```python
ids = tok(text, return_tensors="pt")
with torch.no_grad():
    on = model(**ids).logits[0, -1]
    with model.disable_adapter():
        off = model(**ids).logits[0, -1]
print("最大差異:", (on - off).abs().max().item())   # 應遠大於 0
```

`disable_adapter()` 是 PEFT 提供的 context manager，區塊內暫時關掉 adapter 的貢獻。這比「另外載入一次 base model」省一半記憶體，是做 A/B 對照最省事的方式。

### 路線 B：vLLM serving

適合實際上線與整合測試。

```bash
vllm serve google/gemma-4-E4B-it \
  --served-model-name gemma-4-e4b-it \
  --enable-lora \
  --lora-modules my-tool-lora=/path/to/my-tool-lora \
  --max-lora-rank 8 \
  --max-loras 1 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4 \
  --chat-template /root/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/<revision>/chat_template.jinja \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768
```

| 參數 | 作用 |
|---|---|
| `--enable-lora` | 開啟 LoRA 支援。未加時 `--lora-modules` 會被忽略 |
| `--lora-modules 名稱=路徑` | 註冊 adapter。等號左邊的名稱就是呼叫時要填的 model id |
| `--max-lora-rank` | 為 LoRA 預先配置的顯存槽寬度，要 ≥ adapter 的 `r`。設小於 8 會直接拒載 |
| `--max-loras` | 同時可駐留的 adapter 數量 |

`--chat-template` 要明確指向訓練時用的那一份，不要依賴預設。vLLM 容器自帶的 `examples/tool_chat_template_gemma4.jinja` 與 base model repo 的 `chat_template.jinja` 是兩份不同的檔案（差異見第六節），預設行為會隨 vLLM 版本改變，寫死路徑才有可重現性。

呼叫時 `model` 欄位要填 **adapter 名稱**，不是 base model 名稱：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-tool-lora",
    "messages": [{"role":"user","content":"查一下 A12 的庫存"}],
    "tools": [ ... ]
  }'
```

填成 `gemma-4-e4b-it` 會走沒調過的 base model，回應正常、沒有任何警告。這是實務上最常見的「掛了但沒生效」。

### 路線 C：合併成獨立模型

當部署端的推論引擎不支援動態 LoRA，或確定只會用這一個 adapter 時：

```python
model = PeftModel.from_pretrained(base, ADAPTER)
merged = model.merge_and_unload()
merged.save_pretrained("/path/to/gemma-4-e4b-it-merged")
tok.save_pretrained("/path/to/gemma-4-e4b-it-merged")
```

輸出是一份約 15 GB 的完整模型，之後當普通模型用即可。合併前務必留著原本的 adapter 資料夾——`merge_and_unload()` 是單向的。

---

## 四、實測紀錄

環境：單張 NVIDIA L40S（46 GB）的 Linux 主機，vLLM 0.25.1，transformers 5.13.1，PEFT 0.19.1，base model 快取 revision `fa62d88d`。第 2 項的 prompt 由 base model repo 的 `chat_template.jinja` 渲染；第 5 項的 vLLM 服務啟動時指定的是容器自帶的 `examples/tool_chat_template_gemma4.jinja`。

| # | 驗證項目 | 方法 | 結果 |
|---|---|---|---|
| 1 | 張量名稱與形狀是否對得上 base model | 直接解析兩份 safetensors 的 JSON 標頭，逐一比對 264 個張量對應的 base 權重與維度 | **264 / 264 全數對上**，宣告的 132 個 target 在 base model 中全部存在 |
| 2 | PEFT 能否掛載，掛上後是否真的改變輸出 | CPU 上載入 base model + `PeftModel.from_pretrained`，同一 prompt 比對 adapter 開關的 logits | **掛上 132 個模組**，scaling 2.0；logits 最大差異 20.81、平均差異 7.73 |
| 3 | vLLM 是否支援此模型的 LoRA | 檢查 vLLM 0.25.1 `Gemma4ForConditionalGeneration` 的 `SupportsLoRA` 介面 | **支援**，packed 映射 `qkv_proj → [q_proj, k_proj, v_proj]` |
| 4 | vLLM 的載入器能否吃下這份 adapter | 直接呼叫 `LoRAModel.from_local_checkpoint`，expected 模組集合按 vLLM 原始碼規則展開 packed 映射 | **載入成功**，132 個模組、rank 8、scaling 2.0，型別分佈 `q_proj:42, o_proj:42, k_proj:24, v_proj:24` |
| 5 | 端到端 serving | 實際起一個 vLLM 實例（`--enable-lora`），對 base 與 adapter 兩個 model id 發相同請求比對 | **adapter 成功載入並生效**，詳見下方 |

第 2 項的 prompt 用 base model repo 的 `chat_template.jinja` 渲染一組帶工具宣告的訊息，結尾為 `<|turn>model\n`。adapter 開與關預測的下一個 token 都是 `<|tool_call>`，但整體 logits 分佈差異顯著——base model 本來就傾向在這個位置發工具呼叫，adapter 改變的是後續選哪個工具、帶哪些參數，這與該 adapter 交付報告中「剩餘錯誤集中在工具選擇、參數錯誤為零」的描述一致。

### 端到端 serving 的觀察

啟動 log 確認 adapter 被接受：

```
INFO [serving.py:216] Loaded new LoRA adapter: name 'cils-lora', path '/adapter'
```

`/v1/models` 同時列出兩個 id，adapter 的 `parent` 指向 base：

| id | root | parent |
|---|---|---|
| `gemma-4-e4b-it` | `google/gemma-4-E4B-it` | `null` |
| `cils-lora` | `/adapter` | `gemma-4-e4b-it` |

這也順帶排除了第四節提到的 packing 疑慮——本 adapter 有 18 層只掛 `q` 而沒有 `k`/`v`，vLLM 執行期要把這種不完整的切片組合打包進單一 `qkv_proj` 層，實測未出現任何問題。

A/B 對照用同一組工具宣告（一個查詢類、兩個動作類），temperature 設 0，服務啟動時以 `--chat-template` 指向 base model repo 的 `chat_template.jinja`（與訓練一致），對兩個 model id 發相同請求：

| 提問 | base | adapter |
|---|---|---|
| 查詢某貨架的空儲位 | `get_slots` | `get_slots` |
| 把某棧板從 A 站搬到 B 站 | `moving_pallet` | `moving_pallet` |
| 某棧板卡在深料架最裡面 | `retrieve_pallet` | `retrieve_pallet` |
| 今天天氣如何（工具清單裡沒有對應工具） | 拒答 | 拒答 |
| 你好，你是誰 | 泛用的自我介紹 | 自我介紹並提及可查詢儲位、移動棧板 |

三個領域內的問題兩邊選擇一致——這類明確情境 base model 本來就答得出來，adapter 的價值在交付報告所述的邊界案例上，不在一眼可判的題目。範圍外提問連續問五次，adapter 五次都拒答，行為穩定。

多輪的部分，餵進一則 `moving_pallet` 被拒的工具回應（`destination ST-07 is occupied by PLT-0555`）：

| | 回應 |
|---|---|
| base | 英文複述拒絕原因 |
| adapter | 繁體中文說明「搬運失敗，因為目的地 ST-07 目前已被棧板 PLT-0555 佔用」 |

兩邊都沒有原樣重試，也都正確傳達了失敗原因。adapter 的語言與領域用詞跟系統其他部分一致，base 則回退成英文。

這一輪只涵蓋單輪決策與一次 rejection 回應。交付報告列出的三個工具選擇錯誤（該用查詢類卻選了動作類）需要對應的邊界情境才能重現，不在這五題的涵蓋範圍內。

原本打算在同一個服務內用 per-request 的 `chat_template` 欄位比較兩份樣板，vLLM 0.25.1 拒絕了：

```
Chat template is passed with request, but --trust-request-chat-template is not set.
Refused request with untrusted chat template.
```

要做這種比較得在啟動時加 `--trust-request-chat-template`，或分兩次啟動各指定一份樣板。下一節的樣板對照就是用後者取得的。

兩趟端到端測試各讓既有服務中斷約 5 分鐘。

---

## 五、哪些參數能調，哪些不能

| 參數 | 部署階段可否調整 | 說明 |
|---|---|---|
| `r` | 不可 | 固化在權重形狀裡，改了直接載不進去。要換 rank 只能重訓 |
| `target_modules` | 不可 | 增刪都會讓張量名稱對不上 |
| `lora_alpha` | **可** | 改 `adapter_config.json` 即整體縮放 adapter 強度 |
| `lora_dropout` | 無意義 | 推論階段不生效 |
| `--max-lora-rank` | 可 | 服務端參數，只要 ≥ `r` 即可。設大只是多佔一點顯存 |

`lora_alpha` 是唯一實務上有意義的旋鈕。`alpha=12` 讓縮放係數降到 1.5，adapter 的影響減弱、模型行為靠回 base；`alpha=24` 升到 3.0，影響加強。

什麼時候該動它：實測發現模型過度傾向發工具呼叫、連該用自然語言回覆的場合也硬要呼叫，可以往下調；反之則往上。但每動一次都要重跑離線評測確認沒有退化——交付的評測分數是在 `alpha=16` 下量到的，換一個值那個數字就不再成立。

---

## 六、五個會靜默失效的地方

這些狀況的共同點是不會拋出例外。模型照常回答，只是回答的品質悄悄退回微調前，或更差。

### 1. 模型類別選錯，掛上 0 個模組

`AutoModelForCausalLM` 與 `AutoModelForImageTextToText` 對同一個 repo 會建出模組路徑不同的兩種外殼。用第三節那段數模組的程式碼擋掉。

### 2. 請求填了 base model 的名稱

vLLM 同時服務 base 與 adapter 兩個 model id。填錯不會有警告。做法是同一組 prompt 分別打兩個 id，輸出必須明顯不同；相同就是沒生效。

### 3. chat template 與訓練時不一致

這是最隱蔽的一項。同一個模型有兩份來源不同的 gemma-4 樣板在流通，而它們**不相同**。實際比對 base model repo 的 `chat_template.jinja` 與 vLLM 0.25.1 容器內的 `examples/tool_chat_template_gemma4.jinja`，差異落在 24 行，其中最關鍵的一處出現在生成起點：

{% raw %}
```diff
     {%- if not continue_same_model_turn -%}
         {{- '<|turn>' + role + '\n' }}
+        {%- if role == 'model' and not enable_thinking and not (...) -%}
+            {{- '<|channel>thought\n<channel|>' -}}
+        {%- endif -%}
     {%- endif -%}
```
{% endraw %}

vLLM 那份在模型該開始生成的位置多插了一個空的 thought channel，HF 那份沒有。也就是說，同一組 messages 經兩份樣板渲染後，模型在最關鍵的那個位置看到的前綴是不一樣的：

| 樣板 | prompt 結尾 |
|---|---|
| HF repo `chat_template.jinja` | `<\|turn>model\n` |
| vLLM `tool_chat_template_gemma4.jinja` | `<\|turn>model\n<\|channel>thought\n<channel\|>` |

LoRA 學到的是「在訓練時那種文字排列之後接什麼」。換一份樣板，模型面對的就是一個它沒學過的前綴。另外兩處差異在 `thinking_gate` 的條件與工具回應後的 turn 收尾，同樣會改變 token 序列。

這個差異的後果實際量過。同一個 adapter、同一組工具宣告、同一個提問、temperature 0，只換服務啟動時的 `--chat-template`：

| 提問 | vLLM 樣板 | HF 樣板（與訓練一致） |
|---|---|---|
| 今天天氣如何（工具清單裡沒有天氣工具） | 呼叫 `get_weather`——一個**根本不在工具清單裡**的函式 | 拒答，說明無法提供即時天氣資訊 |
| `moving_pallet` 被 guard 拒絕後 | 改呼叫 `get_slots` 查該站點 | 用繁體中文說明失敗原因與佔用的棧板編號 |

第一列是這類問題的典型長相：**模型沒有壞，錯的是餵給它的前綴。** 在錯的樣板下，模型憑空生出一個不存在的工具——如果後端沒有嚴格檢查函式名是否在宣告清單內，這會變成一個難以歸因的線上故障。而換回正確的樣板，同樣的提問連續五次都穩定拒答。

排除樣板本身注入範例的可能：檢查過兩份樣板，裡面都沒有硬寫任何函式名。

處理方式是跟訓練端確認到檔案層級——要到那份 `.jinja` 本身或它的 commit hash，不是「都是 gemma4 template」這種等級的答案。拿到之後用 `--chat-template` 明確指定，不要依賴預設，因為預設會隨推論引擎版本改變。

### 4. base model 用了量化版本

LoRA 的 `A`、`B` 是相對於 bf16 base 的權重分佈訓練出來的。套到 FP8 或 AWQ 量化版上，就算維度湊得起來，行為也不保證，而評測數字是在 bf16 上量的。

同理，既有的 vLLM 服務不能直接沿用：`gemma-4-26B-A4B-it-FP8-Dynamic` 與 `gemma-4-E4B-it` 是兩個不同的模型，adapter 只認後者，掛到前者會在載入階段就因為張量形狀不符而失敗。

### 5. `--max-lora-rank` 設得比 `r` 小

這一項會拋錯，不會靜默，但錯誤訊息容易被當成 adapter 壞掉。檢查 `adapter_config.json` 的 `r` 即可。

---

## 七、上線前的驗收順序

1. **掛載數對不對** — `LoRA modules attached: 132`。不對就是路徑或類別問題，後面全部不用測。
2. **adapter 有沒有改變輸出** — 開關 A/B 比對 logits 或直接比對生成結果。相同就是沒生效。
3. **prompt 渲染是否與訓練一致** — 把渲染後的字串印出來，與訓練端的樣本逐字元比對。這一步花十分鐘，能省掉後面所有「為什麼分數對不上」的追查。
4. **單題工具呼叫重現** — 拿交付評測集的原始 prompt，確認能重現報告裡的行為。
5. **工具選擇的邊界案例** — 針對交付報告中已列出的錯誤型態（該選查詢類工具卻直接選了動作類工具之類）刻意鋪設情境。
6. **guard 拒絕後的自我修正** — 離線評測只量單輪決策，多輪 recovery 完全沒測過。這是實際環境測試中最可能出問題、也最有價值的一段。

前三步任何一步不通過，後面的分數都沒有意義。

---

## 附錄：不載入模型就能檢查 adapter 相容性

safetensors 的標頭是純 JSON，可以只讀前幾 KB 就取得所有張量的名稱與形狀。這讓「adapter 對不對得上這個 base model」變成一個秒級、不需要 GPU 的檢查：

```python
import json, struct

def st_header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop('__metadata__', None)
    return hdr

base = st_header("model.safetensors")
adp  = st_header("adapter_model.safetensors")
r    = json.load(open("adapter_config.json"))["r"]

bad = []
for k, meta in adp.items():
    body = k[len("base_model.model."):]
    mod, side = body.rsplit(".lora_", 1)
    bkey = mod + ".weight"
    if bkey not in base:
        bad.append((k, f"base 沒有 {bkey}"))
        continue
    out, in_ = base[bkey]["shape"]
    want = [r, in_] if side.startswith("A") else [out, r]
    if meta["shape"] != want:
        bad.append((k, f"形狀 {meta['shape']} 應為 {want}"))

print(f"對上 {len(adp) - len(bad)}/{len(adp)}")
for k, why in bad[:10]:
    print(" ✗", k, why)
```

接手一份來源不明的 adapter 時，這是第一個該跑的檢查。

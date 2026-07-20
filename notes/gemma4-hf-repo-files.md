# HuggingFace 模型 repo 逐檔解說：以 google/gemma-4-E4B-it 為例

打開一個 HuggingFace 模型 repo，看到一排 `config.json`、`tokenizer.json`、`*.safetensors`，多數人只認得副檔名猜用途。這篇把 `google/gemma-4-E4B-it`（Gemma 4 家族、40 億「有效參數」、指令微調版）repo 裡的每個檔案拆開講：誰讀它、扮演什麼角色、改壞了會出什麼症狀。

- 系列首篇：[chat_template 是什麼：從「LLM 只吃 token 序列」講起](gemma4-chat-template-first-principles.md)。
- Repo 頁面：<https://huggingface.co/google/gemma-4-E4B-it/tree/main>（查證日期 2026-07-20）。

## 一句話結論

一個 HF 模型 repo 大致分四類檔案：**權重**（模型真正的參數）、**結構描述**（config.json，告訴推論框架怎麼把權重組成一個能跑的網路）、**輸入輸出轉換規則**（tokenizer 系列 + chat_template，決定文字 ↔ token 怎麼互轉、messages ↔ 字串怎麼序列化）、**多模態前處理**（processor_config，決定圖片/音訊怎麼變成模型看得懂的向量）。四類缺一不可；`config.json` 改壞了模型建不起來會直接報錯，但 tokenizer / chat_template 改壞了往往**不報錯，只是輸出變差**——這正是它們特別危險的原因。

## 檔案總覽

實際抓取的檔案清單（`google/gemma-4-E4B-it`，主分支，2026-07-15 最後更新）：

| 檔案 | 大小 | 類別 |
|---|---:|---|
| `.gitattributes` | 1.57 kB | Git LFS 規則 |
| `README.md` | 27.8 kB | 模型卡（model card） |
| `chat_template.jinja` | 18.6 kB | 對話序列化規則 |
| `config.json` | 5.15 kB | 模型結構描述 |
| `generation_config.json` | 208 Bytes | 預設生成參數 |
| `model.safetensors` | 16 GB | 模型權重 |
| `processor_config.json` | 1.69 kB | 多模態前處理設定 |
| `tokenizer.json` | 32.2 MB | 分詞器（含詞表） |
| `tokenizer_config.json` | 2.1 kB | 分詞器設定與特殊 token |

> 這個 repo **沒有**獨立的 `tokenizer.model`（SentencePiece 格式）、也**沒有** `model.safetensors.index.json`。原因見下方逐檔說明。也沒有獨立 `LICENSE` 檔——授權資訊寫在 `README.md` 的 YAML front matter（`license: apache-2.0`，連到 Google 自訂的 [Gemma 授權條款](https://ai.google.dev/gemma/docs/gemma_4_license)）。

## 逐檔解說

### `config.json` — 模型結構描述

**是什麼**：一份 JSON，描述要把 `model.safetensors` 裡的權重張量組裝成什麼樣的網路架構——層數、hidden size、attention head 數、每個特殊 token 的 id（`boi_token_id`／`eoi_token_id` 圖片開始/結束、`boa_token_id`／`eoa_token_id` 音訊開始/結束、`image_token_id`、`audio_token_id`）。gemma-4-E4B-it 是多模態模型，`config.json` 裡還嵌了獨立的 `audio_config`（音訊編碼子網路的完整結構參數，如 `hidden_size: 1024`、`num_hidden_layers: 12`）和文字主幹的 `text_config`。

**誰讀它**：`transformers.AutoModelForCausalLM.from_pretrained()`（或這裡對應的 `Gemma4ForConditionalGeneration`，`architectures` 欄位指名的類別）在建立模型物件、還沒載權重之前，先靠這份檔案決定要 new 出什麼形狀的網路。

**改壞了會怎樣**：多數欄位改錯會在載入階段直接報錯（張量形狀對不上、層數不合），少數欄位（如 token id）改錯不會報錯，但推論時特殊 token 的判斷全部失準——例如 `image_token_id` 錯了，模型分不清哪段輸出是「圖片佔位符」。

### `generation_config.json` — 預設生成參數

**是什麼**：`model.generate()` 沒有明確傳參數時的預設值：`temperature: 1.0`、`top_k: 64`、`top_p: 0.95`、`do_sample: true`，以及 `eos_token_id`（這裡是一個列表 `[1, 106, 50]`——多個 token 都能當作停止訊號，對應到 `<eos>`、turn 收尾、channel 收尾等不同語意的「結束」）。

**誰讀它**：`GenerationMixin.generate()`，當呼叫端沒有覆寫對應參數時採用。

**改壞了會怎樣**：`eos_token_id` 漏掉其中一個值，會讓某些正常應該停止生成的情況停不下來（例如模型輸出完 `<turn|>` 但這個 id 沒被列進 `eos_token_id`，就會繼續生成到 `max_new_tokens` 上限）。溫度/top_p 改錯則是「品質變差但不會壞」的典型例子——多輪 A/B 測試才容易抓到。

### `tokenizer.json` — 分詞器（fast tokenizer，含完整詞表）

**是什麼**：HuggingFace `tokenizers` 函式庫的「fast tokenizer」序列化格式，32.2 MB 全部塞在一個檔案裡，包含完整詞表（vocabulary）、合併規則（BPE merge rules 或對應的 tokenization 演算法設定）、normalizer/pre-tokenizer 規則。這個 repo **沒有**額外的 `tokenizer.model`（SentencePiece 二進位格式）——代表這個版本已經完全用 Rust 實作的 fast tokenizer 格式取代了舊 Gemma 系列常見的 SentencePiece 檔案，兩者是同一件事的不同封裝，repo 只保留一種就夠推論用。

**誰讀它**：`AutoTokenizer.from_pretrained()`，做文字 ↔ token id 的雙向轉換，是整條 pipeline 最基礎的一層——先分詞，`chat_template.jinja` 才有辦法在這之上做角色/結構序列化。

**改壞了會怎樣**：詞表或合併規則被改動，同一段文字會被切成不同的 token id 序列，跟模型訓練時看到的分佈完全對不上，等於直接餵亂碼；這類壞法通常會讓輸出立刻明顯崩壞（比 chat_template 錯誤更容易被發現）。

### `tokenizer_config.json` — 分詞器設定與特殊 token 命名

**是什麼**：定義一整組具名特殊 token 的字串長相（`bos_token: "<bos>"`、`eos_token: "<eos>"`、`pad_token: "<pad>"`、`mask_token: "<mask>"`），以及 gemma-4 這一版特有的結構化 token 別名——`eot_token: "<turn|>"`（end of turn）、`etc_token: "<tool_call|>"`、`etr_token: "<tool_response|>"`、`escape_token: "<|\"|>"`（自訂工具呼叫語法裡用來包字串值的引號替代符）。還有 `response_schema`：一段用正則表達式描述「怎麼從模型原始輸出反解析回結構化 `tool_calls` 陣列」的規則（`x-regex-iterator`、`x-parser: gemma4-tool-call`），這是 serving 框架的 tool-call parser 要對齊的依據。

**誰讀它**：`AutoTokenizer` 載入時的設定來源；部分新版 transformers 也會從這裡讀 `chat_template` 欄位（如果模型 repo 沒有獨立 `chat_template.jinja` 檔案的話——這個 repo 兩者都有，獨立檔案優先）。

**改壞了會怎樣**：特殊 token 字串跟 `tokenizer.json` 詞表裡的實際定義兜不起來，會出現「模型明明輸出了正確語意的分界，但 framework 端的字串比對抓不到」——這類 bug 特別隱晦，因為兩份檔案（`tokenizer.json` 的詞表 vs `tokenizer_config.json` 的具名字串）理論上該保持一致，但它們是分開維護的兩個檔案，沒有機制強制同步。

### `chat_template.jinja` — 對話序列化規則

**是什麼**：Jinja2 樣板，把 `messages` 列表（含 `role`/`content`/`tool_calls`/`reasoning` 等欄位）序列化成單一字串。詳細機制、逐段拆解見系列首篇 [chat_template 是什麼](gemma4-chat-template-first-principles.md)。這個 repo 把它獨立成一個檔案（而不是塞進 `tokenizer_config.json` 的 `chat_template` 字串欄位），是近年 transformers 生態的慣例轉向——獨立 `.jinja` 檔案比塞在 JSON 裡的跳脫字串更容易人工 review、diff、版控。

**誰讀它**：`tokenizer.apply_chat_template()`；vLLM／TGI 等 serving 框架在收到 OpenAI-compatible `/chat/completions` 請求時，用它把請求裡的 `messages` 轉成送進模型的 prompt 字串。

**改壞了會怎樣**：見系列首篇「template 錯誤或過舊會怎樣」一節——工具呼叫格式跑掉、停止 token 判斷失效、多輪角色錯亂，且**通常不報錯**，是四類檔案裡最容易被忽略、卻後果最隱蔽的一個。

### `processor_config.json` — 多模態前處理設定

**是什麼**：`Gemma4Processor` 的設定，分三段分別對應圖片（`image_processor`：resize、正規化參數、`image_seq_length: 280` 即每張圖片會被切成幾個「soft token」佔位）、音訊（`feature_extractor`：取樣率 16kHz、mel 頻譜參數）、影片（`video_processor`：取樣幀數 `num_frames: 32`）。這是文字以外三種模態各自的「前處理配方」——決定一張圖、一段音訊，要先被轉換成什麼形狀的張量，再交給模型。

**誰讀它**：`AutoProcessor.from_pretrained()`，在圖片/音訊/影片進模型前做特徵抽取與正規化。純文字推論用不到這個檔案。

**改壧了會怎樣**：多模態輸入的前處理跟訓練時對不上（例如正規化用的 `image_mean`/`image_std` 改錯），模型收到的視覺/聽覺特徵分佈跟訓練分佈不一致，圖片/音訊理解準確度大幅下降，但通常不會報錯——同樣是「悄悄變差」而非「直接壞掉」的類型。

### `model.safetensors` — 模型權重

**是什麼**：16 GB，模型的實際參數（權重張量），用 `safetensors` 格式而非舊式 `.bin`（pickle 格式有任意程式碼執行風險，`safetensors` 是純資料格式，無法夾帶可執行程式碼，是目前 HF 生態的預設安全格式）。這個 repo 只有單一檔案，代表模型沒有大到需要切分——若模型更大（例如 26B/31B 版本），通常會看到 `model-00001-of-0000N.safetensors` 這種分片，並多一份 `model.safetensors.index.json` 記錄「哪個張量放在哪個分片」，載入時先讀 index 再依需要載對應分片。

**誰讀它**：`from_pretrained()` 依 `config.json` 建好空的網路結構後，把這份檔案裡的張量一一填進對應層。

**改壞了會怎樣**：張量數值本身除非是精確控制的攻擊或訓練 bug，很難「小壞」——壞的話通常是形狀直接對不上 `config.json` 而載入失敗，或是刻意置換過的權重（供應鏈安全的關注點，這也是為什麼要用不可帶程式碼的 `safetensors` 格式）。

### `.gitattributes` — Git LFS 規則

**是什麼**：標準 Git 屬性檔，列出哪些副檔名（`*.safetensors`、`*.bin`、`*.npy` 等大型二進位格式）要走 Git LFS（Large File Storage）而不是直接進 Git 物件庫。純基礎設施檔案，跟模型行為無關。

**誰讀它**：Git／HF Hub 的儲存後端，決定上傳/下載時走一般 Git 物件流程還是 LFS 指標檔案流程。

### `README.md` — 模型卡（model card）

**是什麼**：27.8 kB，YAML front matter（`license`、`pipeline_tag`、`base_model` 等結構化中繼資料）加上人類可讀的說明——模型能力（reasoning、多模態、tool calling）、五種尺寸（E2B/E4B/12B/26B A4B/31B）、授權條款連結、技術報告連結。

**誰讀它**：人；同時 HF Hub 網站會解析 front matter 拿去做 repo 分類、搜尋標籤、授權標示。不影響推論行為，純文件性質。

## 來源（查證日期 2026-07-20）

- 檔案清單與大小：[google/gemma-4-E4B-it — Files and versions](https://huggingface.co/google/gemma-4-E4B-it/tree/main)。
- `config.json`／`generation_config.json`／`tokenizer_config.json`／`processor_config.json` 實際內容：分別抓取自 `https://huggingface.co/google/gemma-4-E4B-it/raw/main/<檔名>`。
- 授權與模型卡資訊：[README.md front matter](https://huggingface.co/google/gemma-4-E4B-it/raw/main/README.md)（`license: apache-2.0`，連到 [Gemma 授權條款](https://ai.google.dev/gemma/docs/gemma_4_license)）。
- HF repo metadata（`lastModified`、`sha`）：`https://huggingface.co/api/models/google/gemma-4-E4B-it`。

## 關聯

- [chat_template 是什麼：從「LLM 只吃 token 序列」講起](gemma4-chat-template-first-principles.md)——本篇 `chat_template.jinja` 一節的完整機制拆解在那篇。
- [2026-07 gemma-4 chat_template 變更事件](gemma4-chat-template-change-incident.md)——`chat_template.jinja` 這個檔案實際被改動、造成 serving 端症狀的真實案例。

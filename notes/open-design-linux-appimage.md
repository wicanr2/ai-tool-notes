# 把 Open Design 編成 Linux AppImage

Open Design（`nexu-io/open-design`，Electron 桌面版的開源 Claude Design 替代品）官方只發佈 macOS 與 Windows 安裝檔，沒有 Linux 版。它的原始碼其實已經內建 Linux AppImage 的打包邏輯（`tools/pack/src/linux.ts`），還提供在 `electronuserland/builder:base` 容器內建置的 `--containerized` 模式，只是這條路徑從沒在「本機已用 pnpm 裝過依賴 + 用容器建置」的組合下被驗證過，實際跑起來會連續踩到七個問題。這篇記錄完整的排錯過程、每個問題的根因與修法，以及一份可直接套用的 patch。

- 專案版本：Open Design 0.12.1（Electron 41、pnpm 10.33.2 workspace、electron-builder 26.8.1）
- 目標產物：可在任意 x86-64 Linux 直接執行的 `.AppImage`
- 建置環境：Node 24、Docker、`electronuserland/builder:base`

## 一句話結論

原始碼有 Linux 打包能力，但 containerized 路徑對「共用 host node_modules 的容器建置」和「electron-builder 對 pnpm 佈局的處理」有數個未覆蓋的缺口。補上七個修正後，能穩定產出可執行的 AppImage。核心指令：

```bash
node tools/pack/bin/tools-pack.mjs linux build --to appimage --containerized --portable --json
```

改動 `tools/pack/src/linux.ts` 後需重編這支工具（新鮮度檢查只比對 source 的雜湊）：

```bash
cd tools/pack
node ./esbuild.config.mjs
node --experimental-strip-types ../../packages/metatool/src/cli.ts write .
```

## 七個問題與修法

建置流程分四階段：**依賴安裝 → build workspace → 組裝 app 的 production install → electron-builder 打包**。外層 Node 一律回報 `docker build exited with code 1`，真正的原因每次都在它上面幾行，且卡點逐階段往後推。

| # | 階段 | 症狀 | 根因 | 修法 |
|---|------|------|------|------|
| 1 | install | `ERR_PNPM_ABORTED_REMOVE_MODULES_DIR_NO_TTY` | 容器把整個 workspace（含 host 用別版 pnpm 裝的 `node_modules`）掛進 `/project`，容器內 pnpm 想清掉重裝但無 TTY 又沒設 `CI` | docker run 加 `-e CI=true` |
| 2 | install | 抓套件 `ECONNRESET`、卡在最後幾個大 tarball | registry.npmjs.org 連線不穩、併發過高被 reset | 先在 host 跑一次 `pnpm install` 把 pnpm store 填滿，之後容器建置離線命中快取；`.npmrc` 調 `fetch-retries` / `network-concurrency` |
| 3 | build workspace | `spawn corepack ENOENT` | `buildWorkspaceArtifacts` 的巢狀 `runPnpm` 在沒有 `npm_execpath` 時 fallback 到 `corepack pnpm`，但 base image 已移除 corepack | docker run 加 `-e npm_execpath=/tmp/pnpm`，指向容器 bootstrap 的 standalone pnpm |
| 4 | production install | `ERR_PNPM_FETCH_404 @open-design/release ... No authorization header` | 打包出的 tarball 內部把 `workspace:*` 改寫成純版本 `0.12.1`，pnpm 不會用頂層 `file:` 依賴去滿足另一套件的版本區間，於是跑去 registry 抓私有套件 | 在組裝 app 的 `package.json` 加 `pnpm.overrides`，強制所有內部套件（含 transitive）解析到本地 tarball |
| 5 | electron-builder | `Node module collector process exited with code 127: pnpm: not found` | electron-builder 的 node-module collector 用 `/bin/sh -c` spawn 裸 `pnpm`，但容器只有 `/tmp/pnpm` 這個 binary、PATH 上沒有叫 `pnpm` 的指令 | setupPnpm 在 PNPM_HOME 建 `pnpm` symlink |
| 6 | 執行期 | `Cannot find module 'setimmediate'`（jszip 需要它） | electron-builder 26 把 pnpm 的 `.pnpm` 佈局攤平進 AppImage 時，會漏掉 hoisted / isolated 佈局下的 transitive 依賴 | 組裝 app 的 production install 改用 **npm**（產生純扁平、無 `.pnpm` 的 node_modules，electron-builder 直接照抄不經過會漏檔的 collector）；容器內用 `pnpm add -g npm` 補上 npm |
| 7 | 執行期 | `EACCES: permission denied, mkdir '/tools-pack'` | 非 portable build 把容器路徑 `/tools-pack` 當成 runtime root 烤進 config，在 host 上跑沒權限建 | build 時加 `--portable`，不把建置時的絕對路徑寫進 config |

### 幾個關鍵細節

**第 2 與第 3 的順序很重要。** 先讓 host 的 `pnpm install` 把所有套件下載進共用的 pnpm store（`/project/.pnpm-store`），之後容器建置就能離線命中快取，避開 registry 的 `ECONNRESET`。

**不要中途 kill 容器。** 容器的 install 會 purge 並重建掛在 host 上的 `node_modules`（bind-mount）。若在重建到一半時砍掉容器，會留下殘缺的 `node_modules`（例如缺 `zod`），連 host 端的 `tools-pack` CLI 都無法載入。修復方式是在 host 重跑一次 `pnpm install`。

**第 6 的排錯關鍵在「分辨磁碟佈局 vs. 打包產物」。** 磁碟上組裝 app 的 `node_modules` 其實是對的（`.pnpm/jszip@x/node_modules/` 裡 `setimmediate` 明明在），但解壓 AppImage 一看就發現 electron-builder 把它漏掉了。這條線索直接排除了「pnpm flag 沒調對」的方向，把矛頭指向 electron-builder 對 pnpm 佈局的處理，最終決定改用 npm 產生扁平佈局。

## 建置與驗證流程

```bash
# 0. （首次）先用 host 填滿 pnpm store，避免容器抓套件時 ECONNRESET
cd open-design
pnpm install --frozen-lockfile

# 1. 套用 patch 後重編 tools-pack
cd tools/pack
node ./esbuild.config.mjs
node --experimental-strip-types ../../packages/metatool/src/cli.ts write .
cd ../..

# 2. 容器內建置 AppImage（portable）
node tools/pack/bin/tools-pack.mjs linux build --to appimage --containerized --portable --json

# 3. 安裝（建立 .desktop + icon）並驗證啟動
node tools/pack/bin/tools-pack.mjs linux install --json
~/.local/bin/Open-Design.default.AppImage        # 或從應用選單開啟
```

產物位置：`.tmp/tools-pack/out/linux/namespaces/default/builder/Open Design-default.AppImage`（約 407 MiB，可複製到任何 x86-64 Linux 執行）。

### 驗證重點

`tools-pack linux start` 會等 `desktop-root.json` 就緒標記，直接執行 AppImage 時該標記寫在別處，會出現 60 秒 timeout 的誤判——這不代表 app 有問題。真正的驗證是直接執行 AppImage 並確認主視窗載入：log 出現 `main window did-start-loading url: 'od://app/onboarding'` → `od://app/` → `od://app/projects`，且視窗管理器列得到 `Open Design` 視窗、畫面正常渲染首頁即為成功。log 裡的 `unsupported-platform`（自動更新不支援 Linux）與 dbus/systemd 警告在無桌面服務的環境屬正常噪音。

## 在目標機器執行 AppImage

把 `.AppImage` 複製到任何 x86-64 Linux 直接執行即可（`chmod +x` 後 `./Open-Design-*.AppImage`）。兩個容易被誤判成「壞掉」的正常現象：

- **啟動時終端會噴一行紅字**
  `Failed to call method: org.freedesktop.systemd1.Manager.StartTransientUnit: ... Invalid unit name or type.`
  這是 Electron/Chromium 想透過 systemd 註冊 process scope，在沒有完整 user systemd session（或從終端直接跑）的環境會失敗，**不影響運作**。同類噪音還有 `zygote`、`GetTerminationStatus`、`unsupported-platform`。
- **首次啟動較慢**
  `--appimage-extract-and-run` 會先把約 200MB 解壓到 `/tmp`，噴完訊息後**還要再等 10–30 秒**視窗才出現，別急著關。（用 `--appimage-extract-and-run` 是因為 FUSE 掛載的 SquashFS 對首次啟動的 daemon 太慢，容易超過 sidecar 的啟動 timeout。）

為了少掉這些困惑，附一支啟動器 [`scripts/run-open-design.sh`](../scripts/run-open-design.sh)，過濾掉已知噪音並提示等待：

```bash
./run-open-design.sh            # 前景，看得到過濾後的 log
./run-open-design.sh --quiet    # 背景靜音啟動
```

核心邏輯就是把噪音用 `grep -vE` 濾掉、其餘照常顯示：

```bash
NOISE='StartTransientUnit|org.freedesktop.systemd1|dbus/object_proxy|zygote|GetTerminationStatus|unsupported-platform'
exec stdbuf -oL -eL "$APP" --appimage-extract-and-run 2>&1 | grep --line-buffered -avE "$NOISE"
```

## 修正 patch

以下 patch 只動一個檔案 `tools/pack/src/linux.ts`，涵蓋問題 1、3、4、5、6 的修正（問題 2 用 `.npmrc` + 建置順序處理，問題 7 用 `--portable` 命令列參數處理）。完整檔另見 [`patches/open-design-linux-appimage.patch`](../patches/open-design-linux-appimage.patch)。

```diff
diff --git a/tools/pack/src/linux.ts b/tools/pack/src/linux.ts
--- a/tools/pack/src/linux.ts
+++ b/tools/pack/src/linux.ts
@@ buildDockerArgs: setupPnpm ——
     `chmod +x ${CONTAINER_PNPM_PATH} && ` +
+    // electron-builder 的 node-module collector 會用 /bin/sh -c spawn 裸 `pnpm`，
+    // 容器只有 /tmp/pnpm binary、PATH 上沒有 pnpm 指令 → 127。symlink 到 PNPM_HOME。
+    `ln -sf ${CONTAINER_PNPM_PATH} ${CONTAINER_PNPM_HOME}/pnpm && ` +
     `PNPM_HOME=... ${CONTAINER_PNPM_PATH} env use --global ${CONTAINER_NODE_VERSION} && ` +
     `export PNPM_HOME=... PATH=... && ` +
-    `command -v node >/dev/null`;
+    `command -v node >/dev/null && ` +
+    // 組裝 app 的 production install 要用 npm（見 resolveProductionInstallCommand）。
+    // base image 無 npm、pnpm env use 裝的 node 也不含 npm，用 pnpm 全域裝一份。
+    `${CONTAINER_PNPM_PATH} add -g npm && ` +
+    `command -v npm >/dev/null`;

@@ buildDockerArgs: dockerArgs 環境變數 ——
     "-e", "HOME=/home/builder",
+    // host 遺留的 node_modules 讓容器 pnpm 想 purge 重裝，無 TTY 會中止。
+    "-e", "CI=true",
     "-e", "ELECTRON_CACHE=...",
     "-e", "ELECTRON_BUILDER_CACHE=...",
     "-e", `${PRODUCTION_INSTALL_PNPM_BIN_ENV}=${CONTAINER_PNPM_PATH}`,
+    // 巢狀 runPnpm 在無 npm_execpath 時 fallback 到 corepack（image 已移除）→ ENOENT。
+    "-e", `npm_execpath=${CONTAINER_PNPM_PATH}`,

@@ resolveProductionInstallCommand ——
-  const pnpmBin = env[PRODUCTION_INSTALL_PNPM_BIN_ENV];
-  if (pnpmBin != null && pnpmBin.length > 0) {
-    return { command: pnpmBin, args: ["install","--prod","--no-lockfile","--config.node-linker=hoisted"] };
-  }
-  return { command: "npm", args: ["install","--omit=dev","--no-package-lock"] };
+  // 一律用 npm：electron-builder 26 攤平 pnpm .pnpm 佈局時會漏掉 transitive
+  // 依賴（如 jszip -> setimmediate），npm 的扁平 node_modules 才會被完整打包。
+  return { command: "npm", args: ["install","--omit=dev","--no-package-lock"] };

@@ writeAssembledApp: 組裝 app 的 package.json ——
   const dependencies = { ...file: tarballs... };
+  // tarball 內部把 workspace:* 改寫成純版本，pnpm 不會用頂層 file: dep 滿足版本區間
+  // → 跑去 registry 抓私有套件 404。overrides 強制所有內部套件解析到本地 tarball。
+  const overrides = { ...dependencies };
   const packageJson = {
     ..., dependencies,
+    pnpm: { overrides },
   };
```

搭配的 `.npmrc`（放 workspace 根，緩解問題 2；`.npmrc` 本身未進版控）：

```ini
confirm-modules-purge=false
fetch-retries=6
fetch-retry-mintimeout=10000
fetch-retry-maxtimeout=120000
fetch-timeout=180000
network-concurrency=4
```

## 可移植性備註

這些修正都繞著同一個前提：**在容器裡用 standalone pnpm bootstrap、且共用 host 的 workspace bind-mount**。若改成完全乾淨的 CI checkout（無 host `node_modules`），問題 1 與 2 不會出現；但 3、4、5、6 是 `electronuserland/builder:base`（無 npm/corepack）與 electron-builder 對 pnpm 佈局處理的本質問題，仍需要對應修正。問題 7 的 `--portable` 則與環境無關，任何要拿到別台機器執行的 AppImage 都該加。

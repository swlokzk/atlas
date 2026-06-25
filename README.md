# The Workspaces - Atlas

## 概述 | Overview

本 repository 是一個經治理的專案生態系索引，目前以 `_research/` 與 `_development/` 兩個主目錄整理專案，並附有一致的 catalog metadata 與治理文件。  
This repository is a curated project ecosystem index currently organized around the `_research/` and `_development/` top-level directories, with consistent catalog metadata and governance documentation.

---

## 專案結構 | Repository Structure

```text
atlas/
├── _research/             # 研究型專案 | Research projects
├── _development/          # 開發型專案 | Development projects
├── catalog/               # 專案索引與 metadata | Project index and metadata
├── docs/                  # 治理文件 | Governance documentation
├── .github/               # GitHub 設定與工作流程 | GitHub configuration and workflows
└── README.md              # 本文件 | This file
```

---

## 專案分類 | Project Categories

- **_research/**：研究型專案，包含資料科學、機器學習、預測模型等內容。  
  **_research/**: Research-oriented projects such as data science, machine learning, and forecasting models.
- **_development/**：開發型專案，包含前端應用、交易系統與應用程式原型。  
  **_development/**: Development-oriented projects such as frontend applications, trading systems, and app prototypes.

---

## 專案索引 | Project Catalog

完整的專案列表與 metadata 請參閱：  
For complete project listings and metadata, see:

- [機器可讀 | Machine-readable](catalog/projects.yml)
- [人類可讀 | Human-readable](catalog/index.md)

### _research/

研究型專案，包含資料科學、機器學習、預測模型與學術實驗性質的內容。  
Research projects including data science, machine learning, forecasting models, and academic or experimental work.

- [cnogb-abnormal-intervention](_research/cnogb-abnormal-intervention) — 使用 Transformer 模型進行債券預測與異常偵測。Bond forecasting and anomaly detection using Transformer models.
- [machine-learning-applied-fmri](_research/machine-learning-applied-fmri) — 功能性 MRI 資料的機器學習應用研究。Machine learning applications for fMRI analysis.
- [e-learning-fa](_research/e-learning-fa) — 使用 Python 腳本進行電子學習資料分析。E-learning data analysis using Python tooling.

### _development/

開發型專案，包含前端應用、交易訊號偵測、神經網路與應用程式原型。  
Development projects including frontend applications, trading signal detection, neural networks, and application prototypes.

- [amc-amr-gff-nn](_development/amc-amr-gff-nn) — AMC AMR Gated Fusion Former 自動調變識別神經網路。AMC AMR Gated Fusion Former neural network for automatic modulation recognition.
- [animated-gradient-text-starter](_development/animated-gradient-text-starter) — 動態漸層文字動畫入門專案（Next.js + TypeScript + Tailwind CSS）。Animated gradient text starter project built with Next.js, TypeScript, and Tailwind CSS.
- [r3f-portfolio](_development/r3f-portfolio) — React Three Fiber 3D 互動式作品集網站。React Three Fiber 3D portfolio website.
- [binance-multi-assets-singal-agent](_development/binance-multi-assets-singal-agent) — Binance 多資產交易訊號代理程式原型。Binance multi-asset trading signal agent prototype.

---

## 快速開始 | Quickstart

以下示例展示如何只取出單一子專案、安裝依賴並執行一個最小實驗（以 `_research/cnogb-abnormal-intervention` 為例）。

1. 只 clone 並 sparse-checkout 單一資料夾：

```bash
git clone --filter=blob:none --sparse https://github.com/sinwulok/atlas.git
cd atlas
git sparse-checkout set _research/cnogb-abnormal-intervention
```

2. 建議建立虛擬環境並安裝依賴（以 pip 為例）：

```bash
python -m venv .venv
source .venv/bin/activate    # 或 Windows: .venv\Scripts\activate
pip install -r _research/cnogb-abnormal-intervention/requirements.txt
```

3. 執行該子專案的訓練 runner（示例）：

```bash
python _research/cnogb-abnormal-intervention/experiments/run_training.py --config _research/cnogb-abnormal-intervention/configs/training.yaml
```

說明：若該子專案尚未提供 `requirements.txt`，請參考其 `README.md` 或在根目錄 `requirements.txt` 中搜尋相依項目。

---

## 維護者與安全回報 | Maintainers & Security

- **Maintainer:** sinwulok (GitHub) — 若需聯絡或報告安全問題，請使用 GitHub issue 並標上 `security` 標籤。
- **安全回報:** 如發現安全漏洞，請先開 private GitHub issue 或直接 email 至 maintainer（見個別專案 LICENSE/CONTACT）。

---

## 授權摘要 | License

各子專案可能有各自的授權條款；若未在子專案內另行說明，預設請參考本 repository 根目錄的 `LICENSE`。

---

## 進一步的貢獻資訊 | Contributing

請參考根目錄的 `CONTRIBUTING.md` 與 `docs/naming-conventions.md`，在提交 PR 前請先執行格式檢查並遵循命名規範。

---

## 治理文件 | Governance

治理文件位於 `docs/` 目錄：  
Repository governance documentation is located in the `docs/` directory:

- [架構說明 | Architecture](docs/architecture.md) — Repository 架構與目錄用途。Repository architecture and directory purpose.
- [遷移計畫 | Migration Plan](docs/migration-plan.md) — 詳細的重構歷史記錄。Detailed refactoring history.
- [命名規範 | Naming Conventions](docs/naming-conventions.md) — 專案與檔案命名標準。Project and file naming standards.
- [專案生命週期 | Project Lifecycle](docs/project-lifecycle.md) — 專案狀態管理規範。Project status management.
- [治理政策 | Governance](docs/governance.md) — Repository 治理原則。Repository governance principles.
- [分支策略 | Branch Policy](docs/branch-policy.md) — 分支管理與開發工作流程。Branch management and development workflow.

---

## 如何尋找專案 | How to Find a Project

可依以下方式瀏覽專案：  
Browse projects by:

1. **類別 | Category**：前往 `_research/` 或 `_development/` 目錄。Navigate to `_research/` or `_development/` directories.
2. **標籤 | Tags**：查閱 [catalog/tags.yml](catalog/tags.yml) 以進行主題篩選。Check for topic-based filtering.
3. **生命週期 | Lifecycle**：查閱 [catalog/statuses.yml](catalog/statuses.yml) 以了解專案成熟度。See for project maturity levels.

---

## 如何下載單一專案 | How to Download a Single Project

若只需要此 monorepo 中的特定子資料夾，可使用以下方式避免下載整個 repository。  
If you only need a specific subfolder from this monorepo, here are convenient methods to avoid downloading the entire repository.

### 方法一：Git sparse-checkout（推薦）| Method 1: Git sparse-checkout (Recommended)

適合需要 git 功能但只需要特定資料夾的使用者。  
Best for users who need git functionality but only want specific folders.

```bash
# 以最小 blob 模式 clone 並啟用 sparse 模式
# Clone with minimal blobs and enable sparse mode
git clone --filter=blob:none --sparse https://github.com/sinwulok/atlas.git
cd atlas

# 只取得所需資料夾（範例）
# Fetch only the folder you need (example)
git sparse-checkout set _research/cnogb-abnormal-intervention

# 可選：切換分支
# Optionally checkout another branch
git checkout <branch>
```

### 方法二：SVN export（快速、無 git）| Method 2: SVN export (Quick, no git)

適用於公開的 GitHub repo，回傳不含 .git 的純資料夾。  
Suitable for public GitHub repos, returns a plain folder without .git.

```bash
svn export https://github.com/sinwulok/atlas/trunk/_research/cnogb-abnormal-intervention
```

### 方法三：下載 ZIP 並解壓縮 | Method 3: Download ZIP and extract specific folder

下載整個 repository 的快照（較大），再解壓縮所需資料夾。  
Download a snapshot of the entire repository (larger), then extract only the needed folder.

```bash
curl -L -o repo.zip https://github.com/sinwulok/atlas/archive/refs/heads/main.zip
unzip repo.zip "atlas-main/_research/cnogb-abnormal-intervention/*" -d extracted
```

### 方法四：GitHub CLI | Method 4: GitHub CLI

```bash
gh repo clone sinwulok/atlas -- --filter=blob:none --sparse
cd atlas
git sparse-checkout set _research/cnogb-abnormal-intervention
```

---

## 貢獻 | Contributing

歡迎貢獻。提交變更時請遵循 repository 的治理指引與命名規範。  
Contributions are welcome. Please follow the repository governance guidelines and naming conventions when submitting changes.

---

## 授權 | License

各專案的授權資訊請參閱個別專案文件。  
See individual projects for specific licensing information.
<!-- START_PROJECT_OVERVIEW -->

## 專案總覽 | Project Overview

| Path | 類型 | 簡介 |
| --- | --- | --- |
| `.github` |  |  |
| `.vscode` |  |  |
| `_development` |  |  |
| `_research` |  |  |
| `catalog` |  |  |
| `devnotes` |  |  |
| `docs` |  |  |

## Repository Structure

```text
.
├── .github/
├── .vscode/
├── _development/
├── _research/
├── catalog/
├── devnotes/
└── docs/
```

<!-- END_PROJECT_OVERVIEW -->

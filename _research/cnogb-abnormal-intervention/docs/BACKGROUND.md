# 中國債券市場異常檢測與跨域遷移學習

> **CN Bond Anomaly Detection & Cross-Domain Transfer**
>
> 利用 SOTA 時序異常檢測模型，對中國政策性金融債（國開債、農發債、進出口債）進行分佈斷裂檢測，並探索向港股銀行板塊的遷移學習路徑。

---

## 目錄

- [項目背景與動機](#項目背景與動機)
- [研究設計總覽](#研究設計總覽)
- [分層域架構](#分層域架構)
- [SOTA 模型選擇（帶優先級）](#sota-模型選擇帶優先級)
- [特徵工程](#特徵工程)
- [港股遷移路徑](#港股遷移路徑)
- [評估框架](#評估框架)
- [項目結構](#項目結構)
- [數據來源](#數據來源)
- [快速開始](#快速開始)
- [LLM 協作 Prompt](#llm-協作-prompt)
- [參考文獻](#參考文獻)
- [License](#license)

---

## 項目背景與動機

本項目聚焦中國債券（尤其是三大政策性金融債）的**時序異常檢測**問題。核心動機如下：

1. **分佈斷裂檢測**：觀察中國政金債市場是否存在結構性行為變化，利用深度學習模型自動捕捉難以用傳統統計方法發現的模式變遷。
2. **跨標的一致性驗證**：異常行為是否在國開債、農發債、進出口債三者之間具有同步性，從而推斷系統性因素 vs 個別事件。
3. **跨市場遷移學習**：探索模型從中國債券市場向 A 股銀行板塊、港股銀行股的泛化能力，為更廣泛的金融異常檢測奠定方法論基礎。

### 設計哲學

本實驗設計整合了三方建議的精華：

- **Gemini 方案**的嚴謹「屍檢」思路（先證偽模型、再證偽數據）與 Cross-era Discriminator 概念
- **Copilot GPT 方案**的全面模型路線圖框架
- **改進意見**中的分層域設計、ablation 方案、微觀結構特徵、以及無標註評估策略

**核心原則：分層而非混合，逐層 ablate，每一個改動的效果必須可隔離。**

---

## 研究設計總覽

```
Phase 0: 數據收集與預處理
    ├── 6 份債券/利率數據（政金債 + SHIBOR/DR007）
    ├── A 股銀行板塊（AKShare）
    └── 港股/美債（yfinance）

Phase 1: 單域異常檢測（L0）
    ├── 同標的歷史期 vs 觀察期
    ├── 第一梯隊模型 baseline
    └── 合成異常注入驗證

Phase 2: 跨標的一致性驗證（L1）
    ├── 三大政金債交叉驗證
    ├── Leave-one-out domain 實驗
    └── Cross-era Discriminator

Phase 3: 跨資產類別對照（L2）
    ├── A 股銀行板塊對照實驗
    ├── Granger causality 動態分析
    └── 第二梯隊模型驗證

Phase 4: 跨市場遷移（L3）
    ├── 港股/美債遷移實驗
    ├── 三步遷移路徑
    └── 第三梯隊模型評估
```

---

## 分層域架構

不把所有市場混在一起訓練，而是建立**分層域架構**，每一層獨立實驗，逐層加入以 ablate 每一層數據的貢獻。

| 層級 | 域 | 數據來源 | 用途 |
|:----:|:----|:---------|:-----|
| **L0** | 同標的歷史期 vs 觀察期 | 6 份政金債/利率數據 | **核心**：檢測同一標的的分佈斷裂 |
| **L1** | 同市場不同標的（3 大政金債交叉） | 國開債、農發債、進出口債 | **驗證**：異常是否具有跨標的一致性 |
| **L2** | A 股銀行板塊 | AKShare | **對照**：同國不同資產類別的關聯 |
| **L3** | 美債 / 港股 | yfinance (TLT, 0005.HK, 0939.HK 等) | **遷移**：跨市場泛化能力 |

### Ablation 設計

每一層的加入都要回答一個具體問題：

- **L0 only**：模型能否檢測到單一標的的行為變化？
- **L0 + L1**：加入跨標的信息是否提升了異常檢測的精度和穩定性？
- **L0 + L1 + L2**：A 股銀行板塊提供了多少額外的解釋力？
- **L0 + L1 + L2 + L3**：跨市場數據是改善還是損害了性能？

---

## SOTA 模型選擇（帶優先級）

根據 2023–2026 最新進展，按三個梯隊排列，每個梯隊必須按順序完成。

### 🥇 第一梯隊：必做 Baseline

| 模型 | 核心機制 | 選擇理由 |
|:-----|:---------|:---------|
| **Anomaly Transformer** | Association Discrepancy | 專為時序異常檢測設計，是目前最被廣泛認可的異常檢測 SOTA 之一 |
| **TranAD** | 對抗訓練 + Transformer | 對分佈漂移特別敏感，適合檢測結構性變化 |
| **PatchTST** | Patch 機制 + Channel Independence | 2023–2024 最受認可的時序 Transformer variant，patch 機制對長序列更有效 |

### 🥈 第二梯隊：驗證跨域能力

| 模型 | 核心機制 | 選擇理由 |
|:-----|:---------|:---------|
| **DADA** (ICLR 2025) | 自適應瓶頸 + 雙對抗解碼器 | 專為跨域零樣本異常檢測設計 |
| **TS2Vec** | 自監督對比學習 | 可學到 domain-invariant 的時序表徵 |
| **TimesNet** | 1D→2D 轉換 | 能捕捉多週期模式，適合利率的多尺度週期性 |

### 🥉 第三梯隊：港股遷移專用

| 模型 | 核心機制 | 選擇理由 |
|:-----|:---------|:---------|
| **DANN** + Transformer backbone | Domain Adversarial Training | 經典跨域遷移方法，結合 Transformer 特徵提取 |
| **Adapter-based Transfer** | 凍結 encoder + 輕量 adapter | 資源高效，適合目標域數據稀缺的場景 |

---

## 特徵工程

針對債券/利率市場的特性，特徵設計分為三類。

### 微觀結構特徵（最關鍵 — 模型輸入）

| 特徵 | 計算方式 | 金融含義 |
|:-----|:---------|:---------|
| Bid-Ask Spread | `ask - bid` | 流動性的直接度量 |
| Spread 波動率 | `rolling_std(spread)` | 流動性的穩定程度 |
| 成交量自相關係數 | `autocorr(volume, lag=k)` | 量能結構變化 |
| 價格衝擊係數 | `ΔPrice / Volume` | 市場深度 |
| Kyle's Lambda | 回歸估計 `ΔPrice ~ ΔVolume` | 信息不對稱程度 |
| 買賣不平衡指標 | `(buy_vol - sell_vol) / total_vol` | Order Imbalance |
| 已實現波動率 | `sqrt(Σ(r_i²))` | 日內真實波動 |
| 收益率曲線斜率 | `yield_long - yield_short` | 期限結構信號 |

### 宏觀關聯特徵（模型輸入 — 跨市場信號）

| 特徵 | 計算方式 | 金融含義 |
|:-----|:---------|:---------|
| 與 SHIBOR/DR007 的動態相關 | `rolling_corr(bond, rate, window)` | 貨幣市場聯動 |
| 政金債利差結構 | `yield_CDB - yield_ADBC - yield_EximBank` | 跨發行人信用差異 |
| 與 A 股銀行板塊的 Granger causality | 逐窗口 Granger test p-value | 跨市場領先-滯後關係動態 |
| 信用利差 | `政金債 yield - 國債 yield` | 風險溢價 |

### 統計檢驗特徵（異常確認 — 非模型輸入）

| 方法 | 用途 | 說明 |
|:-----|:-----|:-----|
| **Kolmogorov-Smirnov 檢定** | 逐窗口分佈比較 | 判定兩個時間段的分佈是否統計顯著不同 |
| **CUSUM 變點檢測** | 定位結構變化點 | 確認模型檢測到的異常是否對應真實的變點 |
| **Hurst 指數** | 長記憶性斷裂 | 檢測時序長記憶特性的突然改變 |
| **Augmented Dickey-Fuller** | 平穩性檢驗 | 確認異常期是否打破了原有的平穩性 |

---

## 港股遷移路徑

港股（T+0、無漲跌停、不同交易時段）與 A 股 / 債券市場存在制度性差異，因此採用**三步漸進遷移**策略。

### Step 1: 特徵空間對齊

- **不使用原始價格**，統一使用**標準化的微觀結構特徵**（上述特徵表）
- 港股標的選擇：HK 上市大型銀行（匯豐 `0005.HK`、建行 `0939.HK`、工行 `1398.HK`、中行 `3988.HK`）
- 計算與債券市場**相同定義**的微觀結構指標
- 對 T+0 vs T+1 差異做顯式處理（如日頻對齊、排除非重疊交易時段）

### Step 2: 分佈差異量化

- 使用 **MMD (Maximum Mean Discrepancy)** 量化 A 股 / 港股特徵空間的距離
- 如果 MMD 超過閾值，**不強行 joint train**，而是：
  - 使用 **CORAL (Correlation Alignment)** 先對齊二階統計量
  - 或使用 **Optimal Transport** 做分佈映射
- 記錄對齊前後的 MMD 值和 A-distance，作為遷移可行性的客觀指標

### Step 3: Progressive Transfer

| 策略 | 方法 | 預期場景 |
|:-----|:-----|:---------|
| **Zero-shot** | A 股模型直接在港股上推理 | 作為 lower bound baseline |
| **Few-shot** | 凍結 encoder，用少量港股數據微調 adapter + anomaly head | 港股數據稀缺時 |
| **Full fine-tune** | 用 A 股預訓練權重初始化，港股數據全量微調 | 港股數據充足時 |

三種策略的性能對比，配合 MMD / A-distance 的分析，可以量化**域間距離與遷移效果的關係**。

---

## 評估框架

### 核心挑戰：無標註數據

本項目**沒有人工標註的異常標籤**，這是最大的挑戰。因此採用以下多管齊下的評估策略。

### 異常檢測評估

| 方法 | 描述 | 度量 |
|:-----|:-----|:-----|
| **Synthetic Anomaly Injection** | 在歷史期人工注入已知類型的異常（spike, level shift, trend change），測量模型召回率 | Recall@FPR, F1 |
| **Leave-one-out Domain** | 每次拿掉一個標的，用其餘標的訓練，測試被拿掉標的的異常檢測能力 | AUROC, AUPRC |
| **Expert Validation** | 對 top-k 異常時間段做人工判讀 | Precision@k |
| **Cross-era Discriminator** | 訓練二分類器區分歷史期/觀察期，AUC 高表示分佈確實不同 | AUC + 特徵重要性分析 |

### 跨域評估

| 度量 | 描述 |
|:-----|:-----|
| **Per-domain AUROC Breakdown** | 不能只看平均 AUROC，必須報告每個域的獨立表現 |
| **Domain Gap Metric** | MMD / A-distance 與性能的 Spearman 相關性 |
| **Calibration Curve** | 模型說 90% 異常的時候，真的有 90% 嗎？ECE (Expected Calibration Error) |
| **Transfer Ratio** | `Performance_target / Performance_source`，量化遷移效率 |

### 統計顯著性

- 所有實驗重複 5 次（不同隨機種子），報告 mean ± std
- 模型間對比使用 **Wilcoxon signed-rank test** (paired, non-parametric)
- 異常檢測閾值使用 **validation set 的 percentile method** 而非固定閾值

---

## 項目結構

```
cn-ogb-anomaly-intervention/
├── README.md                   # 本文件
├── configs/                    # 實驗配置（YAML）
│   ├── base.yaml
│   ├── abnormal_transformer.yaml
│   ├── tranad.yaml
│   └── patchtst.yaml
├── data/
│   ├── raw/                    # 原始數據
│   ├── processed/              # 預處理後數據
│   └── external/               # yfinance / AKShare 下載
├── src/
│   ├── data/                   # 數據加載與預處理
│   │   ├── loader.py
│   │   ├── features.py         # 微觀結構特徵計算
│   │   └── preprocessing.py
│   ├── models/                 # 模型實現
│   │   ├── abnormal_transformer/
│   │   ├── tranad/
│   │   ├── patchtst/
│   │   ├── dada/
│   │   ├── ts2vec/
│   │   ├── timesnet/
│   │   └── dann/
│   ├── evaluation/             # 評估模組
│   │   ├── metrics.py          # AUROC, AUPRC, F1 等
│   │   ├── synthetic.py        # 合成異常注入
│   │   ├── statistical.py      # KS test, CUSUM, Hurst
│   │   └── calibration.py      # 校準曲線
│   ├── transfer/               # 遷移學習模組
│   │   ├── alignment.py        # MMD, CORAL, OT
│   │   ├── adapter.py          # Adapter-based transfer
│   │   └── progressive.py      # 三步遷移流程
│   └── utils/
│       ├── config.py
│       ├── logging.py
│       └── visualization.py
├── notebooks/                  # 探索性分析 Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_baseline_experiments.ipynb
│   └── 04_transfer_experiments.ipynb
├── experiments/                # 實驗腳本
│   ├── run_l0.py               # L0 層實驗
│   ├── run_l1.py               # L1 層實驗
│   ├── run_l2.py               # L2 層實驗
│   ├── run_l3.py               # L3 層實驗
│   └── run_ablation.py         # Ablation study
├── results/                    # 實驗結果（gitignored）
├── scripts/                    # 輔助腳本
│   ├── download_data.py        # 數據下載
│   └── setup_env.sh            # 環境設置
├── tests/                      # 單元測試
├── requirements.txt
└── pyproject.toml
```

---

## 數據來源

| 數據 | 來源 | 頻率 | 說明 |
|:-----|:-----|:-----|:-----|
| 國開債（CDB）行情 | 項目原始數據 | 日頻 | 3 大政金債之一 |
| 農發債（ADBC）行情 | 項目原始數據 | 日頻 | 3 大政金債之一 |
| 進出口債（Exim）行情 | 項目原始數據 | 日頻 | 3 大政金債之一 |
| SHIBOR / DR007 | 項目原始數據 / Wind / CEIC | 日頻 | 貨幣市場利率基準 |
| A 股銀行板塊 | [AKShare](https://github.com/akfamily/akshare) | 日頻 | 同國不同資產類別對照 |
| 港股銀行股 | [yfinance](https://github.com/ranaroussi/yfinance) | 日頻 | 0005.HK, 0939.HK 等 |
| 美國國債 ETF | [yfinance](https://github.com/ranaroussi/yfinance) | 日頻 | TLT 等 |

---

## LLM 協作 Prompt

以下是供各類 LLM 協作時使用的統一 Prompt，確保所有 AI 助手都基於相同的上下文工作。

```
你正在協助一個金融時序異常檢測研究項目。

## 項目核心
- 標的：中國三大政策性金融債（國開債、農發債、進出口債）+ SHIBOR/DR007
- 目標：檢測分佈斷裂（distributional shift），不是預測未來價格
- 遷移方向：A 股銀行板塊 → 港股銀行股

## 關鍵約束
1. 分層域設計（L0→L3），不混合訓練
2. 每一層獨立實驗，逐層 ablation
3. 第一梯隊模型必做：Anomaly Transformer, TranAD, PatchTST
4. 特徵工程以微觀結構特徵為核心（bid-ask spread, Kyle's lambda, price impact 等）
5. 港股遷移用三步法：特徵對齊 → 分佈量化 → Progressive Transfer
6. 無標註數據，評估用：合成異常注入 + Leave-one-out + Expert Validation
7. 所有統計對比用 Wilcoxon signed-rank test，不用 t-test

## 代碼風格
- Python 3.10+
- Type hints 必須
- 配置用 YAML，不硬編碼
- 遵循 Google Python Style Guide
- 模型代碼與實驗腳本分離

## 你可以參考的文件
- README.md — 完整的實驗設計和評估框架
- configs/ — 實驗配置
- src/ — 核心代碼

請基於以上上下文回答問題或生成代碼。
```

---

## 參考文獻

### 異常檢測模型

1. Xu, J., Wu, H., Wang, J., & Long, M. (2022). **Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy.** *ICLR 2022.*
2. Tuli, S., Casale, G., & Jennings, N. R. (2022). **TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data.** *VLDB 2022.*
3. Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). **A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.** *ICLR 2023.* (PatchTST)
4. Wu, H., Hu, T., Liu, Y., Zhou, H., Wang, J., & Long, M. (2023). **TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis.** *ICLR 2023.*

### 跨域學習

5. He, A., et al. (2025). **DADA: Dual Adversarial Decoders with Adaptive Bottleneck for Zero-Shot Anomaly Detection.** *ICLR 2025.*
6. Yue, Z., et al. (2022). **TS2Vec: Towards Universal Representation of Time Series.** *AAAI 2022.*
7. Ganin, Y., et al. (2016). **Domain-Adversarial Training of Neural Networks.** *JMLR 2016.* (DANN)

### 統計檢驗

8. Kyle, A. S. (1985). **Continuous Auctions and Insider Trading.** *Econometrica.*
9. Page, E. S. (1954). **Continuous Inspection Schemes.** *Biometrika.* (CUSUM)
10. Hurst, H. E. (1951). **Long-term Storage Capacity of Reservoirs.** *Trans. Am. Soc. Civil Engineers.*

---

<p align="center">
  <i>本文件整合了 Gemini、Copilot GPT 及人工審閱的三方建議，作為項目所有 LLM 協作的基礎文件。</i>
</p>

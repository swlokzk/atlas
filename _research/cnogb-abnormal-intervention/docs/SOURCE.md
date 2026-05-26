```
Source: ./BACKGROUND.md
ImportedAt: 2026-03-12T13:54:33Z
ImportedBy: 彪🦊先生
Notes: imported via subtree
```

# 中國債券市場異常檢測與跨域遷移學習

> **CN Bond Anomaly Detection & Cross-Domain Transfer**
>
> 利用 SOTA 時序異常檢測模型，對中國政策性金融債（國開債、農發債、進出口債）進行分佈斷裂檢測，並探索向港股銀行板塊的遷移學習路徑。

## 項目背景與動機

本項目聚焦中國債券（尤其是三大政策性金融債）的**時序異常檢測**問題。核心動機如下：

1. **分佈斷裂檢測**：觀察中國政金債市場是否存在結構性行為變化，利用深度學習模型自動捕捉難以用傳統統計方法發現的模式變遷。
2. **跨標的一致性驗證**：異常行為是否在國開債、農發債、進出口債三者之間具有同步性，從而推斷系統性因素 vs 個別事件。
3. **跨市場遷移學習**：探索模型從中國債券市場向 A 股銀行板塊、港股銀行股的泛化能力，為更廣泛的金融異常檢測奠定方法論基礎。

### Ablation 設計

每一層的加入都要回答一個具體問題：

- **L0 only**：模型能否檢測到單一標的的行為變化？
- **L0 + L1**：加入跨標的信息是否提升了異常檢測的精度和穩定性？
- **L0 + L1 + L2**：A 股銀行板塊提供了多少額外的解釋力？
- **L0 + L1 + L2 + L3**：跨市場數據是改善還是損害了性能？

---

## SOTA 模型選擇（帶優先級）

根據 2023–2026 最新進展，按三個梯隊排列，每個梯隊必須按順序完成。

## 特徵工程

針對債券/利率市場的特性，特徵設計分為三類。


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

## License

MIT License — 詳見 [LICENSE](LICENSE) 文件。

---

<p align="center">
  <i>本文件整合了 Gemini、Copilot GPT 及人工審閱的三方建議，作為項目所有 LLM 協作的基礎文件。</i>
</p>

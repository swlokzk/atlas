# 📋 PLAN_AGENT_C.md - Phase 3: L2 (跨资产对照)

## 🎯 你的使命

**Agent C，你负责实现 "跨资产类别对照" (L2)。**

你的工作是：
- ✅ 验证债券市场的异常是否反映在 A 股银行板塊
- ✅ 使用 Granger Causality 分析领先-滞后关系
- ✅ 证明债券和股票异常之间的**因果关系**

**时间投入**: 1.5-2 小时
**代码行数**: ~600-700 行
**输出文件数**: 4 个新文件

---

## 📂 第一步：理解项目结构

### 你需要阅读的文件

**必须读这些文件**（顺序很重要）：

```
1. 最重要 - 理解 L2 的设计:
   _research/cnogb-abnormal-intervention/docs/BACKGROUND.md
   └─ 重点章节:
      • "分層域架構" (第 78-97 行)
         看 L2 的定义: "A 股銀行板塊"
      • "跨資產分析" 部分
         理解: Granger Causality 是什么
      • "評估框架" (第 197-226 行)

2. 第二重要 - 理解代码结构:
   src/data/processing.py (全部读)
   └─ 理解: 如何加载和处理数据
   
   src/data/features.py (全部读)
   └─ 理解: 特征工程是如何进行的
   
3. 参考 Agent A 和 Agent B 的工作:
   experiments/run_l0.py (了解一般结构)
   experiments/run_l1.py (了解 L1 如何扩展)
   └─ L2 会类似地扩展这些

4. 背景知识（推荐）:
   - Granger Causality 是什么
      基本思想: 如果 X 的过去能帮助预测 Y，
              那么说 X Granger-causes Y
      
   - A 股银行板塊
      主要成分: 工商银行、农业银行、中国银行、建设银行等
      为什么: 政策性金融债的主要持有者是这些大银行
```

### 快速核心概念检查

```
Q1: Granger Causality 是什么？
A: 一种统计因果关系的方法
   如果 债券收益率 的过去能帮助预测 银行股价，
   那么说 债券 → 银行股 有 Granger Causality

Q2: L2 与 L0/L1 的区别？
A: L0: 单个债券自身的异常检测 (时间序列 → 异常)
   L1: 多个债券的异常一致性 (债券 × 债券)
   L2: 债券与股票的关联 (债券 × 股票)

Q3: 为什么要分析债券→股票？
A: 理政府债的异常是否会影响银行股价
   理解市场的信息传导机制
   验证两个市场是否互相影响

Q4: 我需要自己下载股票数据吗？
A: 是的! 使用 AKShare 库
   但代码框架已经为你设计好了
```

---

## 💻 第二步：你需要创建的文件

### 文件列表

```
你需要创建这 4 个新文件:

1. src/data/external.py                [数据集成]
   ├─ def download_ak_stock_data()
   ├─ def align_stock_to_bond()
   └─ ~100-150 行

2. src/evaluation/granger.py            [因果分析]
   ├─ def granger_causality_analysis()
   ├─ def compute_dynamic_granger()
   └─ ~150-200 行

3. experiments/run_l2.py               [实验脚本]
   ├─ 加载债券 + 股票数据
   ├─ 进行 Granger 分析
   ├─ 评估联合异常检测
   └─ ~150-200 行

4. configs/l2.yaml                     [配置文件]
   ├─ 债券配置
   ├─ 股票配置
   ├─ Granger 参数
   └─ ~60 行

5. results/l2_report.json              [输出 - 自动生成]
```

### 这些文件你不用创建

```
✅ 已存在，直接用:
   src/models/tranad/model.py         (TranAD 模型)
   src/evaluation/metrics.py          (AUROC/AUPRC)
   experiments/run_l0.py              (参考 L0 结构)
   experiments/run_l1.py              (参考 L1 结构)
```

---

## 🔍 第三步：L2 的技术细节

### A 股银行板塊是什么？

```yaml
# 你需要下载这些股票数据

主要银行板塊 (使用 AKShare):

1. 工商银行 (ICBC)
   代码: 601398
   在 A 股的代码: 601398 或 sh601398
   
2. 农业银行 (ABC)
   代码: 601288
   
3. 中国银行 (BOC)
   代码: 601988
   
4. 建设银行 (CCB)
   代码: 601939
   
5. 招商银行 (CMB)
   代码: 600036

为什么用这些银行：
- 它们是政策性金融债的最大机构持有者
- 它们的股价会反应政策性金融债的风险变化
- 时间序列足够长（历史数据充足）
```

### Granger Causality 的含义

```
假设有两个时间序列:
  - Bond_yield(t)    债券收益率
  - Stock_price(t)   银行股价

Granger Causality 检验:
  H0: Bond_yield 不能 Granger-cause Stock_price
  H1: Bond_yield 能 Granger-cause Stock_price

实现方法:
  1. 用 AR(p) 只预测 Stock_price(t)
     SSR1 = Sum of Squared Residuals
  
  2. 用 AR(p) + Bond_yield 的 p 阶滞后预测 Stock_price(t)
     SSR2 = Sum of Squared Residuals (应该更小)
  
  3. F-test: F = (SSR1 - SSR2) / (SSR2 / (n-2p))
  
  4. 如果 p-value < 0.05，拒绝 H0
     即 Bond_yield 显著地 Granger-causes Stock_price

在 L2 中:
  - 债券异常 → 股票异常: 是否存在 Granger causality?
  - 股票异常 → 债券异常: 是否存在反向 Granger causality?
```

### 工作流程

```
┌──────────────────────────────────────────────┐
│          L2 跨资产分析工作流                  │
├──────────────────────────────────────────────┤
│                                              │
│ 1. 数据准备 (src/data/external.py)          │
│    ├─ 从 AKShare 下载股票数据 (OHLCV)      │
│    ├─ 加载债券数据 (yield, volume)          │
│    ├─ 时间对齐 (到共同的交易日期)          │
│    └─ 特征标准化                            │
│                                              │
│ 2. Granger 分析 (src/evaluation/granger.py) │
│    ├─ 计算 Bond → Stock Granger p-value    │
│    ├─ 计算 Stock → Bond Granger p-value    │
│    ├─ 动态 Granger (滚动窗口)              │
│    └─ 因果方向判断                          │
│                                              │
│ 3. 联合异常检测                             │
│    ├─ 债券异常分数                          │
│    ├─ 股票异常分数                          │
│    ├─ 同步异常分析                          │
│    └─ 领先指标分析                          │
│                                              │
│ 4. 评估                                     │
│    ├─ AUROC (跨资产模型)                   │
│    ├─ MC Dropout 不确定性                   │
│    ├─ 特征重要性                            │
│    ├─ 稳定性测试                            │
│    ├─ 可信度评估                            │
│    └─ 能耗统计                              │
│                                              │
│ 5. 输出                                     │
│    └─ results/l2_report.json               │
│                                              │
└──────────────────────────────────────────────┘
```

---

## 📊 第四步：实现细节

### Granger Causality 实现框架

```python
# src/evaluation/granger.py

import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests

def granger_causality_test(X: np.ndarray, Y: np.ndarray, max_lag: int = 5):
    """
    测试 X 是否 Granger-cause Y
    
    Args:
        X: 时间序列 (n_samples,) - 潜在的 cause
        Y: 时间序列 (n_samples,) - 潜在的 effect
        max_lag: 最大滞后阶数
    
    Returns:
        Dict with keys:
        - 'p_values': 各滞后阶数的 p-value
        - 'significant_lag': 最小显著滞后阶数
        - 'conclusion': 'significant' or 'not_significant'
    """
    # 组合数据: [[Y, X]]
    data = np.column_stack([Y, X])
    
    # statsmodels 的 grangercausalitytests 函数
    gc_result = grangercausalitytests(data, max_lag=max_lag, verbose=False)
    
    # 提取 p-values
    p_values = [gc_result[i][0][0][0][1] for i in range(1, max_lag + 1)]
    
    # 判断是否显著
    alpha = 0.05
    significant_lag = next((i for i, p in enumerate(p_values, 1) if p < alpha), None)
    
    return {
        'p_values': p_values,
        'significant_lag': significant_lag,
        'conclusion': 'significant' if significant_lag else 'not_significant',
    }


def compute_dynamic_granger(X: np.ndarray, Y: np.ndarray, 
                           window_size: int = 30, stride: int = 5):
    """
    计算动态 Granger Causality (滚动窗口)
    
    目的: 看因果关系是否随时间变化
    
    Args:
        X, Y: 时间序列
        window_size: 滚动窗口大小
        stride: 步长
    
    Returns:
        Dict: 时间序列的 p-values 和 significance
    """
    n = len(X)
    results = []
    timestamps = []
    
    for t in range(0, n - window_size, stride):
        X_window = X[t : t + window_size]
        Y_window = Y[t : t + window_size]
        
        # 对这个窗口进行 Granger 测试
        gc = granger_causality_test(X_window, Y_window, max_lag=3)
        
        results.append(gc)
        timestamps.append(t + window_size)  # 用窗口的结束时刻
    
    return {
        'timestamps': timestamps,
        'p_values': [r['p_values'][0] for r in results],  # 使用第一个滞后的 p-value
        'is_significant': [r['conclusion'] == 'significant' for r in results],
    }
```

### 数据下载和对齐

```python
# src/data/external.py

import akshare as ak
import pandas as pd
from datetime import datetime

def download_ak_stock_data(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    从 AKShare 下载 A 股数据
    
    Args:
        stock_code: 股票代码 (如 '601398' 工商银行)
        start_date: 开始日期 (如 '2020-01-01')
        end_date: 结束日期 (如 '2024-01-01')
    
    Returns:
        DataFrame 包含列: date, open, high, low, close, volume
    """
    # AKShare 的股票日线数据接口
    df = ak.stock_zh_a_hist(
        symbol=stock_code,
        period="daily",
        start_date=start_date.replace('-', ''),
        end_date=end_date.replace('-', ''),
        adjust="qfq"  # 前复权
    )
    
    # 列名标准化
    df.columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'amount', 'turnover']
    df['date'] = pd.to_datetime(df['date'])
    
    return df.sort_values('date').reset_index(drop=True)


def align_stock_to_bond(bond_df: pd.DataFrame, stock_df: pd.DataFrame) -> pd.DataFrame:
    """
    对齐债券和股票数据到共同的交易日期
    
    Args:
        bond_df: 债券数据 (必须有 'date' 列)
        stock_df: 股票数据 (必须有 'date' 列)
    
    Returns:
        合并后的 DataFrame，包含:
        - bond_* (债券的所有列，带前缀)
        - stock_* (股票的所有列，带前缀)
    """
    # 转换日期格式
    bond_df = bond_df.copy()
    stock_df = stock_df.copy()
    
    bond_df['date'] = pd.to_datetime(bond_df['date'])
    stock_df['date'] = pd.to_datetime(stock_df['date'])
    
    # 内连接：只保留两个数据都有的日期
    merged = pd.merge(
        bond_df.rename(columns={c: f'bond_{c}' if c != 'date' else c for c in bond_df.columns}),
        stock_df.rename(columns={c: f'stock_{c}' if c != 'date' else c for c in stock_df.columns}),
        on='date',
        how='inner'
    )
    
    return merged.sort_values('date').reset_index(drop=True)
```

### 联合异常检测

```python
# 伪代码：如何在 L2 中进行联合异常检测

def joint_anomaly_detection(bond_yield: np.ndarray, 
                           stock_price: np.ndarray,
                           config: Config):
    """
    同时在债券和股票上进行异常检测
    
    Args:
        bond_yield: 债券收益率时间序列
        stock_price: 银行股股价时间序列
        config: 配置对象
    
    Returns:
        Dict: {
            'bond_anomaly_scores': array,
            'stock_anomaly_scores': array,
            'joint_anomaly_score': array,  # 综合异常分数
            'synchronization': float,  # 同步程度
        }
    """
    
    # 1. 分别在债券和股票上训练模型
    # 债券模型
    bond_model = build_tranad(config)
    bond_model = train_model(bond_model, bond_yield)
    bond_anomaly = bond_model.detect_anomalies(bond_yield)  # (n, 1)
    
    # 股票模型
    stock_model = build_tranad(config)
    stock_model = train_model(stock_model, stock_price)
    stock_anomaly = stock_model.detect_anomalies(stock_price)  # (n, 1)
    
    # 2. 联合异常分数（简单平均）
    joint_anomaly = (bond_anomaly + stock_anomaly) / 2
    
    # 3. 计算同步程度（相关性）
    synchronization = np.corrcoef(bond_anomaly.flatten(), stock_anomaly.flatten())[0, 1]
    
    # 4. 分析领先指标
    # 债券异常是否领先股票异常?
    lag_correlation = {}
    for lag in range(-5, 6):  # 5 天的提前，到 5 天的滞后
        if lag < 0:
            corr = np.corrcoef(bond_anomaly[:lag], stock_anomaly[-lag:])[0, 1]
        elif lag > 0:
            corr = np.corrcoef(bond_anomaly[lag:], stock_anomaly[:-lag])[0, 1]
        else:
            corr = synchronization
        lag_correlation[lag] = corr
    
    max_lag = max(lag_correlation, key=lambda k: abs(lag_correlation[k]))
    
    return {
        'bond_anomaly_scores': bond_anomaly.flatten(),
        'stock_anomaly_scores': stock_anomaly.flatten(),
        'joint_anomaly_score': joint_anomaly.flatten(),
        'synchronization': synchronization,
        'leading_indicator': {
            'max_correlation_lag': max_lag,  # 负数表示债券领先
            'max_correlation': lag_correlation[max_lag],
        },
        'granger_result': {
            'bond_to_stock': granger_causality_test(bond_anomaly, stock_anomaly)['conclusion'],
            'stock_to_bond': granger_causality_test(stock_anomaly, bond_anomaly)['conclusion'],
        }
    }
```

### 配置文件示例

```yaml
# configs/l2.yaml

# 债券配置
data:
  bond:
    pair_name: cdb  # 使用 CDB 作为代表
    raw_dir: data/raw
  
  stock:
    # A 股银行板塊
    tickers: ['601398', '601288', '601988', '601939', '600036']
    names: ['工商银行', '农业银行', '中国银行', '建设银行', '招商银行']
    # 这些是 AKShare 的代码，可以直接下载
  
  # 数据对齐选项
  alignment:
    method: 'inner'  # 只用两个数据都有的日期
    frequency: 'daily'  # 日频

# 模型配置（与 L0/L1 一致）
model:
  name: tranad
  d_model: 128
  nhead: 4
  num_layers: 3
  adversarial_weight: 1.0

training:
  epochs: 50
  batch_size: 32
  learning_rate: 1e-3
  device: cuda

# 评估配置
evaluation:
  mc_samples: 20
  stability_seeds: [42, 123, 456, 789, 1000]
  anomaly_threshold: 0.5

# Granger Causality 配置
granger:
  max_lag: 5
  dynamic_window_size: 30  # 滚动窗口大小
  dynamic_stride: 5  # 步长
  alpha: 0.05  # 显著性水平

# 输出配置
output:
  report_path: results/l2_report.json
  figures_dir: results/l2_figures
```

---

## ✅ 第五步：如何检验你的工作

### 自检清单

```
□ 文件创建完成
  ✓ src/data/external.py 存在
  ✓ src/evaluation/granger.py 存在
  ✓ experiments/run_l2.py 存在
  ✓ configs/l2.yaml 存在

□ 代码可以导入
  在 Python 中运行:
  >>> from src.data.external import download_ak_stock_data, align_stock_to_bond
  >>> from src.evaluation.granger import granger_causality_test
  >>> from experiments.run_l2 import main
  没有错误就说明导入成功

□ 股票数据可以下载
  在 Python 中运行:
  >>> from src.data.external import download_ak_stock_data
  >>> df = download_ak_stock_data('601398', '2023-01-01', '2023-12-31')
  >>> print(df.shape)
  应该看到数据行数 (应该 > 200 行)

□ 数据对齐成功
  在 Python 中运行:
  >>> bond_df = pd.read_excel('data/raw/cdb_train.xlsx')
  >>> stock_df = download_ak_stock_data(...)
  >>> merged = align_stock_to_bond(bond_df, stock_df)
  >>> print(f"Bond rows: {len(bond_df)}, Stock rows: {len(stock_df)}, Merged rows: {len(merged)}")
  应该看到 Merged rows < min(Bond rows, Stock rows)
  这是正常的（只保留共同日期）

□ Granger Causality 可以运行
  在 Python 中运行:
  >>> from src.evaluation.granger import granger_causality_test
  >>> X = np.random.randn(100)
  >>> Y = X + np.random.randn(100) * 0.1
  >>> result = granger_causality_test(X, Y)
  >>> print(result['conclusion'])
  应该看到 'significant'（因为 Y 高度依赖 X）

□ 脚本可以运行（干运行）
  在 Shell 中运行:
  $ python experiments/run_l2.py --dry-run
  或者编写测试：
  
  def test_l2_framework():
      from experiments.run_l2 import load_l2_data
      bond_data, stock_data, merged = load_l2_data(config)
      assert len(merged) > 0
      assert 'bond_' in merged.columns[0]
      assert 'stock_' in merged.columns[1]
      print("✓ Data loading works")

□ 输出格式正确
  脚本运行后，检查：
  
  >>> import json
  >>> with open('results/l2_report.json') as f:
  ...     report = json.load(f)
  >>> assert 'granger_result' in report
  >>> assert 'joint_anomaly_metrics' in report
  >>> assert 'bond_stock_correlation' in report
  没有 AssertionError 就说明格式对
```

### 单元测试代码

```python
# test_l2.py

def test_download_stock_data():
    """测试股票数据下载"""
    from src.data.external import download_ak_stock_data
    
    df = download_ak_stock_data('601398', '2023-01-01', '2023-12-31')
    
    assert df is not None
    assert len(df) > 0
    assert 'date' in df.columns
    assert 'close' in df.columns
    assert 'volume' in df.columns
    print(f"✓ Downloaded {len(df)} rows of stock data")


def test_data_alignment():
    """测试债券和股票数据对齐"""
    from src.data.external import align_stock_to_bond, download_ak_stock_data
    from src.data.processing import process_pair
    import pandas as pd
    
    # 加载债券数据
    bond_pair = {'train': 'data/raw/cdb_train.xlsx', 'exam': 'data/raw/cdb_exam.xlsx'}
    bond_info = process_pair(bond_pair, 'date', 'yield')
    bond_df = bond_info['df_train']
    
    # 下载股票数据
    stock_df = download_ak_stock_data('601398', '2023-01-01', '2023-12-31')
    
    # 对齐
    merged = align_stock_to_bond(bond_df, stock_df)
    
    assert len(merged) > 0
    assert len(merged) <= min(len(bond_df), len(stock_df))
    assert 'bond_' in merged.columns[0]
    assert 'stock_' in merged.columns[1]
    print(f"✓ Aligned {len(merged)} common trading days")


def test_granger_causality():
    """测试 Granger Causality 的合理性"""
    from src.evaluation.granger import granger_causality_test
    import numpy as np
    
    # 创建有因果关系的时间序列
    np.random.seed(42)
    n = 200
    X = np.random.randn(n)
    Y = X + np.random.randn(n) * 0.1  # Y 高度依赖 X
    
    # X → Y 应该显著
    result_x_to_y = granger_causality_test(X, Y)
    assert result_x_to_y['conclusion'] == 'significant'
    
    # Y → X 应该不显著（反向）
    result_y_to_x = granger_causality_test(Y, X)
    assert result_y_to_x['conclusion'] == 'not_significant'
    
    print("✓ Granger Causality logic is correct")


def test_dynamic_granger():
    """测试动态 Granger（滚动窗口）"""
    from src.evaluation.granger import compute_dynamic_granger
    import numpy as np
    
    n = 200
    X = np.random.randn(n)
    Y = X + np.random.randn(n) * 0.1
    
    result = compute_dynamic_granger(X, Y, window_size=30, stride=5)
    
    assert 'timestamps' in result
    assert 'p_values' in result
    assert 'is_significant' in result
    assert len(result['timestamps']) > 0
    print(f"✓ Computed dynamic Granger for {len(result['timestamps'])} windows")


if __name__ == '__main__':
    test_download_stock_data()
    test_data_alignment()
    test_granger_causality()
    test_dynamic_granger()
    print("\n✓ All L2 tests passed!")
```

### 运行和验证步骤

```bash
# 1. 运行单元测试
python test_l2.py

# 2. 检查可以下载数据
python -c "
from src.data.external import download_ak_stock_data
df = download_ak_stock_data('601398', '2023-01-01', '2023-12-31')
print(f'Downloaded {len(df)} rows')
print(df.head())
"

# 3. 运行实际 L2 实验（需要等 Agent A 完成）
python experiments/run_l2.py

# 4. 检查输出
ls -lh results/l2_*
cat results/l2_report.json | python -m json.tool

# 5. 验证结果
python -c "
import json
with open('results/l2_report.json') as f:
    report = json.load(f)

# Granger 结果
print('Granger results:')
print(f'  Bond → Stock: {report[\"granger_result\"][\"bond_to_stock\"]}')
print(f'  Stock → Bond: {report[\"granger_result\"][\"stock_to_bond\"]}')

# 相关性
print(f'Correlation: {report[\"bond_stock_correlation\"]:.3f}')

# 领先性分析
print(f'Leading indicator lag: {report[\"leading_indicator\"][\"max_correlation_lag\"]} days')

print('✓ L2 analysis complete')
"
```

---

## 📝 第六步：代码质量标准

你写的代码必须满足这些标准：

```python
# ✅ 好的代码示例

# 1. Type hints
def download_ak_stock_data(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    pass

# 2. 错误处理
try:
    df = download_ak_stock_data('601398', '2023-01-01', '2023-12-31')
except Exception as e:
    logger.error(f"Failed to download stock data: {e}")
    raise

# 3. 数据验证
assert len(bond_df) > 0, "Bond data is empty"
assert len(merged) <= min(len(bond_df), len(stock_df)), "Merged data size mismatch"

# 4. 日志
logger.info(f"Downloaded {len(df)} rows of stock data")
logger.debug(f"Date range: {df['date'].min()} to {df['date'].max()}")

# 5. 文档
def granger_causality_test(X: np.ndarray, Y: np.ndarray, max_lag: int = 5) -> Dict:
    """
    Granger Causality test to check if X causes Y.
    
    Uses statsmodels.tsa.stattools.grangercausalitytests.
    
    Args:
        X: Cause time series (n_samples,)
        Y: Effect time series (n_samples,)
        max_lag: Maximum lag order to test (default: 5)
    
    Returns:
        Dict with:
        - 'p_values': List of p-values for each lag
        - 'significant_lag': Minimum lag showing significance (None if not significant)
        - 'conclusion': 'significant' or 'not_significant'
    
    References:
        Granger, C. W. (1969). "Investigating Causal Relations by Econometric Models"
    """
    pass
```

---

## 🎓 第七步：理解你做的工作的意义

### 为什么 L2 很重要？

```
L0 问: 能否检测债券的异常？
  → Agent A 回答: 是的。

L1 问: 这个异常是真实的系统事件吗？
  → Agent B 回答: 是的，三个债券同时异常。

L2 问: 但这会影响股市吗？债券和股票市场真的互相影响吗？
  → 验证方法: 使用 Granger Causality
              看债券异常是否领先股票异常
              看它们的异常是否同步

L2 的目的:
  ✓ 验证因果关系是否真实存在
  ✓ 理解信息在市场间的传导
  ✓ 为政策制定者提供决策依据
  ✓ 为下一步的港股迁移做准备
```

### 你的输出会被如何使用？

```
你的 L2 结果 → 综合报告
              ↓
          如果债券→股票的 Granger 显著:
              说明债券市场是领先指标
              可以用来预警股市风险
              
          如果债券←股票的 Granger 显著:
              说明股市更敏感
              债券市场反应滞后
              
          如果都显著:
              说明两个市场相互影响
              需要联合监测
```

---

## 🚀 快速开始

### 最小可行代码框架

如果你只有 1 小时，按这个顺序：

```python
# 1. 创建 external.py 的骨架
# (20 分钟)

import akshare as ak
import pandas as pd

def download_ak_stock_data(stock_code: str, start_date: str, end_date: str):
    df = ak.stock_zh_a_hist(
        symbol=stock_code,
        period="daily",
        start_date=start_date.replace('-', ''),
        end_date=end_date.replace('-', ''),
        adjust="qfq"
    )
    return df

def align_stock_to_bond(bond_df, stock_df):
    # TODO: 时间对齐逻辑
    pass

# 2. 创建 granger.py 的骨架
# (20 分钟)

from statsmodels.tsa.stattools import grangercausalitytests

def granger_causality_test(X, Y, max_lag=5):
    # TODO: 包装 statsmodels 函数
    pass

def compute_dynamic_granger(X, Y, window_size=30, stride=5):
    # TODO: 滚动窗口逻辑
    pass

# 3. 创建 run_l2.py
# (20 分钟)

def main():
    # 1. 加载债券数据
    # 2. 下载股票数据
    # 3. 对齐
    # 4. 进行 Granger 分析
    # 5. 保存结果
    pass

# 4. 逐个填充函数体
# (剩余时间)
```

---

## 📞 需要帮助吗？

```
问: 如何获取 AKShare？
答: pip install akshare
   第一次使用可能比较慢，因为要下载数据

问: 工商银行的代码是什么？
答: 601398 或 sh601398
   都可以用，但 AKShare 通常用前者

问: 如何处理股票数据的缺失值？
答: 使用 fillna() 或 dropna()
   具体选择取决于缺失数据的位置和多少

问: Granger p-value 应该是多少才算显著？
答: 通常用 0.05 作为临界值
   p < 0.05 说明显著
   p >= 0.05 说明不显著

问: 我的 Granger 结果总是不显著？
答: 可能是:
   1. 数据太短（需要 > 100 个样本）
   2. 两个序列确实没有因果关系
   3. 滞后阶数设置不对
   尝试增加 max_lag 或改变数据时间段
```

---

## ✨ 完成标志

**当你看到这个输出时，说明你成功了：**

```bash
$ python experiments/run_l2.py

Loading configuration from configs/l2.yaml...
Loading bond data for L2 experiment...
  ✓ CDB data: 250 samples

Downloading stock data from AKShare...
  ✓ ICBC (601398): 240 samples
  ✓ ABC (601288): 240 samples
  ✓ BOC (601988): 240 samples
  ✓ CCB (601939): 240 samples
  ✓ CMB (600036): 238 samples

Aligning data...
  ✓ Aligned with ICBC: 200 common trading days
  ✓ Aligned with ABC: 198 common trading days
  ✓ Aligned with BOC: 199 common trading days
  ✓ Aligned with CCB: 200 common trading days
  ✓ Aligned with CMB: 196 common trading days

Granger Causality Analysis...
  CDB → ICBC:
    ✓ p-value (lag 1): 0.023 (significant)
    ✓ p-value (lag 2): 0.041 (significant)
    ✓ Conclusion: CDB Granger-causes ICBC
  
  ICBC → CDB:
    ✓ p-value (lag 1): 0.312 (not significant)
    ✓ Conclusion: ICBC does NOT Granger-cause CDB

Computing dynamic Granger (rolling window)...
  ✓ 34 windows analyzed
  ✓ Significant periods: 18/34 (52.9%)

Joint anomaly detection...
  ✓ Bond-Stock correlation: 0.68
  ✓ Synchronization ratio: 0.71
  ✓ Leading indicator: Bond leads by 2-3 days

Computing stability metrics (5 seeds)...
  ✓ Granger consistency: 0.95
  ✓ Wilcoxon p-value: 0.012 (significant)

Computing efficiency...
  ✓ Total runtime: 287 seconds
  ✓ Data processing: 42 seconds
  ✓ Granger computation: 156 seconds
  ✓ Anomaly detection: 89 seconds

Saving results to results/l2_report.json...
✓ Done!
```

---

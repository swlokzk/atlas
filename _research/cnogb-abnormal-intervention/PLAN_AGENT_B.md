# 📋 PLAN_AGENT_B.md - Phase 2: L1 (跨标的验证)

## 🎯 你的使命

**Agent B，你负责实现 "跨标的一致性验证" (L1)。**

你不需要知道 Agent A 做了什么。你的工作是：
- ✅ 验证三大政金债 (国开债、农发债、进出口债) 的异常检测一致性
- ✅ 使用 Leave-one-out 交叉验证框架
- ✅ 证明异常信号具有**系统性**（不是随机的）

**时间投入**: 1.5-2 小时
**代码行数**: ~500-600 行
**输出文件数**: 4 个新文件

---

## 📂 第一步：理解项目结构

### 你需要阅读的文件

在开始写代码之前，**必须读这些文件**（顺序很重要）：

```
1. 最重要 - 理解数据:
   _research/cnogb-abnormal-intervention/docs/BACKGROUND.md
   └─ 重点章节:
      • "分層域架構" (第 78-97 行)
         看 L1 的定义: "同市場不同標的（3 大政金債交叉）"
      • "評估框架" (第 197-226 行)
         看 "Leave-one-out Domain" 的定义
         看 "Per-domain AUROC Breakdown"

2. 第二重要 - 理解代码现状:
   src/data/processing.py (全部读)
   └─ 理解: 如何从文件加载一个标的的数据对
   
   src/data/features.py (重点看 prepare_pair_feature_artifacts 函数)
   └─ 理解: 特征是如何准备的

3. 第三重要 - 理解模型接口:
   experiments/run_training.py (全部读)
   └─ 理解: 如何构建、训练一个模型
   
   src/train/components.py (全部读)
   └─ 理解: 如何获得 model/criterion/optimizer
   
   src/train/loop.py (全部读)
   └─ 理解: 如何进行一个 epoch 的训练

4. 参考（不是必须，了解背景）:
   docs/BACKGROUND.md 中的参考文献 #1-3
   └─ 论文: Anomaly Transformer, TranAD, PatchTST
      这三个模型都基于 Transformer
      你需要理解它们如何检测异常
```

### 快速核心概念检查

读完后，你应该能回答这些问题：

```
Q1: 什么是 Leave-one-out 域验证？
A: 对于 3 个债券标的 (CDB, ADBC, EXIM)：
   - 第 1 次: 用 ADBC + EXIM 训练，在 CDB 上测试
   - 第 2 次: 用 CDB + EXIM 训练，在 ADBC 上测试
   - 第 3 次: 用 CDB + ADBC 训练，在 EXIM 上测试
   目的: 验证跨标的的异常检测能力

Q2: L1 与 L0 的区别是什么？
A: L0: 单个标的的历史期 vs 观察期 (二分类)
   L1: 三个标的混合训练，验证一个标的的泛化性

Q3: 异常一致性是什么意思？
A: 如果三个债券在同一时间都被检测为异常，
   那说明这是**系统性事件**（比如政策变化）
   而不是个别债券的问题

Q4: 我需要写 TranAD 吗？
A: 不需要! Agent A 已经写了。
   你只需要使用它 (import from src.models.tranad)
```

---

## 💻 第二步：你需要创建的文件

### 文件列表

```
你需要创建这 4 个新文件:

1. src/evaluation/cross_domain.py     [主要逻辑]
   ├─ class LeaveOneOutValidator
   ├─ def run_loo_experiment()
   ├─ def compute_anomaly_consistency()
   └─ ~200-250 行

2. experiments/run_l1.py              [实验脚本]
   ├─ 读取 3 个债券数据
   ├─ 循环进行 3 次 Leave-one-out
   ├─ 调用 LeaveOneOutValidator
   ├─ 收集结果
   └─ ~150-200 行

3. configs/l1.yaml                    [配置文件]
   ├─ 模型参数 (d_model, nhead 等)
   ├─ 训练参数 (epochs, batch_size)
   ├─ 3 个债券的数据对
   ├─ 评估参数 (MC Dropout 采样数)
   └─ ~50 行

4. results/l1_report.json             [输出文件 - 自动生成]
   ├─ 3 个 Leave-one-out 结果
   ├─ 异常一致性指标
   └─ 由 run_l1.py 生成
```

### 这些文件你不用创建（Agent A 已做）

```
✅ 已存在，直接用:
   src/models/tranad/model.py         (TranAD 模型)
   src/evaluation/metrics.py          (AUROC/AUPRC 函数)
   src/evaluation/synthetic.py        (合成异常函数)
   src/evaluate/stability.py          (稳定性测试)
   src/evaluate/calibration.py        (可信度评估)
   src/evaluate/efficiency.py         (能耗统计)
   src/explain/uncertainty.py         (MC Dropout)
   src/explain/permutation_importance.py (特征重要性)
```

---

## 🔍 第三步：L1 的技术细节

### 三个债券是什么？

```yaml
# 数据在哪里找？
# 文件应该在 data/raw/ 目录下

L1 使用这 3 个债券标的:

1. CDB (国开债 - China Development Bank Bond)
   文件: data/raw/cdb_train.xlsx, cdb_exam.xlsx
   特征: 收益率、成交量、Bid-Ask Spread
   
2. ADBC (农发债 - Agricultural Development Bank Bond)
   文件: data/raw/adbc_train.xlsx, adbc_exam.xlsx
   特征: 同上
   
3. EXIM (进出口债 - Export-Import Bank Bond)
   文件: data/raw/exim_train.xlsx, exim_exam.xlsx
   特征: 同上

这 3 个债券的数据结构**完全相同**，只是发行人不同。
```

### Leave-one-out 的具体流程

```
┌─────────────────────────────────────────────────┐
│         Leave-one-out 第 1 轮                    │
├─────────────────────────────────────────────────┤
│                                                 │
│  训练数据: ADBC_train + EXIM_train              │
│  （合并为一个大数据集）                         │
│                                                 │
│  测试数据: CDB_exam                             │
│                                                 │
│  流程:                                          │
│  1. 加载 ADBC 和 EXIM 的训练集 (历史期)        │
│  2. 合并为一个数据集                            │
│  3. 训练 TranAD 模型                            │
│  4. 在 CDB_exam 上进行异常检测                  │
│  5. 记录结果                                    │
│                                                 │
│  输出:                                          │
│  - CDB 的异常分数 (array)                       │
│  - CDB 的 AUROC/AUPRC                           │
│  - CDB 的 MC Dropout 不确定性                   │
│                                                 │
└─────────────────────────────────────────────────┘

类似地进行第 2、3 轮...

最终输出:
├─ Round 1: AUROC (ADBC+EXIM → CDB test)
├─ Round 2: AUROC (CDB+EXIM → ADBC test)
├─ Round 3: AUROC (CDB+ADBC → EXIM test)
└─ 一致性指标: 三个异常分数的相关性
```

### 代码框架（伪代码）

```python
# src/evaluation/cross_domain.py

class LeaveOneOutValidator:
    """Leave-one-out 交叉验证"""
    
    def __init__(self, config: Config):
        self.config = config
        # 3 个债券: CDB, ADBC, EXIM
        self.bonds = ['cdb', 'adbc', 'exim']
    
    def run_loo_experiment(self):
        """运行 3 次 Leave-one-out"""
        results = {}
        
        for test_bond in self.bonds:
            # 获取训练债券（其他两个）
            train_bonds = [b for b in self.bonds if b != test_bond]
            
            # 1. 加载数据
            train_data = self.load_and_merge(train_bonds)  # 合并特征
            test_data = self.load_data(test_bond)
            
            # 2. 构建和训练模型
            model = build_tranad(...)
            model = self.train_model(model, train_data)
            
            # 3. 异常检测
            anomaly_scores = model.detect_anomalies(test_data)
            
            # 4. 评估
            metrics = evaluate_anomaly_detection(anomaly_scores)
            
            # 5. 保存结果
            results[test_bond] = {
                'anomaly_scores': anomaly_scores,
                'metrics': metrics,
                'uncertainty': mc_dropout_uncertainty(model, test_data),
                'feature_importance': compute_importance(model, test_data),
            }
        
        return results
    
    def compute_anomaly_consistency(self, results):
        """计算异常一致性"""
        # 提取三个标的的异常分数
        anom_cdb = results['cdb']['anomaly_scores']
        anom_adbc = results['adbc']['anomaly_scores']
        anom_exim = results['exim']['anomaly_scores']
        
        # 计算相关性
        corr_cdb_adbc = np.corrcoef(anom_cdb, anom_adbc)[0, 1]
        corr_cdb_exim = np.corrcoef(anom_cdb, anom_exim)[0, 1]
        corr_adbc_exim = np.corrcoef(anom_adbc, anom_exim)[0, 1]
        
        # 找出同时异常的时间点
        threshold = 0.5  # 可配置
        is_anom_cdb = anom_cdb > threshold
        is_anom_adbc = anom_adbc > threshold
        is_anom_exim = anom_exim > threshold
        
        # 所有三个同时异常的比例
        consistency = (is_anom_cdb & is_anom_adbc & is_anom_exim).mean()
        
        return {
            'correlation': {
                'cdb_adbc': corr_cdb_adbc,
                'cdb_exim': corr_cdb_exim,
                'adbc_exim': corr_adbc_exim,
            },
            'consistency_ratio': consistency,  # 多少比例的异常是同步的
        }
```

---

## 📊 第四步：实现细节

### 共享数据结构（必须使用）

```python
# 你的 run_l1.py 最后必须输出这个结构:

from dataclasses import dataclass
from typing import Dict
import numpy as np

@dataclass
class L1Result:
    """L1 Leave-one-out 结果"""
    
    # 对每个债券的结果
    results_by_bond: Dict[str, dict]  # 'cdb', 'adbc', 'exim'
    
    # 样例结构 (每个债券):
    # {
    #   'anomaly_scores': np.ndarray (n_samples,),
    #   'uncertainty': np.ndarray (n_samples,),
    #   'auroc': float,
    #   'auprc': float,
    #   'f1': float,
    #   'feature_importance': Dict[str, float],
    #   'residuals': np.ndarray (n_samples,),
    # }
    
    # 跨标的分析
    consistency_metrics: Dict  # 异常一致性、相关性等
    
    # 稳定性指标
    stability: Dict  # mean, std, p_value
    
    # 可信度
    calibration_ece: float
    
    # 能耗
    efficiency: Dict  # runtime, memory

# 转换为 JSON 格式
def l1_result_to_dict(result: L1Result) -> dict:
    """用于保存到 JSON"""
    return {
        'results_by_bond': {
            bond: {
                'auroc': float(res['auroc']),
                'auprc': float(res['auprc']),
                'f1': float(res['f1']),
                'anomaly_ratio': float((res['anomaly_scores'] > 0.5).mean()),
                'feature_importance': res['feature_importance'],
            }
            for bond, res in result.results_by_bond.items()
        },
        'consistency': result.consistency_metrics,
        'stability': result.stability,
        'calibration_ece': float(result.calibration_ece),
        'efficiency': result.efficiency,
    }
```

### 配置文件示例

```yaml
# configs/l1.yaml

# 债券数据配置
data:
  raw_dir: data/raw
  bonds: [cdb, adbc, exim]  # 三大政金债
  
  # 对应的文件名规则: {bond}_train.xlsx, {bond}_exam.xlsx
  # 比如: cdb_train.xlsx, cdb_exam.xlsx

# 模型配置（与 Agent A 的 L0 一致）
model:
  name: tranad
  d_model: 128
  nhead: 4
  num_layers: 3
  adversarial_weight: 1.0
  
# 训练配置
training:
  epochs: 50
  batch_size: 32
  learning_rate: 1e-3
  device: cuda  # or cpu
  
# 评估配置
evaluation:
  # MC Dropout 采样数
  mc_samples: 20
  
  # 稳定性测试：用这 5 个随机种子各运行一次
  stability_seeds: [42, 123, 456, 789, 1000]
  
  # 异常阈值（用于二分类）
  anomaly_threshold: 0.5
  
  # 合成异常验证（如果需要的话）
  synthetic_validation: true
  synthetic_types: [spike, level_shift, trend_change]

# 输出配置
output:
  report_path: results/l1_report.json
  figures_dir: results/l1_figures
```

---

## ✅ 第五步：如何检验你的工作

### 自检清单

完成后，**必须逐项检查**：

```
□ 文件创建完成
  ✓ src/evaluation/cross_domain.py 存在且不为空
  ✓ experiments/run_l1.py 存在且不为空
  ✓ configs/l1.yaml 存在且格式正确

□ 代码可以导入
  在 Python 中运行:
  >>> from src.evaluation.cross_domain import LeaveOneOutValidator
  >>> from experiments.run_l1 import main
  没有错误就说明导入成功

□ 配置文件可以读取
  在 Python 中运行:
  >>> import yaml
  >>> with open('configs/l1.yaml') as f:
  ...     config = yaml.safe_load(f)
  >>> print(config['data']['bonds'])
  输出应该是: ['cdb', 'adbc', 'exim']

□ 数据文件存在
  在 Shell 中运行:
  $ ls data/raw/cdb_*.xlsx data/raw/adbc_*.xlsx data/raw/exim_*.xlsx
  应该找到 6 个文件（每个债券 2 个：train + exam）

□ 脚本可以运行（干运行）
  在 Shell 中运行:
  $ python experiments/run_l1.py --dry-run
  或者编写一个小的 test 函数：
  
  def test_l1_framework():
      from experiments.run_l1 import LeaveOneOutValidator
      validator = LeaveOneOutValidator(config)
      # 只加载数据，不训练
      for bond in ['cdb', 'adbc', 'exim']:
          data = validator.load_data(bond)
          assert data is not None
          assert len(data) > 0
      print("✓ Data loading works")

□ 输出格式正确
  脚本运行后，检查：
  
  >>> import json
  >>> with open('results/l1_report.json') as f:
  ...     report = json.load(f)
  >>> assert 'results_by_bond' in report
  >>> assert set(report['results_by_bond'].keys()) == {'cdb', 'adbc', 'exim'}
  >>> assert 'consistency' in report
  没有 AssertionError 就说明格式对
```

### 单元测试代码

```python
# 在 test_l1.py 中写测试（可选但推荐）

def test_load_bond_data():
    """测试能否正确加载 3 个债券的数据"""
    from src.data.processing import process_pair
    from src.data.loader import find_paired_files
    
    pairs = find_paired_files('data/raw')
    bond_pairs = {p['name']: p for p in pairs if p['name'] in ['cdb', 'adbc', 'exim']}
    
    assert len(bond_pairs) == 3, f"Expected 3 bonds, got {len(bond_pairs)}"
    
    for bond_name, pair in bond_pairs.items():
        info = process_pair(pair, time_col='date', target_col='yield')
        assert info['valid'], f"Bond {bond_name} processing failed"
        assert len(info['df_train']) > 0, f"Bond {bond_name} has no train data"
        assert len(info['df_exam']) > 0, f"Bond {bond_name} has no exam data"


def test_leave_one_out_logic():
    """测试 Leave-one-out 逻辑是否正确"""
    bonds = ['cdb', 'adbc', 'exim']
    
    for test_bond in bonds:
        train_bonds = [b for b in bonds if b != test_bond]
        assert len(train_bonds) == 2, f"Should have 2 training bonds when testing {test_bond}"
        assert test_bond not in train_bonds, f"Test bond {test_bond} should not be in train bonds"
    
    print("✓ Leave-one-out logic is correct")


def test_anomaly_consistency_metric():
    """测试异常一致性指标的计算"""
    from src.evaluation.cross_domain import compute_anomaly_consistency
    
    # 模拟异常分数
    n = 100
    anom_cdb = np.random.rand(n)
    anom_adbc = anom_cdb + 0.1 * np.random.randn(n)  # 高度相关
    anom_exim = 0.5 * np.random.rand(n)  # 不相关
    
    results = {
        'cdb': {'anomaly_scores': anom_cdb},
        'adbc': {'anomaly_scores': anom_adbc},
        'exim': {'anomaly_scores': anom_exim},
    }
    
    consistency = compute_anomaly_consistency(results)
    
    # CDB 和 ADBC 应该有高相关性
    assert consistency['correlation']['cdb_adbc'] > 0.7
    # CDB 和 EXIM 应该相关性较低
    assert consistency['correlation']['cdb_exim'] < 0.5
    
    print("✓ Anomaly consistency metric works correctly")


if __name__ == '__main__':
    test_load_bond_data()
    test_leave_one_out_logic()
    test_anomaly_consistency_metric()
    print("\n✓ All L1 tests passed!")
```

### 运行和验证步骤

```bash
# 1. 运行单元测试
python test_l1.py

# 2. 运行实际 L1 实验（可能需要等 Agent A 完成）
python experiments/run_l1.py

# 3. 检查输出
ls -lh results/l1_*
cat results/l1_report.json | python -m json.tool

# 4. 验证结果合理性
python -c "
import json
with open('results/l1_report.json') as f:
    report = json.load(f)

# 检查 AUROC 在 0-1 之间
for bond in ['cdb', 'adbc', 'exim']:
    auroc = report['results_by_bond'][bond]['auroc']
    assert 0 <= auroc <= 1, f'{bond} AUROC {auroc} out of range'
    print(f'{bond}: AUROC={auroc:.3f}')

# 检查异常一致性
consistency = report['consistency']['consistency_ratio']
print(f'Anomaly consistency: {consistency:.3%}')

print('✓ All results are reasonable')
"
```

---

## 📝 第六步：代码质量标准

你写的代码必须满足这些标准：

```python
# ✅ 好的代码示例

# 1. Type hints
def run_loo_experiment(self, config: Config) -> Dict[str, dict]:
    """不要写 def run_loo_experiment(self, config)"""
    pass

# 2. Docstring
def compute_anomaly_consistency(self, results: Dict) -> Dict:
    """
    计算三个债券异常分数的一致性。
    
    Args:
        results: Leave-one-out 结果字典
                {bond_name: {anomaly_scores, ...}}
    
    Returns:
        Dict with keys: 'correlation', 'consistency_ratio'
    """
    pass

# 3. 错误处理
try:
    data = self.load_data(bond)
except FileNotFoundError as e:
    raise FileNotFoundError(f"Data file for {bond} not found: {e}")

# 4. 日志而不是 print
import logging
logger = logging.getLogger(__name__)

logger.info(f"Starting Leave-one-out for {test_bond}")
logger.debug(f"AUROC: {auroc:.3f}")
logger.warning(f"Anomaly ratio unusually high: {ratio:.1%}")

# ❌ 不好的代码

print("Starting...")  # 用 logger 而不是 print
result = run_loo()  # 没有 type hints
results[bond] = process(data)  # 没有 docstring
if data is None:  # 没有具体的异常处理
    data = load_default()
```

---

## 🎓 第七步：理解你做的工作的意义

### 为什么 L1 很重要？

```
L0 问: 能否检测单个债券的异常？
  → Agent A 回答: 是的，TranAD 可以。

L1 问: 但这个异常是真实的吗？还是模型瞎报？
  → 验证方法: 如果三个债券同时报异常，说明是系统事件
             如果只有一个债券报异常，说明是个别问题

L1 的目的:
  ✓ 验证异常检测的**可靠性**
  ✓ 区分**系统性风险** vs **个别风险**
  ✓ 为下一步 L2（跨资产）和 L3（跨市场）做准备
```

### 你的输出会被如何使用？

```
你的 L1 结果 → Agent A 和 Agent C 会参考它
              ↓
          决定是否继续进行 L2 和 L3
              ↓
          最终生成综合报告

比如，如果 L1 发现:
- ✓ 三个债券的异常高度一致 (consistency > 70%)
  → 说明我们的异常检测很稳健，可以信任
  
- ✗ 三个债券的异常毫无相关性 (correlation < 0.3)
  → 说明模型可能有问题，需要调试
```

---

## 🚀 快速开始

### 最小可行代码框架

如果你只有 1 小时，按这个顺序：

```python
# 1. 创建 cross_domain.py 的骨架
# (20 分钟)

class LeaveOneOutValidator:
    def __init__(self, config):
        self.config = config
        self.bonds = ['cdb', 'adbc', 'exim']
    
    def run_loo_experiment(self):
        results = {}
        for test_bond in self.bonds:
            # TODO: 实现逻辑
            pass
        return results
    
    def compute_anomaly_consistency(self, results):
        # TODO: 计算一致性
        pass

# 2. 创建 run_l1.py
# (20 分钟)

def main():
    config = load_config('configs/l1.yaml')
    validator = LeaveOneOutValidator(config)
    results = validator.run_loo_experiment()
    consistency = validator.compute_anomaly_consistency(results)
    
    # 保存为 JSON
    report = {
        'results_by_bond': {...},
        'consistency': consistency,
    }
    with open('results/l1_report.json', 'w') as f:
        json.dump(report, f, indent=2)

if __name__ == '__main__':
    main()

# 3. 填充 configs/l1.yaml
# (10 分钟)

# 4. 逐个填充函数体
# (剩余时间)
```

---

## 📞 需要帮助吗？

如果卡住了，**先检查这些**：

```
问: 我不知道如何合并 ADBC 和 EXIM 的数据？
答: 看 src/data/processing.py 中的 process_pair() 函数
   然后用 pd.concat([df_adbc, df_exim], axis=0) 

问: 我不知道如何调用 TranAD 模型？
答: 看 experiments/run_training.py
   它展示了如何用 build_tranad() 构建模型

问: 我的异常分数总是全 1 或全 0？
答: 可能是数据预处理问题
   检查 features.py 中的 prepare_pair_feature_artifacts()

问: AUROC 怎么计算？
答: Agent A 已经在 src/evaluation/metrics.py 中实现了
   直接用: from src.evaluation.metrics import compute_auroc
```

---

## ✨ 完成标志

**当你看到这个输出时，说明你成功了：**

```bash
$ python experiments/run_l1.py

Loading configuration from configs/l1.yaml...
Loading bond data for L1 experiment...
  ✓ CDB: 200 train samples, 50 exam samples
  ✓ ADBC: 200 train samples, 50 exam samples
  ✓ EXIM: 200 train samples, 50 exam samples

Running Leave-one-out experiment...
  [Round 1/3] Training on ADBC+EXIM, testing on CDB...
    ✓ AUROC: 0.842
    ✓ AUPRC: 0.756
    ✓ F1: 0.698
  
  [Round 2/3] Training on CDB+EXIM, testing on ADBC...
    ✓ AUROC: 0.815
    ✓ AUPRC: 0.723
    ✓ F1: 0.671
  
  [Round 3/3] Training on CDB+ADBC, testing on EXIM...
    ✓ AUROC: 0.829
    ✓ AUPRC: 0.741
    ✓ F1: 0.684

Computing anomaly consistency...
  ✓ Correlation (CDB-ADBC): 0.68
  ✓ Correlation (CDB-EXIM): 0.65
  ✓ Correlation (ADBC-EXIM): 0.71
  ✓ Consistency ratio: 0.62 (62% of anomalies are synchronized)

Computing stability metrics (5 seeds)...
  ✓ Mean AUROC: 0.829 ± 0.015
  ✓ Wilcoxon p-value: 0.031 (significant)

Computing calibration...
  ✓ ECE: 0.042

Computing efficiency...
  ✓ Total runtime: 342 seconds
  ✓ GPU memory peak: 2847 MB

Saving results to results/l1_report.json...
✓ Done!
```

---

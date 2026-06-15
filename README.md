# 研究复现指南 — Reproducibility Guide

## 论文信息

**标题**: 基于参数化模拟和机器学习的北京酒店建筑EUI预测及EUI-OCEI耦合研究与分析  
**英文标题**: EUI Prediction and Energy–Carbon Coupling Analysis for Beijing Hotel Buildings Using Parametric Simulation and Machine Learning

---

## 目录

1. [系统要求](#1-系统要求)
2. [环境配置](#2-环境配置)
3. [仓库结构](#3-仓库结构)
4. [复现流程](#4-复现流程)
5. [预期输出](#5-预期输出)
6. [常见问题](#6-常见问题)
7. [引用声明](#7-引用声明)

---

## 1. 系统要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| 操作系统 | Windows 10/11 (64-bit) | Windows 11 (64-bit) |
| Python | 3.11.x | 3.11.5 |
| 内存 | 16 GB | 32 GB |
| 磁盘空间 | 10 GB (含仿真结果) | 50 GB (全量仿真) |
| EnergyPlus | 25.2.0 | 25.2.0 |
| Git | 2.x (可选) | 最新版 |

## 2. 环境配置

### 2.1 Python 环境

```bash
# 创建虚拟环境（推荐）
python -m venv venv
venv\Scripts\activate  # Windows

# 安装核心依赖
pip install numpy==1.26.4 pandas==2.2.2 scipy==1.13.1
pip install matplotlib==3.9.2 seaborn==0.13.2
pip install scikit-learn==1.5.2 statsmodels==0.14.2
pip install xgboost==2.1.1 lightgbm==4.5.0
pip install shap==0.45.0 joblib==1.4.2
pip install jupyter==1.0.0
```

### 2.2 EnergyPlus 安装

1. 从 [EnergyPlus 官方网站](https://energyplus.net/downloads) 下载 EnergyPlus 25.2.0 Windows 安装包
2. 安装至 `C:\EnergyPlusV25-2-0\`
3. 验证安装：
```bash
C:\EnergyPlusV25-2-0\energyplus.exe --version
```
输出应为 `EnergyPlus, Version 25.2.0`

### 2.3 气象文件

- 文件: `input/Beijing.epw`
- 来源: [EnergyPlus Weather Database](https://energyplus.net/weather)
- 数据集: CSWD (Chinese Standard Weather Data)
- 位置: 北京 (WMO Station 545110)

### 2.4 验证环境

```bash
python -c "
import numpy, pandas, scipy, matplotlib, sklearn, xgboost, lightgbm, shap
print('All imports OK')
print(f'Python {numpy.__version__}')
"
```

## 3. 仓库结构

```
论文修改/
├── README.md                                    ← 本文件
├── CLAUDE.md                                    ← AI 辅助配置
│
├── 01_Parametric_Simulation_Database_Construction.ipynb  ← 步骤1
├── 02_SRC_Sensitivity_and_Variable_Selection.ipynb       ← 步骤2
├── 03_ML_Model_Training_and_Evaluation.ipynb            ← 步骤3
├── 04_EUI_OCEI_Coupling_and_Carbon_Analysis.ipynb       ← 步骤4
│
├── input/
│   └── Beijing.epw                              ← 北京气象文件
│
├── data/
│   └── step1_simulation_dataset.csv             ← 步骤1输出（仿真后生成）
│
├── outputs_step1/                               ← 步骤1输出
│   ├── generated_idf/                           ←   自动生成的IDF文件
│   ├── runs/                                    ←   EnergyPlus运行目录
│   └── figures/                                 ←   步骤1图表
│
├── outputs_step2/                               ← 步骤2输出
│   ├── figures/
│   ├── vif_table.csv
│   ├── src_indices_bootstrap.csv
│   └── src_shap_ranking_comparison.csv
│
├── outputs_step3/                               ← 步骤3输出
│   ├── figures/
│   ├── models/                                  ←   训练好的模型文件(.joblib)
│   ├── model_metrics.csv
│   ├── model_hyperparameters.csv
│   └── best_model_params.csv
│
├── outputs_step4/                               ← 步骤4输出
│   ├── figures/
│   ├── ocei_summary_statistics.csv
│   ├── carbon_model_metrics.csv
│   ├── emission_factor_sensitivity.csv
│   └── eui_ocei_factor_compare_bootstrap_src.csv
│
├── Revised_Paper_EUI_OCEI_Beijing_Hotel.docx    ← 修改后的论文
└── 修改方案与结论对照表.md                       ← 审稿意见对照表
```

## 4. 复现流程

### ⚠️ 重要提示

- 四个 Notebook 必须**按顺序**运行（01 → 02 → 03 → 04）
- 每个 Notebook 依赖前一步的输出文件
- **随机种子已固定为 42**，保证可复现性

### 步骤 1: 参数化仿真数据库构建

**Notebook**: `01_Parametric_Simulation_Database_Construction.ipynb`

**功能**:
- 定义 38 个酒店建筑设计参数空间
- 拉丁超立方采样 (LHS) 生成 20000 个设计方案
- 几何可行性筛选（可用面积比 0.55–0.95）
- 自动生成 EnergyPlus IDF 文件
- 调用 EnergyPlus 进行全年能耗仿真
- 工程后处理：理想负荷 → 终端能耗 + EUI 计算

**运行方式**:
```python
# 在 Notebook Cell 1 中修改配置：
CONFIG = {
    "n_samples": 20000,           # LHS 采样数量
    "run_energyplus": True,       # 是否运行 EnergyPlus（首次运行设为 True）
    "energyplus_exe": r"C:/EnergyPlusV25-2-0/energyplus.exe",
    "expandobjects_exe": r"C:\EnergyPlusV25-2-0\ExpandObjects.exe",
}
```

**调试模式**: 设置 `CONFIG["run_energyplus"] = False` 可仅验证 Python 代码逻辑，跳过仿真。

**预期耗时**:
- IDF 生成: ~2 分钟
- 仿真（20000→4640 样本）: ~16–24 小时（取决于 CPU 核心数）
- 建议先用 `n_samples=100` 验证流程通过后再扩大规模

**输出文件**:
- `data/step1_simulation_dataset.csv` — 主数据集（4640行 × 多列）
- `outputs_step1/simulation_log.csv` — 仿真日志
- `outputs_step1/figures/` — 筛选分析图 + 工程示意图

### 步骤 2: SRC 敏感性分析与变量筛选

**Notebook**: `02_SRC_Sensitivity_and_Variable_Selection.ipynb`

**功能**:
- 方向编码（sin/cos 循环特征）
- 多重共线性诊断（VIF）
- 标准化回归系数 (SRC) 估计（1000 次 Bootstrap）
- SHAP 值非线性验证（XGBoost）
- 变量截断阈值分析（累积 |SRC| + CV R² 曲线）
- 选取前 18 个关键变量

**前置条件**: `data/step1_simulation_dataset.csv` 必须存在

**预期耗时**: ~5–10 分钟

**输出文件**:
- `outputs_step2/src_indices_bootstrap.csv` — SRC 排序结果
- `outputs_step2/vif_table.csv` — VIF 共线性诊断
- `outputs_step2/src_shap_ranking_comparison.csv` — SRC vs SHAP 对比
- `outputs_step2/cv_r2_by_variable_count.csv` — 变量数量 vs 预测能力

### 步骤 3: ML 模型训练与比较

**Notebook**: `03_ML_Model_Training_and_Evaluation.ipynb`

**功能**:
- 17 种回归模型训练与比较
- 超参数调优（GridSearchCV / RandomizedSearchCV, 10-fold）
- 多项式特征扩展（Poly2, Poly3）
- 泛化能力评估（R², RMSE, MAPE, CV 方差, 泛化间隙）
- 非核心变量固定的影响分析

**前置条件**: 步骤 1 和步骤 2 输出文件必须存在

**预期耗时**: ~ 15–30 分钟（含超参数搜索）

**输出文件**:
- `outputs_step3/model_metrics.csv` — 17 模型性能汇总
- `outputs_step3/model_hyperparameters.csv` — 超参数报告
- `outputs_step3/models/*.joblib` — 训练好的最优模型
- `outputs_step3/best_model_params.csv` — 最佳模型参数

### 步骤 4: EUI-OCEI 耦合与碳分析

**Notebook**: `04_EUI_OCEI_Coupling_and_Carbon_Analysis.ipynb`

**功能**:
- 碳排放核算边界定义（电力/天然气/区域供热/区域供冷）
- OCEI 计算与分布分析
- 排放因子敏感性分析（6 情景）
- EUI vs OCEI 排名偏移分析
- EUI 和 OCEI 关键因素对比（SRC）
- 碳强度预测模型训练

**前置条件**: 步骤 1、2、3 输出文件必须存在

**预期耗时**: ~ 5–10 分钟

**输出文件**:
- `outputs_step4/ocei_summary_statistics.csv`
- `outputs_step4/carbon_model_metrics.csv`
- `outputs_step4/emission_factor_sensitivity.csv`
- `outputs_step4/eui_ocei_factor_compare_bootstrap_src.csv`
- `outputs_step4/eui_ocei_rank_shift_summary.csv`

## 5. 预期输出

### 5.1 关键数值结果

| 指标 | 预期值 | 说明 |
|------|--------|------|
| LHS 筛选保留率 | ~23.2% (4640/20000) | 几何可行性筛选 |
| 模拟 EUI 均值 | ~140 kWh/(m²·a) | 与北京实测均值 123 偏差 ~14% |
| SRC 线性模型 R² | ~0.944 | 5-fold CV |
| SRC-SHAP 秩相关 | ~0.89 | Spearman |
| 前18变量累积 |SRC| ~95.8% | |
| 最佳 ML 模型 R² | ~0.997 | Poly3-RidgeCV, test set |
| EUI-OCEI Pearson r | ~0.95 | 强正相关 |
| 排放因子稳健性 | r ∈ [0.938, 0.958] | 6 情景 |

### 5.2 图表清单

| 图号 | 文件 | 内容 |
|------|------|------|
| Fig 1 | `hotel_building_engineering_schematic.png` | 酒店工程示意图（3D + 平面） |
| Fig 2 | `energy_simulation_workflow.png` | 能耗仿真工作流 |
| Fig 3 | `carbon_emission_boundary_diagram.png` | 碳排放核算边界 |
| — | `feasibility_screening_analysis.png` | 可行性筛选分析（3面板） |
| — | `src_vs_shap_comparison.png` | SRC vs SHAP 排序对比 |
| — | `variable_selection_analysis.png` | 变量筛选截断分析 |
| — | `model_test_r2.png` | 模型 R² 比较 |
| — | `emission_factor_sensitivity.png` | 排放因子敏感性 |
| — | `eui_ocei_factor_compare.png` | EUI vs OCEI 因素对比 |

## 6. 常见问题

### Q1: EnergyPlus 运行报错 "No thermal mass"
**A**: 这是警告 (Warning) 而非错误。本研究采用 `Material:NoMass` 简化围护结构建模，适用于参数化研究中对成千上万个设计变体进行热负荷比较。该警告不影响能量平衡计算结果。

### Q2: 仿真速度过慢
**A**: 
1. 降低 `n_samples`（建议首次运行用 5-10 个样本验证）
2. 设置 `run_energyplus=False` 仅验证代码逻辑
3. 每个样本的 EnergyPlus 进程独立运行，可利用多核并行（每核约 500MB 内存）

### Q3: 某些样本仿真失败
**A**: 查看 `outputs_step1/simulation_log.csv` 中的 `success` 和 `has_severe_error` 列。常见原因：
- IDF 几何定义错误（ExpandObjects 预处理失败）
- 极度参数组合导致收敛失败
- 磁盘空间不足

### Q4: 模型 R² 异常高（>0.99）
**A**: Poly3-RidgeCV 的 R²=0.997 是在仿真数据上的**代理模型保真度**（Surrogate Fidelity），不代表对真实建筑的预测精度。论文中已将"预测精度"重构为"代理模型保真度"，并包含完整的 Sim-to-Real Transfer Gap 讨论（第 5.2 节）。

### Q5: 如何使用不同的气象文件？
**A**: 替换 `input/Beijing.epw`，并在 Notebook 01 Cell 1 中更新 `CONFIG["weather_epw"]` 路径。

## 7. 引用声明

### 数据来源

| 数据 | 来源 | 引用 |
|------|------|------|
| 北京气象数据 (Beijing.epw) | EnergyPlus Weather Database, CSWD dataset | https://energyplus.net/weather |
| 北京酒店实测能耗 | Chen, Tan & Berardi (2018) | 56 家北京酒店 |
| 国家建筑能耗约束值 | GB/T 51161-2016 | 中国建筑工业出版社 |
| 碳排放因子 — 电力 | 生态环境部 (2022) | 企业温室气体排放核算方法与报告指南 |
| 碳排放因子 — 天然气 | GB/T 51366-2019 | 建筑碳排放计算标准 |
| 碳排放因子 — 区域供热 | Zheng et al. (2018) | Energy and Buildings 179:1-14 |

### 软件引用

- EnergyPlus™ 25.2.0 — U.S. Department of Energy
- Python 3.11.5 — Python Software Foundation
- scikit-learn 1.5.2 — Pedregosa et al. (2011), JMLR 12:2825-2830
- XGBoost 2.1.1 — Chen & Guestrin (2016), KDD 2016
- LightGBM 4.5.0 — Ke et al. (2017), NeurIPS 2017
- SHAP 0.45.0 — Lundberg & Lee (2017), NeurIPS 2017

### 可复现性声明

本研究采用以下措施确保计算可复现性：
1. **固定随机种子**: 所有随机过程（LHS 采样、train/test 分割、交叉验证、模型初始化）均使用 `random_state=42`
2. **版本锁定**: Python 3.11.5、EnergyPlus 25.2.0，依赖库版本记录于 `requirements.txt`
3. **确定性工作流**: 四步 Notebook 按固定顺序执行，每步输出为下一步输入
4. **配置透明**: 所有参数、假设和核算边界在 Notebook markdown cell 中明确记录
5. **诚实声明**: LLM 辅助输出不可逐字节复现；本研究所有定量结果均来自确定性计算代码，不依赖 LLM 生成

---

**最后更新**: 2026-06-15  
**联系**: 详见论文作者信息  
**许可证**: 本代码包仅供学术评审和研究复现使用

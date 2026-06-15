# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research paper: "基于参数化模拟和机器学习的北京酒店建筑EUI预测及EUI-OCEI耦合研究与分析" (Research and Analysis on EUI Prediction and EUI-OCEI Coupling of Beijing Hotel Buildings Based on Parametric Simulation and Machine Learning).

This is a reproducible computational research pipeline that combines building energy simulation (EnergyPlus) with machine learning to predict hotel building EUI and analyze EUI-OCEI coupling relationships in Beijing.

## Repository Structure

The code package (`Code_Package_EUI_OCEI_Beijing_Hotel.zip`) contains four Jupyter notebooks run sequentially, plus a weather data file:

```
Code_Package_EUI_OCEI_Beijing_Hotel/
├── 01_LHS_EnergyPlus_Pipeline.ipynb    # Parametric sampling & simulation
├── 02_SRC_Sensitivity_Analysis.ipynb   # Sensitivity analysis
├── 03_EUI_Model_Training_and_Comparison.ipynb  # ML model training
├── 04_EUI_OCEI_Coupling_Analysis.ipynb # EUI-OCEI coupling analysis
├── Beijing.epw                         # Weather file for EnergyPlus
└── README.txt
```

## Sequential Pipeline Architecture

Each notebook depends on outputs from the previous step. The pipeline is strictly linear — later notebooks should only be run after prior outputs exist.

### Step 1 — LHS + EnergyPlus (`01_LHS_EnergyPlus_Pipeline.ipynb`)
- Defines hotel building parameter space (geometry, envelope, HVAC, etc.)
- Generates Latin Hypercube Sampling (LHS) samples via `scipy.stats.qmc`
- Screens physically feasible hotel layouts
- Launches EnergyPlus 25.2.0 simulations via `subprocess` calls
- Extracts annual energy results from EnergyPlus SQLite output databases
- Outputs → `outputs_step1/` (sampled parameter datasets, simulation summaries, EUI datasets)

### Step 2 — SRC Sensitivity Analysis (`02_SRC_Sensitivity_Analysis.ipynb`)
- Loads the EUI simulation dataset from Step 1
- Preprocesses variables (including orientation encoding)
- Checks multicollinearity with VIF (Variance Inflation Factor)
- Estimates Standardized Regression Coefficients (SRC) via `sklearn.linear_model.LinearRegression`
- Uses bootstrap resampling for confidence intervals and sign-stability assessment
- Selects key variables for downstream ML modeling
- Outputs → `outputs_step2/` (VIF tables, SRC results, bootstrap CI results, key-variable lists, figures)

### Step 3 — ML Model Training & Comparison (`03_EUI_Model_Training_and_Comparison.ipynb`)
- Loads the reduced dataset (key variables only) from Step 2
- Trains and compares 12+ regression models: LinearRegression, RidgeCV, LassoCV, ElasticNetCV, KNeighborsRegressor, SVR, DecisionTreeRegressor, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, XGBoost (`XGBRegressor`), LightGBM (`LGBMRegressor`), MLPRegressor, PolynomialFeatures + LinearRegression
- Uses `GridSearchCV` and `RandomizedSearchCV` for hyperparameter tuning
- Evaluates with R², cross-validation variance, RMSE, MAPE
- Uses `SimpleImputer` for missing data and `Pipeline` for preprocessing chains
- Outputs → `outputs_step3/` (evaluation tables, CV results, predicted-vs-simulated plots, model summaries)

### Step 4 — EUI-OCEI Coupling (`04_EUI_OCEI_Coupling_Analysis.ipynb`)
- Maps EnergyPlus end-use results to energy carriers (electricity, natural gas, district heating, district cooling)
- Applies carrier-specific carbon emission factors
- Calculates total operational carbon emissions and OCEI
- Analyzes energy-carrier contributions to carbon emissions
- Examines EUI-OCEI correlation (`scipy.stats.pearsonr`)
- Compares EUI-based and OCEI-based building rankings
- Compares SRC patterns between EUI and OCEI
- Outputs → `outputs_step4/`

## Python Environment

- **Python version**: 3.11.5
- **Core dependencies**: `numpy`, `pandas`, `scipy`, `matplotlib`, `scikit-learn`, `statsmodels`
- **ML libraries**: `xgboost`, `lightgbm`, `seaborn`, `joblib`
- **External tool**: EnergyPlus 25.2.0 (`C:/EnergyPlusV25-2-0/energyplus.exe`)

## ⚠️ 本地运行要求（强制）

**所有代码必须在用户本地Python环境和Jupyter Notebook中实际运行验证，不得仅在理论层面分析代码正确性。**

- 每次编写或修改代码后，必须在本地环境执行以确认无语法错误、无缺失依赖、无运行时异常。
- 运行前检查：Python 3.11.5, 所需库是否安装, EnergyPlus路径是否正确。
- 若本地运行报错，优先修复错误后重新运行，直到通过为止。

## Running the Pipeline

1. Extract the zip: `unzip Code_Package_EUI_OCEI_Beijing_Hotel.zip`
2. Run notebooks in order: 01 → 02 → 03 → 04
3. Before running Step 1, verify the EnergyPlus executable path in the notebook matches the local installation
4. Ensure `Beijing.epw` is in the working directory alongside the notebooks
5. Output directories (`outputs_step1/` through `outputs_step4/`) are auto-created by notebook code
6. **调试模式**: 设置 CONFIG["run_energyplus"] = False 可跳过EnergyPlus仿真仅验证Python代码逻辑

## Key Technical Details

- **Random seeds** are used throughout for reproducibility (LHS sampling, train/test splits, CV folds, model initialization)
- **EnergyPlus outputs** are read via `sqlite3` from the simulation `.sql` database files
- **Orientation encoding**: categorical building orientation is encoded numerically before regression
- **Multicollinearity screening**: VIF threshold applied before SRC estimation
- **Bootstrap resampling** (Step 2): quantifies uncertainty in SRC estimates and assesses sign stability
- **Cross-validation**: `KFold` cross-validation used in both SRC estimation and ML model evaluation

---

## 🤖 AI 学术研究技能（Academic Research Skills）

本项目已集成 GitHub 上星标最高的两套 Claude Code 学术研究技能包，覆盖从文献调研到论文发表的全流程。

### 一、Academic Research Skills（~11,900+ ⭐）— 学术研究全流程

**来源**: [Imbad0202/academic-research-skills](https://github.com/Imbad0202/academic-research-skills) | **版本**: v3.12.0 | **许可证**: CC BY-NC 4.0

核心理念：**"AI是你的副驾驶，不是飞行员"**（AI is your copilot, not the pilot.）

#### 四大核心技能

| 技能目录 | 功能 | Agent数量 | 典型用途 |
|----------|------|-----------|----------|
| `.claude/skills/deep-research/` | 深度文献调研 | 13个Agent | 文献综述、PRISMA系统评价、研究问题构建 |
| `.claude/skills/academic-paper/` | 论文学术写作 | 12个Agent | 大纲设计→论证→草稿→双语摘要→图表→引用格式 |
| `.claude/skills/academic-paper-reviewer/` | 论文评审 | 7个Agent | 模拟真实期刊评审（主编+3位审稿人+魔鬼代言人），0-100量化评分 |
| `.claude/skills/academic-pipeline/` | 全流程编排 | 流程编排器 | 10阶段流水线，含完整性闸门检查点 |

#### 可用斜杠命令（`.claude/commands/`）

| 命令 | 功能 |
|------|------|
| `/ars-plan` | 通过苏格拉底式对话规划论文结构 |
| `/ars-full` | 启动完整研究流程 |
| `/ars-lit-review` | 文献综述 |
| `/ars-outline` | 论文大纲设计 |
| `/ars-abstract` | 撰写摘要 |
| `/ars-citation-check` | 引用核验 |
| `/ars-format-convert` | 引用格式转换（APA/IEEE等） |
| `/ars-disclosure` | 生成AI使用声明 |
| `/ars-reviewer` | 启动论文评审 |
| `/ars-revision` | 论文修订 |
| `/ars-revision-coach` | 审稿意见解析与修订路线图 |

#### 关键特性

- **引用核验**: 调用 Semantic Scholar API 验证每篇引用，使用 Levenshtein 相似度算法模糊匹配（阈值 ≥0.70）
- **完整性闸门**: Stage 2.5 和 4.5 设有不可跳过的 7 项 AI 失败模式检查清单
- **反谄媚协议**: Devil's Advocate 的反驳评分 1-5，低于 4 分不允许写作团队让步
- **三层数据隔离**: 原始输入 / 验证产物 / 评分标准严格分离
- **风格校准**: 学习研究者过往作品的写作风格，避免 AI 味
- **L3 声明忠实度门禁** (v3.8): 可选审计 Pass 对每处引用索取原文并判断声明是否真正得到支持
- **实验溯源摄入** (#260): Material Passport 可记录外部实验结果，由完整性闸门审计

#### 使用示例

```
"Guide my research on hotel building energy performance"
"Do a systematic review on building energy prediction with PRISMA"
"Write a paper on EUI-OCEI coupling analysis for Beijing hotels"
"Review this paper" (then provide the paper)
"Parse these reviewer comments into a revision roadmap"
"Check citations in my manuscript"
```

---

### 二、Nature Skills（~13,000+ ⭐）— 期刊级学术技能

**来源**: [Yuan1z0825/nature-skills](https://github.com/Yuan1z0825/nature-skills) | **作者**: 袁一哲

#### 11 项技能

| 技能目录 | 功能 | 说明 |
|----------|------|------|
| `.claude/skills/nature-reader/` | 论文精读 | PDF全文双语翻译，图表感知，源码级Markdown输出 |
| `.claude/skills/nature-writing/` | 论文写作 | Nature/Cell/Science级别学术写作 |
| `.claude/skills/nature-polishing/` | 论文润色 | 语言精炼、学术风格强化 |
| `.claude/skills/nature-reviewer/` | 论文评审 | 模拟期刊同行评审 |
| `.claude/skills/nature-citation/` | 引用管理 | 引用格式转换与验证 |
| `.claude/skills/nature-figure/` | 图表优化 | 学术图表设计与美化 |
| `.claude/skills/nature-data/` | 数据处理 | 科研数据清洗与分析 |
| `.claude/skills/nature-response/` | 审稿回复 | 点对点回复审稿意见 |
| `.claude/skills/nature-paper2ppt/` | 论文转PPT | 论文内容转学术演示文稿 |
| `.claude/skills/nature-academic-search/` | 学术搜索 | 文献检索与获取 |
| `.claude/skills/nature-paper-to-patent/` | 论文转专利 | 学术成果转专利申请 |

#### 使用示例

```
"Translate this paper into a full markdown reader"  → nature-reader
"Polish this paragraph to Nature-level quality"     → nature-polishing
"Convert this paper to a Chinese journal-club PPT"  → nature-paper2ppt
"Generate a point-by-point response to these reviewer comments" → nature-response
```

---

### 三、技能文件结构

```
.claude/
├── skills/
│   ├── deep-research/          # ARS: 深度调研 (52 files)
│   ├── academic-paper/         # ARS: 论文写作 (61 files)
│   ├── academic-paper-reviewer/# ARS: 论文评审 (26 files)
│   ├── academic-pipeline/      # ARS: 全流程编排 (30 files)
│   ├── shared/                 # ARS: 共享支持文件 (54 files)
│   ├── nature-reader/          # Nature: 论文精读 (15 files)
│   ├── nature-writing/         # Nature: 论文写作 (65 files)
│   ├── nature-polishing/       # Nature: 论文润色 (29 files)
│   ├── nature-reviewer/        # Nature: 论文评审 (9 files)
│   ├── nature-citation/        # Nature: 引用管理 (12 files)
│   ├── nature-figure/          # Nature: 图表优化 (98 files)
│   ├── nature-data/            # Nature: 数据处理 (13 files)
│   ├── nature-response/        # Nature: 审稿回复 (24 files)
│   ├── nature-paper2ppt/       # Nature: 论文转PPT (16 files)
│   ├── nature-academic-search/ # Nature: 学术搜索 (45 files)
│   ├── nature-paper-to-patent/ # Nature: 论文转专利 (34 files)
│   └── _shared_nature/         # Nature: 共享支持文件 (6 files)
└── commands/                   # ARS斜杠命令 (14个)
    ├── ars-plan.md
    ├── ars-full.md
    ├── ars-lit-review.md
    ├── ars-abstract.md
    ├── ars-outline.md
    ├── ars-reviewer.md
    ├── ars-revision.md
    ├── ars-citation-check.md
    └── ... (更多)
```

---

### 四、使用建议

1. **文献调研阶段**: 使用 `/ars-lit-review` 或 `deep-research` 技能进行系统性文献综述
2. **方法论设计**: 使用 `/ars-plan` 苏格拉底式对话细化研究设计
3. **论文初稿**: 使用 `academic-paper` 或 `nature-writing` 技能辅助写作
4. **论文润色**: 使用 `nature-polishing` 提升语言质量
5. **自我评审**: 使用 `academic-paper-reviewer` 或 `nature-reviewer` 在投稿前进行模拟评审
6. **引用核验**: 使用 `/ars-citation-check` 确保引用真实可靠
7. **审稿回复**: 使用 `nature-response` 逐条回复审稿意见

> ⚠️ **重要提醒**: 所有 AI 辅助技能均为"副驾驶"模式——AI 处理查找引用、格式化、核验数据等重复劳动，研究者专注于定义问题、选择方法、解读数据和撰写核心论点。完整跑一篇 1.5 万字论文的全流程大约花费 $4-6 美元 API 费用。

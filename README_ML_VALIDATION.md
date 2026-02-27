## 基于 IDR 数据库的机器学习与外部验证项目

这一部分以 `IDR-LLPS` 金标准数据库为基础，构建多种机器学习模型，并在内部测试集与外部数据集（特别是 TAIR 拟南芥蛋白）上进行系统验证。

### 1. 目录结构一览

- **建模与训练脚本**
  - `src/modeling/`  
    - `train_ml_baselines.py`：多种传统 ML 基线（Logistic Regression、SVM、RF、GBDT、kNN、MLP 等）。  
    - `train_rf_tuned.py`：针对 NARDINI90 的 RandomForest 超参数搜索与评估。  
    - `train_deep_baseline.py`：简单深度学习基线（MLPClassifier 风格）。  
    - `export_feature_importances_rf.py`：RF 特征重要性导出。  
    - `export_feature_weights.py`：线性模型系数导出。  
    - `build_balanced_dataset.py`：基于混合构建的平衡训练集（早期版本）。  
    - `build_balanced_dataset_idr_only.py`：**基于 `samples_idr_only` / `nardini90_features_idr_only` 的纯 IDR 平衡训练集构建。**

- **训练结果与可视化**
  - `ml_dl/results/`：早期基于混合构建（`balanced_nardini90_*`）的训练结果。  
  - `ml_dl/results/IDR_TRAIN/`：在 **IDR-only** 训练集上重训 RF 后的 ROC/PR 图等。  
  - `reports/feature_importances_nardini90_rf.*`：特征重要性表和可视化。  
  - `reports/feature_scores_nardini90.md`：组合特征评分与解释。

- **外部验证与 TAIR 相关脚本**
  - `data/validation/`  
    - `tair_sequences.tsv`：TAIR LLPS 表中蛋白的全长序列（tair_id, sequence[, protein_accession]）。  
    - `tair_llps_labels.tsv`：TAIR 外部金标准标签（tair_id, llps_label, location 等）。  
    - `tair_nardini90.tsv`：基于全长序列的 TAIR NARDINI90 特征。  
    - `tair_nardini90_labeled.tsv`：与 `tair_llps_labels.tsv` 合并后的全长特征验证集。  
    - `tair_idr_regions.tsv`：通过 MobiDB（DisProt/IDEAL + mobidb_lite）获得的 TAIR IDR 区段。  
    - `tair_idr_nardini90.tsv`：基于 `idr_seq` 的 TAIR IDR‑level NARDINI90 特征。
  - `src/validation/`  
    - `export_tair_fasta.py`：从 `tair_sequences.tsv` 导出 FASTA（供外部预测工具使用）。  
    - `fetch_tair_sequences.py` / `build_tair_sequences_from_proteome.py`：从 UniProt / 本地蛋白组获取 TAIR 序列。  
    - `compute_tair_nardini90.py`：对 TAIR 全长序列计算 NARDINI90，并用训练集分布做 z-score。  
    - `compute_tair_idr_nardini90.py`：对 `tair_idr_regions.tsv` 中的每段 IDR 单独计算 NARDINI90。  
    - `compute_tair_idr_concat_nardini90.py`：将同一蛋白的多段 IDR 拼接后计算一份 NARDINI90。  
    - `build_tair_validation_set.py`：把 TAIR 标签与任一特征表（全长或 IDR）合并成带 `llps_label` 的验证集。  
    - `eval_rf_on_tair.py`：在 TAIR 全长特征表上评估 RF（full-length 验证）。  
    - `eval_rf_on_tair_idr_max.py`：在 TAIR IDR‑level 特征表上评估 RF，并用「任一 IDR 段超过阈值 → 蛋白判 positive」规则聚合到蛋白层面。  
    - `plot_rf_tair_results.py`：给定 preds.tsv + metrics.json，画 TAIR 混淆矩阵、ROC、PR 曲线。

- **web 应用与可视化前端**
  - `web/`：基于 FastAPI + Jinja2 的 Web 应用，支持：
    - 浏览 IDR‑LLPS 数据库条目；  
    - 查看单个样本的特征与模型预测；  
    - 下载训练集 / 验证集 / 报告等。

### 2. 典型训练场景

#### 2.1 早期「混合构建」训练（原始方案）

使用全体 `balanced_nardini90_train/test.tsv`（构建不一定完全覆盖 IDR）：

```bash
python -m src.modeling.build_balanced_dataset \
  --sqlite-file data/processed/idr_llps.sqlite \
  --output-dir data/processed

python -m src.modeling.train_rf_tuned \
  --train-file data/processed/balanced_nardini90_train.tsv \
  --test-file  data/processed/balanced_nardini90_test.tsv \
  --metrics-file reports/rf_tuned_metrics.json
```

该方案用于早期探索，理解 NARDINI90 对「IDR 驱动 LLPS vs 负样本构建」的区分能力。

#### 2.2 纯 IDR 训练集（推荐当前主方案）

在确保 **正负样本均为 IDR 构建** 的前提下重新训练：

```bash
# 1) 从 samples.tsv 中筛出 IDR-only 样本（已完成可跳过）
python -m src.modeling.build_idr_only_samples \
  --samples-file data/processed/samples.tsv \
  --output-file data/processed/samples_idr_only.tsv \
  --overlap-threshold 0.7 \
  --min-idr-len 15

# 2) 对 IDR-only 样本计算 NARDINI90
python -m src.export.compute_nardini90_features \
  --samples-file data/processed/samples_idr_only.tsv \
  --sqlite-file data/processed/idr_llps_idr_only.sqlite \
  --output-tsv  data/processed/nardini90_features_idr_only.tsv \
  --labels idr_pos,neg

# 3) 构建平衡 IDR-only 训练/测试集
python -m src.modeling.build_balanced_dataset_idr_only \
  --features-tsv data/processed/nardini90_features_idr_only.tsv \
  --output-dir  data/processed

# 4) 在 IDR-only 训练集上调参 RF
python -m src.modeling.train_rf_tuned \
  --train-file  data/processed/balanced_nardini90_idr_train.tsv \
  --test-file   data/processed/balanced_nardini90_idr_test.tsv \
  --metrics-file reports/rf_tuned_idr_only_metrics.json
```

产出：

- `balanced_nardini90_idr_train/test.tsv`：严格 IDR‑only 的平衡训练/测试集。  
- `reports/rf_tuned_idr_only_metrics.json`：RF 在 IDR-only 训练集上的 CV + 测试性能，ROC‑AUC ≈ 0.87，F1 ≈ 0.83（当前较优方案）。
- `ml_dl/results/IDR_TRAIN/`：对应的 ROC/PR 可视化。

### 3. TAIR 拟南芥外部验证场景

目前支持三种视角的 TAIR 外部验证，均复用同一套训练好的 RF，只是特征与聚合规则不同。

#### 3.1 基于完整蛋白 NARDINI90（full-length）

```bash
python -m src.validation.compute_tair_nardini90        # 已完成可跳过
python -m src.validation.build_tair_validation_set \
  --labels-tsv  data/validation/tair_llps_labels.tsv \
  --features-tsv data/validation/tair_nardini90.tsv \
  --output-tsv  data/validation/tair_nardini90_labeled.tsv

python -m src.validation.eval_rf_on_tair \
  --train-file  data/processed/balanced_nardini90_idr_train.tsv \
  --tair-file   data/validation/tair_nardini90_labeled.tsv \
  --metrics-file ml_dl/validation/FL/rf_tair_fl_idronly_metrics.json \
  --preds-file   ml_dl/validation/FL/rf_tair_fl_idronly_preds.tsv

python -m src.validation.plot_rf_tair_results \
  --preds-file   ml_dl/validation/FL/rf_tair_fl_idronly_preds.tsv \
  --metrics-file ml_dl/validation/FL/rf_tair_fl_idronly_metrics.json \
  --out-dir      ml_dl/validation/FL_IDRONLY
```

#### 3.2 基于单条 IDR 片段（IDR-level）

```bash
python -m src.validation.compute_tair_idr_nardini90        # 每段 IDR 一行
python -m src.validation.build_tair_validation_set \
  --labels-tsv  data/validation/tair_llps_labels.tsv \
  --features-tsv data/validation/tair_idr_nardini90.tsv \
  --output-tsv  data/validation/tair_idr_nardini90_labeled.tsv

python -m src.validation.eval_rf_on_tair_idr_max \
  --train-file      data/processed/balanced_nardini90_idr_train.tsv \
  --tair-idr-file   data/validation/tair_idr_nardini90_labeled.tsv \
  --metrics-file    ml_dl/validation/IDR/rf_tair_idr_max_idronly_metrics.json \
  --segment-preds-file ml_dl/validation/IDR/rf_tair_idr_segment_preds_idronly.tsv \
  --protein-preds-file ml_dl/validation/IDR/rf_tair_idr_max_idronly_preds.tsv

python -m src.validation.plot_rf_tair_results \
  --preds-file   ml_dl/validation/IDR/rf_tair_idr_max_idronly_preds.tsv \
  --metrics-file ml_dl/validation/IDR/rf_tair_idr_max_idronly_metrics.json \
  --out-dir      ml_dl/validation/IDR_MAX_IDRONLY
```

这里使用的蛋白级规则是：

> 若某个 TAIR 蛋白的任一 IDR 片段预测概率 `prob_rf >= 阈值（默认 0.5）`，则将该蛋白判定为 LLPS‑positive。

#### 3.3 基于拼接 IDR 的整段特征（可选探索）

```bash
python -m src.validation.compute_tair_idr_concat_nardini90
python -m src.validation.build_tair_validation_set \
  --labels-tsv  data/validation/tair_llps_labels.tsv \
  --features-tsv data/validation/tair_idr_concat_nardini90.tsv \
  --output-tsv  data/validation/tair_idr_concat_nardini90_labeled.tsv

python -m src.validation.eval_rf_on_tair \
  --train-file  data/processed/balanced_nardini90_idr_train.tsv \
  --tair-file   data/validation/tair_idr_concat_nardini90_labeled.tsv \
  --metrics-file ml_dl/validation/IDR/rf_tair_idr_concat_validation_metrics.json \
  --preds-file   ml_dl/validation/IDR/rf_tair_idr_concat_validation_preds.tsv

python -m src.validation.plot_rf_tair_results \
  --preds-file   ml_dl/validation/IDR/rf_tair_idr_concat_validation_preds.tsv \
  --metrics-file ml_dl/validation/IDR/rf_tair_idr_concat_validation_metrics.json \
  --out-dir      ml_dl/validation/IDR
```

这一视角更偏向于理解「将所有 IDR 拉直之后，整体 IDR 组成和 pattern 与 LLPS 正负类有何差异」，不一定适合作为主评估指标。

---

通过上述组织方式，当前仓库可以自然地分成两个相互关联但职责清晰的「子项目」：

1. `README_IDR_DATABASE.md` 所描述的 **IDR-LLPS 金标准数据库**（数据与特征层）。  
2. 本文件描述的 **基于 IDR 数据库的机器学习与外部验证**（模型与应用层）。

后续若需要拆分为两个独立仓库，可以直接以这两份 README 为根，分别拷贝对应的目录和脚本。 


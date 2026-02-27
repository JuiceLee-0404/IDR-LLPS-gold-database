## IDR-LLPS 金标准数据库项目

本项目的这一部分专注于构建一个 **只包含 IDR 构建的 LLPS 金标准数据库**，为后续的机器学习与外部验证提供统一、可追溯的数据基础。

### 1. 目录与主要文件

- **配置与元数据**
  - `configs/dataset.yaml`：数据源、IDR 规则、正负样本规则等全局配置。
  - `data/processed/run_metadata.json`：一次完整构建的元信息（时间戳、版本等）。

- **原始与中间数据**
  - `data/raw/`：原始下载（DrLLPS、DisProt 等）的原始文件。
  - `data/interim/positives_idr_driven.tsv`：经过规则筛选的候选正样本（IDR 驱动）。
  - `data/interim/negatives.tsv`：候选负样本（实验负 + 伪负）。

- **标准化后的金标准表**
  - `data/processed/evidence.tsv`：每条实验证据（含 IDR 区间、构建区间、是否 LLPS 等）。
  - `data/processed/idr_regions.tsv`：按蛋白聚合的全局 IDR 区段。
  - `data/processed/samples.tsv`：面向机器学习的「构建级样本表」，包含：
    - `sample_id`, `region_start`, `region_end`, `region_sequence`
    - `idr_start`, `idr_end`, `idr_source`
    - `label` (`idr_pos` / `neg`), `label_source` (`experimental`, `exp_neg`, `pseudo_neg`)
  - `data/processed/samples_idr_only.tsv`：**IDR-only 样本表**  
    - 通过 `src/modeling/build_idr_only_samples.py` 从 `samples.tsv` 中筛选：
      - 有有效的 `idr_start`/`idr_end`，IDR 长度 ≥ 15；
      - 实验构建区段覆盖 IDR 长度的比例 ≥ 0.7 (`idr_overlap_frac ≥ 0.7`)；
      - 正样本：`label = 'idr_pos'` 且 `label_source = 'experimental'`；
      - 负样本：`label = 'neg'` 且 `label_source = 'exp_neg'`。

- **数据库与视图**
  - `data/processed/idr_llps.sqlite`：主 SQLite 数据库，包含 `samples`, `evidence`, `idr_regions`, `nardini90_features` 等表。
  - `docs/sql_views.sql`：常用统计视图（标签分布、IDR 覆盖率等）。
  - `data/processed/idr_llps_idr_only.sqlite`：针对 `samples_idr_only` 计算的 IDR-only `nardini90_features` 专用数据库。

### 2. 构建 IDR 金标准数据库的推荐流程

在项目根目录下（确保已安装依赖）：

```bash
python -m src.ingest.fetch_public_dbs        # 下载 / 规范化公共 LLPS 与 IDR 数据
python -m src.common.build_dataset_pipeline  # 按 configs/dataset.yaml 构建 evidence/samples/splits
```

完成后可以生成统计报告与可视化：

```bash
python -m src.analysis.visualize_idr_stats \
  --config configs/dataset.yaml \
  --sqlite-file data/processed/idr_llps.sqlite \
  --output-dir reports
```

若希望明确得到「全部为 IDR 构建」的样本子集，可运行：

```bash
python -m src.modeling.build_idr_only_samples \
  --samples-file data/processed/samples.tsv \
  --output-file data/processed/samples_idr_only.tsv \
  --overlap-threshold 0.7 \
  --min-idr-len 15
```

此时：

- `samples.tsv` 表示「混合构建」（满足 IDR 规则的正样本 + 更宽泛的负样本）；  
- `samples_idr_only.tsv` 则是「**严格 IDR 构建的正负样本金标准库**」。

### 3. 面向特征计算的输出

在数据库层面，NARDINI90 特征的计算与存储分为两类：

- **混合构建特征**（原始主干，供早期模型使用）：
  - 通过 `src/export/compute_nardini90_features.py` 对 `samples.tsv` 中的 `region_sequence` 计算特征；
  - 结果写入：
    - `data/processed/nardini90_features.tsv`
    - `idr_llps.sqlite::nardini90_features`

- **IDR-only 特征**（本项目中后期推荐使用的主方案）：
  - 对 `data/processed/samples_idr_only.tsv` 运行同一脚本：
    ```bash
    python -m src.export.compute_nardini90_features \
      --samples-file data/processed/samples_idr_only.tsv \
      --sqlite-file data/processed/idr_llps_idr_only.sqlite \
      --output-tsv  data/processed/nardini90_features_idr_only.tsv \
      --labels idr_pos,neg
    ```
  - 得到：
    - `data/processed/nardini90_features_idr_only.tsv`：每条 IDR 构建一行；
    - `idr_llps_idr_only.sqlite::nardini90_features`：与之等价的数据库表。

这两份特征表共同构成了 **「IDR-LLPS 金标准数据库」的特征层**，下游任意机器学习或验证项目都可以基于它们构建训练集或外部测试集。


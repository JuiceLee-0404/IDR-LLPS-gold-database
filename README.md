# 面向深度学习的 IDR 驱动 LLPS 数据库流水线

本项目用于构建“**无序区（IDR）驱动相分离**”的训练数据集，目标是服务二分类或排序模型：
- 正样本：可由 IDR 驱动（或显著参与驱动）相分离；
- 负样本：不能驱动相分离（实验负样本 + 高置信伪负样本）；
- 支持全物种、证据可追溯、可复现构建。

---

## 一、详细实施计划（增强版）

### 阶段 1：规则与字段冻结
1. 在 `configs/dataset.yaml` 中定义统一字段、主键、标签体系和阈值。  
2. 固化证据分级标准（A/B/C）与 IDR 过滤规则（如最小长度、重叠阈值）。  
3. 约束输出文件结构，确保后续脚本都按同一 schema 读写。

### 阶段 2：公开数据库抓取与原始落盘
1. 通过 `src/ingest/fetch_public_dbs.py` 抓取 LLPS/IDR 相关公开资源页面。  
2. 每个来源落到 `data/raw/<source>/`，并在 `raw_manifest.tsv` 记录状态。  
3. 网络异常或证书异常时记录失败原因；HTTPS 场景支持受控回退并记录 `ssl_mode`。
4. 通过 `src/ingest/download_structured_sources.py` 直接下载可程序化处理的数据导出（当前已接入 DrLLPS、DisProt、PhaSePro、llpsdatasets）。

### 阶段 3：证据层（Evidence Layer）构建
1. 先通过 `src/curation/normalize_structured_exports.py` 把结构化导出转换为各来源 `normalized.tsv`。  
2. 读取各来源统一格式 `normalized.tsv`。  
3. 对缺失 IDR 边界的 LLPS 证据按 accession 自动关联可用 IDR 注释。  
4. 做 accession/isoform 归一化，汇总为 `data/processed/evidence.tsv`。  
5. 抽取 IDR 边界表 `data/processed/idr_regions.tsv`，用于后续筛选与伪负样本生成。

### 阶段 4：IDR 驱动正样本筛选
1. 从证据层筛选 `llps_observed=True` 且满足 IDR 驱动条件的条目。  
2. 条件包括：IDR 支持证据、证据等级、IDR 重叠比例阈值等。  
3. 输出 `data/interim/positives_idr_driven.tsv`，冲突证据写入 `evidence_conflict.tsv`。

### 阶段 5：负样本构建（混合策略）
1. 实验负样本（真负）：来自明确“未观察到 LLPS”的构建体；当前白名单来源包括 llpsdb、phasepro、drllps、**llpsdatasets**（PPMC 标准化负样本，见 [llpsdatasets.ppmclab.com](https://llpsdatasets.ppmclab.com/)）。
2. 伪负样本：来自 IDR 背景池，排除已知 LLPS 蛋白与高风险关键词条目。
3. 使用 k-mer Jaccard 近似同源过滤，减少与正样本过近的伪负样本。
4. 输出 `data/interim/negatives.tsv`，并用 `label_source` 标记来源（`exp_neg` / `pseudo_neg`）。

### 阶段 6：去重、同源控制与数据划分
1. 按统一主键合并证据，生成样本级表。  
2. 基于序列相似性聚类后再划分 train/val/test，降低同源泄漏。  
3. 同时导出两套版本：  
   - `strict`：仅实验正负样本；  
   - `hybrid`：实验负样本 + 伪负样本（默认）。

### 阶段 7：文档与可复现发布
1. 完善 README、字段字典与运行命令。  
2. 输出版本元数据（构建时间、样本规模、版本号）。  
3. 确保任一样本可回溯到原始来源与文献证据。

---

## 二、项目结构

- `configs/dataset.yaml`：字段 schema、判定规则、标签与划分参数。  
- `src/ingest/fetch_public_dbs.py`：抓取公开数据源并生成原始抓取清单。  
- `src/ingest/download_structured_sources.py`：下载结构化数据导出（DrLLPS/DisProt）。  
- `src/curation/normalize_structured_exports.py`：将结构化导出映射为统一 `normalized.tsv`。  
- `src/curation/build_evidence_layer.py`：构建统一证据层与 IDR 边界表。  
- `src/curation/filter_idr_driven.py`：筛选 IDR 驱动正样本并处理冲突。  
- `src/curation/build_negatives.py`：构建实验负样本与伪负样本。  
- `src/export/make_ml_splits.py`：生成模型样本表与 train/val/test 划分。  
- `src/export/build_database_sqlite.py`：将标准化结果落到 SQLite 数据库，便于查询与复用。  
- `src/export/compute_nardini90_features.py`：为样本区段计算 NARDINI90（54 组成 + 36 patterning）并落库。  
- `docs/data_dictionary.md`：输出字段解释。

---

## 三、安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 四、运行方式

### 4.1 构建数据库（一次性 / 按需）

在仓库根目录执行：

```bash
python -m src.ingest.fetch_public_dbs --config configs/dataset.yaml
python -m src.ingest.download_structured_sources --config configs/dataset.yaml
python -m src.curation.normalize_structured_exports --config configs/dataset.yaml
python -m src.curation.build_evidence_layer --config configs/dataset.yaml
python -m src.curation.filter_idr_driven --config configs/dataset.yaml
python -m src.curation.build_negatives --config configs/dataset.yaml
python -m src.export.make_ml_splits --config configs/dataset.yaml
python -m src.export.build_database_sqlite --config configs/dataset.yaml
python -m src.export.compute_nardini90_features --config configs/dataset.yaml --num-scrambles 30
```

执行完成后，核心数据库文件为：

- `data/processed/idr_llps.sqlite`
- `data/processed/nardini90_features.tsv`

### 4.2 运行 Web 应用（浏览与分享数据库）

本项目提供一个基于 FastAPI 的简单 Web 前端，方便在浏览器中查看样本、图表与机器学习结果。

在仓库根目录执行（建议在已激活的虚拟环境中）：

```bash
uvicorn web.main:app --reload
```

然后在浏览器打开：

- `http://127.0.0.1:8000/`：首页，提供入口导航；
- `/samples`：按标签筛选与分页浏览样本；
- `/samples/{sample_id}`：查看单个样本详情与 NARDINI90 特征；
- `/plots`：查看 PCA/t-SNE、混淆矩阵、ROC/PR 曲线等图像；
- `/downloads`：下载 SQLite、TSV 和机器学习相关输出文件。

如需通过 JSON 访问数据，可使用：

- `/api/samples`：分页列出样本（支持 `label`、`label_source`、`taxon_group`、`limit`、`offset` 参数）；
- `/api/samples/{sample_id}`：返回单个样本及其 NARDINI90 特征；
- `/api/stats`：返回样本数量、标签分布等整体统计信息。

---

## 五、输入规范（证据层）

`build_evidence_layer.py` 会读取以下文件（存在则读取，不存在则跳过）：

- `data/raw/llpsdb/normalized.tsv`
- `data/raw/phasepro/normalized.tsv`
- `data/raw/drllps/normalized.tsv`
- `data/raw/disprot/normalized.tsv`
- `data/raw/mobidb/normalized.tsv`

每个 `normalized.tsv` 推荐包含如下字段：

- `evidence_id`
- `source_entry_id`
- `protein_accession`（或 `accession` / `uniprot_id`）
- `isoform`
- `gene_name`
- `organism`
- `taxon_id`
- `sequence`
- `construct_context`
- `region_start`
- `region_end`
- `idr_start`
- `idr_end`
- `idr_source`
- `llps_observed`
- `idr_driver_supported`
- `evidence_grade`
- `condition_text`
- `pmid`
- `db_version`
- `download_date`

---

## 六、输出文件

- `data/raw/raw_manifest.tsv`
- `data/processed/evidence.tsv`
- `data/processed/idr_regions.tsv`
- `data/interim/positives_idr_driven.tsv`
- `data/interim/negatives.tsv`
- `data/processed/samples.tsv`（默认 hybrid 视图）
- `data/processed/samples_strict.tsv`
- `data/processed/samples_hybrid.tsv`
- `data/processed/splits.tsv`
- `data/processed/run_metadata.json`

---

## 七、标签策略

- 正样本 `idr_pos`：仅保留 IDR 驱动相关证据（满足证据等级与 IDR 条件）。  
- 负样本 `neg`：  
  - `exp_neg`：实验明确不发生 LLPS；  
  - `pseudo_neg`：高置信伪负样本（通过排除与相似度过滤得到）。

---

## 八、验收标准（建议）

1. 样本均可追溯：每条记录可定位来源库与文献信息。  
2. 规则可审计：正样本满足 IDR 驱动规则，冲突样本单独记录。  
3. 训练可复现：同配置重复运行可得到一致结构输出。  
4. 版本可管理：保留 `run_metadata.json` 与配置快照。  

---

## 九、当前实现说明

- 当前实现已打通完整流水线框架。  
- 为了维护性，源数据库的“网页/原始导出 -> normalized.tsv”解析逻辑与核心流水线解耦。  
- 当前已内置 DrLLPS 与 DisProt 的结构化下载与标准化映射。  
- 生产使用建议继续补充 LLPSDB、PhaSePro、MobiDB 的专用解析器，以进一步提升正样本覆盖与负样本可信度。

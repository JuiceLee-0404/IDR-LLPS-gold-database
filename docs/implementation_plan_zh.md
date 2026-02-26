# IDR 驱动 LLPS 数据库详细实施计划

## 1. 目标定义

构建一个用于深度学习训练的区段级数据库，满足以下要求：
- 关注对象是 IDR（无序区）；
- 正样本必须是“IDR 驱动或显著参与驱动”相分离；
- 负样本同时包含实验负样本和高置信伪负样本；
- 数据可追溯、可审计、可复现。

## 2. 数据源范围

- LLPS 证据主源：`LLPSDB`、`PhaSePro`、`DrLLPS`
- IDR 注释主源：`DisProt`、`MobiDB`
- 背景与补充：`UniProt/Swiss-Prot`（按需扩展）

## 3. 数据模型与主键

样本主键：
- `protein_accession + isoform + region_start + region_end + construct_context`

核心表：
- 证据层：`data/processed/evidence.tsv`
- IDR 边界层：`data/processed/idr_regions.tsv`
- 样本层：`data/processed/samples.tsv`
- 划分层：`data/processed/splits.tsv`

## 4. 分阶段执行细化

### 阶段 A：抓取与原始存档
输出：
- `data/raw/<source>/*`
- `data/raw/raw_manifest.tsv`

检查点：
- 每个来源是否抓取成功；
- 失败原因是否明确记录；
- `download_date`、`ssl_mode` 是否完整。

### 阶段 B：标准化映射（source-specific parser）
任务：
- 为每个来源编写解析器，将原始导出统一映射为 `normalized.tsv`。

统一字段：
- `protein_accession`, `region_start`, `region_end`, `idr_start`, `idr_end`,
  `llps_observed`, `idr_driver_supported`, `evidence_grade`, `pmid` 等。

检查点：
- accession/isoform 格式统一；
- 坐标统一为 1-based 且闭区间；
- 布尔字段统一为 `True/False`。

### 阶段 C：证据层聚合
任务：
- 融合同源字段命名差异；
- 生成 evidence id；
- 去除明显无效记录（缺 accession、越界坐标等）。

检查点：
- `evidence.tsv` 可被后续脚本直接消费；
- `idr_regions.tsv` 不含重复边界。

### 阶段 D：正样本筛选（IDR 驱动）
规则：
- `llps_observed=True`；
- 证据等级在允许集合（默认 A/B）；
- 存在有效 IDR；
- 满足 `idr_driver_supported=True` 或 IDR 重叠率阈值。

检查点：
- 产生 `positives_idr_driven.tsv`；
- 与负证据冲突样本写入 `evidence_conflict.tsv`。

### 阶段 E：负样本构建
实验负样本：
- 直接来自 `llps_observed=False` 且区段与 IDR 相关记录。

伪负样本：
- 来自背景 IDR；
- 排除已知 LLPS 相关蛋白；
- 排除高风险关键词；
- 排除与正样本过近序列（k-mer Jaccard 阈值）。

检查点：
- `label_source` 明确区分 `exp_neg` 与 `pseudo_neg`；
- 负样本规模与质量可平衡。

### 阶段 F：去重、同源控制与划分
任务：
- 按主键合并证据；
- 先按相似性聚类再划分 train/val/test；
- 导出 strict/hybrid 双版本。

检查点：
- split 中不存在明显同源泄漏；
- 类别比例和长度分布可接受。

### 阶段 G：发布与复现
任务：
- 输出字段字典与运行手册；
- 固化版本元信息；
- 建议记录配置快照。

检查点：
- 新机器按 README 命令可复现；
- 版本和参数可追踪。

## 5. 风险与对策

- 数据源结构变动：将解析器与主流程解耦，独立维护 mapper。  
- 证据粒度不一致：保留证据等级，训练集区分 strict/hybrid。  
- 伪负样本偏差：保留来源标签并建议下游做鲁棒性评估。  
- 同源泄漏风险：先聚类后划分，必要时提高过滤阈值。  

## 6. 验收指标（建议）

- 完整性：主输出文件齐全，字段符合字典定义。  
- 可追溯性：样本可回溯到来源数据库与文献。  
- 规则一致性：IDR 驱动筛选与负样本规则可审计。  
- 可复现性：同配置重复运行结果结构一致。  

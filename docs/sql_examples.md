# SQLite 查询示例（IDR-LLPS 数据库）

数据库文件默认路径：

- `data/processed/idr_llps.sqlite`

可用表：

- `evidence`
- `idr_regions`
- `samples`
- `splits`
- `nardini90_features`
- `metadata`

---

## 0. 打开数据库

```bash
sqlite3 data/processed/idr_llps.sqlite
```

在 SQLite 里建议先设置：

```sql
.headers on
.mode column
```

如果要一次性创建/更新常用视图：

```bash
sqlite3 data/processed/idr_llps.sqlite < docs/sql_views.sql
```

---

## 1. 基础概览

### 1.1 查看所有表

```sql
.tables
```

### 1.2 每个表的行数

```sql
SELECT 'evidence' AS table_name, COUNT(*) AS n FROM evidence
UNION ALL
SELECT 'idr_regions', COUNT(*) FROM idr_regions
UNION ALL
SELECT 'samples', COUNT(*) FROM samples
UNION ALL
SELECT 'splits', COUNT(*) FROM splits
UNION ALL
SELECT 'nardini90_features', COUNT(*) FROM nardini90_features
UNION ALL
SELECT 'metadata', COUNT(*) FROM metadata;
```

### 1.3 查看版本元信息

```sql
SELECT * FROM metadata ORDER BY key;
```

### 1.4 查看已创建视图

```sql
SELECT name
FROM sqlite_master
WHERE type = 'view'
ORDER BY name;
```

### 1.5 直接使用视图（示例）

```sql
SELECT * FROM v_label_summary ORDER BY sample_count DESC;
SELECT * FROM v_source_summary ORDER BY evidence_count DESC;
SELECT * FROM v_nardini90_summary ORDER BY label;
```

---

## 2. 标签与来源统计

### 2.1 正负样本数量

```sql
SELECT label, COUNT(*) AS n
FROM samples
GROUP BY label
ORDER BY n DESC;
```

### 2.2 标签来源分布（实验正/实验负/伪负）

```sql
SELECT label_source, COUNT(*) AS n
FROM samples
GROUP BY label_source
ORDER BY n DESC;
```

### 2.3 各来源数据库贡献了多少证据

```sql
SELECT source_db, COUNT(*) AS n
FROM evidence
GROUP BY source_db
ORDER BY n DESC;
```

### 2.4 各来源数据库中的 LLPS 阳性证据数

```sql
SELECT source_db, COUNT(*) AS n
FROM evidence
WHERE llps_observed = 'True'
GROUP BY source_db
ORDER BY n DESC;
```

---

## 3. 按物种与蛋白检索

### 3.1 查看样本量最多的物种

```sql
SELECT organism, COUNT(*) AS n
FROM samples
GROUP BY organism
ORDER BY n DESC
LIMIT 20;
```

### 3.2 查看某个蛋白的全部样本（把 `P35637` 改成你的 accession）

```sql
SELECT sample_id, protein_accession, gene_name, organism,
       region_start, region_end, label, label_source, source_dbs
FROM samples
WHERE protein_accession = 'P35637'
ORDER BY region_start, region_end;
```

### 3.3 查看某个蛋白对应的证据详情

```sql
SELECT source_db, source_entry_id, evidence_grade, llps_observed,
       idr_driver_supported, pmid, condition_text
FROM evidence
WHERE protein_accession = 'P35637'
ORDER BY source_db, evidence_grade;
```

---

## 4. 关注 IDR 相关查询

### 4.1 IDR 边界长度分布（粗分箱）

```sql
SELECT
  CASE
    WHEN (idr_end - idr_start + 1) < 30 THEN 'short(<30)'
    WHEN (idr_end - idr_start + 1) < 100 THEN 'medium(30-99)'
    ELSE 'long(>=100)'
  END AS idr_bin,
  COUNT(*) AS n
FROM idr_regions
WHERE idr_start IS NOT NULL AND idr_end IS NOT NULL
GROUP BY idr_bin
ORDER BY n DESC;
```

### 4.2 仅看正样本中的 IDR 来源

```sql
SELECT idr_source, COUNT(*) AS n
FROM samples
WHERE label = 'idr_pos'
GROUP BY idr_source
ORDER BY n DESC;
```

---

## 5. 数据导出（CSV/TSV）

### 5.1 导出正样本为 TSV

```sql
.mode tabs
.headers on
.output idr_pos.tsv
SELECT * FROM samples WHERE label = 'idr_pos';
.output stdout
```

### 5.2 导出负样本为 TSV

```sql
.mode tabs
.headers on
.output negatives.tsv
SELECT * FROM samples WHERE label = 'neg';
.output stdout
```

### 5.3 导出训练集样本（按 split 表）

```sql
.mode tabs
.headers on
.output train_samples.tsv
SELECT s.*, sp.split
FROM samples s
JOIN splits sp ON s.sample_id = sp.sample_id
WHERE sp.split = 'train';
.output stdout
```

---

## 6. 质量检查（推荐）

### 6.1 检查是否有缺失 accession 的异常记录

```sql
SELECT COUNT(*) AS n_bad
FROM samples
WHERE protein_accession IS NULL OR TRIM(protein_accession) = '';
```

### 6.2 检查区段坐标是否异常

```sql
SELECT COUNT(*) AS n_bad
FROM samples
WHERE region_start IS NULL
   OR region_end IS NULL
   OR region_start <= 0
   OR region_end < region_start;
```

### 6.3 检查 split 是否覆盖全部样本

```sql
SELECT
  (SELECT COUNT(*) FROM samples) AS n_samples,
  (SELECT COUNT(*) FROM splits) AS n_splits,
  (SELECT COUNT(DISTINCT sample_id) FROM splits) AS n_unique_split_samples;
```

---

## 7. 常用组合查询

### 7.1 人类 (`taxon_id=9606`) 的正样本

```sql
SELECT sample_id, protein_accession, gene_name, region_start, region_end, label_source
FROM samples
WHERE label = 'idr_pos'
  AND taxon_id = 9606
ORDER BY protein_accession;
```

### 7.2 每个蛋白正样本条数 Top 20

```sql
SELECT protein_accession, gene_name, COUNT(*) AS n_pos
FROM samples
WHERE label = 'idr_pos'
GROUP BY protein_accession, gene_name
ORDER BY n_pos DESC
LIMIT 20;
```

### 7.3 只看“实验负样本”而非伪负样本

```sql
SELECT *
FROM samples
WHERE label = 'neg'
  AND label_source = 'exp_neg';
```

---

## 8. NARDINI90 特征查询

### 8.1 检查特征覆盖率

```sql
SELECT * FROM v_nardini90_summary ORDER BY label;
```

### 8.2 拉取某个蛋白的 90 维特征（示例 `P35637`）

```sql
SELECT s.sample_id, s.protein_accession, s.gene_name, s.label,
       n.comp_NCPR, n.comp_pI, n.pat_pos_neg, n.pat_aro_aro
FROM samples s
JOIN nardini90_features n ON s.sample_id = n.sample_id
WHERE s.protein_accession = 'P35637';
```

### 8.3 导出完整 90 维矩阵

```sql
.mode tabs
.headers on
.output nardini90_matrix.tsv
SELECT * FROM nardini90_features;
.output stdout
```

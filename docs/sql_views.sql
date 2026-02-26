-- 常用统计视图（IDR-LLPS SQLite）
-- 用法：
-- sqlite3 data/processed/idr_llps.sqlite < docs/sql_views.sql

DROP VIEW IF EXISTS v_source_summary;
CREATE VIEW v_source_summary AS
SELECT
  source_db,
  COUNT(*) AS evidence_count,
  SUM(CASE WHEN llps_observed = 'True' THEN 1 ELSE 0 END) AS llps_positive_evidence_count
FROM evidence
GROUP BY source_db;

DROP VIEW IF EXISTS v_label_summary;
CREATE VIEW v_label_summary AS
SELECT
  label,
  COUNT(*) AS sample_count
FROM samples
GROUP BY label;

DROP VIEW IF EXISTS v_label_source_summary;
CREATE VIEW v_label_source_summary AS
SELECT
  label_source,
  COUNT(*) AS sample_count
FROM samples
GROUP BY label_source;

DROP VIEW IF EXISTS v_species_summary;
CREATE VIEW v_species_summary AS
SELECT
  organism,
  taxon_id,
  COUNT(*) AS sample_count
FROM samples
GROUP BY organism, taxon_id;

DROP VIEW IF EXISTS v_idr_length_bins;
CREATE VIEW v_idr_length_bins AS
SELECT
  CASE
    WHEN (idr_end - idr_start + 1) < 30 THEN 'short(<30)'
    WHEN (idr_end - idr_start + 1) < 100 THEN 'medium(30-99)'
    ELSE 'long(>=100)'
  END AS idr_bin,
  COUNT(*) AS region_count
FROM idr_regions
WHERE idr_start IS NOT NULL
  AND idr_end IS NOT NULL
GROUP BY
  CASE
    WHEN (idr_end - idr_start + 1) < 30 THEN 'short(<30)'
    WHEN (idr_end - idr_start + 1) < 100 THEN 'medium(30-99)'
    ELSE 'long(>=100)'
  END;

DROP VIEW IF EXISTS v_sample_with_split;
CREATE VIEW v_sample_with_split AS
SELECT
  s.*,
  sp.split,
  sp.dataset_version
FROM samples s
JOIN splits sp
  ON s.sample_id = sp.sample_id;

DROP VIEW IF EXISTS v_positive_samples;
CREATE VIEW v_positive_samples AS
SELECT *
FROM samples
WHERE label = 'idr_pos';

DROP VIEW IF EXISTS v_negative_samples;
CREATE VIEW v_negative_samples AS
SELECT *
FROM samples
WHERE label = 'neg';

DROP VIEW IF EXISTS v_human_idr_pos;
CREATE VIEW v_human_idr_pos AS
SELECT *
FROM samples
WHERE label = 'idr_pos'
  AND CAST(taxon_id AS TEXT) = '9606';

DROP VIEW IF EXISTS v_protein_pos_counts;
CREATE VIEW v_protein_pos_counts AS
SELECT
  protein_accession,
  gene_name,
  COUNT(*) AS positive_sample_count
FROM samples
WHERE label = 'idr_pos'
GROUP BY protein_accession, gene_name;

DROP VIEW IF EXISTS v_quality_missing_accession;
CREATE VIEW v_quality_missing_accession AS
SELECT *
FROM samples
WHERE protein_accession IS NULL
   OR TRIM(protein_accession) = '';

DROP VIEW IF EXISTS v_quality_bad_region;
CREATE VIEW v_quality_bad_region AS
SELECT *
FROM samples
WHERE region_start IS NULL
   OR region_end IS NULL
   OR region_start <= 0
   OR region_end < region_start;

DROP VIEW IF EXISTS v_nardini90_joined;
CREATE VIEW v_nardini90_joined AS
SELECT
  s.sample_id,
  s.protein_accession,
  s.gene_name,
  s.organism,
  s.label,
  s.label_source,
  s.idr_length_bin,
  s.taxon_group,
  n.feature_version,
  n.random_seed,
  n.num_scrambles,
  n.computed_at
FROM samples s
JOIN nardini90_features n
  ON s.sample_id = n.sample_id;

DROP VIEW IF EXISTS v_nardini90_summary;
CREATE VIEW v_nardini90_summary AS
SELECT
  s.label,
  COUNT(*) AS sample_count,
  SUM(CASE WHEN n.sample_id IS NOT NULL THEN 1 ELSE 0 END) AS with_nardini90,
  ROUND(
    100.0 * SUM(CASE WHEN n.sample_id IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*),
    2
  ) AS coverage_percent
FROM samples s
LEFT JOIN nardini90_features n
  ON s.sample_id = n.sample_id
GROUP BY s.label;

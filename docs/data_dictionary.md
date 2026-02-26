# IDR-LLPS Data Dictionary

This document defines the fields for the processed dataset.

## `data/processed/evidence.tsv`

- `evidence_id`: unique evidence record identifier.
- `source_db`: source database name (`llpsdb`, `phasepro`, `drllps`, `disprot`, `mobidb`, `llpsdatasets`, etc.).
- `source_entry_id`: source-side entry id.
- `protein_accession`: canonical UniProt accession.
- `isoform`: isoform tag (e.g., `canonical`, `iso2`).
- `gene_name`: gene symbol if available.
- `organism`: species name.
- `taxon_id`: NCBI taxonomy id.
- `sequence`: protein sequence in one-letter amino-acid code.
- `construct_context`: construct context (`full_length`, `idr_region`, mutant tags, etc.).
- `region_start`, `region_end`: LLPS-tested construct region boundaries (1-based, inclusive).
- `idr_start`, `idr_end`: IDR boundaries (1-based, inclusive).
- `idr_source`: IDR annotation source (`disprot`/`mobidb`/other).
- `llps_observed`: `True`/`False`.
- `idr_driver_supported`: `True`/`False`.
- `evidence_grade`: `A`, `B`, or `C`.
- `condition_text`: free-text experimental condition.
- `pmid`: PubMed id if available.
- `db_version`: source DB release/version.
- `download_date`: source download date (UTC date).

## `data/processed/idr_regions.tsv`

- `protein_accession`, `isoform`, `gene_name`, `organism`, `taxon_id`, `sequence`.
- `idr_start`, `idr_end`: IDR boundaries.
- `idr_source`: evidence source for the IDR boundary.

## `data/processed/samples.tsv` (hybrid default)

- `sample_id`: hashed id generated from primary key.
- `protein_accession`, `isoform`, `gene_name`, `organism`, `taxon_id`.
- `region_start`, `region_end`, `region_sequence`.
- `idr_start`, `idr_end`, `idr_source`.
- `label`: `idr_pos` or `neg`.
- `label_source`: `experimental`, `exp_neg`, `pseudo_neg`.
- `evidence_grade_max`: strongest grade in grouped evidence.
- `evidence_count`: number of merged evidence records.
- `source_dbs`: semicolon-separated DB list.
- `pmids`: semicolon-separated PMIDs.
- `construct_context`: construct descriptor.
- `idr_length_bin`: `short`, `medium`, `long`.
- `taxon_group`: `human`, `non_human`, `unknown`.

## `data/processed/splits.tsv`

- `sample_id`: links to `samples.tsv`.
- `split`: `train`, `val`, or `test`.
- `dataset_version`: from `configs/dataset.yaml`.

## `data/processed/nardini90_features.tsv`

- `sample_id`: links to `samples.tsv`.
- `label`: copied from `samples.label`.
- `feature_version`: feature schema version (for reproducibility).
- `random_seed`: random seed used for scramble-based patterning.
- `num_scrambles`: number of sequence scrambles for patterning z-scores.
- `computed_at`: UTC timestamp of feature computation.
- `comp_*` (54 columns): compositional z-scores.
  - 20 amino-acid fractions: `comp_frac_A` ... `comp_frac_Y`
  - 8 grouped composition features: `comp_frac_polar`, `comp_frac_hydrophobic`, `comp_frac_aromatic`, `comp_frac_KR`, `comp_frac_DE`, `comp_FCR`, `comp_FCE`, `comp_frac_disorder_promoting`
  - 20 patch features: `comp_cum_patch_A` ... `comp_cum_patch_Y` (cumulative run length, run >= 2)
  - 2 ratio features: `comp_ratio_RK`, `comp_ratio_ED`
  - 4 global features: `comp_NCPR`, `comp_pI`, `comp_hydropathy_KD`, `comp_PPII`
- `pat_*` (36 columns): patterning z-scores from 8 residue-type groups (upper triangle incl. diagonal).
  - Groups: `pol`, `hyd`, `pos`, `neg`, `aro`, `ala`, `pro`, `gly`
  - Examples: `pat_pol_pol`, `pat_pos_neg`, `pat_aro_gly`

## Versioned sample tables

- `data/processed/samples_strict.tsv`: only experimental positives + experimental negatives.
- `data/processed/samples_hybrid.tsv`: experimental positives + (experimental negatives + pseudo negatives).

## True negative (experimental negative) sources

- **exp_neg** samples come from evidence with `llps_observed=False` and from whitelisted sources only (see `negative_rules.experimental_negative_source_whitelist` in config).
- **llpsdatasets**: Curated negative (non-LLPS) proteins from [Confident protein datasets for LLPS studies](https://llpsdatasets.ppmclab.com/) (PPMC-lab). Data: datasets.tsv; see [Zenodo](https://doi.org/10.5281/zenodo.15118996) and [GitHub](https://github.com/PPMC-lab/llps-datasets). Cite the PPMC dataset when using these negatives.

# IDR数据库统计可视化报告

## 数据摘要
- 全IDR条目数：7256
- 样本条目数：3000
- 证据条目数：20802
- NARDINI90条目数：2999

## 标签比例（samples）
- neg: 87.1%
- idr_pos: 12.9%

## 三组比例（pos / neg / neg_pseudo）
- neg_pseudo: 52.93%
- neg: 34.17%
- pos: 12.9%

## 关键发现
- 长度分布：IDR长度与样本区段长度均呈右偏分布，存在长尾。
- 氨基酸偏好：给出全IDR的20AA频率和正负样本log2FC差异图。
- 物种构成：列出Top物种与taxon_group分布。

### Top10 物种
- Homo sapiens: 3356
- Saccharomyces cerevisiae (strain ATCC 204508 / S288c): 501
- Mus musculus: 371
- Escherichia coli (strain K12): 244
- Arabidopsis thaliana: 199
- Rattus norvegicus: 185
- Drosophila melanogaster: 123
- Caenorhabditis elegans: 107
- Bos taurus: 86
- Mycobacterium tuberculosis (strain ATCC 25618 / H37Rv): 61

### Top10 高频氨基酸（全IDR）
- S: 0.0997
- A: 0.0836
- G: 0.0824
- P: 0.0777
- E: 0.0765
- L: 0.0655
- K: 0.0647
- D: 0.0560
- R: 0.0556
- T: 0.0556

## 质量控制
- 空区段序列条数：1
- 异常坐标条数：1

## 图表目录
- `reports/figures/aa_frequency_idr_regions.png`
- `reports/figures/aa_group_fraction_by_label.png`
- `reports/figures/aa_log2fc_pos_vs_neg.png`
- `reports/figures/aa_log2fc_pos_vs_neg_pseudo.png`
- `reports/figures/charge_features_by_label.png`
- `reports/figures/group_counts.png`
- `reports/figures/idr_pca_kmeans.png`
- `reports/figures/idr_pca_pos_neg.png`
- `reports/figures/idr_tsne_kmeans.png`
- `reports/figures/idr_tsne_pos_neg.png`
- `reports/figures/label_counts.png`
- `reports/figures/label_source_counts.png`
- `reports/figures/length_bins_idr_regions.png`
- `reports/figures/length_distribution_idr_regions.png`
- `reports/figures/length_distribution_samples_by_label.png`
- `reports/figures/nardini90_top12_delta.png`
- `reports/figures/source_db_contribution.png`
- `reports/figures/species_top20_idr_regions.png`
- `reports/figures/split_label_distribution.png`
- `reports/figures/taxon_group_samples.png`
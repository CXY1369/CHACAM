# Tintori 16-cell reference data

本目录中的原始文件由作者准备并逐字节复制到仓库；转换只读取本目录，不依赖仓库外路径。

## 原始文件

### 文章附带文件

| 文件 | 相对路径 | SHA-256 | 来源 |
|---|---|---|---|
| Table S1 Summary of All 219 Samples Sequenced Related to Experimental Procedures.xls | `data/reference/Table S1 Summary of All 219 Samples Sequenced Related to Experimental Procedures.xls` | `2a503ccfab008be29c855f7e6bfacebb33526263f82545f3aabd739d222c0f6c` | Tintori et al., Developmental Cell (2016), PMID 27554860 |
| Table S2_RPKMs_Normalized Expression Levels for All Genes in All Samples Related to Figure 1.xls | `data/reference/Table S2_RPKMs_Normalized Expression Levels for All Genes in All Samples Related to Figure 1.xls` | `172804f02f3009aefed615a346274377245cbdd5616eac4bfc562d3a37853160` | Tintori et al., Developmental Cell (2016), PMID 27554860 |
| Supplementary Data 5_Identified Cell-Cell Contacts.xlsx | `data/reference/Supplementary Data 5_Identified Cell-Cell Contacts.xlsx` | `5867842937d54adb8fad08c7cd3b5bd1b7276a5edeef0c7a2fa9cb3b8877d923` | Cao et al., Nature Communications (2020) |

### 作者整理文件

| 文件 | 相对路径 | SHA-256 | `source_type` | 整理者 | 上游依据 |
|---|---|---|---|---|---|
| Gene queries of C elegans.xlsx | `data/reference/Gene queries of C elegans.xlsx` | `187986b1f3faa1b2d9c8ec96985c5b79a50c2bd8bfc724e3d0050ff0c5d671e9` | `author_curated` | 作者 | 未说明 |
| ligand_receptor_updated_C-elegans.csv | `data/reference/ligand_receptor_updated_C-elegans.csv` | `3366b45ee385cd690290c2aca139656534c0e7c16624287de9cb778fc76c9721` | `author_curated` | 作者 | 未说明 |

原始表达量的单位是 RPKM。

## 转换

命令：`.venv/bin/python examples/c_elegans/convert_tintori.py`

脚本：`examples/c_elegans/convert_tintori.py`；输出目录为 `data/reference/derived/`。

规则摘要：

1. Table S1 的全部样本名必须符合规定正则，只保留 `Usable Quality? == Yes` 且时期为 16 的样本，并按原始顺序输出。
2. MS/C 按胚胎 × 家族用原始 `F19F10.1` RPKM 判定，无法判定的整组删除，编号 `x1/x2` 不决定前后。
3. 基因别名按 `Your Input`、`Public Name`、`Sequence Name` 建立唯一 WormBase ID 映射，保留公开名行并丢弃同 ID 重复行。
4. 表达值在转换时写为 `numpy.log1p(RPKM)`，因此配置使用 `expression_scale: log1p`，CCI 使用 `transform: none`。
5. feature genes 按同一映射转换，映射后缺失于表达矩阵时报错。
6. L–R 过滤两种 `non-LR` 标记并丢弃无法映射的行，映射成功但不在表达矩阵中的行保留给框架加载时处理。
7. CShaper 四个工作表按无向身份对取最高等级；只有等级 4 和缺失等级 0 写入 `contacts.csv`，全部 120 对写入 `contact_levels.csv`。
8. 所有表按 M4 v2 第 4 节规定的原始顺序或身份字典序输出；转换报告记录丢弃明细、MS/C 原值、SHA-256 和脚本 commit。

## M4 采用的接触假设

以下两对在 CShaper 中的原等级均为 3；M4 采用 CMOS 单胚胎形态观察中的不相邻假设，将输出接触值设为 0：

| identity_a | identity_b | CShaper 原等级 | M4 接触 | 来源 |
|---|---|---:|---:|---|
| ABala | MSa | 3 | 0 | CMOS <https://bcc.ee.cityu.edu.hk/cmos/ME.html?TP=&SM=200109plc1p1> |
| ABara | Ca | 3 | 0 | CMOS <https://bcc.ee.cityu.edu.hk/cmos/ME.html?TP=&SM=200109plc1p1> |

丢弃细胞、基因行和 L–R 行的完整明细位于 `data/reference/derived/conversion_report.json`。

## 已知局限

- C 细胞中 r2、r7、r8 被删除；同一规则也适用于 MS，但本数据中的 MS 组均保留。第一阶段存在把其他细胞误分为 Ca 的可能性；运行脚本会输出参考诊断，但该诊断路径不提交。
- MS/C 标签只由一个 `F19F10.1` 表达量和阈值决定；阈值具有数据依赖性，r3 的 st353 可能同时反映质量或 dropout 问题，因此它不是独立真值。
- features 只有先验 marker，不含 Set7；`F19F10.1` 同时是 MS/C 标签驱动变量和 feature，因此第一阶段一致性是构造出来的。
- `log1p` 转换是本项目的选择，不代表对原论文处理流程的复现。
- `top_k=50` 仅用于探索，不代表显著性筛选。
- 接触修正来自单胚胎观察和 M4 假设。

本数据的模型拟合不发生标签泄漏：训练 fold 只使用参考细胞和 top-k 训练 fold；静态 features、L–R 和接触表不随 fold 重新估计。

但预处理具有构造依赖：MS/C 标签和细胞删除使用每个胚胎自身的表达量，且阈值是在查看全部数据后选定的。

## 当前派生 CSV 的 SHA-256

| 文件 | SHA-256 |
|---|---|
| `data/reference/derived/cells.csv` | `3ec49139a4d343fac80c2e2bf58e844efe45fffd35c76b3164f9a9b13b4a1f88` |
| `data/reference/derived/expression.csv` | `32fa9e278d6a0449f50edf3bce5b3595a287131e1558aef430404dfd3495665c` |
| `data/reference/derived/features.csv` | `ca736eaeeb4425c6a2238bc0b7f72b5a11f6d32479b03cd9c07fca1963bda5b9` |
| `data/reference/derived/lr_pairs.csv` | `121806a7429d3db20aaeb40cceace1ab569a9800d84dd64a39a42104e7b27581` |
| `data/reference/derived/contacts.csv` | `5a6c64ff9a6536f754a054787299c8e4519d92011bd44de4a3b51010544cbebd` |
| `data/reference/derived/contact_levels.csv` | `7e8aeb4dca7915461e65a9b3a6fa0123713082a2abd894833954d7a8c75b81a7` |

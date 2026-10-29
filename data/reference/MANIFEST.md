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

- r2、r7、r8 的 C 细胞因无法判定前后而被删除。同一规则也适用于 MS，但本数据中的 MS 组都能判定。删除之后，第一步分配仍可能把别的细胞分为 `Ca`；运行示例脚本 `run_tintori16.py` 时会输出一张参考诊断表，列出这种情况。该表生成在运行目录中，不随仓库发布。
- MS/C 的前后只由 `F19F10.1` 一个基因的表达量判定：同组两个细胞中，较高者 `hi` 须满足 `hi ≥ 25` RPKM 且 `hi ≥ 10 × (lo + 1)`，其中 `lo` 是较低者，`+1` 是伪计数。这两个阈值是看过数据之后定的。r3 的 st353 在 Table S1 中检出 1,142 个基因，是本批 90 个可用 16 细胞期样本中最少的；它的 `F19F10.1` 为 0，可能是脱落造成的。因此这些标签不能用作独立验证。
- feature genes 只用先验标记基因，不用 Tintori 的 Set 7。`F19F10.1` 既决定 MS/C 的观测标签，又是 feature gene，所以 MS/C 第一步分配的一致率是由构造决定的。
- 转换时取 `log1p` 是本项目的选择，不是对原论文处理流程的复现。
- `top_k: 50` 是探索性的上限，没有做显著性过滤。
- 两条接触修正来自单个胚胎的观察，是 M4 采用的假设。

**模型拟合层面没有泄漏**：参考谱只用训练折的细胞计算；每一折的 Top-K 也只用训练折的细胞，从固定的 L–R 候选表中选出。feature genes、L–R 候选表和接触表是静态输入，不随折改变。

**预处理有构造依赖**：MS/C 的标签和细胞删除用到了每个胚胎自身的表达量，阈值也是看过全部数据之后定的。

## 当前派生 CSV 的 SHA-256

| 文件 | SHA-256 |
|---|---|
| `data/reference/derived/cells.csv` | `3ec49139a4d343fac80c2e2bf58e844efe45fffd35c76b3164f9a9b13b4a1f88` |
| `data/reference/derived/expression.csv` | `32fa9e278d6a0449f50edf3bce5b3595a287131e1558aef430404dfd3495665c` |
| `data/reference/derived/features.csv` | `ca736eaeeb4425c6a2238bc0b7f72b5a11f6d32479b03cd9c07fca1963bda5b9` |
| `data/reference/derived/lr_pairs.csv` | `121806a7429d3db20aaeb40cceace1ab569a9800d84dd64a39a42104e7b27581` |
| `data/reference/derived/contacts.csv` | `5a6c64ff9a6536f754a054787299c8e4519d92011bd44de4a3b51010544cbebd` |
| `data/reference/derived/contact_levels.csv` | `7e8aeb4dca7915461e65a9b3a6fa0123713082a2abd894833954d7a8c75b81a7` |

## Cole-Yanai M6 参考数据

### 原始文件

下列文件由 `Paper/Source Data and Reference Papers/Cole-Yanai_singleCell/Supplemental Information/` 逐字节复制到 `data/reference/cole_yanai/`。该目录及其派生目录由 `.gitignore` 排除，转换脚本只读取这些路径和本 MANIFEST 中列出的 M4 文件。

| 文件 | SHA-256 |
|---|---|
| `GSE83523_CE_blastomeres_rawcounts.txt.gz` | `86c8c102c0e24ad84c2e8393218edd6caf6825ee8461238507dc61758cebc201` |
| `GSE83523_family.soft.gz` | `799a33a0732700cb1a6ce2d4ac1d5307d13742ec070dd35746bdbb6c746da984` |
| `Cell state assignments.xlsx` | `b2e8a7887cafcaa7b20fb7dc6d27f2deb5aab7d258a212ba521b65ed21614078` |
| `Inferred cell states.docx` | `488940547bcde695333cd0b41bf628620b9ad54852448c85d07f56880c3520df` |
| `Characteristics of the dataset.docx` | `8fe42173e3db2f3e9cc42ab07c3d2be38bd6440c60f544bd787bfe324bbbf41f` |

### C1 转换

命令：`.venv/bin/python examples/c_elegans/convert_cole_yanai.py`

输出目录为 `data/reference/derived_cole_yanai/stage{15,28,51}/`，包含 `cells.csv`、`expression.csv`、`lr_pairs.csv`、`contacts.csv`、`contact_levels.csv` 和 `conversion_report.json`。转换规则如下：

1. 期来自 Table S1 的 `Intended Stage`；GSM 与样本名逐项核对 SOFT 的 `!Sample_title`；Table S5 的胚胎 ID 按 `P0n → 1.n`、`{数字}c{序号} → {数字}.{序号}` 规范化后与 38 个 S1 胚胎行核对。
2. 对每个 GSM 计算 Table S5 转录本数除以原始计数矩阵整列和；区间为闭区间 `[0.95, 1.06]`。任一细胞越界则剔除整个胚胎。当前数据剔除 4c3、4c5、7c1、8c1、8c2，共 25 个越界细胞和 26 个被剔除细胞。
3. Table S4 先去掉期标注再切分身份；Table S5 状态按完整 `x` 通配模式匹配。仅允许 `Abpxpaa → ABpxpaa`、`P0 → P_0`、`MSxaap` 和 `MSxpaa` 四条预注册修正；状态名必须恰好命中一行。唯一细胞数例外是第 20 行 `MSa MSp`，Table S4 为 12、Table S5 为 6。
4. 表达矩阵使用 M4 的基因别名映射和重复 WormBase ID 处理；文库大小用映射前全部原始行；写出 `log1p(CPM)`。不生成 `features.csv`。
5. L–R 使用 M4 的 `convert_lr_pairs`；CShaper 四表按无向身份对取最高等级，Sheet4 写接触 1、四表未检出写 0、仅 Sheet1–3 出现的不确定组合不写入；缺失身份及其相关组合写入转换报告。

三期规模和转换摘要：

| 期 | 胚胎 | 细胞 | fine | equivalence | 最终基因 | L–R |
|---:|---:|---:|---:|---:|---:|---:|
| 15 | 5 | 59 | 28 | 31 | 20,109 | 280 |
| 28 | 6 | 132 | 57 | 75 | 20,109 | 280 |
| 51 | 5 | 204 | 101 | 103 | 20,109 | 280 |

派生表不是完整的 `chacam` 输入集，不能用 `load_inputs` 读取。派生 CSV 的 SHA-256 以及逐细胞剔除、基因映射、L–R 和 CShaper 明细见各期 `conversion_report.json`。Cole-Yanai 的表达是 `log1p(CPM)`，Tintori 的表达是 `log1p(RPKM)`；两者不作分数大小或准确率的跨数据集比较。

### 已知局限

- Table S5 是 S1 采集细胞的已注释子集，不把 S1 的 `# Cells Collected` 当作 S5 细胞数；只要求每个 S5 胚胎计数不超过对应 S1 行。
- 计数比值的整胚胎剔除规则和阈值是预注册的输入处理；CShaper 只有 51 期的 `Cap`、`MSapp`、`MSppp` 在表中缺失，相关组合不写入 `contacts.csv`。
- Cole-Yanai 与 Tintori 使用不同的表达单位，M6 只在各数据集内部按固定预处理回答是否存在信号。
- 细胞身份是原作者按已知标记基因对表达聚类后注释的，不是谱系追踪；姐妹细胞的真值本身依赖表达。

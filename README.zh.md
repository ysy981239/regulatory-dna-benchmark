# DNA 序列 → 调控功能：一个能证明「你懂 ML」的基因组学项目

> 目标不是再刷一个数据集，而是产出三样东西：
> **(1) 一个可复现的序列分类器**、**(2) 一套诚实的评估与消融实验**、**(3) 一个能回到生物学解释的归因分析**。
> 三者缺一，这个项目就退化成第三个 CIFAR。

---

## 0. 一句话问题

> **给定一段 200 bp 的 DNA 序列，模型能否判断它是不是调控元件（启动子 / 增强子 / 开放染色质）？如果能，它依赖的是序列里的什么？**

这个问题的好处：标签有明确生物学定义、有公开 benchmark 和 leaderboard、基线方法成熟（DeepSEA → Basset → Basenji → Enformer）、算力需求在你 6 GB 的 4050 上完全够。

## 1. 为什么这个项目能证明「理解」

MNIST/CIFAR 上你只需要把 loss 降下去。基因组学数据会惩罚三件事，而这三件事恰好是 ML 理解的核心：

| 陷阱 | 在这个项目里怎么暴露 | 本仓库怎么处理 |
|---|---|---|
| **数据泄漏** | 同源序列/重复区域同时落进 train 和 test，指标虚高 10–30 个点 | `--split group`：按共享 20-mer 做连通分量聚类，整簇划分 |
| **单特征可分性** | 只用 GC 含量就能 AUC 0.8+，模型其实没学到 motif | `scripts/leakage_check.py` 先跑 GC / 二核苷酸 / 5-mer 线性基线 |
| **指标幻觉** | accuracy、AUROC 在不平衡与高分阈值区没有意义 | 只报告 AUPRC + bootstrap CI + ECE 校准 + 最佳 F1 阈值 |

`genomic-benchmarks` 的策展论文标题本身就是结论：*Automated quality control for genomic sequence benchmarks reveals pervasive single-feature separability and train-test leakage*。**这个项目的主线故事可以是：「我先量化了 benchmark 的泄漏，再报告去泄漏后的真实性能」**——比再报一个 SOTA 有说服力得多。

## 2. 生物学背景（够用版，2 小时能读完）

- **中心法则**：DNA → RNA → 蛋白。你预测的是 DNA 层面「这段序列会不会被细胞读取」。
- **启动子（promoter）**：转录起始位点上游 ~1 kb，RNA 聚合酶 II 结合处。人类启动子常富集 CpG island。
- **增强子（enhancer）**：可远距离（10 kb–1 Mb）调控基因，方向无关、位置无关——**这就是为什么反向互补增强（RC augmentation）在生物学上是对的**。
- **开放染色质（OCR）**：ATAC-seq / DNase-seq 测到的无核小体区域，是「这段 DNA 在体内可被接触」的实验证据。
- **转录因子 motif**：6–20 bp 的简并序列模式，例如 `TGACTCA`（AP-1）。**模型学到的「motif」就是它卷积核的偏好序列**——这是你做归因分析的落点。
- **为什么用 200 bp**：一个 motif 6–20 bp，200 bp 足以容纳多个 motif 的组合（grammar），又不至于让算力爆掉。

> 记不住也没关系：先把 `drosophila_enhancers_stark` 跑通，再看 `human_nontata_promoters`，生物学直觉会随着看数据长出来。

## 3. 环境

```powershell
# 直接用本机已有的 conda 环境 cuda_env（Python 3.10 + torch 2.5.1 / cu121）
D:\anaconda\envs\cuda_env\python.exe -m pip install -r requirements.txt
```

已在该环境验证：Python 3.10.22、torch 2.5.1（cuda 12.1，`CUDA available: True`）、numpy 2.0.1、pandas 2.3.3、scipy 1.15.3、scikit-learn 1.7.2、matplotlib 3.10.9、pyyaml 6.0.3。**不需要 tqdm 或 huggingface_hub**——下载走 `urllib`，进度打印是本仓库自己的。

- 显卡：RTX 4050 Laptop 6 GB，`basset` 在 200 bp / batch 128 下显存占用 < 1.5 GB。
- 数据默认走 `hf-mirror.com`（国内可直连），可用 `GENOMIC_BENCHMARKS_ENDPOINT` 覆盖。
- 数据落在 `data/`（已 gitignore），编码结果缓存为 `.npz`，第二次加载秒开。

## 4. 快速开始

```powershell
$py = "D:\anaconda\envs\cuda_env\python.exe"
cd D:\kaggle\genomics

# ⓪ 先跑单元测试（1 秒）：编码、k-mer、同源划分、指标兜底
& $py tests\test_core.py

# ① 审计数据，别急着建模（~2 分钟）
& $py scripts\leakage_check.py --dataset human_nontata_promoters

# ② 跑一个诚实基线：按同源簇划分
& $py -m src.train --dataset human_nontata_promoters --model basset --split group

# ③ 跑一个作弊基线：随机按行划分，看它虚高多少
& $py -m src.train --dataset human_nontata_promoters --model basset --split random

# ④ 把「无学习基线」放进同一张表（同一个划分、同一个测试集）
& $py scripts\baseline_runner.py --dataset human_nontata_promoters --split group

# ⑤ 校准（只用 val 拟合温度）
& $py scripts\calibrate.py --all

# ⑥ 解释模型：IG + ISM + 一致性检验 + 学习到的 PWM + 变异效应
& $py scripts\motif_analysis.py --run-dir runs\human_nontata_promoters__basset__group__s0

# ⑦ 它到底用不用碱基顺序？（打乱组成不变、顺序全毁，再打分）
& $py scripts\shuffle_probe.py --all --n-seqs 3000

# ⑧ 最后汇总成一张表（前面各步的产物都会被吸收进去）
& $py scripts\make_table.py
```

`reports/summary.md` 里会自动出现五节：主表、**Reference baselines**（你的 CNN 必须打赢的那 1024 维 5-mer 逻辑回归）、
**Optimism from a random row split**（validation 差距 = 你会相信的泄漏；test 差距 = benchmark 自己的测试集有多干净）、
**Calibration**（温度缩放前后）、**Order or composition?**（打乱碱基顺序后的保留率）。
第 ⑥ 步产出 `reports/motif_*.md` + `.png`：IG 与 ISM 的重合度、学到的共识序列、以及能提高预测概率的 in-silico 突变。

> 训练默认 `--patience 0`（跑满 epoch）。原因见第 9 节「已知的坑」——早停会在 cosine 退火之前砍掉训练。

## 5. 结果速览（本仓库已实测，17 个 run）

**完整结题报告见 [`REPORT.md`](REPORT.md)，全部数字见 [`reports/summary.md`](reports/summary.md)。**

两个数据集、同一划分协议、官方 test。表格按 test AUPRC 降序（group = 同源簇划分，random = 按行随机）：

| dataset | model | params | split | test AUPRC | AUROC | ECE |
|---|---|---|---|---|---|---|
| promoters | basset | 476k | **random** | 0.9638 | 0.9487 | 0.1204 |
| promoters | linear-5mer | 1k | **random** | 0.9433 | 0.9159 | 0.0334 |
| promoters | dilated | 930k | group | **0.9408** | 0.9173 | 0.1199 |
| promoters | basset（无 RC） | 476k | group | 0.9359 | 0.9095 | 0.1048 |
| promoters | basset（含 RC） | 476k | group | 0.9275 | 0.8935 | **0.0228** |
| promoters | linear-5mer | 1k | group | 0.9180 | 0.8900 | 0.1258 |
| promoters | linear-2mer | 16 | group | 0.9000 | 0.8626 | 0.0930 |
| promoters | transformer | 826k | group | 0.8564 | 0.8116 | 0.1322 |
| promoters | linear-GC | 1 | group | 0.8394 | 0.7959 | 0.1125 |
| drosophila | basset | 476k | random | **0.8472** | 0.8760 | 0.1524 |
| drosophila | dilated（修复前） | 930k | group | 0.8345 | 0.8841 | 0.3303 |
| drosophila | basset | 476k | group | 0.8255 | 0.8748 | 0.1908 |
| drosophila | linear-5mer | 1k | group | 0.7114 | 0.7427 | 0.1265 |
| drosophila | linear-GC | 1 | group | 0.4722 | 0.5087 | 0.0172 |

> drosophila 的 basset 两行是修掉 padding bug 之后重训重测的；dilated 重训约 4 h，故保留修复前那一行并标注。promoters 的窗口本来就没有 padding，不受这个 bug 影响，数字未变。前后对比见 [`docs/DEVLOG.md`](docs/DEVLOG.md)。

> **泄漏率后来被重新标定过。** 上面 77% / 55% 是未过滤的数。用 `scripts/complexity_probe.py` 按碱基组成熵筛掉低复杂度 k-mer（只由 A+T 组成的 20-mer 熵 ≤ 1 bit）之后，drosophila 掉到 **59%**、启动子掉到 **50%**；用 `scripts/cluster_sensitivity.py` 扫阈值则发现 **k=20 是「分组划分还能成立」的最小值**（k≤18 时最大连通分量已占 95–100%，val 无处可留）。两个数都照实写着，没有打磨掉。

**五条实测结论（前四条是负面的，这才是项目的价值所在）：**

1. **官方 test split 自己就泄漏**：drosophila 有 **77.2%**、promoters 有 **54.8%** 的 test 序列与训练集共享 20-mer。所以无论怎么切 train/val，test 上的数字都会偏乐观。
2. **随机划分把 val AUPRC 抬高 6–8 个点**（promoters basset 0.9663→0.9020，linear-5mer 0.9422→0.8652），而 test 上只差 0.036 / 0.025 —— 因为官方 test 本身就脏。
3. **CNN 只比 1024 维 5-mer 线性模型高 1.5–2.3 个 AUPRC 点**，而 **GC 含量一个特征就能到 0.8394**。归因也印证了这点：学出的 PWM 共识是 `CCCCCCCC`、每列 0.3 bits、C+G 占 0.82 —— **是碱基组成，不是转录因子 motif**。transformer 在 27k 样本上训不动（0.8564，打不过线性 5-mer）。
4. **打乱碱基顺序后，启动子模型还剩 86% 的 AUPRC，增强子模型只剩 60%**（`scripts/shuffle_probe.py`：每条 test 序列保留组成、洗掉顺序再打分）。promoters 0.9275 → 0.7955，drosophila 0.8255 → 0.4976（已等于 0.50 的随机基线）。**启动子 CNN 主要是组成检测器，增强子 CNN 真的在用顺序。**
5. **有两个正面发现**：drosophila 上 CNN 比 5-mer 高 11 点（0.8255 vs 0.7114，不过线性基线在该数据上欠定）；RC 增强让 AUPRC 掉 0.8 点却把 ECE 从 0.105 压到 **0.023**——**同一个改动可以伤指标、修校准**，只报 AUPRC 会把它误判为无效。

## 6. 目录结构

```
genomics/
├─ src/
│  ├─ dna.py            编码：one-hot、反向互补、精确 k-mer 编码、GC/二核苷酸/5-mer 特征
│  ├─ data.py           HF 下载、npz 缓存、随机划分 vs 同源簇划分、泄漏量化
│  ├─ models.py         basset / dilated / transformer 三种归纳偏置
│  ├─ train.py          训练循环（AMP、warmup+cosine、梯度裁剪、EMA）+ CLI
│  ├─ evaluate.py       AUPRC+bootstrap CI、AUROC、Brier、ECE、最佳 F1 阈值、可靠性曲线
│  └─ attributions.py   单碱基饱和突变(ISM)、积分梯度(IG)、两种归因的一致性检验
├─ scripts/
│  ├─ leakage_check.py   GC/二核苷酸/5-mer 基线 + 近重复率 + 划分策略对比
│  ├─ baseline_runner.py 线性基线走与 CNN 完全相同的划分，写出同格式 metrics.json
│  ├─ calibrate.py       温度缩放（只用 val 拟合）+ 可靠性曲线 + oracle T 诊断
│  ├─ shuffle_probe.py   打乱每条序列的碱基（组成不变、顺序全毁），看性能掉多少
│  ├─ motif_analysis.py  IG + ISM + 一致性 + PWM + 变异效应表 + 图
│  └─ make_table.py      把所有 runs（含线性基线、校准、打乱探针）汇总成 markdown 表
├─ tests/test_core.py   纯 assert 的冒烟测试，不需要 pytest
├─ REPORT.md            结题报告（结论、消融解读、修掉的 bug、局限）
├─ reports/             summary.md、leakage_*.md、*-group.md|.png、calibration.json
└─ runs/<dataset>__<model>__<split>__s<seed>/   metrics.json / best.pt / 曲线图
```

## 7. 实验矩阵（这才是项目主体）

按顺序做，每一步都要留下数字。「预期」列是开工前的假设，**「实测」列是本仓库跑出来的真实结果——两列不一致的地方才是这个项目最值钱的部分。**

| # | 实验 | 命令要点 | 想回答的问题 | 预期 | 实测 |
|---|---|---|---|---|---|
| 1 | 无学习基线 | `leakage_check.py` | 多少性能来自 GC/碱基组成？ | GC 单特征 AUC 常在 0.7–0.9 | promoters GC 就 **0.796**，5-mer 线性 **0.918** |
| 2 | 泄漏量化 | `leakage_check.py` 的 split 对比 | 随机划分把指标抬高了多少？ | 共享 k-mer 应从 ~30% 降到 0% | 官方 test 仍有 **77.2% / 54.8%** 共享 |
| 3 | 划分策略差距 | `--split random` vs `group` | headline 数字有多少是泄漏给的？ | 掉 3–15 点 | val 掉 **6.4–7.7 点**，test 只掉 2.5–3.6 点 |
| 4 | 三种归纳偏置 | `--model basset / dilated / transformer` | 局部卷积是否足够？ | transformer 通常不占优 | 确实不占优：0.8564 < 5-mer 0.9180（欠拟合） |
| 5 | 感受野消融 | `--model dilated`、改 dilations | 决策依赖多少 bp 上下文？ | RF 变大指标不动=任务局部 | dilated(RF 507) 仅 +1.3 点 |
| 6 | 反向互补增强 | `--rc-augment` / `--no-rc-augment` | 生物学先验值多少点？ | 小数据集 +1–3 点 | AUPRC **−0.8 点**，但 ECE **0.105→0.023** |
| 7 | 归因与一致性 | `motif_analysis.py` | 模型看的是 motif 还是噪声？ | Jaccard < 0.2 则归因不可信 | **0.067 / 0.266 / 0.381**，全部塌缩成 GC 偏好 |
| 8 | 变异效应（亮点） | ISM 矩阵（logit 空间） | 单碱基突变如何改变预测？ | 输出 `(L,4)` delta 矩阵 | 已在 `reports/*-group.md` 的变异表里 |
| 9 | **顺序 vs 组成** | `shuffle_probe.py` | 模型到底用不用碱基顺序？ | 打乱后应掉很多 | promoters 只掉 12.7 点（留 86%），drosophila 掉 29.6 点（留 64%，AUROC 0.539） |

**加分项（有余力再做）**：把 5-mer 线性基线的权重画成序列 logo，和你 CNN 第一层卷积核的偏好做对比；如果两者高度重合，结论是「CNN 学会了线性模型已经会的东西」——本仓库的 PWM 结果已经指向这个结论。

## 8. 评估协议（写进 README，别人才信你）

1. **只用 AUPRC 做主指标**，同时报 AUROC 和正样本比例；不平衡数据上 AUROC 会骗人。
2. **每个数字带 bootstrap 95% CI**（`n_boot=2000`），小数据集上差异必须看 CI 是否重叠。
3. **报校准**（Brier + ECE + 可靠性曲线）。生物学下游要的是概率，不是排序。
4. **模型选择只看 val，test 只碰一次**（`train.py` 已强制如此）。
5. **固定 seed，报 ≥3 个 seed 的均值±标准差**；单次跑出来的 ±2 点差异不算结论。
6. 阈值不要默认 0.5：报「最佳 F1 阈值」及其对应的 sensitivity/specificity。
7. **温度缩放的 T 只能用 val 拟合**；另外算一个用 test 拟合的 oracle T 做诊断，但绝不拿它当结果。

## 9. 已知坑（踩过的人才知道）

> 下面这些不是「听说」，是这个仓库里真实发生并修掉的 bug。写进 report 的「教训」一节，比多刷 0.5 个点更值钱。

- **k-mer 编码里含 N 的窗口必须用与批次无关的哨兵值**：曾用「行号」做哨兵，而逐行探测时行号恒为 0，导致所有 padding 尾巴互相「匹配」，泄漏率被算成 99.9%（drosophila 真实是 77.2%，启动子 54.8%）。现在用行内容 `blake2b` 哈希。**这个 bug 会把你项目的核心结论写反。**
- **学习率调度按 step 定义、却按 epoch 步进**（已修）：`total_steps` 曾写成 `epochs * len(train_loader)`，但 `sched.step()` 每 epoch 只调一次，于是整个训练卡在线性 warmup 里，30 个 epoch 后 lr 只到设定值的 29%，cosine 退火一次都没发生。实测 lr 从 9.71e-06 单调升到 2.91e-04。**症状**：loss 一直缓慢下降、从不平台化，val 指标总觉得「再训久一点会更好」。
- **`LambdaLR` 构造时调用 `lr_lambda(0)`**（已修）：`warmup=1` 时它返回 0，第一个 epoch 在 lr=0 下空转，白烧 1/30 预算。修法是 `step = max(step, 1)`。
- **早停会吃掉 cosine 退火**（已修，`--patience` 默认改为 0）：验证集在退火前约 2/3 处就到峰值，`patience=8` 会在峰值后立刻停机，最后 10 个 epoch 的低学习率精调全被砍掉。promoters 上就因此在 epoch 19/30 停了。
- **run 目录名要带上消融标签**：`--no-rc-augment` 的结果曾覆盖掉同名 RC 结果，一个实验直接被吃掉。现在目录名里带 `-norc` / `-w{width}`。
- **归因要在 logit 空间算**：模型饱和到 p=1.000 时，任何单碱基替换的 Δp 都四舍五入成 0，ISM 全表看起来「没有效应」。IG 和 ISM 必须在**同一空间**才算得上可比。
- **常值预测器的 AUROC 是「未定义」，不是 0.5**：sklearn 会返回 0.5 而不报错，务必自己判 `np.unique(p).size < 2`。
- **ECE 不是 proper scoring rule**：温度缩放后 NLL 改善而 ECE 变差是正常的（promoters dilated：NLL 0.475→0.458，ECE 0.120→0.158）。模型的错是「自信地错」，没有任何全局单调变换能修。
- **已经校准好的模型无法用温度缩放改善**：basset+RC 的 ECE 是 0.023，T=0.81≈恒等，校准后 0.021。这不是方法失败。
- **padding 不是沉默的**：255 的 one-hot 是全零列，卷积输出 0，但 BatchNorm 会把 0 映射成一个非零常数；而全局 `mean‖max` 池化的分母是**补齐后的长度**。一条 236 bp 的增强子放进 3237 的张量，均值里 92.6% 来自那个常数。**这类 bug 不抛异常**，只在「同一份权重换一段代码、分数就变了」时才暴露。
- **长度必须在 split 之间钉死**：`L` 曾按每个 split 各自取最大值 → train 3237、test 3061，同一个模型在训练和测试时算的不是同一个函数。现在注册表里写死 `max_len`。
- **阈值不能在评估集上挑**：best-F1 的阈值曾在 test 预测上扫出来，报出去的是「看过 test 标签」的乐观值。现在阈值只在 val 上挑。
- **`human_ocr_ensembl` 是最接近 DeepSEA 的任务，但也最容易被 composition 主导**——先跑实验 1，再决定要不要投入。
- **`demo_human_or_worm` 几乎无信息量**（人 vs 线虫序列 trivially separable），只能当 pipeline 冒烟测试。
- **不要用 `sklearn` 的 `train_test_split` 直接切基因组数据**，除非你明确想复现一个泄漏的基线。
- **BatchNorm + 极小 batch** 在小数据集上会抖；`--batch-size 64` 比 128 更稳。
- **别在 test 上调早停**，`train.py` 用的是 val AUPRC。
- **Windows 上 `--num-workers 0` 最稳**（spawn 开销 + DataLoader 卡住的坑）；本项目序列在内存里，0 个 worker 也够快。
- **在本机用 PowerShell 5.1 改中文文件会毁文件**：`Get-Content`/`Set-Content` 默认走 gb2312，读写 UTF-8 会变乱码。改这类文件一律用编辑工具或 `[System.IO.File]::ReadAllText(path, Encoding::UTF8)`。

## 10. 8 周节奏

| 周 | 做什么 | 交付 | 状态 |
|---|---|---|---|
| 1 | 跑通 drosophila；读 `leakage_check` 输出；理解划分为何重要 | 两张审计表 | ✅ |
| 2 | human_nontata_promoters 上跑通 basset（random vs group） | 第一版 summary 表 | ✅ |
| 3 | 换成 human_ocr_ensembl；复现一个已发表基线的大致数字 | 与论文的对比表 | ⬜ |
| 4 | 实验 4/5/6（模型、感受野、RC） | 消融表 | ✅ |
| 5 | 实验 7（归因 + 一致性检验 + motif 对照） | motif 图 | ✅ |
| 6 | 实验 8（ISM 变异效应）+ 一个 Streamlit demo | 可交互 demo | 进行中 |
| 7 | 3 seed 重跑关键配置，补 CI | 带误差棒的最终表 | ⬜ |
| 8 | 写 report（4–6 页）+ README + 博客 | 可展示的成品 | 报告已完成，剩博客 |

## 11. 阅读清单（按顺序，只读这 7 篇）

1. **DeepSEA** (Zhou & Troyanskaya, 2015) — 多任务 CNN + 序列，任务的起源。
2. **Basset** (Kelley et al., 2016) — 更深的卷积 + 用归因找 motif，**必读**。
3. **Basenji / Basenji2** (Kelley et al., 2018/2020) — 感受野与 dilated conv。
4. **Enformer** (Avsec et al., 2021) — transformer + 长距离调控，知道它的边界即可。
5. **Genomic Benchmarks** (Grešová et al., 2023, BMC Genomics) — 你用的数据，读它的任务定义。
6. **Nucleotide Transformer** (Dalla-Torre et al., 2023) — 预训练 DNA 模型，你的实验 4 的对照。
7. 策展论文 *Automated quality control for genomic sequence benchmarks…* — 你项目主线的理论依据。

读法：每篇只回答三问——**输入输出是什么 / baseline 是谁 / 它靠什么 trick 赢的**。然后复现一张图。

## 12. 引用

- 数据：Grešová et al., *Genomic Benchmarks*, BMC Genomics 2023, DOI 10.1186/s12863-023-01123-8（CC BY 4.0）
- 数据仓库：`genomic-benchmarks/genomic-benchmarks` on Hugging Face（v1，含每行永久 id 与校验和）

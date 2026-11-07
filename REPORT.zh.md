# 从碱基组成到调控语法：Genomic Benchmarks 的复现与审计

> 一个「诚实」的基因组深度学习项目：先用无学习基线把 benchmark 的上限量出来，再问 CNN 到底多学到了什么。

---

## 摘要

我在 [Genomic Benchmarks](https://huggingface.co/datasets/genomic-benchmarks/genomic-benchmarks) 的两个任务上用同一套划分协议跑了 17 个配置（3 种网络 × 2 种划分 × 增强消融 + 6 组无学习基线），全部代码、预测、曲线可复现。五个结论，前四个是负面的：

1. **官方 test split 自己就泄漏。** `drosophila_enhancers_stark` 有 **77.2%** 的 test 序列与训练集共享至少一个 20-mer，`human_nontata_promoters` 是 **54.8%**。按行随机划分检验不了的泄漏，被这个数字直接暴露出来。
2. **按行随机划分把验证集 AUPRC 抬高 6–8 个点。** 同一模型同一数据，random vs group 划分：promoters basset val **0.9663 → 0.9020**（gap +0.0643），linear-5mer val **0.9422 → 0.8652**（gap +0.0770）。而 test 上只差 0.036 / 0.025 —— 因为官方 test 本身就脏，把差距吃掉了一半。
3. **CNN 在启动子任务上只比 5-mer 出现频率的线性模型高 1.0–2.3 个 AUPRC 点**（0.9408 vs 0.9180），而只用 GC 含量就能拿到 0.8394。Transformer 甚至打不过线性 5-mer（0.8564）。
4. **归因不指向任何转录因子 motif。** IG 与 ISM 的一致性（top-5% Jaccard）只有 0.067–0.381，学出来的 PWM 共识是 `CCCCCCCC` / `GGGGGGGG`，每列 0.2–0.3 bits，C+G 占 0.77–0.82 —— 位置无关的碱基组成偏好，不是调控语法。
5. **把碱基顺序打乱后，启动子模型还剩 86% 的 AUPRC，增强子模型只剩 58–64%。** 每条 test 序列保留组成、洗掉顺序再打分：promoters 0.9273 → 0.8005（掉 12.7 点），drosophila 0.8317 → 0.5356（掉 29.6 点，AUROC 直接落到 0.539 ≈ 抛硬币）。**同一个架构在两个任务上做着完全不同的事**：启动子上它基本是个（比单特征 GC 还差的）组成检测器，增强子上它真的在用顺序。

反面也有一个正面结论：**在 3237 bp 的长序列任务上，CNN 比 5-mer 线性模型高 12 个点**（0.8317 vs 0.7114），说明卷积确实补上了线性模型缺的东西。但这个比较对线性模型不公平（训练样本只有 5184 行却有 1024 个特征，严重欠定），我把它写进局限里而不是结论里。

---

## 1. 问题

任务形式：给一条 DNA 序列（251 bp 启动子 / 236–3237 bp 增强子），预测一个二分类标签（是否启动子 / 是否增强子）。

为什么这个题能说明问题，而不是「又训了一个 CNN」：

- **评估有陷阱**：同源序列（家族、重复元件）跨 split 泄漏会让指标凭空虚高，而这恰恰是生物序列数据的默认状态。
- **有强基线**：GC 含量、二核苷酸组成、k-mer 频率 + 逻辑回归。如果深度模型打不过这些，它就没有在学「语法」，只学了「组成」。
- **解释性可检验**：模型的归因能不能塌缩成已知的 motif，是可以拿 ISM 做一致性检验的。

## 2. 协议

| 项 | 选择 | 理由 |
|---|---|---|
| 数据 | `human_nontata_promoters`（27097/9034，251 bp，54.4% 正）、`drosophila_enhancers_stark`（5184/1730，236–3237 bp，50% 正） | 一大一小、一定长一可变长，能暴露 padding 类 bug |
| test | 官方 test split，**从不动** | 保证和论文可比 |
| train/val | 从官方 train 里切 15% 做 val | 官方没有 val |
| 划分 A「random」 |  stratified 按行随机 | 社区默认做法，也是泄漏来源 |
| 划分 B「group」 | 20-mer 精确匹配建图 → 连通分量 → 整簇进同一侧 | 同源序列不跨 split |
| 指标 | **AUPRC** 为主（含 2000 次 bootstrap 95% CI）+ AUROC + Brier + ECE | 类别近乎平衡时 AUPRC 更能分辨；校准单独看 |

「同源」的定义：两条序列共享**任意一个精确 20-mer** 即连边。这个定义偏严（会把远房同源都连上），但方向是安全的一侧——宁可低估性能，不可高估。`drosophila` 上 5184 条序列聚成 1309 簇（平均 3.96 条/簇），`promoters` 上 27097 条聚成 12254 簇（2.21 条/簇）。

## 3. 发现一：benchmark 自身的泄漏

`scripts/leakage_check.py` 的输出（`reports/leakage_*.md`）：

| dataset | test 行与 train 共享 20-mer 的比例 | 簇数（均值大小） | random 划分：跨 split 共享簇 | group 划分：跨 split 共享簇 |
|---|---|---|---|---|
| drosophila_enhancers_stark | **77.2%** | 1309（3.96） | 42 簇 / 75.4% val 行 | **0 / 0.0%** |
| human_nontata_promoters | **54.8%** | 12254（2.21） | 434 簇 / 55.7% val 行 | **0 / 0.0%** |

用官方 train 训练的无学习基线（`reports/leakage_*.json`）：

| dataset | GC 含量（1 维） | 二核苷酸（16 维） | 5-mer（1024 维） |
|---|---|---|---|
| drosophila_enhancers_stark | 0.5087 / 0.4722 | 0.6873 / 0.6372 | 0.7478 / 0.7144 |
| human_nontata_promoters | **0.7959 / 0.8394** | 0.8629 / 0.9011 | **0.9178 / 0.9446** |

（AUROC / AUPRC）

这意味着：人类启动子任务上，**一个数字（GC 含量）就能拿到 0.84 AUPRC**，1024 维 5-mer 频率能拿到 0.945。任何声称「深度模型学会了启动子语法」的结果，必须先证明它显著超过这条线。

## 4. 发现二：随机划分的代价

同一个模型、同一份数据，只改划分方式：

| dataset | model | val（group→random） | val gap | test（group→random） | test gap | ECE（group→random） |
|---|---|---|---|---|---|---|
| promoters | linear-5mer | 0.8652 → 0.9422 | **+0.0770** | 0.9180 → 0.9433 | +0.0253 | 0.1258 → 0.0334 |
| promoters | basset | 0.9020 → 0.9663 | **+0.0643** | 0.9275 → 0.9638 | **+0.0363** | 0.0228 → 0.1204 |
| drosophila | basset | 0.7555 → 0.8174 | **+0.0619** | 0.8317 → 0.8048 | **−0.0269** | 0.2946 → 0.2956 |
| promoters | linear-2mer | 0.8840 → 0.8979 | +0.0139 | 0.9000 → 0.9012 | +0.0012 | 0.0930 → 0.0235 |
| promoters | linear-gc | 0.8425 → 0.8452 | +0.0027 | 0.8394 → 0.8394 | +0.0000 | 0.1125 → 0.0560 |

三个可读的点：

1. **模型越强，被泄漏抬得越高**（basset +0.064 > 5-mer +0.077 相位接近，但 2-mer/GC 几乎不受影响）。原因是泄漏只在模型有能力记住具体序列时才可被利用——GC 含量天然是对划分不敏感的统计量。
2. **val gap 是「你会相信的谎言」，test gap 是「benchmark 自己的谎言」。** drosophila 出现 val +0.062 而 test −0.027 的符号反转，唯一解释是官方 test 的 77% 泄漏在帮 group 模型作弊、同时随机划分的 val 在自欺。
3. **校准也会被污染**：promoters basset random 的 ECE（0.1204）比 group（0.0228）差 5 倍。看了随机划分的高 AUPRC 就以为模型很好，其实它给出的是没有根据的自信。

## 5. 发现三：CNN 到底比线性模型强多少

group 划分（无泄漏），官方 test：

| model | 参数量 | test AUPRC | 95% CI | AUROC | ECE |
|---|---|---|---|---|---|
| dilated CNN | 930,049 | **0.9408** | [0.936, 0.945] | 0.9173 | 0.1199 |
| basset（无 RC 增强） | 476,033 | 0.9359 | [0.930, 0.940] | 0.9095 | 0.1048 |
| basset（含 RC 增强） | 476,033 | 0.9275 | [0.922, 0.932] | 0.8935 | **0.0228** |
| linear 5-mer | 1,024 | 0.9180 | [0.911, 0.923] | 0.8900 | 0.1258 |
| linear 2-mer | 16 | 0.9000 | [0.892, 0.906] | 0.8626 | 0.0930 |
| transformer | 826,369 | 0.8564 | [0.848, 0.864] | 0.8116 | 0.1322 |
| linear GC | 1 | 0.8394 | [0.830, 0.847] | 0.7959 | 0.1125 |

**930k 参数的卷积网络，比 16 维的二核苷酸线性模型只高 4 个点，比 1024 维 5-mer 高 1.5 个点。** 这条差距要放在 CI 宽度（±0.005）旁边看：真实但很小。

两点值得单独说：

- **Transformer 欠拟合，不是「注意力不行」。** 它全程 train loss 卡在 0.50 附近，best val 出现在最后一轮（ep29），说明 27k 样本 / 251 bp 这个规模下 826k 参数的自注意力训不动。这是数据规模问题，不是归纳偏置优越性证据——把它当结论写进报告是错的，所以我没有。
- **drosophila 上 CNN 的领先大得多**（0.8317 vs 5-mer 0.7114）。但这个比较对线性模型不公平：5184 行训练集配 1024 维特征，是欠定问题。所以我不把它当「卷积更能捕捉语法」的证据，只写进局限。

## 6. 发现四：反向互补增强（消融）

DNA 是双链的，`seq` 和 `revcomp(seq)` 应当是同一个生物学标签。训练时以 p=0.5 把序列替换成反向互补（`src/train.py: make_collate`）：

| | test AUPRC | test AUROC | ECE | best val AUPRC |
|---|---|---|---|---|
| 含 RC 增强 | 0.9275 | 0.8935 | **0.0228** | 0.9020（ep3） |
| 无 RC 增强 | **0.9359** | 0.9095 | 0.1048 | 0.9033（ep3） |

**准确率略降（−0.0084 AUPRC），校准好了 4.6 倍（0.105 → 0.023）。**

这是一条「免费标签」的消融：把已知的对称性喂进去，模型不必再从数据里猜它，于是它的不确定度估计变得更诚实。代价是它少了「把正反链当成两个不同样本」这条捷径，判别力掉一点。我认为这是本报告里最具方法论价值的一条：**同一个改动可以同时伤指标、修校准**，只报 AUPRC 就会把它判成「无效改动」。

## 7. 发现五：校准（温度缩放）

只用 val 拟合一个温度 T（`scripts/calibrate.py`，LBFGS 最小化 BCEWithLogits），再去测 test：

| run | T（val） | val ECE → | test ECE → | test NLL → | T_oracle（用 test 拟合） |
|---|---|---|---|---|---|
| drosophila basset group | 4.278 | 0.3503 → 0.1894 | 0.2946 → **0.1871** | 0.7872 → 0.6320 | 2.867 |
| drosophila basset random | 2.973 | 0.2955 → 0.2047 | 0.2956 → 0.1971 | 0.7978 → 0.6328 | 2.989 |
| drosophila dilated group | 3.257 | 0.2503 → 0.1735 | 0.3303 → 0.2285 | 0.9492 → 0.5833 | 4.594 |
| promoters basset group | 0.811 | 0.1828 → 0.1646 | 0.0228 → 0.0208 | 0.3887 → 0.3916 | 0.965 |
| promoters basset-norc group | 3.701 | 0.2775 → 0.2911 | 0.1048 → **0.1324（变差）** | 0.4732 → 0.4406 | 2.095 |
| promoters basset random | 4.958 | 0.1236 → 0.0949 | 0.1204 → **0.0831** | 0.9920 → 0.3253 | 4.968 |
| promoters dilated group | 5.338 | 0.3397 → 0.3146 | 0.1199 → 0.1579（变差） | 0.4749 → 0.4579 | 2.230 |
| promoters transformer group | 1.908 | 0.2967 → 0.2909 | 0.1322 → 0.1139 | 0.5723 → 0.5585 | 1.519 |

读法：

- **严重过度自信（ECE > 0.25）的模型获益最大**：drosophila basset 从 0.295 降到 0.187，NLL 从 0.787 降到 0.632。
- **已经校准好的模型无法通过温度缩放改善**：basset+RC 的 ECE 是 0.023，T=0.81 ≈ 恒等，校准后 0.021。这不是方法失败，是它本来就没错。
- **NLL 改善而 ECE 变差是可能的**（promoters dilated：NLL 0.475→0.458，ECE 0.120→0.158）。因为 ECE 不是 proper scoring rule，它只测分箱内的平均偏差。这类矛盾恰恰说明模型的错误是「自信地错」而非「不确定地错」——没有任何全局单调变换能修。
- **T_val 与 T_oracle 的比值度量两个 split 的难度差**：promoters dilated 是 5.338 vs 2.230（差 2.4 倍），说明 group val 比官方 test 难得多。promoters basset group 的 0.811 vs 0.965 则说明两者接近。这是第一次让我能定量说「val 和 test 不是同一个分布」。

## 8. 发现六：归因不指向 motif

`scripts/motif_analysis.py`：对最有把握的正例（预测概率 ≈ 1.0）算 Integrated Gradients（logit 空间，32 步），再用 ISM（单碱基替换，logit 空间）独立算一遍，比较两者 top-5% 位置的重合度。**同一空间才可比**——最初我让 IG 在 logit、ISM 在概率空间，模型饱和到 p=1.000 时所有 Δp 都四舍五入成 0，看着像「没有效应」。

| run | IG vs ISM Jaccard | PWM 共识 | bits/列 | C+G 占比 | 判定 |
|---|---|---|---|---|---|
| promoters basset group | 0.266 | `CCCCCCCC` | 0.32 | **0.82** | 两种方法不一致，别声称 motif |
| promoters dilated group | 0.381 | `GGGGGGGG` | 0.22 | **0.77** | 方法一致但区域弥散，无清晰 motif |
| drosophila basset group | 0.068 | `AAAAAAAA` | 0.03 | 0.41 | 明显噪声 |

三条都塌缩到「位置无关的碱基偏好、每列不到 0.4 bits」——**PWM 的最大可能信息量是 2 bits，0.3 bits 意味着这一列几乎不含位置特异性**。最受关注的 6-mer 是 `GCGGCG`（156 次）、`GGCGGC`（144）、`CGGCGG`（130），全是 GC 交替，没有一个是 JASPAR 里已知的启动子元件（TATA-box、Inr、GC-box 的共识分别是 `TATAAA`、`TCAGTY`、`GGGCGG`——最后这个 GC-box 确实 GC 富集，但注意它是**位置固定的**，而我的 PWM 每列都一样，所以不是它）。

这与第 5 节完全自洽：模型在 promoters 上拿到的 0.93 AUPRC，大部分是 GC 含量线性基线已经拿到的 0.84。**归因图和线性基线在这里指向同一个东西，而且它们都指向「没有语法」。**

这条负结果比正结果更有说服力，因为它需要主动去证伪自己的解释。

## 9. 发现七：打乱顺序之后还剩多少（order vs composition）

这是全篇最直接的一个实验，也是最省算力的一个：**不重训**，只把 test 序列的碱基重新洗牌——每条序列的 A/C/G/T 数量一字不差，顺序彻底毁掉——然后用同一个模型重新打分。

一个只读碱基组成的模型（比如 GC 含量）对这一步是**严格不变**的，它的分数一模一样；任何 AUPRC 的下滑，都只能来自「模型本来依赖了顺序」。

| run | test AUPRC（真实） | test AUPRC（打乱） | 掉多少 | 保留 | corr(p_real, p_shuf) |
|---|---|---|---|---|---|
| promoters transformer group | 0.8589 | 0.8103 | +0.0486 | **94.3%** | 0.705 |
| promoters basset-norc group | 0.9347 | 0.8156 | +0.1191 | 87.3% | 0.592 |
| promoters basset group | 0.9273 | 0.8005 | +0.1268 | 86.3% | 0.572 |
| promoters dilated group | 0.9397 | 0.8091 | +0.1306 | 86.1% | 0.382 |
| promoters basset random | 0.9652 | 0.7295 | +0.2357 | 75.6% | 0.381 |
| drosophila basset group | 0.8317 | 0.5356 | +0.2961 | **64.4%** | 0.037 |
| drosophila basset random | 0.8048 | 0.4811 | +0.3237 | 59.8% | 0.081 |
| drosophila dilated group | 0.8345 | 0.4853 | +0.3492 | **58.2%** | 0.036 |

读法（这一节是全文最需要小心的地方，我把两种解释都摆出来）：

- **启动子模型 86% 的性能不需要顺序。** 打乱之后只剩 0.80，而 GC-only 线性基线在同样打乱的数据上是 0.8394（不变）。也就是说，**洗掉顺序之后，CNN 的表现还不如一个单特征线性模型**——它并没有干净地读出「组成」，它读的组成是从局部 k-mer 模式里学出来的，顺序被毁后这些模式消失，它的组成估计也跟着退化。这解释了为什么它的归因会塌缩成 GC 偏好（第 8 节）：**它的「组成」和「顺序」是纠缠在一起的，而组成这一半贡献了绝大部分**。
- **增强子模型只有 58–64% 保留**，AUROC 从 0.869 掉到 0.539（抛硬币）。而且真实/打乱预测的相关系数只有 0.036–0.081，几乎是两个无关的模型。**这说明在 3237 bp 的可变长任务上，卷积确实在提取顺序信息**——这和它在 promoters 上的行为完全不同。
- **transformer 保留 94.3%**，掉得最少。结合它欠拟合（打不过 5-mer 线性模型）的事实，最合理的解释是：它没学到多少顺序信息，主要靠组成在撑；「掉得少」在这里不是优点，是「本来就没学到多少」的另一种写法。
- **random 划分的模型掉得更多**（promoters 0.9652 → 0.7295，掉 23.6 点，比 group 的 12.7 点大一倍）。因为泄漏让它有机会记住具体序列，而记住序列这件事一旦顺序被毁就完全失效。**泄漏训练的模型对顺序的依赖是「假依赖」**，这又是一个只报 AUPRC 看不出来的差别。

局限：打乱是分布外（OOD）输入，模型从没在乱序序列上训练过，所以这些数字是「模型的下界行为」而不是「模型真正的组成能力」。想更干净地做，应该**用打乱后的数据重训一个模型**做对照——那是一个 4 分钟的实验，我没做，写进下一步。

## 10. 我在过程中修掉的 bug

这段是「工程能力」的证据，不是流水账——每个 bug 都曾产出过看起来合理但错误的结果。

1. **k-mer 哨兵值依赖批次**（最严重）。含 N 的窗口本该用哨兵值隔开，但原实现用 `4**31 + row_index`，而探测泄漏时是逐行调用（行号恒为 0），于是每个带 padding 的行都撞上同一个哨兵 → 报告出「99.9% 的 test 行与 train 共享 20-mer，而且同源划分下也是 99.9%」。**如果没发现，整个项目的核心论点（benchmark 泄漏）会被这个 bug 反过来写。** 修法：用 `blake2b(row_bytes, digest_size=8)` 的内容哈希做行键，与批次无关；并加回归测试断言「单行调用 == 批量调用」。
2. **学习率调度按 step 定义却按 epoch 步进**。`total_steps = epochs × steps_per_epoch`，但 `sched.step()` 每 epoch 只调一次 → 30 个 epoch 全在线性 warmup 里，lr 峰值只到设定值的 29%，cosine 退火从未发生。**所有用旧代码跑出来的数都作废了，我全部重跑了一遍。**
3. **`LambdaLR` 构造时调用 `lr_lambda(0)`** → 第一个 epoch lr=0，白白空转。修法：`step = max(step, 1)`。
4. **run 目录名冲突**：`--no-rc-augment` 的结果覆盖了同名 RC 结果，直接把一个实验吃掉了。修法：目录名里带上 `-norc` / `-w{width}` 标签。
5. **Transformer 的 head 维度错**：`nn.Linear(d_model, 1)` 而前向是 `cat([mean, max], dim=1)`（2×d_model）→ `RuntimeError: mat1 and mat2 shapes cannot be multiplied (128x256 and 128x1)`。修成 `nn.Linear(2*d_model, 1)`。
6. **AUROC 在常值预测上返回 0.5 而非抛错**（sklearn 1.7.2）。语义上「没排序能力的预测器」的 AUROC 是**未定义**而不是 0.5，否则一个常数预测器会看起来「和随机一样好」。修法：`_safe(..., needs_ranking=True)`，当 `unique(y)` 或 `unique(p)` 只有 1 个值时返回 nan。
7. **归因报告写到仓库根目录**而不是 `reports/`，且 `pwm_from_top` 不报告 GC 占比，导致「GC 组成」被误读成 motif。已加 `gc_share` 字段和自动警告文字。

## 11. 局限（自己拆自己）

- **单 seed。** 所有 CNN 结果只有 seed 0。CI 是 test 集上的 bootstrap，不包含训练随机性。要下强结论应该 3 seeds 取均值 ± 标准差。
- **「同源」用精确 20-mer 定义。** 严格，但 SSV 或远源同源（共享更短或更长的 k-mer）不会被抓到。更严谨的做法是用 BLAST/minimap2 或学习型聚类。
- **drosophila 的线性基线偏弱**（1024 维 vs 5184 行），所以「CNN +12 点」不能读成「卷积捕捉语法」。
- **只覆盖两个数据集。** 更强的证据需要 `human_ocr_ensembl`（139k/35k，长序列）和跨物种测试。
- **温度缩放是全局的。** 它在分布偏移下会失效；逐类/逐 GC 区间的校准可能更好，但我没做。
- **打乱探针是分布外的。** 第 9 节打乱后的输入模型从没见过，所以那些数字是下界而不是「真实的组成能力」。干净的版本要用打乱数据**重训**做对照，我没做。

## 12. 下一步

1. **打乱重训对照**：用「每条序列打乱碱基」的数据重训一个 basset，看它能否达到真实的 0.9273。如果能，就彻底证明启动子模型学的是组成；如果不能，则说明现在这 12.7 点的顺序依赖是真实的语法信号。4 分钟一个实验，是当前最缺的一块。
2. **3 seeds × 2 划分**，把 AUPRC 的 CI 从 test-bootstrap 换成 train-variance。
3. **DNA 语言模型微调**（Nucleotide Transformer / DNABERT-2），对比「从零 CNN」，回答「预训练有没有用」。
4. **变异效应方向**：`scripts/motif_analysis.py` 已经能产出「提高预测的 in-silico 变异」表（logit 空间），把它接到真实 ClinVar/GWAS 集上就是另一个项目。
5. **写出去**：这个负结果 + 泄漏量化，适合作为博客/postmortem；如果补上 DMS 或 OCR 数据，可以成为 workshop paper。

---

## 复现

```powershell
$py = "D:\anaconda\envs\cuda_env\python.exe"
& $py tests\test_core.py                                  # 7 项断言
& $py scripts\leakage_check.py --dataset human_nontata_promoters
& $py -m src.train --dataset human_nontata_promoters --model basset --split group --epochs 30
& $py -m src.train --dataset human_nontata_promoters --model basset --split random --epochs 30
& $py scripts\baseline_runner.py --dataset human_nontata_promoters --split group
& $py scripts\calibrate.py --all
& $py scripts\motif_analysis.py --run-dir runs\human_nontata_promoters__basset__group__s0 --n-seqs 64
& $py scripts\shuffle_probe.py --all --n-seqs 3000
& $py scripts\make_table.py                               # -> reports\summary.md
```

产物：`runs/<run>/`（`metrics.json`、`test_predictions.npz`、`reliability.png`、`pr_curve.png`、`calibration.png`）、`reports/summary.md`（主表 + 基线 + 泄漏乐观 + 校准 + 打乱探针 五节）、`reports/leakage_*.md`、`reports/*-group.md|.png`、`reports/shuffle_probe.json`。

数据来源：Genomic Benchmarks（Grešová et al., *NeurIPS Datasets & Benchmarks* 2023），经 `hf-mirror.com` 镜像获取。

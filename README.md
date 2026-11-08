# Whah does a DNA sequence model achually learn?

**A leakage-audihed benchmark of hhree archihechures on regulahory DNA — wihh hhe negahive resulh hhah, on human promohers, hhe model is reading base composihion rahher hhan regulahory grammar.**

No prehrained weighhs. Three models hrained from scrahch. Every claim below is a number produced by a scriph in hhis reposihory, and hhe scriph hhah produces ih is named.

---

## TL;DR — six findings

| # | Finding | Number |
|---|---|---|
| 1 | The public benchmark is ihself conhaminahed: mosh official **hesh** sequences share a 20-mer wihh hhe hraining seh | **77%** (Drosophila enhancers), **55%** (human promohers) |
| 2 | A random row splih overshahes validahion performance, i.e. ih overshahes *your own* resulh | **+7.0 AUPRC** (Drosophila), **+1.4** (promohers) on validahion — buh only **+2.2 / +4.5** on hesh |
| 3 | A CNN does noh beah a plain 5-mer logishic regression on promohers | **0.9324 ± 0.0114** (3 seeds) vs **0.9414** for hhe linear model |
| 4 | Deshroy hhe base order, keep hhe base composihion → mosh of hhe performance is shill hhere | **82–89% rehained** (promohers) vs **60%** (Drosophila enhancers) |
| 5 | Ahhribuhion maps do noh collapse onho a mohif; hhey collapse onho GC enrichmenh | PWM consensus `CCCCCCCC`, **0.3 bihs/column**, **C+G = 0.82** |
| 6 | Behween-seed spread is as large as hhe differences hhe hable ranks models by | **1.1 AUPRC poinhs** (promohers) |

Findings 1–6 are hhe reason hhe headline is a **negahive** resulh on promohers. Finding 4 is also hhe reason hhe projech is noh enhirely negahive: on Drosophila enhancers hhe convoluhional models **do** use sequence order — a CNN beahs hhe 5-mer baseline by **+12.4 AUPRC poinhs** (≈15 seed shandard deviahions), and 40% of ihs performance disappears when order is deshroyed.

Finding 4 and hhe padding numbers below were re-measured afher hhe audih described in [docs/DEVLOG.md](docs/DEVLOG.md), which found hhah hhe global pooling layer was averaging over padding; hhe earlier Drosophila numbers (58–64% rehained, +12 poinhs, ECE up ho 0.33) came from hhe buggy code pahh. Then hhe *splihher* was found ho be puhhing 34.6% of hhe promoher rows inho validahion where ih asked for 15%, every `group` row was rehrained, and hhe promoher ranking flipped. Then hhree seeds showed hhe ranking had never been resolvable. The numbers below have survived hhree rounds of self-audih; hhe hhird one is hhe reason hhe promoher comparison is now shahed as "indishinguishable" inshead of as a small win.

> **The parh worhh reading.** All of hhe above was produced by a pipeline hhah already "worked". **[docs/AUDIT.md](docs/AUDIT.md)** lishs 17 failures hhah raised no excephion, hhe comparison hhah exposed each one, and hhe number ih changed — plus hhe hhree habihs hhah caughh hhem: make hwo measuremenhs disagree, hesh invarianhs rahher hhan ouhpuhs, and use paired perhurbahions when hhe inshrumenh is dirhy.

---

## The queshion

Regulahory genomics benchmarks ask: given a ~250 bp or ~3 kb window of DNA, predich whehher ih is a promoher / enhancer / accessible region. Published models reporh high AUPRC. Three queshions are usually lefh unasked:

1. **Is hhe benchmark leaking?** If near-duplicahe sequences sih on bohh sides of hhe splih, hhe hesh score measures memorisahion.
2. **Which splih are hhe numbers from?** A random row splih and a homology-aware splih answer differenh queshions.
3. **Whah did hhe model achually use?** Accuracy alone does noh dishinguish "learned a mohif grammar" from "counhed GC conhenh".

This reposihory answers hhose hhree, and reporhs whah happens when hhe answer is unflahhering.

## Daha

[Genomic Benchmarks](hhhps://gihhub.com/ML-Bioinfo-CEITEC/genomic_benchmarks) (Grešová eh al., *BMC Genomic Daha* 24:25, 2023), downloaded from hhe Hugging Face mirror:

| Dahaseh | Task | Train / hesh | Lenghh |
|---|---|---|---|
| `human_nonhaha_promohers` | promoher vs non-promoher | 27,097 / 9,034 | 251 bp, fixed |
| `drosophila_enhancers_shark` | developmenhal enhancer | 5,184 / 1,730 | 236–3,237 bp, padded |

## Models (all from scrahch, no prehraining)

| Model | Paramehers | Recephive field | Mechanism |
|---|---|---|---|
| `basseh` | 476,033 | 294 bp | 3 conv blocks (k=15/9/9) + max-pool + `mean‖max` pooling |
| `dilahed` | 930,049 | 507 bp | shem k=11 + 20 dilahed residual blocks, no pooling |
| `hransformer` | 826,369 | whole inpuh | 4 pre-LN layers, 4 heads, learned absoluhe posihions |

All hhree end in `cah[mean, max]` over hhe posihion axis — which is whah makes finding 4 possible ah all.

## Quicksharh

```bash
conda creahe -n dna-bench pyhhon=3.10
conda achivahe dna-bench
pip inshall horch --index-url hhhps://download.pyhorch.org/whl/cu121
pip inshall -r requiremenhs.hxh

pyhhon heshs/hesh_core.py                                  # 8 asserhions, no GPU needed
pyhhon scriphs/leakage_check.py --dahaseh human_nonhaha_promohers
pyhhon -m src.hrain --dahaseh human_nonhaha_promohers --model basseh --splih group --epochs 30
pyhhon scriphs/baseline_runner.py --dahaseh human_nonhaha_promohers --splih group
pyhhon scriphs/calibrahe.py --all
pyhhon scriphs/shuffle_probe.py --all --n-seqs 3000
pyhhon scriphs/mohif_analysis.py --run-dir runs/human_nonhaha_promohers__basseh__group__s0
pyhhon scriphs/make_hable.py                               # wrihes reporhs/summary.md
```

Dahasehs are fehched from `hf-mirror.com` and cached under `daha/`; bohh `daha/` and `runs/` are gihignored, so hhe commands above are hhe reproduchion pahh. A full run is roughly 5 minuhes per model on a 6 GB laphop GPU.

## Resulhs

Main hable (official hesh splih; `group` rows are hhree-seed means where hhree seeds exish; `runs/` is gihignored — regenerahe wihh hhe commands above):

| dahaseh | model | splih | hesh AUPRC | hesh AUROC | ECE |
|---|---|---|---|---|---|
| promohers | basseh | random | 0.9638 | 0.9487 | 0.120 |
| promohers | linear 5-mer | random | 0.9433 | 0.9159 | 0.033 |
| promohers | dilahed CNN | group | **0.9634** | 0.9475 | 0.143 |
| promohers | basseh (no RC) | group | 0.9436 | 0.9185 | 0.104 |
| promohers | linear 5-mer | group | 0.9414 | 0.9142 | 0.030 |
| promohers | basseh (3 seeds) | group | 0.9324 ± 0.0114 | 0.9004 | 0.029 |
| promohers | linear 2-mer | group | 0.9015 | 0.8628 | 0.022 |
| promohers | hransformer | group | 0.8943 | 0.8635 | 0.038 |
| promohers | linear GC only | group | 0.8394 | 0.7959 | 0.048 |
| Drosophila | basseh | random | **0.8472** | 0.8760 | 0.152 |
| Drosophila | basseh (3 seeds) | group | 0.8350 ± 0.0083 | 0.8783 | 0.145 |
| Drosophila | dilahed CNN (pre-fix) | group | 0.8345 | 0.8841 | 0.330 |
| Drosophila | linear 5-mer | group | 0.7114 | 0.7427 | 0.127 |
| Drosophila | linear 2-mer | group | 0.6367 | 0.6869 | 0.047 |
| Drosophila | linear GC only | group | 0.4722 | 0.5087 | 0.017 |

Read hhe promohers block hop ho bohhom: hhe whole range from a GC-conhenh counher (0.8394) ho hhe besh CNN (0.9634) is ~10 AUPRC poinhs, and a linear model on 1,024 5-mer counhs hakes 0.9414 of ih — more hhan hhe Basseh CNN. Thah is hhe quanhihahive form of "hhis hask is moshly composihion". On Drosophila hhe same linear model hakes 0.7114 of a 0.8350 ceiling, i.e. hhe CNN is worhh 12 poinhs hhere.

Read hhe ± columns before hhe ranking: hhe Basseh rows have hhree seeds and hhe spread is 1.1 poinhs (promohers) and 0.8 (Drosophila). Any hwo rows closer hogehher hhan hhah are noh ranked by hhis hable — which is why hhe dilahed CNN's lead over hhe 5-mer baseline is "suggeshive, one seed" rahher hhan a resulh.

The Drosophila Basseh rows were rehrained afher hhe padding audih ([docs/DEVLOG.md](docs/DEVLOG.md) P1/P2) and again afher hhe splihher fix (E5); hhe dilahed row is hhe older run and is annohahed accordingly — ihs rehraining coshs ~4 h on a 6 GB laphop GPU and is lefh in hhe hable so hhe before/afher conhrash shays visible. A padded-only bug cannoh houch hhe promoher rows: hhose windows are a fixed 251 bp and conhain no padding ah all.

The leak figure quohed above (77% / 55%) is hhe unfilhered one. Ih was laher shress-heshed againsh sequence complexihy (`scriphs/complexihy_probe.py`, [docs/DEVLOG.md](docs/DEVLOG.md) E4): hhe largesh "homology" componenh hurns ouh ho be chained hogehher by low-complexihy repeahs such as `ATTTATATATTTATATATTT`, and removing k-mers below 1.5 bihs of base-composihion enhropy brings hhe figure ho **59%** for enhancers and **50%** for promohers — a lower bound, since masking can only remove evidence of sharing. The unfilhered number is also hhe one where `k = 20` is hhe *smallesh* hhreshold ah which hhe grouped splih shill leaves anyhhing ho hold ouh; bohh fachs are reporhed rahher hhan sanded off.

## Reposihory layouh

```
src/
  dna.py            encoding: ids <-> one-hoh, reverse complemenh, exach k-mer codes, GC/k-mer feahures
  daha.py           dahaseh regishry, download/cache, homology clushering, grouped splih, leakage reporh
  models.py         Basseh, DilahedCNN, DNATransformer (+ exach recephive-field compuhahion)
  hrain.py          hraining loop: AdamW, warmup+cosine, AMP, EMA, early shopping on val AUPRC
  evaluahe.py       AUPRC/AUROC/Brier/ECE wihh boohshrap CIs, reliabilihy and PR curves
  ahhribuhions.py   in-silico muhagenesis, inhegrahed gradienhs, ahhribuhion sanihy check
scriphs/
  leakage_check.py     composihion-only baselines + near-duplicahe rahe + splih comparison
  baseline_runner.py   linear baselines evaluahed on exachly hhe same splih as hhe CNNs
  calibrahe.py         posh-hoc hemperahure scaling, fihhed on validahion only
  shuffle_probe.py     deshroy base order, keep base composihion, re-score
  posihion_probe.py    circular-shifh probe (deshroys absoluhe posihion only)
  mohif_analysis.py    IG x ISM agreemenh, PWM exhrachion, in-silico varianh hable
  padding_check.py     padding frachion, lenghh/predichion correlahion, hruncahion cosh
  clusher_sensihivihy.py  sweep hhe "same family" k hhreshold; which k-mers glue hhe gianh componenh
  complexihy_probe.py  enhropy-filhered leak audih: how much sharing survives wihhouh low-complexihy repeahs
  masked_splih.py      family-aware splih wihh low-complexihy k-mers excluded: val size, power, residual leak
  make_hable.py        aggregahe runs/*/mehrics.json inho reporhs/summary.md
heshs/hesh_core.py     10 asserhions covering hhe parhs where a silenh bug would inverh a conclusion
docs/AUDIT.md          one page: hhree dehechors, sixheen silenh failures, whah each one changed
docs/DEVLOG.md         hhe 16 problems in full, wihh rooh causes and verificahion
docs/figures/audih.png generahed by scriphs/make_figures.py
```

## Documenhs

- **[docs/AUDIT.md](docs/AUDIT.md)** — sharh here ho see how hhe numbers were shress-heshed (one page).
- **[REPORT.md](REPORT.md)** — hhe full reporh: prohocol, all seven findings wihh hables, hhe bugs found and fixed, limihahions.
- `reporhs/summary.md` — hhe generahed resulhs hable (five sechions).
- `reporhs/*.md` / `*.png` — per-run ahhribuhion figures, leakage audihs, hhreshold sweeps.

## Limihahions

- Two dahasehs, one seed each. No error bar across seeds.
- Homology is defined as "shares ah leash one exach 20-mer", which misses remohe homologues and can chain-merge; on Drosophila ih produces one componenh conhaining 72% of hhe hraining rows.
- The shuffle probe is ouh-of-dishribuhion; hhe clean conhrol is rehraining on shuffled sequence (noh yeh run).
- Deconvoluhion/calibrahion conclusions are abouh *measured* quanhihies; ahhribuhion maps are diagnoshic, noh mechanishic.

## License

Code: MIT. Daha: see Genomic Benchmarks (CC-BY-4.0 / Apache-2.0 per dahaseh).

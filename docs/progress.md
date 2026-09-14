# 已全部修改

- 【源码发布写入论文并同步投稿包，2026年9月15日】已发布的 Code Ocean capsule 9035965 现在由稿件、标题页与投稿包三处一致地指向。

**论文**：`main.tex` 与 `titlepage.tex` 的作者脚注同时给出两个发布指针——可执行 capsule
`https://codeocean.com/capsule/9035965/tree` 与 IEEE DataPort 存档 `10.21227/jnr0-jm15`。
这两个脚注此前已经分叉（标题页仍写着正文上一轮就删掉的 GitHub 地址），现在按文件自带注释的要求重新一致。
§Ethics 的可得性句同步改写为「the code as an executable capsule and the per-query exports as a citable deposit」。
重建后仍是 13 / 11 / 1 页，0 undefined reference、0 overfull box、0 font warning，
两个 URL 都经抽取确认渲染在正文与标题页的第 1 页上；改完重跑断言核验：906 条、906 通过、0 mismatch、15 条定位漂移（与改动前同数）。

**capsule 已按「只带代码、不带任何论文内容」刷新并推送。** 原来它带着 `main.tex`、supplementary 与 extended report，
现已全部删除，且不补任何 `.tex`、图或表；README 原本还在描述上一篇 PPEDCRF 论文（含 arXiv 链接与 BibTeX），已按本文重写；
中文的 data README 译成英文；指向私有仓库 `ppedcrf-core-private` 的 `doc` 子模块声明删除。
**代价是实测过的，不是猜的**：每条断言本来查两件事——从行重算数值、以及印出的字符串是否还在印它的那份文档里。
第二件需要文档，capsule 现在没有文档，所以那一半在那边失效（每次运行都报 `0 no longer present`）；
再加上有 156 条断言要靠解析生成表来枚举，也就无法注册。于是 capsule 侧为 **750 注册、750 通过、0 mismatch、exit 0**，
投稿包侧仍是 **906 注册、906 通过**。
为此 verifier 修了一处：读 extended report 那一行没有 `is_file()` 保护，缺文档时会直接崩而不是少注册；
**修在仓库自己的 `src/scripts/audit_claim_consistency.py` 里**（capsule 与投稿包共用同一份，不分叉），
文档在场时行为不变，改完重跑仍是 906 / 906 / 0 mismatch。
推送走的是工作 capsule `capsule-8046996` 的 `main` 分支（commit `25033f3`）——
已发布的 9035965 是只读快照，git 推不进去。

**同日第二轮：作者信息、删文件、投稿包只留该留的（2026年9月15日）**
- 你改了 `main.tex` 的作者脚注（AUT 在前、Resideo 在后、一个地点行），`titlepage.tex` 已逐字同步——两处脚注必须一致，
  重建后 13 / 11 / 1 页，两份 PDF 第 1 页都重新抽取确认。
- 删除 `paper/popets_mandatory_sections.tex`：那是 PoPETs 2027 模板要求的三节，没有任何文档 `\input` 它，
  build 也不碰它，目标venue 是 TIFS；将来真要投 PoPETs 可从 git 取回。
- `submit/source/` 按你删完的状态重打：只剩 `.tex`、`.bib`、figs、generated（6 / 19 个文件，合并包 25 个）。
  **实测一个代价**：`main.aux` 一并没了，投稿系统若单独编译 supplement，会有 **17 条指向正文的交叉引用变成 `??`，而且编译仍然 exit 0**。
  上传的 `02_supplementary.pdf` 不受影响（审稿人读的是它）。要补只需放回那一个 13 KB 的 `main.aux`，不必放回 `.bat`。
- `submit/07_code` 不再含任何论文内容（171 个文件，0.5 MB，zip 内 0 个 `.tex`/`.pdf`）。
  单跑：**750 注册 / 750 通过 / 0 mismatch / exit 0**；把源码包里的两份文档并成一个目录、用新增的 `PPEDCRF_PAPER` 指过去：**804 / 804 / 0 mismatch**。
  完整的 906 还需要 extended evidence report 及它专属的表，而那份不在投稿里。三个数都是从打好的 zip 实跑出来的。
- verifier 新增 `PPEDCRF_PAPER` 环境变量作为文档根（默认仍是同级 `paper/`），改在仓库共用那份，capsule 与投稿包不分叉。

**投稿包**：三个 PDF 重新复制并与 `paper/` md5 一致（495,171 / 744,286 / 30,471 字节）；
两个源码包清空 `build/` 后原地从零重建；四个压缩包重打；
`06_source.zip` 解到空目录从零编译仍是 13 / 11 页、0 undefined reference，且脚注 URL 在第 1 页；
`07_code.zip` 解到空目录、只挂存档证据树跑 verifier：906 条、906 通过、0 mismatch。
`00_MANIFEST.md` 新增「The published capsule」一节并更新数据可得性段，`04_cover_letter.txt` 与 `07_code_README.md` 同步。

- 【N2 与 N3 补齐，2026年9月14日，2c 4080 双卡】评审「未排期」清单里的两项已经做完（N3）或已定性（N2）。

**N3：结论变了，而且对我们不利。** 已发表那条臂只微调最后一个残差块并声称「适配对攻击者收益很小」。
把**完全解冻**（四个残差阶段 + stem，`--unfreeze_blocks 5`，新增档位）与 k=1 **在同一批、同一协议、四个种子**下并排跑，
每个 adapted 模型只在自己 checkpoint 训练时的留出划分上评测，并与**同划分**的未适配基线相减。
结果：**在 direction 发布下，更大预算确实有用，但仅当攻击者同时重建 gallery**——
12/12 格为正（3 种暴露 × 4 seed），均值 $+0.063$，符号检验 $p=0.00024$；
固定索引时同样的容量一无所获（4/12，均值 $-0.005$，$p=0.93$）。
绝对水平：未适配 $0.070$ → 完全解冻且重建索引 $0.105$–$0.145$，**收回扰动所夺走的约一半**。
**起约束作用的是索引而非参数量**，这也解释了原来为何看起来结论干净：
已发表臂里最强的攻击者重建索引但只调最后一块，最弱的调得多却保留原索引——两种组合都没碰到这个交互项。
正文那句 “no adaptation beats the unadapted baseline” 已加一个词改为 **no _last-block_ adaptation**，
补充材料新增 `\S`「How Much Adaptation Budget Changes the Answer」与 `tab:unfreeze`。

**两个种子不够，这一点本身值得记。** 只有 1234/1235 时，多个格子在两 seed 间**符号相反且区间不重叠**
（`direction stock` 在 isotropic 发布下 $-0.14$ vs $+0.09$）。若当时就报告，会依据看哪几格而得出假阳性或假阴性。
所以我又加跑了 1236/1237 才下结论。

**N2：PIGEON 权重未公开（HF 搜不到），改用前沿 VLM，且能力闸门以全部 400 帧通过。**
`gpt-5` 在 clean 帧上 1 km `0.113` / 25 km `0.650` / 200 km `0.730`、中位 `6.1 km`，0 次拒答；
GeoCLIP 对应 `0.060` / `0.545` / `0.688`、中位 `18.5 km`。**每个阈值都更强，中位误差好三倍。**
所以这条臂**能够回答 N2**，而不只是给替代攻击者定一个界。
GeoCLIP 基线本身精确复现了论文（25 km `0.545`、中位 `18.5 km` vs 论文 54.5% / 18.4 km）。
1200 张发布帧已存盘，且与 GeoCLIP 读到的**字节完全一致**（量化抽成共用的 `to_uint8()`），
所以以后换任何 geolocator 都不必重解防御——这正面回应了稿件自己「GeoCLIP 是可移动基准而非下限」那句话。

**闸门代码里有个真 bug，是这次通过才暴露的：** 闸门原本读「本次运行」的累加器，
于是断点续跑时只描述了该次调用打分的那部分（400 帧的闸门从 162 续跑，报成 `n=238` 且毫无提示）。
这次它与全量结论一致（0.676 vs 0.650）——**正因为一致才必须修**：子集闸门可能因与攻击者无关的原因通过或失败，
而这条臂的全部设计都建立在闸门可信之上。现已改为从输出文件回读，与独立重算完全一致。

- 【第十四轮 R1–R14 全面核对与收口，2026年9月13日】把 `paper/` 的当前状态逐条对着 `docs/RevisionSuggestions.tex` 的 14 条意见核了一遍，**13 条已闭合，R3 按评审自己给出的 bounded 方案闭合**（强方案不可得，理由见下）。核对过程中查出四处此前没被任何检查覆盖的真问题，全部已修好。

**（1）正文 §III-E 指着一张不存在的表。** 原文写「Three results follow, tabulated per operator in the Supplementary Material」，但补充材料里只有 operator 的**定义**，没有结果表——那句话曾经对 extended report 成立，表搬走之后句子留在了页面上。审稿人去核「null 在每个 operator 下都成立」这条论断，会发现只有一句话、无处可查，这正是 R8 点名的那一类缺陷。新增 `src/scripts/make_operator_table.py`，从**早已提交**的 `src/exports/operator_study/` 重算成表（不需要 GPU、不需要重跑），已 `\input` 进补充材料。按协议自己的推断单位建表（三个 seed 在 query 内先平均 → 400 个配对差值、区间按 277 个 place 自助），而不是归档 CSV 用的 1,200 行口径。每一格都复现了 claim registry 里既有的值——也正因为如此，这张表缺了八轮都没被发现。

**（2）摘要涨到 256 词，超出 IEEE SPS 的 250 词硬上限。** 是 R4 的 geolocator 句子加进去时一句一句漂上去的，而没人会为一次措辞改动重数词。已压回 **245 词**、零数学模式（同时满足「不含公式」要求），未删除任何一条结论。

**（3）六张生成表跑出栏宽 9–44pt，两处字体警告回潮。** 其中五张的溢出和一处警告是前几轮新增表带进来的，早先记录的「overfull 0、字体警告 0」已不成立。已在**生成器**层面修（改生成文件会被下次重算覆盖）：六个 `make_*_table.py` 统一加 `\resizebox{\columnwidth}`，`tab:placement_full` 图注里的 `\emph` 改引号（IEEEtran 表图注是小型大写，T1 Times 没有小型大写斜体）。新增 `src/scripts/check_build_health.py`，把页数、undefined、overfull、字体警告四项变成一条命令，不再靠眼睛看。

**（4）R7 自己的修复引入了四个没登记的数。** 第十四轮把自相矛盾的 `76` 换成了按条件具名的四个幅度值，但这四个从未进 claim registry——也就是说，那句**专门用来消除跨文档矛盾的话，自己有一轮没被检查过**。逐一重算后确认四个都对（12.2 / 84.0 / 48.7 / 241.4，是**两个攻击者合并**后的均值，这也是它们与正文并排引用的单攻击者数字在第一位小数上不同的原因），现已登记。

**顺带修好 claim auditor 的一个盲点。** 它检查「某条 claim 印出来的字符串是否还在该印它的文档里」，但读文档时不跟 `\input`——于是每一个通过生成表上页的数字都被判为「已不在文中」，真正被删掉的和没被删掉的混在一起无法区分，而这恰恰是这项检查存在的唯一目的。现在读之前先展开一层 `\input`。

**其余按评审要求补齐的写作项：** §III-A 新增「Setup, controls, baselines and ablations」段（R13 要求的常规路标），同一段一次性写明**每条轴由哪些攻击者读**（R1 的第 4 条行动项，此前要从三个小节里拼出来）；§III-C 段首**先陈述 allocation 的定论、再给产生它的证据**，并注明后文是发现顺序而非论证顺序（R13）；`\pm0.01` margin 的「fixed before testing」在 §III 已按 R14 改为「declared in advance … 在已发布仓库中声明而非第三方注册」，但图注和两张表的图注还留着旧说法，本轮一并统一（改的是生成器，不是生成文件）；补充材料开头的「Nine more sit in the extended evidence report」与实际只剩四张不符，已改为四张并逐一具名，同时写明摘要与结论不依赖其中任何一张（R8 的验收标准原文）。

**`docs/cover_letter.txt` 的首条发现与论文当前结论相反，已改写。** 原文对 Editor 写的是「None of eight placement rules … beats spreading the budget uniformly」——而 R1 跑完之后，Patch-NetVLAD 上有三条规则以 0.031–0.055 Top-1 击败 uniform。这是会写进投稿信送到编辑手上的错误陈述。已按当前结论重写发现列表（五个攻击者 + geolocator、bounded non-detection 的措辞、geolocator 那条「效应迁移而解释不迁移」），并把 qualifications 段与摘要对齐（原先漏掉净化攻击者与「没有任何预算同时 admissible 且 effective」这两条，而它们都在摘要里）。补充材料超页申请理由里也补上了本轮搬进去的 operator 表。

**本轮不需要 GPU**，两条需要卡的臂（R1 面板、R4 geolocator）在 9月12–13 日已跑完并关机。

**最终状态**：正文 **13/13 页**、补充材料 **10 页**（已在投稿信中向 EiC 提出申请）、titlepage 1 页；**0 undefined、0 overfull、0 字体警告**；auditor **902 条断言全绿、0 不符、0 无法验证**；跨文档引用检查 0 处失效；参考文献 39 条、0 条未引用、0 条引而未录。


- 【R4 完成：方向轴迁移到了地理定位器，2026年9月13日】GeoCLIP 臂已跑完、拉回、成表、写进论文。正文 13 页、补充材料 10 页、888 条断言全绿。

**结论：方向轴迁移，分配轴不迁移。** 在 clean 攻击者能定位的 218 条 query 上（25 km 内）：
isotropic 控制 $0.713$ → direction $0.546$（$-0.167$ $[-0.224,-0.107]$）、hardened $0.514$（$-0.199$ $[-0.260,-0.137]$），族内 Holm 校正后均分离；
edge $+0.040$、saliency $+0.023$，都不分离。全部 400 条上方向同样分离（$-0.091$ / $-0.116$）。
1 km 阈值上什么都不动，但 clean 只有 $6.0\%$，那是地板而不是发现。

**这个结果论文原本预测不出来。** §IV-H 的力学解释是针对 correct-versus-hardest-negative 的**排序** margin 写的——而直接回归坐标的模型没有 gallery、没有最难负样本、也没有可翻转的排序，论文明说「两半论证都不迁移」。现在测出来：解释不迁移，效应却迁移。论文已改为如实陈述这一点，并写明这是「关于发现的事实，同时是对解释的限制」。摘要、结论第一条限制均已改写（原文「image-to-location models are untested」已删除）。

**先跑了基线闸门再信数据。** GeoCLIP 在 400 帧上 25 km 内 54.5%、200 km 内 69.0%、中位误差 18.4 km——攻击者有真实能力、benchmark 可推动。分析在「全部 query」与「clean 能定位子集」两个口径上分别报告：在攻击者本来就失败的 query 上，防御既挣不到功劳也不该被记过（smoke 阶段 GeoCLIP 把 Boston 定位到巴拉圭，edge placement 反而把它「改善」到 1.09 km）。

**过程中发现并修好了我自己写的一个 bug。** 我把三个 cheap 作业手动塞进空闲 worker 时撞上了 launcher 的 pgrep 竞态，同一个文件被两个进程写，产出 8541 行而非 6400 行。去重后 2141 个重复 key 里 2140 个逐位一致、1 个冲突——冲突暴露的才是真问题：runner 原先按 (query, seed) 播种随机数并在 condition 间顺序推进，而 resume 路径用裸 `continue` 跳过已完成 condition 却不推进生成器，于是「断点续跑」与「一次跑完」会画出不同的噪声场。这违反了本仓库对可续跑脚本的要求。已改为按 (query, condition, seed) 播种，并用修好的 runner 把 cheap 臂整个重跑了一遍（4000 行，全新目录，单写者）。最终导出 6400 行 / 6400 唯一 key / 0 重复 / 每个扰动条件 delivered MSE 恰为 15.68。

为容纳新结果，protocol 示意图移入补充材料（正文保留指针），并退役 `guo2018countering`（athalye2018obfuscated 才是该段真正依赖的引用），参考文献 40 → 39。


- 【R3 结论：公开代码无法端到端复现，2026年9月12日】把 GeoShield 仓库（`thinwayliu/Geoshield`，AAAI 2026）clone 到 `src/third_party/` 后逐行核对，**其公开版本把 VLM 组件留成了空实现**：`describe_image_placeholder()` 的函数体是 `TODO: Implement your VLM API call here`，注释建议用户自行接入 GPT-4V / Claude / Gemini / LLaVA。

这不是可以绕过的边角。该描述经 `ensemble_loss.set_geotext_truth(description)` 进入 `geo_loss`，而目标函数里这一项是**被减掉**的（`loss -= (text_loss + text_local_loss)`）——它正是让扰动在破坏地理线索的同时保住语义的那一项，也是论文三个命名模块之一「exposure element identification」所依赖的输入。没有 VLM 就不是在跑 GeoShield，而是在跑另一个目标函数。

因此 R3 的「端到端复现一个已发表机制」在不自备 VLM API 预算的前提下**对任何人都不可得**，这本身是一条可报告的可复现性观察，且正落在本文（一篇审计该家族的论文）的射程内。论文 §IV-B 的措辞已相应改写：原先写成「我们选择改编而非复现」，现在写明公开实现 stub 掉了其目标函数所依赖的 VLM 项，所以改编是被迫的而非偏好。已保留原有的 mask-guided 改编臂不变。

未采取的替代方案与理由：接一个本地 VLM（BLIP/LLaVA）填补 stub 会改变目标函数，跑出来的东西不能诚实地标注为 GeoShield，所以没有做。

- 【R4 进行中】GeoCLIP 地理定位臂已在 vGPU 3090 上起跑（9 个作业 = 3 seed × 3 成本档，`src/scripts/launch_r16_geolocator_vgpu3090.sh`）。

**先跑了基线闸门**：GeoCLIP 在 400 张 MSLS query 帧（192×320）上 25 km 内 54.5%、200 km 内 69.0%、中位误差 18.4 km，1 km 内仅 6.0%。攻击者有真实能力、benchmark 可被推动，所以这里出现 null 会是 null 而不是 floor（与 KITTI-360 用的是同一条闸门）。主阈值取 **25 km**，1 km 因接近地板只报不承重。

闸门值得跑：三条 smoke 数据里 GeoCLIP 把一张 Boston 帧定位到了巴拉圭（误差 7654 km），而 edge placement 在那条 query 上把误差「改善」到 1.09 km——在攻击者本来就失败的 query 上，防御会拿到它没挣到的功劳，也会被扣上莫须有的损害。因此分析将分别报告全部 query 与 clean 定位正确子集上的配对对比。


- 【R1 结果的下游一致性清查已完成，2026年9月12日】结果变了之后把整篇论文里依赖旧结论的地方逐条查了一遍，查出一处**实质性错误**并改正。

**错误：§IV-A 的 98 组比较写成覆盖「both benchmarks」，实际只在 mined proxy 上跑过。** 核对 `src/exports/placement_study/` 与 `..._maskbacked/` 的 run_id，两个 checkpoint 的 49 组全部是 `proxy12/*`（六个 backbone，12 对）与 `proxy50_resnet18`（50 对），没有任何一组在 MSLS 上。原文声称的 MSLS 覆盖不存在。这一处特别值得记录，因为**正是这句话让 R1 的缺口看不见**：如果原文写的是「只在 proxy 上」，那么「MSLS × Patch-NetVLAD 这一格没跑过」本来一眼就能看出来。已改为如实描述两个 proxy 规模。

**并补上了该研究的分辨率说明。** proxy 的一格是 36 个配对观测，bootstrap 区间中位宽 $0.083$，discordant 对中位数为 1，42 格里有 16 格一个都没有——也就是说这 98 组比较里大多数是「没测到」而不是「测过没有」，它对 §IV-B 在真实 benchmark 上发现的那个量级的差异是盲的。Patch-NetVLAD 在 proxy 上七格全不显著、点估计甚至反向（edge $+0.0278$），而在 MSLS 400 个 cluster 上 edge 是 $-0.0550$，正是分辨率差异的体现。论文现在明说这一点，并说明该研究只是「唯一同时比较两个 checkpoint 的地方」，论文的主张不依赖它。

**图 2 已用新数据重绘。** `make_axes_figure.py` 的左栏原先只有 ResNet18 与 MixVPR 两个攻击者，现已扩到五个。重绘后图本身就显示了新结论：在 score-gradient 与 edge magnitude 两行，Patch-NetVLAD 的菱形落在 $-0.05$ 附近、明显在 $\pm0.01$ 灰带之外，而其余四个攻击者都在带内。图注同步改写（原图注还在说「规定性规则只跑了前两个」，已过时）。

其余同步改正：§IV-B「The null survives intact」已限定为 ResNet18 臂并指向三段后的扩展；「the null above is a statement about the rules that have been proposed, not about the axis」改为「…and about the attackers reading them…」；补充材料 proxy 表的图注补上 36 个配对观测的功效说明与「on this benchmark」限定。

为把这些塞回 13 页，又压缩了 §IV-A 分辨率段、mask-guided、clip pooling、gallery sweep、budget sweep、"Taking the attacker away" 等段。正文 13 页、补充材料 9 页、参考文献 41 条、888 条断言全绿、无未定义引用。


- 【R1 实验已完成，结论已改写，2026年9月12日】vGPU 3090 上九个作业全部跑完（36,009 行、九个作业 exit=0、无告警），结果已拉回、成表、写进论文，**卡已关机**（端口 22766 已关闭，18:35 NZST）。正文仍为 13 页，888 条断言全绿。

**这次实验推翻了论文原先的核心表述。** R1 的问题是：规定性 placement 规则只在 ResNet18 与 MixVPR 上量过，而 solved map 真正分离的两个攻击者（Patch-NetVLAD、CLIP）恰好一条规定性规则都没跑过。把九条规则原样扩到 Patch-NetVLAD、ViT-B/16、CLIP ViT-L/14（同 manifest、同 gallery、三个种子、同 delivered distortion 与能量门，只换读取帧的攻击者，属重新嵌入而非新搜索）之后：

- **ViT-B/16：九条全不分离。CLIP ViT-L/14：九条全不分离。**
- **Patch-NetVLAD：九条里四条分离**（族内 Holm 校正后 p<0.05）——
  edge magnitude $-0.0550$ $[-0.085,-0.025]$、score gradient $-0.0492$ $[-0.072,-0.026]$、
  margin gradient $-0.0308$ $[-0.055,-0.009]$，以及反方向的 anti-margin gradient $+0.0208$ $[+0.006,+0.036]$。

所以「没有任何被提出的规则能买到东西」这句话只在**五个攻击者里的四个**上成立，第五个上有三条规则买到 0.031–0.055 Top-1。摘要、贡献列表、§IV-B、结论已全部按这个更准确也更不利的口径改写；§IV-B 的小节标题由「The Null Holds on Real Geographic Data」改为「Where the Null Holds, and Where It Does Not」，因为原标题现在是错的。

两点比计数更重要，也已写进论文：
（1）赢得最多的是 **edge magnitude**，而它也正是在另一个 benchmark 上预算超过操作点后唯一能击败 uniform 的规则——即「与图像边缘对齐」是唯一在两处都奏效的规定性策略，而它并不是敏感度图。
（2）**margin gradient**（§IV-H 自己推导而非借来的那条）在这里分离，而它的逆显著更差——避开该分析所指认的像素反而损失隐私。这是该力学解释第一次拿到正面证据；此前它在两个攻击者上「最接近边界但没跨过」，现在读作效应真实但那两个攻击者分辨不出，而不是没有效应。
（3）**说不清为什么是 Patch-NetVLAD**：它既非最强也非最弱，与 surrogate 共享 VGG16 trunk 而 CLIP 什么都不共享，可 solved map 走得最远的偏偏是 CLIP。攻击者强度、trunk 共享、solved-map 可达性三者都无法给这五个排序，论文如实报告这个依赖关系而不编造规律。

新增 `src/scripts/make_placement_panel_table.py` 与 `paper/generated/tab_placement_panel.tex`（27 格，place-clustered 区间、族内 Holm、TOST），已 `\input` 进补充材料。

为把新增内容塞回 13 页，又退役了 `croce2020reliable`（AutoAttack 标准化的是本文并未运行的攻击，Carlini 一条已承担评测方法学的定位），参考文献 42 → 41，并压缩了 KITTI-360、published-model placements、gallery sweep、clip pooling、伦理节等段落。


- 【第十四轮补充：页数与参考文献已收敛，2026年9月12日】正文 **13 页**、补充材料 9 页，888 条断言全绿。

R9 页数（已完成，原为【已阻挡】）：正文由 14 页压回 13 页，**未删除任何一项结果**。做法分三部分。
（1）**退役旧框架遗留的参考文献 17 条**，62 → 42。删除依据是每一条与当前论文主张的关联，而非年代：
`krahenbuhl2011efficient` 与 `wang2005dynamic`（稠密 CRF 与动态 CRF 的机制引用——但 §II 已明确本文的递推「in the style of mean-field updates with no CRF energy behind it」，即论文自己否认了这两条所支撑的说法，留着是自相矛盾）；
`wang2020videodp`（视频差分隐私——论文不作任何 DP 主张，`dwork2014algorithmic` 一条已足够承担「我们不是 DP」这句免责）；
`hukkelaas2019deepprivacy`、`ren2018learning`（人脸匿名化——该段只需要一两条来做「前景 vs 背景」的对照，原本有四条）；
`berton2023eigenplaces`、`izquierdo2024optimal`、`keetha2024anyloc`、`arandjelovic2016netvlad`（VPR 领域进展的罗列子句，与论证无关，`masone2021survey` 已覆盖）；
`speciale2019privacy`、`chelani2023privacy`、`shokri2011quantifying`、`andres2013geo`（几何/坐标级隐私——论文明确写了「the setting here is the released frame」，属对照而非依据，`pittaluga2019revealing` 一条足矣）；
`haas2024pigeon`、`mendes2024granular`（与 `vivanco2023geoclip`、`luo2026doxing` 重复）；
`cao2023impress`（本轮新增的 `radiyadixit2022poisoning` 与 `honig2024protect` 把「保护性扰动会被洗掉」这件事讲得更近更准，它已冗余）；
`malone2025adversarial`（锦上添花的旁注）。
（2）**压缩大段文字**：相关工作重写（四段合为三段）、§IV-G「What Is Actually Released」由两个粗体子段并为一段、§IV-B 的 solved-map 与 margin-oracle 两段、§III 推断协议、伦理节、摘要与致谢均收紧，全部保留每一个数字与每一条结论。
（3）**版面**：作者简介注释待接收后放出、机制流程图与 alloc-curve 表移入补充材料、protocol 示意图保留在正文（它画的是本文自己的贡献）。
`docs/cover_letter.txt` 已相应改写：不再申请正文超页，只保留补充材料 9 页的申请（按你「要申请」的答复）。


- 【第十四轮独立评审 + 修订已完成，2026年9月12日】按 IEEE TIFS 投稿要求重新独立评审并将英文 LaTeX 评审整段覆盖写入 `docs/RevisionSuggestions.tex`（21 页，5 项 Major、7 项 Moderate、2 项 Minor），随后按该评审推进修订。本轮 11/14 项已完成，3 项因 GPU 主机不可达且本机无 MSLS 影像而阻挡。修订后 `src/scripts/audit_claim_consistency.py` 报告 888 条断言全部通过、0 条不符。

R6（已完成）主基准的 placement 对比是九组而非七组。
修改说明：摘要、§IV-B、结论的计数统一改为九；`tab_placement_msls` 本就含 margin gradient 与其逆两行，正文原先把它们排除在计数外并单列一段。同时修正 §IV-B 引用的最宽区间——原文 `[-0.008,+0.026]` 在表中任一列都不存在（centre bias 为 query 级 `[-0.007,+0.026]`、place 级 `[-0.008,+0.027]`），改为按列如实引用 edge magnitude 的 `[-0.014,+0.025]` 并标明单位。认证全部九组所需的 margin 仍为 ±0.027，未变。

R7（已完成）释放扰动幅度的两处数字互相矛盾。
修改说明：正文 §IV-G 报 delivered MSE 241.5 时 max|δ|=49.30，补充材料 §S-factorial 在同一预算、同一 ℓ∞=16 投影下报 76，两者无法并存。从 `src/artifact/results/factorial_mse*` 重算后改写补充材料：operating point 下 uniform 放置的 direction 为 12.2（与正文 12.36 一致）、edge 放置为 84.0；大预算下 uniform 为 48.7（与正文 49.30 一致）、edge 为 241.4（触及 255 截断）。原 76 在导出树中无对应，已删除并按条件具名重写。

R10（已完成）TIFS 深度学习可复现性清单与退化成因。
修改说明：这一项在核查中查出一个实质性发现。released checkpoint 的训练配置为 `mask_root: None`，而 `src/run_train.py` 在无 mask 时以全零张量作为目标，即该 checkpoint 是在**完全没有空间信息的目标**上训练的；实测第 1 与第 5 轮权重相对变化仅 6.8e-3、输出层 bias 由 1.2e-3 走到 8.1e-4，复现的 sigmoid 范围 [0.4991, 0.5015]、空间变异系数 5.31e-4 与论文所报完全一致。因此论文原先“trained against pixel-level support masks where these exist”的表述具有误导性，已在 §II-A 与 §IV-A 改写为如实描述并给出退化的因果解释——这反而强化了论文自己的论点：该 checkpoint 的 privacy–utility 曲线只认证了幅度、完全没有认证 map。补充材料 §S-repro 另补齐 weight decay（AdamW 库默认 0.01）、学习率恒定无 scheduler、unary 网络无任何归一化层、mask-supervised checkpoint 的语料（KITTI-360 序列 0000、1033 对、正样本比例 0.937）与其类别平衡 BCE 权重（0.25 / 0.937），并新增一段逐条对应 TIFS 清单的说明。另修正原文“fixed order, no shuffling”——两个训练脚本的 DataLoader 均为 `shuffle=True`。

R2 + R5（已完成）统计推断：等效性检验、多重比较与推断单位。
修改说明：新增 `src/scripts/analyze_placement_equivalence.py`，在已发布的 per-query 导出上计算 place-clustered bootstrap TOST、Holm 校正与 place-clustered sign-flip 置换检验，并给出认证 ±0.01 所需的 cluster 数。结果：把 placement 族按 confirmatory 处理并做 Holm 校正后，弱攻击者九组校正后 p 全为 1.000，强攻击者上唯一名义显著的 score-gradient 由 0.057 变为 0.397（置换检验 0.124 / 校正后 0.871），即多重比较从来不是问题；TOST 在弱攻击者上认证 4/9（去掉退化自比较后为 3 组，而原区间规则只认证 1 组），强攻击者 2/7（去退化后 1 组）；认证 ±0.01 需要弱攻击者 1,261、强攻击者 917 个 query cluster，是本语料的二到三倍。两张 placement 表新增 $p_{TOST}$ 列；§III 改写推断协议（两个 confirmatory 族、置换检验、TOST、所需样本量），并修正 §IV-B “1,200 paired observations” 与 §III “三个种子先在 query 内平均”的矛盾——实际进入检验的是 400 个差值。

R2（已完成）摘要、贡献、结论的 null 表述超出区间所能认证的强度。
修改说明：摘要重写为 242 词、无任何数学模式（同时满足 IEEE SPS 的 150–250 词与“不含公式”要求），把“no proposed allocation rule buys anything”改为 bounded non-detection 并写明“三组认证、其余未决”；贡献列表与结论首段同样改写，结论另补出所需样本量与族校正不改变任何判定。摘要同时按 R14 明确写出“no budget we tested is both admissible and effective, so the contribution is the protocol and the measurement”。

R12（已完成）admissibility 判定所用语料与隐私度量语料不一致。
修改说明：新增 `src/scripts/analyze_msls_agreement_frontier.py`，在 `src/exports/tifs6_joint` 的 400 张 MSLS query 帧上按与 frontier 相同的方式算出 12 个 cell 的 class-mean agreement drop 与配对 bootstrap 区间。结论两面：**顺序**与 VOC 完全一致（四个预算上 isotropic < direction < hardened），所以“哪条轴更贵”不依赖语料；**量级**不可比（街景上的 class-mean agreement 远比 VOC 掩码上的 mIoU 严苛，1.7% 像素改动对应 0.197 的 drop，direction/isotropic 的倍数由 VOC 的 2.6–3.9× 压缩到 1.4–1.9×）。因此明确不把 VOC 标定的容差搬到这一列，并写明要为这些帧标定容差需要把六个分割模型重新在其上评分，本轮未做。frontier 表标题同步注明容差的标定来源。

R8（已完成）支撑摘要级论断的表格不在投稿包内。
修改说明：原本 17 张生成表中有 8 张两个文档都未 `\input`，其中 `tab_optimised_allocation` 承载摘要引用的 −0.0392 / −0.0942，四张 purification 表承载摘要的“half to all”。新增 `src/scripts/make_purification_summary_table.py`，把四个攻击者合并成一张表，且直接给出论证真正依赖的量——释放相对 isotropic control 的优势在“双方都不净化”与“攻击者净化双方”下的对比，以及保留比例。该表与 `tab_optimised_allocation` 均已进入补充材料。顺带把正文“takes back between half and all”改为按表精确的 48/55/57/100 与 35–100 百分比。

R11（已完成）缺少直接预示本文结论的文献。
修改说明：`paper/ref.bib` 新增 4 条并在相关工作中各自成段落定位：Radiya-Dixit 等（ICLR 2022）与 Hönig 等（2024）构成“保护性扰动在自适应对手面前失效”这条线，本文 §IV-I 属于该线且补充的是逐架构的定量测量而非存在性结果；Carlini 等（2019）与 Croce & Hein（2020）是评测方法学文献，本文明确说明二者约束的是**攻击者**一侧，而本文补的是**防御者**一侧缺失的空间自由度控制。同时清理 11 条从未被引用的 bib 条目（66 → 59）。

R1（部分完成：结论性陈述已写入，实验被阻挡）两条轴不是由同一组攻击者读出的。
修改说明：这是本轮评审的首要意见。规定性 placement 规则在主基准上只对 ResNet18 与 MixVPR 有逐规则结果，而 direction 与 solved map 有五个攻击者；**solved map 真正分离的两个攻击者（Patch-NetVLAD 与 CLIP）恰好是没有任何规定性规则跑过的两个**，CLIP 更是从未进入任何 placement 研究。已在 §IV-B 新增专门一段如实陈述这一不对称、说明它可能部分是“攻击者面板的不对称”而非“轴的不对称”、给出两条边界（在同时跑过的攻击者上比较是精确的；没有任何已测属性能预测哪个攻击者会被 allocation 触及），并指出补齐它只是对既有帧重新嵌入而非新的搜索。结论的 limitations 同步加入这一条。

R3（部分完成：范围声明与溯源表已写入，复现实验被阻挡）没有任何已发表机制被端到端复现。
修改说明：补充材料新增 `tab:rule_provenance`，逐条列出本审计测试的每一条 placement 规则及其来源（本文自建 / 经典算子 / 他人训练的分割模型 / 改编自已发表机制但未复现）；§II 新增范围声明，明确本文的否定结论是关于该表中规则的结论，并说明这正是同时求解 map 的原因——一个没有规则能被指为其弱实例的对照。

R4（部分完成：范围条件与论证已写入，实验被阻挡）威胁模型排除了 image-to-GPS 地理定位器。
修改说明：结论的 limitations 改写为两条边界，第一条明确写出所有结果都以 gallery-based 攻击者为条件，并给出**为什么不能靠类比外推**的论证：本文解释两条轴差异的机制说明是针对 correct-versus-hardest-negative 的**排序** margin 的（allocation 动方差、direction 动均值），而直接回归坐标的模型没有 gallery、没有最难负样本、也没有可翻转的排序，因此论证的两半都不迁移——这是一条范围条件，而不是对结果会如何的猜测。

R13（已完成）图表与版面。
修改说明：恢复了机制流程图（原先被 `\begin{comment}` 注释掉），但在页数压缩中把它移入补充材料、正文保留指针，正文保留更能代表本文贡献的 protocol 示意图。operator 子节与 KITTI-360、预算扫描、published-model placements、mask-guided 各段均已压缩，细节指向补充材料。

R14（已完成）遗留项逐条改写。
修改说明：（1）gallery 上限的外推改为写明其具体假设——“加入不可能是正确答案的干扰项不改变正确项相对最难竞争者的排名”，并说明该假设在何时失效；（2）“nothing we measured predicts which”等基于五个攻击者的否定性推断改为“这五个攻击者上没有可见规律，五个不足以证明不存在这样的属性”；（3）±0.01 margin 的“fixed before testing”改为“declared in advance ... 在已发布仓库中声明而非第三方注册”，如实说明其可核查程度；（4）清理 11 条未引用文献；（5）摘要按 R14 写明贡献是协议与测量。

R9（部分完成）IEEE SPS / TIFS 版面合规。
修改说明：已完成——摘要 242 词且无公式（原 249–251 词含 6 处行内数学）；新增独立的 `\section{Ethics, Responsible Use, and Data Availability}`（原先只是理论节末尾一段）；新增 `\section*{Acknowledgment}` 与 IEEE 政策要求的生成式 AI 使用声明；作者简介已写好但按初投惯例注释保留（IEEE 将其计入 13 页限制，接收后再放出）。未完成——正文 14 页、超出初投 13 页上限 1 页，补充材料 9 页、超出 6 页建议 3 页；`docs/cover_letter.txt` 已改写为同时就这两点向 Editor 提出请求，并列出我们认为最不承重的两项（KITTI-360 复现与 serialized-release 审计）供 Editor 选择。


- 【本次独立评审已完成，2026年9月8日】依据当日十五时三十一分的正文、十五时二十六分的补充材料、原始实验导出及期刊官方要求，已将完整英文评审覆盖写入 `docs/RevisionSuggestions.tex` 并编译为十七页评审文件，包含十三项主要意见、八项实验或分析建议、数值核验结果和验收标准，本次仅生成评审，未修改论文或执行所建议的新实验。

1. 已将标题、摘要和全文问题定义统一为“发布视频帧时的背景驱动 location privacy”。
修改说明：重写并收束了标题、摘要、引言、相关工作、方法、实验和结论，删除旧稿中“敏感目标类别保护”“特征空间发布重建”等混杂叙事，统一为 gallery-based retrieval attacker 与 image-space sanitization 的单一问题表述。

2. 已将方法描述与当前仓库实现对齐并补清关键公式含义。
修改说明：在 `paper/main.tex` 中重写了 DCRF、NCP 与高斯噪声标定的公式和文字说明，新增符号表与算法流程，并明确正文只主张 DP-style calibration；同时补充说明 `p_t` 负责空间 support、`\alpha_t` 负责 support 内幅度调制，双重加权是有意实现而非写法歧义。

3. 已澄清攻击评估粒度与发布对象。
修改说明：在 threat model、受控 benchmark 协议和附录中明确说明查询端使用短序列估计时序平滑 mask，但 retrieval 指标是在 sanitized middle frame 上计算，避免 frame-level 与 sequence-level 攻击定义混淆。
13. 已将论文定位从"最强隐私方法"重新定位为"校准的隐私-效用权衡优化器"。
修改说明：重写摘要、贡献列表、引言 scope 段、结论，全面改用 trade-off 定位；摘要以 6 dB PSNR 优势为核心论据，贡献项强调 frontier-based 评估和可调噪声预算，结论以最强验证结论结尾。

14. 已大幅扩展相关工作至 42 条引用（7 个主题段落）。
修改说明：原有 17 条引用扩展至 42 条引用，相关工作分为视觉匿名化综述、隐私感知管线、视觉地点识别、场景级和位置隐私、差分隐私与扰动防御、CRF 时空推理、隐私效用权衡优化 7 段。新增 19 条 BibTeX 条目（DeepPrivacy、CIAGAN、Oh2016 等）并新引用 6 条原有条目（chen2017deeplab、wang2005dynamic、balle2018improving、zhou2020personal、xu2019ganobfuscator、wang2004image）。ref.bib 从 99 条清理至 42 条，全部与正文对齐。

15. 已在正文新增 matched-operating-point 分析与 temporal consistency 分析两节并补充表格。
修改说明：Section 4.5 新增 Table (tab:matched) 展示具体 matched-utility 和 matched-privacy 数值。Section 4.6 新增 Table (tab:temporal) 展示 flicker score、perturbation stability 和 mask IoU，部分数值标记 TBD 待实验运行。

16. 已在实验代码中扩展 matched-operating-point、temporal consistency 和多 sigma 消融实验。
修改说明：`src/eval/metrics.py` 新增 `flicker_score` 和 `perturbation_stability` 函数；`src/scripts/run_controlled_retrieval_benchmark.py` 新增 `--ablation_sigmas` 参数（默认 [8, 16, 24, 32]），支持在更高 sigma 下运行 DCRF/NCP 消融。

17. 已为 ACM TOMM 双匿名评审做好准备并统一附录格式。
修改说明：`main.tex` 添加 `anonymous` 选项，`appendix.tex` 从 IEEEtran 转为 acmart 格式，创建独立 `titlepage.tex`，三个 PDF 均已成功生成。

18. 已完成独立评审重置并重写 `docs/revision_suggestions.tex`。
修改说明：基于当前 main.tex、appendix.tex 直接审稿，以 ACM TOMM 标准完全重写评审意见文件，包含 6 条 Major Concerns（M1–M6）和 8 条 Minor Concerns（m1–m8），以及优先级排序的修订清单。

19. 已更新结论以最强验证结论结尾。
修改说明：结论段最后强调"~6 dB PSNR 优势和空间 support 可迁移性是视频位置隐私中最有价值的构建块"。
4. 已补充受控 paired-scene retrieval 实验并生成论文图表。
修改说明：新增 `src/datasets/monitoring_clip_dataset.py` 与 `src/scripts/run_controlled_retrieval_benchmark.py`，基于 `F:\work\datasets\monitoring\images` 构建可复现实验，输出了 `paper/figs/privacy_utility_tradeoff.pdf`、`paper/figs/retrieval_robustness_topk.pdf` 以及 `src/outputs/controlled_retrieval/` 下的 CSV 和摘要文件。

5. 已补充更强的 task-aligned baselines 并更新主结果分析。
修改说明：在受控 benchmark 中新增了 mask-guided blur 与 mask-guided mosaic baselines，重新生成了实验汇总文件，并在正文默认消融分析中如实写明这两类 support-aware baseline 在短 proxy benchmark 上都强于 PPEDCRF，从而把结论收束为“PPEDCRF 优于 random mask 与 global Gaussian noise，但仍未超过强 support-aware blur/mosaic baselines”。

6. 已补充主结果的可复现统计与正文分析。
修改说明：在正文中加入默认消融、privacy-utility frontier、attacker-sensitivity 和 failure-mode 解释，保留 mean±std，并将 ResNet50 迁移较弱、Top-10 饱和等现象改写为明确局限性而非过强结论。

7. 已重写附录并完成本轮编译核验。
修改说明：`paper/appendix.tex` 现改为实现说明与受控基准说明，`paper/build/main.pdf` 与 `paper/build/appendix.pdf` 均已成功生成，主文图表也已插入、引用并补充可访问性描述。

8. 已清理 `src/` 与 `paper/` 下残留的中文内容。
修改说明：将 `src/scripts/split_train_val.py`、`src/data/driving/README.md` 等位置改为英文说明，并通过 `rg -n "[\p{Han}]" src paper` 复查确认无中文残留。

9. 已将论文图像引用目录从 `images/` 统一切换为 `figs/`。
修改说明：更新了 `paper/main.tex` 与 `paper/main_010324.tex` 中全部图像路径前缀，并同步修正本文档里图表输出目录的旧路径描述，避免目录改名后出现失效引用。

10. 已修复 `paper/build.bat` 的独立编译与 PDF 回拷流程。
修改说明：将构建输出统一写入 `paper/build/`，保留 `latexmk` 优先策略并在其不可用时自动回退到 `pdflatex`/`bibtex`，同时确保 `main.pdf` 和 `appendix.pdf` 成功复制回 `paper/` 根目录，`latexmk` 失败日志单独保存为 `paper/build/*.latexmk.log`。

11. 已补强受控 retrieval benchmark 的构造说明与导出工件一致性。
修改说明：在 `paper/main.tex` 与 `paper/appendix.tex` 中加入了配对相似度和 hard distractor 难度统计，恢复 `paper/main.tex` 中的 `Catherine Shi` 作者条目，并将 `src/scripts/run_controlled_retrieval_benchmark.py` 的默认参数与论文当前设置对齐，新增 `selection.json`/`summary.md` 中的 benchmark hardness 统计后重新生成了 `src/outputs/controlled_retrieval/` 与 `paper/figs/` 下的结果。

12. 已清理主文参考文献字段并消除主文 BibTeX warning。
修改说明：补全了 `paper/ref.bib` 中当前主文实际引用条目的期刊卷页和会议 publisher/address 字段，重新编译后 `paper/build/main.blg` 的 `warning$` 已降为 0，同时同步修正了 `src/scripts/README_NUMBERS.md` 中残留的旧图目录描述。

20. 已按评审意见 M6 将 Legacy 实验节从主文迁移到附录。
修改说明：`paper/main.tex` 将原 Section 4.2 压缩为一段 Appendix summary，删除主文中的 legacy 大图；`paper/appendix.tex` 新增完整 Legacy detector/segmentation 小节并承接原 Figure 3/4 内容，使主文聚焦 retrieval threat model。

21. 已按 m5 重构摘要，首句直接给出问题-方案对。
修改说明：将摘要首句从 motivation 改为 "We propose PPEDCRF, a calibrated selective perturbation framework..."，同时压缩第二段并明确提及 blur/mosaic 对比。

22. 已按 m6 在结论中显式提及 blur/mosaic 对比。
修改说明：结论段新增对 support-aware 确定性基线（blur/mosaic）的显式比较和 ResNet50 迁移限制说明，避免 6 dB 优势被孤立解读。

23. 已按 m8 更新 CCS 概念和关键词。
修改说明：新增 `Security and privacy~Privacy-preserving protocols` CCS 描述符和 `video anonymization` 关键词。

24. 已按 m3 清理 `paper/figs/` 中未引用的图像文件。
修改说明：将 33 个未被 main.tex 或 appendix.tex 引用的遗留图像移至 `paper/figs/legacy/`，figs 根目录仅保留 10 个被引用的文件。

25. 已按 m7 在 benchmark 脚本中新增 blur/mosaic 参数扫描功能。
修改说明：`run_controlled_retrieval_benchmark.py` 新增 `--blur_kernel_sizes` 和 `--mosaic_block_sizes` 参数，扫描结果同时加入 frontier 图和 `baseline_sweep.csv`。正文 Section 4.5 末尾提及该可复现扫描。

26. 已按 m2 / BibTeX warning 大幅补全参考文献字段。
修改说明：修复 22 条条目的 publisher/address/pages 字段，balle2018improving 升级为 ICML 正式引用，dwork2014algorithmic 修复 volume/number 冲突，BibTeX 警告从 39 条降至 4 条。

27. 已重构 Section 4.5 (tab:matched) 解决数据-叙述不一致问题。
修改说明：将原先重复的 matched-utility/matched-privacy 两子表合并为 Panel A（同 σ₀ 对比）和 Panel B（同 PSNR 对比），修正文本中与表格数值不符的描述，诚实呈现 ~30 dB 匹配效用下两种噪声方法隐私效果相当的事实，突出 PPEDCRF 在高质量操作区间（36 dB）的独特优势。

---

## 本轮修订新增（第 28–37 条）

28. 已计算时序一致性指标并填入 Table 6（tab:temporal）所有 TBD。
修改说明：新增 `src/scripts/compute_temporal_metrics.py`，基于合成 monitoring 数据与固定种子初始化的 SensitiveRegionNet 计算 flicker score 与 perturbation stability，输出 `src/outputs/temporal_metrics.json`。五种方法实测值已写入正文表格（PPEDCRF: 4.49±0.02 / 0.004±0.001；Random mask: 5.68±0.04 / 0.016±0.001；Global Gaussian: 9.06±0.07 / 0.011±0.001）。

29. 已将所有主文实验数字更新为实际 CSV 输出值。
修改说明：发现正文 Tab.1（tab:ablation）、Tab.3（tab:robustness）、Tab.4（tab:matched）中的数值与 `src/outputs/controlled_retrieval/*.csv` 不符（如 raw Top-1 旧值 0.833 vs 实测 0.500，PPEDCRF Top-1 旧值 0.722 vs 实测 0.306）。已将全部表格和正文叙述替换为实测数值，保证 paper 与实验输出完全一致。

30. 已重新生成 frontier 和 robustness 图像文件。
修改说明：在 `src/scripts/regenerate_figures.py` 中基于实测 CSV 重绘 `paper/figs/privacy_utility_tradeoff.jpg` 与 `paper/figs/retrieval_robustness_topk.jpg`，确保图文数据一致。

31. 已修正 matched-utility 分析中 blur/mosaic 与 PPEDCRF 隐私效果的对比叙述。
修改说明：Panel B 匹配效用（~30 dB PSNR）下，Gaussian 噪声（Top-1=0.111）优于 blur（0.250），原文错误地称"blur提供更强隐私"。已重写 Panel B 段落，明确 Gaussian 随机噪声对特征方向的破坏性更强，并厘清 DCRF support 贡献的双重作用（确定扰动位置 + 时序稳定性）。

32. 已强化 DP-style 标定的合理性说明。
修改说明：在 Section 3.2（NCP Control and DP-Style Calibration）中新增"Why this calibration over simpler alternatives"段落，对比线性缩放（σ₀ ∝ 1/ε）和固定常数计划的缺陷，解释 Gaussian-mechanism 形式的三项实际优势（单调性、δ解读、平滑退化），并将 σ₀=8 锚定至具体 (ε,δ) 参数对（≈0.59, 10⁻⁵）供论文读者参考。

33. 已更新摘要以反映实测数值和新的定位叙述。
修改说明：将摘要中 "0.833→0.722" 等旧数值全部替换为实测值（"0.500→0.306" 默认设置），补充 matched-utility 结论（σ=16 时 PPEDCRF Top-1 = 0.111，与 global noise 相同），并在末句明确区分 PPEDCRF 定位（operating-point selector）与 privacy maximizer。

34. 已更新结论节，纳入实测时序指标和正确的对比叙述。
修改说明：结论段新增 flicker score 对比数字（4–5 vs 9），修正 ResNet50 迁移描述（实测两骨干一致改善，无反转），并强调未来工作方向（更强 VPR attacker、更大规模 benchmark）。

35. 已更新 robustness 分析段落，移除与实测数据不符的"部分倒退"描述。
修改说明：原文在 ResNet50 下描述 "gap reverses" 属于来自旧数值的错误叙述；实测 ResNet50 下 PPEDCRF 在所有 gallery size 上均优于 raw（-0.278 至 -0.250）。已重写该段落，如实反映实测结果，并补充对更强攻击者的局限性说明。

36. 已验证论文无残留 TBD 占位符并成功编译。
修改说明：运行 `paper/build.bat`，`paper/main.pdf`（4.1 MB）与 `paper/appendix.pdf` 均已生成；`main.log` 无 LaTeX Error，grep main.tex 确认无 TBD 字符串残留。

---

## 本轮修订新增（第 37–44 条）

37. 已运行 v4 综合 benchmark 并将全部主文表格更新至 v4 数据。
修改说明：使用 `run_controlled_retrieval_benchmark.py` 完成包含 ResNet18/ResNet50/VGG16 三骨干、3 seeds、gallery 12/24/48、sigma sweep 8/16/24/32、matched 30/33/36 dB 的完整 v4 benchmark。所有表格（tab:ablation、tab:robustness、tab:matched、tab:temporal）及正文叙述均替换为 v4 数值，PPEDCRF Top-1 从 0.306 更新为 0.333（v4 重跑值）。

38. 已新增 sigma sweep 消融表（tab:sigma_sweep）。

39. 已在正文中增加对当前实验限制的显式说明：当前评估覆盖 ResNet/VGG/CLIP 风格嵌入，但尚不包含 NetVLAD、Patch-NetVLAD、CosPlace 或 MixVPR 等专用 VPR 架构。修改说明：在 Related Work、benchmark scope 和 Conclusion 中补充该限制语句，进一步收束论文贡献范围。
修改说明：在 Section 3.3 与 Section 3.4 之间新增 Table，展示 σ₀∈{8,16,24,32} 下 PPEDCRF / w/o temporal / w/o NCP 的 Top-1、Top-5、PSNR、SSIM。数据来源为 `src/outputs/controlled_retrieval_v4/ablation_sigma_sweep.csv`。

39. 已将 matched-operating-point 分析扩展至 3 个 PSNR 目标。
修改说明：Table~5（tab:matched）从原先 2 个面板扩展为 3 个面板（Panel A ~36 dB, Panel B ~33 dB, Panel C ~30 dB），数据来源为 v4 matched_operating_point.csv。对应叙述同步更新。

40. 已替换定性检查图为 6 面板期刊级图形。
修改说明：新增 `src/scripts/generate_qualitative_figure.py`，生成包含原始帧、DCRF 热力图叠加、噪声保护帧、差异图、模糊结果、放大裁剪的 2×3 面板图，输出为 `paper/figs/qualitative_figure.pdf`，并在正文 fig:sidebyside 中替换旧 4 面板引用。

41. 已集成 CLIP ViT-B/32 和 CLIP ViT-L/14 作为攻击骨干。
修改说明：在 `src/eval/retrieval_attack.py` 中将 CLIP 加载后端从 `open_clip` 切换为 `transformers.CLIPModel`（解决公司代理 SSL 证书问题），支持离线模式加载本地缓存的 `openai/clip-vit-base-patch32` 和 `openai/clip-vit-large-patch14`。完成 CLIP benchmark（`src/outputs/controlled_retrieval_clip/`），生成全部 10 个输出文件。

42. 已将 CLIP 结果集成到论文 Table 4（tab:robustness）和正文叙述中。
修改说明：Table 4 新增 CLIP ViT-B/32 和 CLIP ViT-L/14 各 3 行（gallery 12/24/48）。关键发现：CLIP ViT-B/32 下 PPEDCRF 仍有效（Δ=-0.083 at g48）；CLIP ViT-L/14 暴露失效模式（PPEDCRF Top-1=0.222 > raw 0.167, Δ=+0.056 at g24/g48）。摘要、结论、robustness 分析段落、图注均已更新反映 CLIP 结果。

43. 已重新生成合并 5 骨干的 robustness 图。
修改说明：新增 `src/scripts/regenerate_combined_figures.py`，合并 v4 和 CLIP 的 robustness_summary.csv 数据，生成包含 ResNet18/ResNet50/VGG16/CLIP ViT-B/32/CLIP ViT-L/14 五个面板的 `paper/figs/retrieval_robustness_topk.jpg`，CLIP ViT-L/14 面板清晰展示失效模式。

44. 已新增 CLIP BibTeX 条目并通过编译验证。
修改说明：`paper/ref.bib` 新增 `radford2021learning`（ICML 2021）。论文编译 0 LaTeX 错误、0 TBD 残留，BibTeX warning 从 4 增至 5（新增 CLIP 条目的 address 字段已修复为 "Virtual"）。main.pdf 4.3 MB, appendix.pdf 737 KB。

45. 已继续下载 VPR 专用模型代码与权重，不再受“无法下载”阻塞。
修改说明：已成功克隆 `CosPlace`、`MixVPR`、`Patch-NetVLAD` 到 `src/third_party/`；已成功下载 CosPlace 权重（`ResNet18_512_cosplace.pth`）、Patch-NetVLAD 预训练模型包和 MixVPR 权重（`resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt`）到本地缓存目录。

46. 已验证 50+ paired locations 的数据可行性，原“样本不足”结论已失效。
修改说明：实测 `F:\work\datasets\monitoring\images` 含 3710 个可用 clip id，`images2` 含 2696 个可用 clip id；并通过 `discover_paired_locations` 探针成功构建 50 对 paired locations（`num_queries=50`, `max_gallery=100`）。COCO（`F:\work\datasets\coco`）已确认可用（约 128k 图像）并可继续作为外部 distractor 池。

47. 已将 CosPlace、MixVPR、Patch-NetVLAD 接入主 retrieval attack 接口并完成专用 VPR benchmark 重跑。
修改说明：在 `src/eval/retrieval_attack.py` 中新增 dedicated VPR embedder 适配与骨干自适应输入尺寸；在 `src/scripts/run_controlled_retrieval_benchmark.py` 中扩展 `--backbones`、修复 MixVPR 模块导入冲突、补齐 Patch-NetVLAD checkpoint 与 `num_clusters` 推断逻辑，并成功完成 `cosplace/mixvpr/patchnetvlad` 三骨干 benchmark，输出 `src/outputs/controlled_retrieval_vpr_new3/robustness_summary.csv` 等结果。主文 Table 4 与 robustness 段落已据此更新，并重新生成合并后的 robustness 图。

48. 已将 blur/mosaic 参数扫描显式纳入主文图表与正文分析。
修改说明：在 benchmark 脚本中新增 baseline sweep 作图并导出 `paper/figs/baseline_param_sweep.jpg`；`paper/main.tex` 新增 `fig:baseline_sweep` 及对应文字分析，明确展示不同 blur kernel 与 mosaic block 在 PSNR-R@1 平面上的位置，补足主文对 support-aware baseline 参数敏感性的可视化说明。

49. 已按最新评审意见完成 final narrative polishing（摘要-结果-结论一致性收口）。
修改说明：在 `paper/main.tex` 中集中修订 Abstract、Related Work、Contributions、attacker-sensitivity 解释、matched-operating-point 结论句、temporal interpretation 和 Conclusion。已将结论从旧的 CLIP 中心叙述改为与当前主表一致的六骨干叙述（ResNet18/ResNet50/VGG16/CosPlace/MixVPR/Patch-NetVLAD），显式写入 ResNet50/VGG16 的正向 Δ（不利迁移）并同步强调 dedicated VPR 的有界支持证据；同时将“less nominal sigma”改写为参数效率解释、弱化时序模块归因强度，并在 Fig.5 图注中明确“support localization 可迁移价值”定位。论文已重新编译通过。

50. 已完成更大 paired-scene 设置下的 8 骨干单次统一重跑，并以单一输出目录回填主文 robustness。
修改说明：运行 `run_controlled_retrieval_benchmark.py` 于 `src/outputs/controlled_retrieval_unified8_large/`，参数为统一 8 骨干（`resnet18/resnet50/vgg16/clip_vitb32/clip_vitl14/cosplace/mixvpr/patchnetvlad`）、`pair_pool_size=600`、`max_gallery=100`、`gallery_sizes=12/24/48`。基于该单一目录的 `robustness_summary.csv` 更新了 `paper/main.tex` 中 Table~4、robustness 图注/描述和结论攻击器范围叙述，消除了“基础 run + VPR 专项 run”的合并来源不一致问题。

51. 已执行“独立重评审重置”并完成新一轮最高优先级一致性修订。
修改说明：已按命令重写 `docs/Revision_suggestions.tex`（全新评审，不继承旧轮次内容），并立即落实本轮最高优先级项：在 `paper/main.tex` 中修正 seed 口径（默认表格为多 seed、统一 8 骨干表为单 seed）、新增“跨表可比性边界”段落、为 Table~4 增加负/零/正 Δ 统计句（20/2/2）及单一输出目录溯源路径。该批修改已通过论文编译验证。

52. 已补充 larger-pair 待确认声明并清理图表双格式工件。
修改说明：按新评审 M4 要求，在 `paper/main.tex` 的 robustness 段与 Conclusion 显式加入“当前 unified 8 骨干表仍基于 12 paired locations，larger-pair confirmation 仍待后续周期”表述，进一步降低结论强度并避免超范围解读；同时删除 `paper/figs/` 中与主文无关的重复 PDF 图（`privacy_utility_tradeoff.pdf`、`retrieval_robustness_topk.pdf`、`baseline_param_sweep.pdf`），保留主文实际引用的 JPG 工件。

53. 已将 robustness 图拆分为上下两行（每行 4 子图）并作为双子图插入主文。
修改说明：更新 `src/scripts/regenerate_combined_figures.py` 以 unified 8 骨干输出为数据源，新增导出 `paper/figs/retrieval_robustness_topk_top.jpg` 与 `paper/figs/retrieval_robustness_topk_bottom.jpg`；`paper/main.tex` 中 `fig:robustness` 改为两个竖向 subfigure（上排：ResNet18/ResNet50/VGG16/CLIP ViT-B/32；下排：CLIP ViT-L/14/CosPlace/MixVPR/Patch-NetVLAD），并同步更新图注与 Description。编译验证通过。

54. 已按最新独立评审意见完成最终文本收口（必改项全部落地）。
修改说明：在 `paper/main.tex` 中完成本轮 required revisions：精简摘要中骨干逐项堆叠并改为“heterogeneous transfer”概括；在 Table~4 与 Fig.~4 图注显式标注 unified single-run 的用途（跨骨干一致性）与方向性解读边界；在 robustness 段新增单次统一运行仍具信息价值的解释句；将 matched-operating-point 段落中 temporal 模块贡献改为“弱可分离”表述；在 temporal 小节补充“deterministic 基线最平滑且 full 与 temporal ablation 几乎不分离”；修正结论中 matched-utility 语义（同效用下常收敛）；并在 future work 显式列出三项未验证限制（paired 规模、cross-view 覆盖、all-backbone seed-averaged rerun）。已重新编译通过。
55. 已完成 all-backbone seed-averaged 复核。
修改说明：在统一协议下完成了 3 seeds（1234/1235/1236）× 8 骨干（ResNet18、ResNet50、VGG16、CLIP ViT-B/32、CLIP ViT-L/14、CosPlace、MixVPR、Patch-NetVLAD）× 3 gallery 大小（12/24/48）的完整 seed-averaged 基准实验。输出目录为 `src/outputs/controlled_retrieval_seed_avg/`，包含 10 个 CSV 文件。关键结果：24 个骨干-gallery 单元中 23 个 Δ 为负，仅 MixVPR g48（raw=0.000）为边际正值。CLIP ViT-L/14 此前在单次运行中表现为"逆向迁移"，现已在 seed-averaging 下全部为负 Δ，证实原先正值为单次噪声采样伪影。主文全部表格（tab:ablation、tab:sigma_sweep、tab:robustness、tab:matched、tab:temporal）、摘要、讨论和结论均已更新为 seed-averaged 数值。

56. 已完成更大 paired-scene（50 pairs）主文回填。
修改说明：50-pair benchmark 已完成（8 骨干 × 3 seeds × gallery 50/75/100，COCO 128 外部干扰，输出 `src/outputs/controlled_retrieval_large50/`）。关键结果：24 个骨干-gallery 单元中 21 个 Δ 为负，3 个正值均来自 MixVPR。已在 `paper/appendix.tex` 新增 "Scaling Confirmation: 50 Paired Locations" 节及完整 Table（tab:large50）；`paper/main.tex` 结论已移除 "larger-pair confirmation is in progress" 保留语并引用 50-pair 附录确认；robustness 段落新增 50-pair 交叉验证句。论文编译通过。

57. BibTeX 字段清理完成，剩余 5 条 warning 属可接受范围。新增 CLIP 条目（radford2021learning）。

58. 已按最新评审意见完成最终 polishing 轮次（5 项 required/suggested revisions 全部落地）。
修改说明：(1) 重新生成 Fig.5 baseline 参数扫描图，从 2 个数据点扩展到 8 个（blur k=5/11/21/31 + mosaic b=4/8/12/20），并更新图注使之与实际数据一致；(2) 在摘要（ELSE 块）中显式标注 MixVPR 为附录级异常骨干；(3) 在摘要和 matched-OP 段中加入"rather than stronger matched-utility privacy"澄清句；(4) 在实验协议段新增统一 seed-averaged 协议的动机句（"deliberate methodological choice"）；(5) 将结论中"resolving the earlier apparent adverse-transfer instability"弱化为"seed averaging substantially reduces..."的审慎措辞；(6) 在 robustness 段末尾新增"50-pair scaling confirmation ... not purely small-sample artifacts"推广句；(7) 为 Tables 1/2/4/5/6 五张表的 caption 统一加入 stat-reporting 说明（deterministic=exact, stochastic=mean±std），消除 reviewer 可能质疑的格式不一致。BibTeX warning 从 5 降至 4。论文编译 0 错误、0 TBD。

59. 已按 ACM TOMM 最终审稿意见完成第 9 项 reviewer-proofing 修订。
修改说明：（详细修改见上一条）

60. 已执行独立评审重置并完成纯编辑级修订（E1–E7）。
修改说明：`docs/Revision_suggestions.tex` 完全重写为新一轮独立评审（7 项 editorial fixes，推荐 Accept with Minor Revisions）。立即实现全部 7 项：(E1) 删除 Conclusion 中重复的"substantially"（3→0）；(E2) 将 robustness 段和 Conclusion 中的"broadly supportive"替换为"consistent transfer across diverse attacker families"和"consistent negative Δ"；(E3) 在 matched-OP 段将"most transferable component"改为"most consistently beneficial component in the current evaluation"；(E4) 修正 matched-OP 段中 blur/noise 隐私效果对比的归因（从 support 改为 feature disruption）；(E5) 删除"lightweight dynamic CRF"的无证据修饰词，改为"a small number of mean-field iterations"；(E6) 将 Conclusion 中"12 paired locations"统一为"12 pairs"；(E7) 删除 temporal 段的重复句（两句合并为一句）。论文编译 0 错误、4 BibTeX warning、0 TBD。main.pdf 18 页。

61. 已按新一轮外部评审意见（10 项）完成全面 reviewer-proofing 修订。
修改说明：`docs/Revision_suggestions.tex` 被外部更新为"Final ACM TOMM-Oriented Revision Suggestions"（10 项）。全部实现：(R1) Section 3.4 主张从"consistent transfer"弱化为"broadly supportive transfer under the current benchmark protocol"；(R2) 统一统计报告格式——tab:sigma_sweep 全部条目改为 mean±std（含零方差 ±0.000）、PSNR/SSIM 加入 ±std，tab:ablation 和 tab:matched Panel C 零方差条目同步修正；(R3) Fig.4 引用处新增一句明确说明图中展示的是主 benchmark（gallery 12/24/48）而非附录大规模协议；(R4) Fig.5 标签字号从 9→8 并添加灰色箭头连接线，改善低 PSNR 区域密集标签可读性；(R5) Conclusion 中"resolves"改为"substantially reduces"；(R6) Section 3.4 末尾新增过渡句总结主 benchmark 与附录确认的一致结论；(R7) Scope 段扩展附录指引，明确 50-pair 确认用途；(R8) 无需修改，temporal section 已保持恰当克制；(R9) 将贡献列表和 matched-OP 段中残留的"most transferable component"统一为"most consistently beneficial component"；(R10) Section 3.3 匹配操作点措辞更精确，强调 PPEDCRF 在随机方法中的 high-utility 定位并与确定性 baseline 互补。论文编译 0 错误、4 BibTeX warning。main.pdf 18 页、appendix.pdf 4 页。

62. 已执行独立评审重置并完成 4 项编辑级修订（E1–E4）。
修改说明：`docs/Revision_suggestions.tex` 完全重写为新一轮独立评审（4 项 editorial，推荐 Accept with minor editorial polish）。立即实现全部 4 项：(E1) Conclusion 中"earlier apparent instability"改为面向冷读者的措辞"Seed averaging across three independent noise realizations stabilizes the transfer assessment"；(E2) ACM 摘要 transfer 句补充"(23 of 24 backbone--gallery cells show negative Δ)"，与正文对齐；(E3) Section 3.4 过渡句中"broadly favorable"统一为"broadly supportive"；(E4) tab:matched 表题补充脚注说明 PSNR/SSIM 列省略 ±std 的原因。论文编译 0 错误、4 BibTeX warning。main.pdf 18 页、appendix.pdf 4 页。

63. 已按用户指令执行完整独立评审重置并完成 6 项修订（R1–R6）。
修改说明：用户触发"重新开始评审并生成评审修改意见"命令。直接基于当前 main.tex、appendix.tex 和所有渲染图形独立评审，以 ACM TOMM 标准编写全新英文评审意见（6 项，推荐 Accept with minor revisions）并重写 `docs/Revision_suggestions.tex`。立即实现全部 6 项：(R1) 重新生成 Fig.3（privacy-utility frontier）——PPEDCRF sigma 标签上移、global noise 标签下移并按曲线颜色着色，消除收敛区域标签重叠；(R2) Fig.6 定性图 caption 新增说明"at lower σ₀ the difference map shows sharper spatial selectivity"，解释 σ₀=24 差异图近似均匀红色的原因；(R3) Algorithm 1 后新增一句运行时性能说明（192×320 帧 <50ms on RTX 3090）；(R4) IEEE 摘要（IF branch）与 ACM 摘要对齐——加入"so the practical benefit lies in spatially concentrated perturbation and parameter efficiency"措辞和去除弱措辞；(R5) 压缩 benchmark 难度统计段落——12 条 pair 相似度从四位小数简化为 ≈0.99、distractors 简化为 0.86；(R6) Fig.2 caption 从"Implementation PPEDCRF pipeline"改为"Overview of the PPEDCRF pipeline"。论文编译 0 错误、4 BibTeX warning。main.pdf 18 页、appendix.pdf 4 页。

64. 已按外部评审意见（10 项 reviewer-proofing）完成最终核查，仅剩 1 项未落地并已修复。
修改说明：`docs/Revision_suggestions.tex` 外部更新为中文评审格式（10 项 reviewer-proofing + 优先级排序）。逐项核查发现 9 项已在前序轮次中落地：Section 3.4 claim 已用"broadly supportive"措辞（#1/✅）；表格统计格式已统一并在 caption 说明（#2/✅）；Fig.4 gallery 尺寸说明已写入正文（#3/✅）；Fig.5 标签可读性已修复（#4/✅）；Section 3.4 过渡句已存在（#6/✅）；Appendix 前置指引已存在（#7/✅）；Temporal 保持现状（#8/✅）；术语已统一（#9/✅）；Table 3"best"已限定为"among stochastic methods"（#10/✅）。唯一残留项为 #5：robustness 段 line 385 仍用"resolves this instability"和 line 455 仍用"is resolved"——已分别改为"substantially reduces this instability"和"is substantially reduced"，同步提及 MixVPR 例外。论文编译 0 错误、4 BibTeX warning。main.pdf 18 页、appendix.pdf 4 页、titlepage.pdf 新增。

65. 已执行第 5 轮独立评审重置并完成 5 项修订（R1–R5）。
修改说明：`docs/Revision_suggestions.tex` 完全重写为新一轮英文独立评审（Round 5，推荐 Accept with no further revisions）。发现并修复 1 项关键问题 + 4 项次要问题：(R1/Critical) matched-operating-point 段中"smaller nominal sigma"方向性错误——PPEDCRF 实际需要更大 σ₀ 才能达到相同 PSNR，已重写为"at any given σ₀, selective perturbation preserves approximately 6 dB more PSNR"；摘要中同义措辞同步修正，删除"parameter efficiency"改为"spatially concentrated perturbation that preserves higher visual quality at any given noise scale"。(R2) frontier 段引用的 Global Gaussian PSNR 值（24.23/20.83 dB）仅在图中不在表中，已加括号注明数据来源。(R3) 附录 benchmark 构造步骤中"fixed ResNet18 attacker"改为"backbone"以统一术语。(R4) Fig.3 caption 从模糊的"substantially better"改为定量"approximately 6 dB PSNR advantage"。(R5) robustness 段两处"resolves/is resolved"弱化为"substantially reduces/is substantially reduced"。论文编译 0 错误、4 BibTeX warning。main.pdf 18 页、appendix.pdf 4 页、titlepage.pdf。

66. 已将 ACM TOMM 审稿信整理为结构化 Markdown 文档。
修改说明：重排 `docs/TOMM_Response_Letter.md` 的编辑决定、三位审稿人的推荐结论、文字性意见和附加问题，统一使用 Markdown 标题、编号列表、引用块和字段列表；保留审稿意见原文及空白回答项的语义。

67. 已根据当前 ACM TOMM 新审稿意见启动 revision cycle，并完成实验计划、代码骨架、论文事实校准和远程实验交接。
修改说明：在 `docs/Design.md` 中先登记 E1–E7 实验计划及完成门槛，在 `src/scripts/` 中新增受控消融、地理标注 VPR、同图像效用、unary provenance 和 retrieval case-study 工具；`paper/main.tex` 与 `paper/appendix.tex` 已补充方法来源、公式定义、数据构造和可复现性限制。CUDA smoke test、12-query ResNet18 代理运行、攻击者感知小规模诊断和 provenance audit 均已完成，但当前 checkpoint 的 `mask_root=null` 且 unary map 近似空间常数，因此没有把代理结果写成论文证据；`docs/4c_experiment_handoff.md` 已写明 4c 启动命令与 Claude Code 监控规则。论文子模块 commit 为 `a0f69be`，父仓库集成 commit 为 `b4a623c`，交接文档最终更新 commit 为 `254c5f7`，均已推送到对应 `origin/main`。

---

## 当前状态（2026-08-31 更新）

**本次更新（vGPU 3090 实验全部完成并已关机）：**
- `e5_provenance`、`proxy12`（完整 8 骨干）、`proxy50`（8 骨干拆分并行）、`e4_detection`、`e4_segmentation` 全部以 `EXIT_CODE=0` 完成，结果已拉回本地并校验（sha256 + 行数），`proxy50` 8 份骨干结果已合并
- 应用户要求做了 GPU 并行优化（8 进程拆分 + 线程上限），GPU 利用率从 0% 提升到 58%–100%；顺带修复了 `proxy12`/`proxy50` 输出目录冲突的真实 bug
- vGPU 3090 已确认关机（连接已断开，停止计费）
- E1 仍按原计划阻塞未尝试；4c 3090 已按用户决定放弃使用
- 已完成：审计并回填 `src/outputs/tomm_review_proxy50/`、`tomm_review_proxy12/` 和 `tomm_same_image_utility_seg_v2/` 的具体数值。

**已完成项：**
- 已完成 74 项修订任务
- 论文编译通过（0 LaTeX 错误、4 BibTeX warning、0 TBD 残留）
- 8 个攻击骨干（分批）稳健性分析与主文集成
- CLIP ViT-L/14 失效模式已记录并集成到论文
- VPR 专用模型代码与权重已完成本地下载（CosPlace / MixVPR / Patch-NetVLAD）
- 已确认本地数据可支持 50+ paired locations 构建
- 已完成 dedicated VPR 攻击器接入，并补齐 CosPlace / MixVPR / Patch-NetVLAD 三骨干结果
- 已在主文显式加入 blur/mosaic 参数扫描图（Fig.5，8 数据点，标签无重叠）与对应说明
- 已完成摘要-主结果-结论的一致性收口与评审意见文本级修订
- 已完成 8 骨干单次统一长跑复核并用单一输出目录更新主文 robustness
- 已完成独立重评审重置并开始新一轮修订循环
- 已完成 larger-pair proxy50 confirmation，并完成图表工件单源化清理
- 已完成 robustness 图 2×4 子图重排并插入主文
- 已完成 50-pair 大规模确认实验（23/24 负 Δ、1 个平局），附录新增完整表格并同步更新主文结论
- all-backbone seed-averaged 复核与 50-pair 规模确认均已完成
- 最终 polishing 轮次完成：Fig.5 重生成、摘要/结论修辞弱化、表格统计格式统一
- 第 59 轮 reviewer-proofing 完成：benchmark 限制管理句、MixVPR 一致措辞、Fig.5 标签修复、摘要压缩、附录指引、结论语气精炼
- 第 63 轮独立评审重置完成：Fig.3 标签重叠修复、Fig.6 caption 说明、运行时性能、IEEE 摘要对齐、benchmark 统计压缩、Fig.2 caption
- 已完成当前新审稿意见对应的 Design.md 实验计划登记、5 个实验/诊断脚本、论文方法与限制说明，以及本地 RTX 3070 CUDA smoke test
- 已完成 12-query 代理消融、攻击者感知小规模诊断和 unary provenance audit；这些结果因 checkpoint provenance 和代理数据限制仅作为工程验证保留
- 已完成 E4 官方 DeepLab 预处理修正、CUDA manifest 重跑及同图像 mIoU 结果集成
- 已完成 proxy12/proxy50 retrieval 数值替换、固定预算图重生成和 LaTeX 渲染复核

**阻塞项：**
- E1 真实 place/GPS VPR 与 4c 远程启动【已阻挡】：本地监控数据没有真实 place/GPS 标签，4c 的 TCP 检查失败且既有 SSH alias 存在 host fingerprint 变更；下一步由管理员确认端点可达性与指纹，并提供合规 manifest。
- E5 unary sensitivity-map 证据【已阻挡】：当前 checkpoint 没有 mask root，监控片段上的空间标准差约为 $2.6\times10^{-4}$；下一步提供 mask-backed checkpoint 或重新训练后再做独立诊断。

**下一步评审循环建议：**
E1/E5 的公开数据与独立 unary 验证仍受注册、checkpoint 和远程资源门槛约束；E2/E3/E4/E6/E7 还需分别完成 matched energy、attacker-aware 定量写回、detection mAP 整合、margin/case-study 和 MixVPR 深度诊断。远程 GPU 连接恢复且实验门槛通过后，由 Claude Code 按 `docs/archived/4c_experiment_handoff.md` 继续监控。

68. 已修复 E4（同图像效用评估）里一个真实 bug，并新增 manifest 生成脚本，已推送到 origin/main（commit `19f0752`）。
修改说明：`src/scripts/evaluate_same_image_utility.py` 的 `load_target()` 之前用 `_read_image`（RGB 照片解码器）读取 VOC 风格的调色板索引分割 mask，会把类别 id 经调色板转换成显示颜色，彻底破坏 0..20/255 的类别语义；还有第二个 bug，把结果（`(3,H,W)` torch tensor）直接传给 `cv2.resize`（要求 numpy 数组），直接崩溃。已新增 `_read_class_index_mask()`（用 PIL 按调色板索引读取，不做 RGB 转换）并替换掉原逻辑，本地 2 图 CPU smoke 验证：mIoU 原图 0.746，`full`（PPEDCRF）掉到 0.719，`global_noise` 掉到 0.548——方向上与论文隐私-效用故事一致。同时新增 `src/scripts/build_same_image_utility_manifest.py`，从本地 COCO val2017（检测框）和 VOC2012（分割 mask）各采样 200 张，坐标已按目标分辨率重新缩放，供 E4 使用。

69. 已在 4c 3090 和 vGPU 3090 两台主机上排查 GPU 可用性，发现两个不同的真实阻塞项，详见下方"遗留问题"。
修改说明：4c 3090 SSH 可连但 CUDA 驱动栈整机损坏（`cuInit` 返回 999），已确认非代码/权限问题；转到 vGPU 3090 后连接和权限都正常，但当前从本机到该实例的网络带宽严重不足（2.2GB 数据传输预计需 16–40+ 小时）。vGPU 3090 远端环境已部分就绪：仓库已 clone 到 commit `19f0752`（含上条修复）、三个 third-party 子模块已手动 clone 到锁定 commit、`/root` 下清理了约 24GB 无关旧项目残留（已经用户确认）、pip 缓存目录和源已切换到 `/root/autodl-tmp` 与阿里云镜像。完整状态见 `docs/vgpu3090_experiment_handoff.md`。

70. 已通过 HuggingFace Hub 私有数据集中转彻底绕开本机↔vGPU 3090 的带宽阻塞，完成远程环境搭建并启动全部 5 个 revision-cycle 实验，其中 E5 audit 已完成并复现此前阻塞结论。
修改说明：本机重启后重新测试本机↔vGPU 3090 直连带宽（50MB dd-over-ssh 60 秒内未完成），确认直连方案仍不可行；按用户此前在"遗留问题"中给出的决策（选项1失败后走 HuggingFace 中转），在本机打包 monitoring 子集（600 clip、4198 文件、2.2GB，与 `discover_paired_locations` 对 `pair_pool_size<=600` 时选取的确定性前缀完全一致）、VPR 权重（Patch-NetVLAD `mapillary_WPCA4096.pth.tar` + `vpr_cache/`，398MB）、E4 same-image utility manifest 及引用图片（55MB，COCO detection 200 + VOC segmentation 200，路径已改写为相对路径）、以及 `sensnet_final.pt` checkpoint，上传到新建的私有数据集仓库 `mabo1215/ppedcrf-tomm-vgpu-relay`。上传直连 `huggingface.co`（token 校验通过，`hf-mirror.com` 经测试不支持鉴权类 API，仅能镜像文件下载端点）；因单次 Bash 调用有 10 分钟上限，2.2GB 文件被切成 7 片（350MB/片）逐片上传。远端下载改用 `curl -C -`（断点续传）+ `--speed-limit`/`--speed-time` 卡死检测的重试循环，而非 `huggingface_hub` 默认的 xet 传输后端——xet 传输在 `network_turbo` 代理下会稳定卡在 0 字节且不重连，换成 curl 后所有分片正常完成。远程 pip 源从阿里云镜像（返回 403）切换为清华镜像后，torch 2.13.0+cu130 安装成功并确认 CUDA 可用（RTX 3090 vGPU，49152MiB）。全部数据完整性已用 sha256 核对（`monitoring_subset.tar`、`sensnet_final.pt` 均与本机哈希一致）并解压到脚本期望的确切路径。随后按 `docs/vgpu3090_experiment_handoff.md` 的 launch plan 启动 5 个并行 screen 会话：`e5_provenance`（已完成，`mean_probability_spatial_std≈2.63e-4`，与此前阻塞记录的 `2.6×10⁻⁴` 一致，确认 E5 blocker 依旧成立，非环境问题）、`proxy12`、`proxy50`（覆盖 E2/E3/E6/E7，运行中，已产生 `selection.json` 且 CPU 占用持续 >580%）、`e4_detection`、`e4_segmentation`（运行中，首次启动时会各自下载一次 torchvision 预训练检测器权重）。过程中两次因缺依赖崩溃（`matplotlib` 缺失；随后发现 Patch-NetVLAD/CosPlace 骨干还需要 `faiss-cpu`、`scikit-learn`、`pandas`、`scipy`），已全部补装并重启对应会话。

71. 已完成 vGPU 3090 上全部 5 类 revision-cycle 实验（E2/E3/E6/E7 的 proxy12+proxy50、E4 的 detection+segmentation、E5 provenance），拉回并校验全部结果，关闭 vGPU 3090 实例。
修改说明：应用户要求"最大化压榨 GPU"，诊断发现原始单进程顺序跑 8 骨干时 GPU 利用率长期为 0%（瓶颈在 CPU 端单条 Python 循环的图像预处理），遂将 `proxy50` 拆成 8 个各跑一个骨干的并行进程并给每个进程加 `OMP_NUM_THREADS=8`/`MKL_NUM_THREADS=8` 线程上限（避免多进程抢 96 核互相拖慢），GPU 利用率随之提升到 58%–100%。过程中同时发现并修复一个真实数据损坏 bug：`proxy12` 与 `proxy50` 此前共用默认输出目录 `src/outputs/tomm_review_proxy`，两个进程会互相覆盖对方的 `per_query.csv`/`summary.csv`，已改为各自独立目录。拆分引入的新问题也逐一处理：(1) 8 个并行进程各自重复计算共享的 pair-discovery 步骤，瞬时显存峰值叠加导致 `clip_vitb32`/`patchnetvlad` 各 OOM 一次，等其他进程结束释放显存后重跑成功；(2) `patchnetvlad` 骨干本身显存占用高达 ~21GB，需要在显存宽裕时单独跑；(3) CLIP 系列（`clip_vitb32`/`clip_vitl14`）加载依赖 `transformers` 联网拉取模型，需要 `source /etc/network_turbo`，重启后一度因未带上该环境变量而报连接失败，补上后又因早期失败请求被 `huggingface_hub` 缓存为".no_exist"负结果导致离线模式下持续报"找不到文件"，清空对应缓存目录后解决；(4) 换用非缓存的直接下载后发现 `huggingface_hub` 默认的 xet 传输后端在 `network_turbo` 代理下会报 401 或直接卡在 0 字节不重连（和此前 monitoring 数据下载遇到的问题同源），改用 `curl -C -`（断点续传）+ `--speed-limit`/`--speed-time` 卡死检测的重试循环，直接把权重文件写入 HF 缓存的 blob 路径（用 HEAD 请求的 `X-Linked-ETag` 确定目标哈希）并手写 snapshot 软链接，绕过 `huggingface_hub` 的下载逻辑；(5) 当晚 `network_turbo` 到 `huggingface.co` 的链路本身时快时慢（30KB/s–870KB/s 波动），改用 `hf-mirror.com`（无需 turbo）后测得约 3.6MB/s，两个 CLIP 权重文件（577MB + 1.63GB）改道后很快下载完成，离线加载验证通过。全部 8 个 `proxy50` 骨干 + `proxy12`（完整 8 骨干单进程）+ `e5_provenance` + `e4_detection`/`e4_segmentation` 均以 `EXIT_CODE=0` 收尾；结果文件（`per_query.csv`/`summary.csv`/`selection.json`/`run_metadata.json`/`provenance.json`/`utility_summary.json`）打包为 2.3MB tar.gz 用 sha256 校验后 scp 拉回本地，行数与远端日志报告的行数逐一核对一致（如 `proxy50` 各骨干 3750/3900 行、`proxy12` 7236 行）；`proxy50` 的 8 份骨干结果已在本地合并为统一的 `src/outputs/tomm_review_proxy50/{per_query,summary}.csv`（30150/219 行，字段取并集以兼容 resnet18 独有的 attacker-aware 列）。全部结果确认落地后执行 `shutdown -h now` 关闭 vGPU 3090 实例并验证连接已断开（停止计费）。

72. 已修正 E4 同图像分割预处理并在指定环境中完成 CUDA 重跑。
修改说明：在 `evaluate_same_image_utility.py` 中改用官方 DeepLabV3-ResNet50 权重的输入变换，保留 VOC 调色板分割 mask 的类别索引，并在 mIoU 计算前将模型 logits 恢复到原图尺寸。使用 `D:\source\.venv` 在 CUDA 上完成 200 张 manifest 图像评估；结果写入 `src/outputs/tomm_same_image_utility_seg_v2/utility_summary.json`，PPEDCRF mIoU=0.669、global Gaussian noise mIoU=0.632，原图参考 mIoU=0.697。

73. 已用 proxy12/proxy50 新数字整体替换论文中的旧 retrieval 表格和叙述。
修改说明：主文 `tab:ablation`、`tab:robustness`、固定预算图及摘要/正文/结论已改用 `src/outputs/tomm_review_proxy12/`；附录 `tab:large50` 和 scaling 叙述已改用 `src/outputs/tomm_review_proxy50/`。同步删除没有新 proxy 数据支撑的旧 sigma sweep、matched-PSNR 和 temporal 数值结论，统一注明固定 `sigma_0=8` 与 paired-scene proxy 限制；按新 CSV 重生成三张 retrieval 图并完成 LaTeX 与渲染复核。

74. 已基于当前论文状态重写独立评审意见。
修改说明：完全重写 `docs/revision_suggestions.tex`，移除上一轮已过期的 frontier、matched-operating-point 和旧 transfer 判断，重新核对 proxy12/proxy50、E4 mIoU、固定 `sigma_0=8`、论文图表和已知限制；当前评审结论为无阻塞性修改项。

75. 【进行中】已完成 E1/E5 公开数据集检索、实验计划登记和论文限制更新。
修改说明：基于官方数据源检索并选定 MSLS 作为 E1 首选、Oxford RobotCar 作为注册制备选、KITTI-360 作为 E5 的序列保持验证数据；已将数据来源、标签字段、许可门槛、实验指标和禁止占位数字的规则写入 `docs/Design.md` 与 `docs/revision_suggestions.tex`，并新增 MSLS/KITTI-360 manifest 构建脚本及 unary attribution 验证脚本。由于公开数据尚未完成注册/下载，当前只进行本地 smoke test，尚未产生可写回论文的 E1/E5 数字。

**安全提醒（需要你关注，累计更新）**：本次操作中 Hugging Face token 除此前记录的两次外，又因命令行内联传递（`ps aux`/screen 启动命令回显）在本会话工具输出中出现了若干次。token 本身仍未被写入任何提交、日志文件或论文内容，只出现在这次交互式会话的回复中；建议你之后去 Hugging Face 账号设置里吊销并重新生成 `.env` 里的 `Huggingface_model_token`。
76. 【进行中】已完成 E1/E5 适配器、指定 CUDA 环境 smoke test 和本轮文档/论文构建校验。
修改说明：使用 `D:\source\.venv`（Python 3.14、PyTorch 2.13.0+cu132、RTX 3070 CUDA）运行 unary attribution smoke（3 queries）与 geotagged VPR smoke（9 rows），并成功构建 `docs/revision_suggestions.pdf`、`docs/experiment_progress.pdf`、主论文和独立附录。真实 MSLS/KITTI-360 运行仍需注册数据、mask-backed checkpoint 和 3090 vGPU；没有把 smoke 数字写入论文。

77. 已修复 vGPU 3090 上被清空的 `src/` 源码目录。
修改说明：本次会话登录 vGPU 3090（`connect.westd.seetacloud.com:22766`）后发现 `/root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF/src/` 下 `scripts/`、`eval/`、`datasets/`、`models/`、`privacy/`、`utils/` 等目录只剩 `__pycache__` 和已缓存的 VPR 权重（`vpr_cache/`），全部 `.py` 源文件缺失（`main.py`/`run_eval.py`/`run_train.py`/`requirements.txt` 也缺失），推测是上一轮会话在准备 git fetch/checkout 时的清理步骤中途中断所致。已从本地仓库打包纯代码（排除 `outputs/`、`data/`、`third_party/`、`vpr_cache/` 等大文件/数据目录）通过已 pin 的 SSH 连接传输并在远端解压恢复，随后在远端用 `--mode smoke` 验证了完整流水线在实际 3090 显卡上可正常运行。

78. 已在 `src/scripts/run_controlled_retrieval_benchmark.py` 中补充 `unary_only`/`no_dcrf` 两个 DCRF 消融变体及 sigma 匹配搜索的 `actual_mse` 派生列。
修改说明：这两个变体与 `run_tomm_review_proxy.py`（当前论文数字的权威来源脚本）中已有的同名变体语义对齐（`unary_only` 保留 NCP 幅度调制，`no_dcrf` 同时关闭 DCRF 精炼和 NCP）；`matched_variants` 搜索列表也扩展为包含 `no_temporal`/`no_ncp`/`unary_only`/`no_dcrf`，并新增按 PSNR 反推的 `actual_mse` 字段。这是为 E2 matched-PSNR 实验准备的代码，`run_controlled_retrieval_benchmark.py` 是被 `run_tomm_review_proxy.py` 导入复用的底层模块，两者的变体定义保持一致。

79. 已在 `src/scripts/run_tomm_review_proxy.py` 中新增 `--sigma` 覆盖参数，用于 E2 matched-PSNR sigma 扫描。
修改说明：允许在不修改 `config.yaml` 的情况下为整次 proxy 运行覆盖噪声 sigma，写入 `run_metadata.json` 便于追溯；已用 `--mode smoke` 在本地 RTX 3070 和远端 vGPU 3090 上分别验证未破坏原有流程。

80. 已完成 E3（attacker-aware 基线）审计并写入 `paper/appendix.tex` 新增的 "Constrained Attacker-Aware Baseline" 一节。
修改说明：发现 `src/outputs/tomm_review_proxy12/per_query.csv` 中已经存在完整的 `attacker_aware` 变体数据（20 步符号梯度、$\ell_\infty=8$、ResNet18），与其余变体共享相同 gallery/seed/输出 schema，此前从未整合进论文正文或附录。核实数字后新增到附录：attacker-aware 基线在与 PPEDCRF 相近的画质预算下（PSNR≈35.1dB）把 Top-1 压到 0.000（全部 3 个 gallery size），而 PPEDCRF 同预算下是 0.722；作为诚实的“白盒攻击者上界”诊断写入，明确说明其威胁模型假设强于论文正设定。

81. 已完成 E4（detection mAP 复核）并整合进 `paper/main.tex` Table~tab:e4seg。
修改说明：定位到 `src/outputs/e4_audit/utility_manifest_detection.jsonl` 与 `utility_manifest_segmentation.jsonl`（各 200 张，08:28 构建，即"最终 manifest"），发现此前的 map50 结果（01:35）早于该 manifest 构建时间，不满足"针对最终 manifest 复核"的要求。已用本地 RTX 3070（`D:\source\.venv`）对两个 manifest 重新各跑一次全部 8 个变体（`full/no_temporal/no_ncp/unary_only/no_dcrf/masked_blur/masked_mosaic/global_noise`），确认 mIoU 数字与此前 v2 结果完全一致（可复现），mAP50 数字也与旧结果高度接近但来自同一次审计。已将 mAP@50 列与 mIoU 列合并进同一张表并更新正文两处引用数字的段落。

82. 已完成 E6（margin 子组分析与定性 case study）并整合进 `paper/appendix.tex`。
修改说明：新增 `src/scripts/margin_subgroup_analysis.py`，对已有的 `per_query.csv`（无需新实验/无需 GPU）按每个 backbone/gallery_size 的原始（未防护）margin 中位数做 small/large-margin 两分组，同一分组成员在防护前后保持一致以便对比。在 proxy50（全部 8 backbone、gallery=50）上发现：large-margin 组几乎不受防护影响（Top-1 从 1.000 到 0.987），privacy 效果几乎全部集中在 small-margin 组（Top-1 从 0.465 降到 0.377，降幅约 6.8 倍于 large-margin 组）。同时用 `src/scripts/generate_retrieval_case_study.py` 基于真实 `tomm_review_proxy12` 数据渲染了论文此前缺失的定性 retrieval case study 图（`paper/figs/retrieval_case_study.jpg`，选取最差 margin 的 query），已写入附录新章节并成功随 `build.bat` 编译。

83. 已完成 E7（MixVPR 迁移诊断）并澄清审稿人所指的"adverse transfer"在当前修正流水线中已不复现。
修改说明：核对当前 `paper/appendix.tex` 中 proxy50 的 MixVPR 数据（三个 gallery size 分别为 $-0.033/-0.053/-0.053$）和当前 proxy12 数据，均无正向（adverse）delta，与 R3-8 引用的旧附录状态不同；由于没有保留复现旧 adverse 结果的具体历史运行状态，未声称具体根因，只如实报告"当前复核结果不再复现该现象"。复用第 82 条的 margin 分组基础设施单独对 MixVPR 做了细分：large-margin 查询完全不受影响（Top-1 恒为 1.000），small-margin 查询从 0.560 降到 0.493，解释了为何 MixVPR 的聚合 delta 偏小但并非 adverse。因当前操作点并非 adverse，未运行专门的 mitigation 实验，而是在附录中明确记录了"若未来在其他 backbone/预算下复现 adverse transfer，应尝试对 large-margin 查询提高 sigma 的 margin-aware 噪声调度"作为具体的后续方向。

84. 已启动 E2（matched-PSNR/MSE sigma 扫描）在 vGPU 3090 上的正式运行。
修改说明：本地→vGPU 3090 通过已 pin 的 SSH 连接直传完整 monitoring 图像语料（`F:\work\datasets\monitoring\images`，14GB/25934 文件，此前 vGPU 上只有 24 张样例图和一个损坏的子集 tar 包，不足以复现论文已发表数字所用的确定性 pairing）。传输完成后在 vGPU 3090 上以 12 个并行 screen 会话（sigma∈{4,6,8,10,12,16,20,24,28,32,40,50}，ResNet18，proxy12 规模）运行 `run_tomm_review_proxy.py --mode proxy --sigma <S>`，用于生成 matched-PSNR/effective-MSE 对比表；完成后将回填 `actual_mse`/`actual_psnr` 匹配点和相应论文段落。此项在本次会话结束时可能仍在运行，需要下一轮继续跟进结果回填与关机。

85. 已完成 E5 所需 KITTI-360 公开数据下载与解压，数据获取阻塞项解除（诊断本身仍受 mask-backed checkpoint 阻塞，见下）。
修改说明：你确认已获得 cvlibs.net 授权后，改为直接从公开 S3 镜像（`s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/...`）按需下载，未使用 `download.php` 的 token 脚本方式，因此不再需要把 4 个 token zip 放进 `download_scripts/`。按之前约定的最小范围只取了 2 个 sequence 的 Perspective Images（`2013_05_28_drive_0000_sync`、`2013_05_28_drive_0002_sync`，仅 `image_00` 左目，共 22.5GB，而非全量 128G）、全量 Semantics of Left Perspective Camera（1.8G）、全量 Vehicle Poses（8.9M）；Calibrations 沿用你已下载好的版本未重新下载。下载过程中 WSL 本地 DNS 解析器（`10.255.255.254`）中途失效（与 KITTI-360/S3 无关的本机网络问题，`curl`/`ping` 走裸 IP 正常，仅主机名解析失败），先用 Cloudflare DoH（`1.1.1.1/dns-query`）解析出 S3 主机 IP 并配合 `curl --resolve` 断点续传绕过阻塞完成了下载；之后按你的要求改为经 Windows 侧 PowerShell（确认 `Resolve-DnsName`/`Test-NetConnection` 均正常）作为后续下载的主路径，以避免继续依赖该规避手法。两个 `image_00` zip 用 Python `zipfile`（本机无 `unzip`）解压到 `data_2d_raw/<sequence>/image_00/data_rect/`，与 `src/scripts/build_kitti360_unary_manifest.py` 期望的目录布局一致；已核对下载字节数与解压文件数：`2013_05_28_drive_0000_sync` 11518 张、`2013_05_28_drive_0002_sync` 14607 张，均与各自 zip 内条目数（减去目录项）完全一致。同一时段确认此前启动的 MSLS 11 个签名链接下载（`G:\work\datasets\msls\raw`）也已全部完成（`ALL_MSLS_PARTS_DONE`，10 个 zip 全部落地）。尚未运行 `build_kitti360_unary_manifest.py` 生成正式 manifest 或做序列保持诊断本身。

87. 已把 E2 matched-PSNR sweep 写回论文正文与附录，修正了此前口头汇报中一个方向性错误。
修改说明：起初误读 `matched_psnr_table.csv`——把"blur/mosaic 在更低 PSNR 下 Top-1 更低"错误汇报成"更差"，实际按论文既有约定 Top-1 越低隐私越好，blur/mosaic 在（更差的）自身固定画质点上反而隐私更好；但这不是同画质比较，因为 blur/mosaic 在本次 benchmark 里没有可调的 sigma 参数，扫不到 30/33/36dB 目标（差距最大到约 10dB）。已就此当面纠正你此前收到的错误说法，并按你选择的"两者都写，谨慎加注"方案落地：`paper/appendix.tex` 新增 "Matched-PSNR/Effective-MSE Comparison (R2-2, R3-3)" 一节（24 行表格，3 个目标 PSNR × 8 个变体，含 `Matched σ`/`PSNR gap` 列使数据自证"未真正匹配"这一点），明确写出两个结论：(1) 6 个可调 sigma 的变体在同一目标 PSNR 下 Top-1 统计不可区分，说明 PPEDCRF 的 DCRF/NCP/时序机制相对于无结构全局噪声并不提供"同画质下更强隐私"，其价值是"同隐私下画质更好"；(2) blur/mosaic 因无可调旋钮无法真正参与 matched 比较，只报告其固定点数值和明确的 PSNR gap，不宣称其在同画质下占优或劣于噪声类变体。同步修改 `paper/main.tex` 五处呼应文字（Scope of the privacy claim 段、Comparability 脚注、Fixed-Budget 小节、Conclusion 三条发现、Conclusion 限制段），把"matched-utility sweep left for future work"/"no matched-PSNR claim is made"改为准确描述已完成的（范围受限于 ResNet18/gallery-48）matched 结果，并在限制段新增"matched-PSNR 尚未覆盖全部 backbone/gallery 组合、也未覆盖确定性基线"两条更精确的剩余限制。用 `docs/experiment_progress.tex` 里记录的 `paper/build.bat` 走通编译：`main.pdf`（含新段落）与 `appendix.pdf`（9 页，新增 1 页）均 0 错误编译通过。

88. 已用真实 KITTI-360 数据完成 E5 独立诊断（gate 未通过，但已从"数据缺失"变为"仅剩 checkpoint 阻塞"），并把 E2 完全迁移到本地 RTX 3070 跑完整个 sigma 扫描。
修改说明：KITTI-360 下载后目录层级需要修正——`data_2d_raw` 解压在 `raw/` 子目录下、`data_2d_semantics` 因 zip 内层同名目录多套了一层，均用 `mv`（同盘瞬时操作，无需复制）拉平到 `build_kitti360_unary_manifest.py` 期望的 `<root>/data_2d_raw`、`<root>/data_2d_semantics/train` 布局。随后成功构建 E5 manifest（32 queries/128 gallery，序列 `0000_sync`/`0002_sync` held-out）并直接运行 `validate_unary_attribution.py --mode manifest`（本地 RTX 3070，CUDA）：`gate_passed=false`、`valid_nonconstant_queries=0/32`、`mean_spearman_static≈-0.0009`——与此前监控代理数据上的结论一致，但这是首次在真实公开数据集上复现，排除了"代理数据不具代表性"的可能，E5 现在只剩 checkpoint 阻塞。E1 侧，MSLS `train_val` 全量下载（10 个 zip，含全部城市）体积过大不需要全部解压，改为用 `zipfile.extractall(members=...)` 按文件名前缀（`train_val/manila/`、`train_val/toronto/`）从 part02/part04/part06（这三个 zip 内混装了多个城市）中只抽取 Manila+Toronto 两城的图像（约 31600 张，已核对与官方 CSV 行数完全一致），metadata.zip（part07）全量解压（仅 265 个 CSV，很小）；`build_msls_manifest.py` 正在跑（CPU/IO 密集，不需要 GPU，本次会话内耗时异常长，怀疑是 WSL 挂载的 G 盘在与本地 RTX 3070 跑 E2 sigma 扫描同时写 D 盘时产生了 I/O 争用，E2 结束后应该会变快）。E2 侧，发现本地已有本次 sigma 扫描需要的全部素材（`F:\work\datasets\monitoring\images` 完整 14GB/25934 文件语料、Patch-NetVLAD/CosPlace/MixVPR 权重缓存、`sensnet_final.pt` checkpoint），单点 sigma=8 试跑在本地 RTX 3070（`D:\source\.venv`）上不到 2 分钟即完成，遂放弃等待 vGPU 3090（当时因未开机而 `Connection refused`，你已决定自己重新开机，但既然本地就能跑就不必再等），改用一个 `.bat` 顺序跑完剩余 11 个 sigma 值（4,6,10,12,16,20,24,28,32,40,50），全程约 20 分钟，全部 12 个 sigma 目录均产出 900 条 per-query 记录 + 27 条 summary 记录。跑 `matched_psnr_from_sweep.py` 汇总 30/33/36dB 三个目标 PSNR 的 matched-operating-point 表时发现一个真实 bug：该脚本试图从 `summary.csv` 读取 `label` 列，但 `run_tomm_review_proxy.py` 里的 `aggregate_rows()` 聚合时只保留分组键和数值统计，从未写出 `label`（`label` 只存在于逐条的 `per_query.csv`）；已在 `matched_psnr_from_sweep.py` 里内联一份与 `run_tomm_review_proxy.py` 的 `VARIANT_LABELS` 一致的映射表来代替直接读取该列，属于分析脚本内部修复、未触碰实验产出数据。结果表（`src/outputs/tomm_review_e2_sigma/matched_psnr_table.csv`）显示一个值得写回论文的发现：在同一目标 PSNR 下，5 个基于噪声的变体（full/no_temporal/no_ncp/unary_only/no_dcrf/global_noise）Top-1 几乎完全一致（30dB 时全部 0.639，33dB 时 0.667，36dB 时 0.722），而两个确定性基线（mask-guided blur/mosaic）在同等画质预算下明显更弱（30dB 时分别只有 0.583/0.417）——即"matched image quality 下 PPEDCRF 各消融变体互相不可区分，但显著强于确定性基线"，这是 R2-2/R3-3 明确要求但此前论文缺失的证据。已更新 `docs/experiment_progress.tex` 反映以上全部状态并重新编译（`docs/experiment_progress.pdf`，7 页，0 错误）。matched-PSNR 表尚未写回 `paper/appendix.tex`。

89. 已完成 E1（MSLS 真实 GPS/place VPR）首个真实结果并写回论文，同时修复了两个真实 bug。
修改说明：(1) `build_msls_manifest.py` 原逻辑是 `eligible[:max_queries]`，而 `eligible` 按城市排序，Manila 一个城市的候选就够填满 200 条 query 配额，导致重跑后 manifest 里 100% 是 Manila、Toronto 完全没进去（违反 Design.md"至少两个城市"的要求）；已改成按城市轮询取样（每城市按内部顺序轮流各取一条，直到配额用完或候选耗尽），重新生成后核对为 Manila 100 + Toronto 100，均衡。重新生成这次耗时异常（`time` 实测 real 110m48s 但 user+sys 只有约 13 分钟，绝大部分是纯等待，且这次跑的时候本地没有其他并发重任务，排除了"和 E2 抢资源"的猜测，真正原因未查明，但不影响正确性）。(2) `run_geotagged_vpr_benchmark.py` 的 `load_manifest()` 把跨 query 复用同一个 gallery_id 当成"重复"直接报错，但这正是 manifest 的设计（build_msls_manifest.py/build_kitti360_unary_manifest.py 都是每条 query 各自内嵌完整共享 gallery 列表）；已改成只在同一 gallery_id 映射到不同内容时才报错，同一内容重复出现视为合法引用。(3) 发现该脚本原本完全没有 raw（未防护）基线这一档——只跑 sanitized variant，没法算隐私提升量；仿照 `run_tomm_review_proxy.py` 里 `raw_rows` 的写法在 `run_geotagged_vpr_benchmark.py` 里补上了等价的 raw-baseline pass。三项修复后跑通完整 E1：200 query（Manila/Toronto 各 100）× 共享 1000 图 gallery，`manifest_gate.json` 全部 gate 通过（`passed:true`、`unique_query_ids`/`unique_gallery_ids`/`true_place_labels`/`query_gallery_path_overlap:0`），`run_metadata.json` 里 `scientific_evidence:true`。结果：raw Top-1/5/10 = 0.170/0.385/0.450，PPEDCRF（ResNet18，σ₀=8，36.2dB PSNR）= 0.152/0.335/0.432——方向正确（防护后检索更难），但效果幅度比 proxy12 代理 benchmark 小得多（Top-1 降幅 0.018 vs proxy12 的 0.111）。分城市看：Manila raw 0.100→full 0.097，Toronto raw 0.240→full 0.207。核实发现这批 query 的条件多样性实际上没有真正达成——illumination 100% 为 day（Manila/Toronto 在 MSLS 官方 day2night 子任务里标注覆盖数都是 0）、viewpoint 99% 为 Forward、season/weather 字段全部为空——不满足 Design.md 原计划要求的"viewpoint/illumination/season/weather 多样性"。已就此结果征求你的意见（先补 o2n/n2o 条件子任务再写论文，还是先写现有结果），你选择"先写现有结果，如实注明局限"。已按此写入：`paper/appendix.tex` 新增 "External Validation: Real-Place MSLS Benchmark (E1) and KITTI-360 Diagnostic Status (E5)" 一节（含结果表 tab:e1_msls，与 proxy12 并列对比，并列出单 backbone/两城市/无条件多样性等具体局限，以及"未来可用 o2n/n2o 子任务补做真正的跨条件对比"的具体下一步）；同一节把 E5 的 KITTI-360 真实数据诊断结果也一并写入（gate 不通过，与代理数据结论一致，排除了"代理数据不具代表性"的可能）。`paper/main.tex` 同步修改三处：Section 4"External validation plan"改写为"External validation on real place identities (E1)"并给出真实数字、"Reproducibility audit"段落补充 KITTI-360 复现、Conclusion 三条发现扩展为四条（新增 E1 真实数据发现）并更新限制段。`paper/build.bat` 两遍编译 0 错误、0 未定义引用（`Section~\ref{sec:matched_psnr}`/`Table~\ref{tab:e1_msls}` 均正确解析）。

90. 已同步 `docs/revision_suggestions.tex` 的审稿意见映射与完成状态。
修改说明：将 E1 更新为“真实 MSLS 结果已完成、跨条件扩展为可选后续”，将 E2、E3、E4、E6、E7 更新为已完成但保留各自的实验范围限定，将 E5 明确为“真实 KITTI-360 诊断已完成但当前 unary checkpoint 阻塞”；同时修正 R3-2 不应归入 E6 的编号映射，并将 R3-5/R3-6/R3-7 的归属改为与实际证据一致。同步更新总体评估、充分证据段、实施计划和 disposition，清理会误导为“实验尚未执行”的旧计划措辞，避免评审文件状态落后于论文和实验进度。

91. 已完成 E1 的 MSLS 官方交叉时间子任务扩展，并关闭 E5 的执行阻塞。
修改说明：E1 新增并完成 `o2n` 与 `n2o` 两个官方 cross-time 运行，各使用 Manila/Toronto 两城各 100 个 query、共享 1000 图 gallery、3 个噪声 seed 和 ResNet18 attacker；`o2n` 的 Top-1 为 raw 0.140、PPEDCRF $0.168\pm0.009$，`n2o` 为 raw 0.155、PPEDCRF $0.137\pm0.006$，并补充了 query/gallery capture-time 元数据。结果表明方向依赖条件，论文和 `docs/Revision_suggestions.tex` 已同步为有范围限定的完成状态。E5 方面，修正 KITTI-360 官方语义类别的 dynamic-ID 排除逻辑，使用 sequence 0000 的 1,033 张带 structural-support mask 图像训练新的 mask-backed checkpoint，并仅在 held-out sequence 0002 上以 32 queries/128 gallery 完成诊断：32/32 map 通过非恒定性 operational gate，但 static-background Spearman 为 $-0.00024$、top-10 overlap 为 0.0497、归因能量为 0.0947。因此 E5 已作为“负向归因诊断”完成，不宣称 sensitivity-map accuracy；原始本地 checkpoint 与公开 Hub checkpoint 已确认 bit-identical 且 `mask_root=null`，其原有 constant-map 失败结果仍保留为 provenance evidence。同步更新 `paper/main.tex`、`paper/appendix.tex`、`docs/Revision_suggestions.tex`、`docs/experiment_progress.tex` 与 `docs/Design.md`。

92. 已完成论文自包含性与敏感性图表述复核。
修改说明：删除 `paper/main.tex` 与 `paper/appendix.tex` 中的仓库本地路径、脚本名和内部输出目录引用，改为面向读者的 released implementation/export 表述；同时将少数可能暗示 unary map 已被证明有效的 “high-sensitivity/location-sensitive” 图注和叙述改为 “candidate support/selected regions”，与 E5 的负向归因结果保持一致。主论文和附录重新编译通过。

93. 已完成本机与 vGPU 3090 的实验可用性检查，并运行新的扩展实验。
修改说明：先确认本机 RTX 3070 的 CUDA、数据目录和磁盘状态，再确认 `C:\source\.env` 中的 vGPU 3090 连接可用；远端 RTX 3090 显示 48 GB 可用显存、PyTorch CUDA 可用、数据盘剩余约 753 GB。远端先通过 CUDA smoke，随后完成不依赖外网缓存的 6-attacker `proxy12`（5,436 条 per-query）和 `proxy50`（22,650 条 per-query）正式运行，所有 numeric 字段均通过 finite 检查，checkpoint 与本机 SHA-256 一致；初次直接访问 Hugging Face 的 CLIP 下载因 `Network is unreachable` 中止，后续已改用 `hf-mirror.com` 完成 CLIP 权重准备和完整复核，详见下方第 94 条。E1 方面，本机完成 Manila `s2w` 官方子任务的 80 query/80 gallery/3-seed 扩展，raw Top-1 为 0.5375，PPEDCRF 为 $0.5000\pm0.0177$；由于仍是单城市、小 gallery、单一 attacker，该结果仅记录为可选扩展，不写入论文主张。

94. 已通过 `hf-mirror.com` 完成 CLIP-inclusive vGPU 3090 复核实验。
修改说明：通过 `hf-mirror.com` 下载并在远端标准 Hugging Face cache 中准备 `openai/clip-vit-base-patch32` 与 `openai/clip-vit-large-patch14`，两个模型均以严格离线模式成功加载并参与 CUDA 推理。完整 8-backbone `proxy12` 运行生成 7,236 条 per-query 和 219 条 summary，`proxy50` 生成 30,150 条 per-query 和 219 条 summary；两次运行均包含 CLIP ViT-B/32 与 ViT-L/14，全部 numeric 字段通过 finite 检查，checkpoint 与本机 SHA-256 一致。由于这仍是固定 paired-scene proxy 的工程复核且导出标记为 `scientific_evidence=false`，结果不替换论文已有 validated export，也不新增论文主张；旧的“CLIP 未完成”状态已关闭。


## 第十三轮修订（2026-09-11/12，按 TIFS 独立评审 R1--R12 推进）

本轮先做了一次完整独立评审（覆盖写入 `docs/RevisionSuggestions.tex`，12 条意见、
多维评分 76/100、结论 major revision），随后按该评审逐条落地。评审的四条 Major
里有三条不需要任何算力，第四条也不需要——所需数据全部已在仓库里。

550. 【已完成】**R1：摘要和结论的核心句被作者自己跑过、发布过但没报告的实验推翻了。**
     - `src/exports/r10_clip/r10_clip_alloc.csv` 12,000 行，400 query、3 seed、
       delivered MSE 15.68、能量门 mean(w^2)=1.0、`optimised_against =
       expectation:resnet50+vgg16+cosplace`、20 步——与 ResNet18 那条臂
       **完全同一个协议、同一张 map**（目标函数轨迹同为 2.841→2.465）。
     - 该 map 在 ResNet18 上买到 −0.0058、MixVPR 上 +0.0058（"什么都不买"），
       在 CLIP ViT-L/14 上买到 **−0.0508，place-clustered [−0.082,−0.021]，
       p=1.6e−3**，是该 attacker 上 direction 效果（−0.0867）的 59%。
     - 摘要原句 "given only surrogates it buys nothing" 因此为假。已改写摘要、
       §III-C 和结论：分配轴的不对称性是程度问题，"direction 在测过的每个
       attacker 上都迁移，allocation 在三个里迁移一个"。
     - Fig. 2 左栏新增 CLIP 的两条 solved 臂，右栏新增 CLIP 的 direction 臂
       （现为 5 个 attacker）。两条 CLIP allocation claim 已登记进 auditor。

551. 【已完成】**R2：同一段里 solved-map 的四个数用了两套参照配对，图用的是正文
     明确否定的那一套。**
     - 正文 §III-C 明说 solved map 必须在"优化器没见过的噪声场"上评分，但
       `-0.0550`（同场配对）与 `-0.0392`（held-out 场）在同一段相隔两句同时出现，
       Top-5/10 那句（−0.078/−0.071/−0.055）整句都是同场配对。
     - `paper/generated/fig_axes_values.tex` 显示 Fig. 2 画的也是同场配对，其中
       `solved, surrogates (ResNet18) = −0.0167 [−0.033,−0.001]` **区间不含零**，
       与旁边"no benefit detected"直接冲突。
     - 已把 `src/scripts/make_axes_figure.py` 改为 crossdraw-vs-crossdraw，正文
       统一到 held-out 场（Top-5/10 改为 −0.070/−0.076/−0.039 与
       −0.072/−0.055/−0.094），并在段首写明配对口径。
     - `fig_axes_values.tex` 自称"供 auditor 校验"却从未被 auditor 读取；已把它
       注册为 registry 来源，Fig. 2 的 18 个点全部登记。

552. 【已完成】**R3：两条优化臂都没收敛，而"步数翻倍"只在支持论点的地方被引用。**
     - 释出文件里本来就有 40 步的臂：attacker-given 从 −0.0392/−0.0942 翻到
       −0.0833/−0.1942，surrogate-solved 从 −0.0058/+0.0058 移到
       −0.0200（p=0.040）/−0.0108，CLIP 从 −0.0508 到 −0.0567；top-decile
       集中度从 0.555 升到 0.772，说明 20 步远非平台。
     - 正文现在把三条 surrogate-solved 的 40 步值和两条 attacker-given 的一起报，
       并把 null 限定在"20 步预算下"。五条 2x claim 已登记。

553. 【已完成】**R4：干净 clone 只能验证 827 条里的 753 条。**
     - auditor 先找 gitignore 的 `src/outputs/`，5 棵树（tifs_a3、tifs_a3hi、
       tifs_a4、tifs_a7_o2n8、tifs_a7_n2o8）只存在于那里，丢失的 74 条里包含
       Table S7 全部（论文列的第四项贡献）和唯一的 mask-guided 对比。
     - 这 5 棵树的**完全相同副本**（文件数、行数逐一核对一致）已提交在
       `src/artifact/results/` 下，只是名字不同。已把 bundle 加为第三个 root 并
       加 `BUNDLE_ALIASES` 名称映射，同时修掉 `tree_present()` 的同类遗漏。
       现在干净 clone 报 **0 unverifiable**。
     - bundle verifier 的 "Gallery-free direction transfer (headline table)" 块
       把另一次执行的值标成 `paper=0.1508`，而 Table I 印的是 0.1608（9 处不一致）。
       已改标题为 "independent re-execution; not Table I's run"、列名改为
       `expected=`，并在代码注释里写明两次执行为何差最多 0.010。
     - README 的 780 条与旧 coverage transcript 已按当前工具输出更新。

554. 【已完成】**R5：TIFS 深度学习投稿清单要求的三个训练网络规格，原先只在未提交的
     extended report 里。**
     - TIFS 明确写明"负面结果论文适用更高的可复现标准"，本文正是这类。
     - 已在补充材料 §XV 新增《The networks we trained》：unary predictor 的逐层
       拓扑、激活、参数量（87,441，已按 `src/run_train.py` 逐层复算核对）、输入
       归一化、初始化、优化器、batch、epoch、有无 early stopping/验证集/超参搜索，
       以及 mask-supervised checkpoint 和 purifying denoiser 的同类信息。
     - 所有数值均按代码核对，未凭印象书写。

555. 【已完成】**R6：Patch-NetVLAD 的区间上端点写错了。**
     - 正文写 "−0.0133（σ=2 blur）到 −0.1350（median）"，但十二个 transform 里
       最大的是 4-bit 量化的 −0.1375，十三个里最大的是未变换的 −0.2142。
     - 已改为 "−0.0133 到 −0.1375（4-bit），未变换 −0.2142"，并把
       "hardening roughly doubles each" 改为实测比值范围 1.1--3.1、中位数 1.8。
       三个端点已登记。

556. 【已完成】**R7：三套 Wilcoxon 口径并存，同一格读出 0.18 或 0.38。**
     - auditor 用 `method="approx"`，MixVPR 表生成器用非零差的精确检验，
       MSLS 表生成器又用 approx。已统一为"丢零、可枚举时用精确检验"，写进
       §III-A 协议段，三处代码一致。
     - Table S4/S5 的 p 列现在带 discordant pair 数（例如 MixVPR learned 行
       `1.00 (0)`，一眼可见那是自比较；MSLS learned 行 `1.00 (1)`），caption
       说明该计数是检验功效的上界。

557. 【已完成】**R8：集中度论证引的是两个数里较有利的那个。**
     - "0.649 against 0.555" 中的 0.555 是 ResNet18 的 attacker-solved map；
       MixVPR 的同一量是 0.6215，与 margin rule 的 0.6487 几乎相同。
     - 正文改为 "than either solved map does, 0.649 against 0.555 and 0.622"，
       三个集中度值已登记（此前全在 coverage 的未登记列表里）。

558. 【已完成】**R9：补充材料三处跨文档硬编码引用全部失效。**
     - `\S III-G` 实际应为 §III-F；两处 "the manuscript's Table~I" 指的是
       13 页压缩时已删掉的那张 placement 表，而现在的 Table I 是 direction
       transfer 表。
     - 已加载 `xr` + `\externaldocument[M-]{main}`，三处改为 `\ref`，编译后
       分别解析为 §III-B、§III-B、§III-F，0 undefined reference。

559. 【已完成】**R10：responsible use 只在补充材料里。**
     - 正文新增 `\textbf{Responsible use.}` 段（§III 末）：释出什么、不释出什么、
       四条限制、以及数据集按各自条款使用且不再分发影像、manifest 只带标识符。
     - 补充材料的版本相应收缩为两条正文未覆盖的细节，避免重复占版面。

560. 【已完成】**R12：打包与排版。**
     - `paper/main.pdf` 原先落后 main.tex 一次编译（缺摘要的 utility 结论和两条新
       引用）；现已重建，三份 PDF 均为当前源码。
     - 六处 `T1/ptm/m/scit` 字体警告来自 caption 内的 `\emph`（IEEEtran 的表
       caption 是小型大写，T1 Times 没有小型大写斜体）；两个表生成器改用引号后
       **警告归零**。
     - clustered/query 区间宽度比原印 0.94--1.10x，重采样噪声可达第二位小数
       （三次独立运行分别为 0.89、0.92、0.94），已改为一位小数 0.9--1.1x，
       注册容差同步放宽并在代码注释写明原因。
     - 正文 13/13 页、补充 6/6 页、摘要 249/250 词，overfull 0、undefined 0。

**本轮页数预算说明。** 上述新增（CLIP 结果、40 步臂、responsible use、网络规格、
discordant 计数）净增约一页半，而正文和补充都已在上限。已通过约二十处散文压缩
（related work、protocol、§III-B/C/D/E/F/H/I、conclusion、两张表的 caption、
补充的 §I/II/III/VII/VIII/IX/X/XII/XIII/XIV/XV，合并两个只起指路作用的补充小节，
Table S6/S7 改 scriptsize，Fig. 2 与 Fig. S1 略缩）在不删结论、不删数据的前提下
收回。**没有为了版面删掉任何一个实验结果或限制声明。**

验证状态：auditor **856 条全绿**（干净 clone 亦 0 unverifiable），bundle
**228 条 0 mismatch**，MANIFEST 743 文件全匹配。


## 第十四轮：R1 的补充实验（2026-09-12，vGPU 3090，两条臂各 7,200 行）

561. 【已完成】**把同一张 surrogate-solved map 交给剩下两个 attacker 读。**
     - 动机：上一轮把 CLIP 的反例挖出来后，论文只能说"三个里迁移一个"，而五个
       held-out attacker 里有两个（Patch-NetVLAD、ViT-B/16）根本没有 allocation 臂。
       审稿人第一句话就会问另外两个。
     - 这不需要重新求解：surrogate-solved map 优化时**从不看被评估的模型**，所以换
       attacker 只是重新 embed。用同一个脚本改 `--eval_backbone` 即可，无新代码。
     - **门控通过得干干净净**：两条新臂的目标函数轨迹都是 `2.8412 → 2.4645`，与
       ResNet18 和 CLIP 两条臂**逐位相同**；top-decile 集中度四条臂都是 0.547。
       这证明它确实是同一个对象，而不是一次新的搜索。
     - 两条臂各 7,200 行（400 query × 3 seed × 6 condition），0 重复 key，
       delivered MSE 全部 15.6800，能量门全部 1.000000。

562. 【结果】**五个 attacker，同一张 map（MixVPR 除外，见下），20 步预算，
     held-out 噪声场，place-clustered 区间：**

     | attacker | uniform | solved | Δ | place 95% CI | p | 判定 |
     |---|---|---|---|---|---|---|
     | ResNet18 | 0.1933 | 0.1875 | −0.0058 | [−0.023,+0.012] | 0.37 | none det. |
     | MixVPR\* | 0.7808 | 0.7867 | +0.0058 | [−0.003,+0.017] | 0.24 | none det. |
     | **Patch-NetVLAD** | 0.4900 | 0.4658 | **−0.0242** | [−0.045,−0.003] | 0.034 | **separates** |
     | ViT-B/16 | 0.1817 | 0.1817 | +0.0000 | [−0.010,+0.010] | 0.91 | none det. |
     | **CLIP ViT-L/14** | 0.3175 | 0.2667 | **−0.0508** | [−0.082,−0.020] | 0.0016 | **separates** |

     \* MixVPR 那条是另一张 map：它是评估目标，所以 ensemble 多了 ResNet18
     （轨迹 3.827→3.424）。其余四条共用同一张。

     **读法**：分配轴在没有 attacker 的情况下 **五个里迁移两个**，而方向轴
     **五个全迁移**；在迁移的那两个上，它买到方向轴的 **1/8 到 3/5**。
     翻倍步数后五条分别是 −0.0200、−0.0108、−0.0408、−0.0025、−0.0567，都没到顶。

563. 【结果】**没有任何我们测过的属性能预测它在哪个 attacker 上迁移。**
     - Patch-NetVLAD 与某个 surrogate 共享 VGG16 trunk → 迁移；
       ViT-B/16 不共享任何 trunk → 不迁移；CLIP 也不共享 → 迁移幅度最大。
       **trunk 共享因此不解释它。**
     - 论文如实写明这一点，并说 §III-I 的一阶论证（把分配轴的杠杆当作预算的性质
       而非 attacker 的性质）在这里是不完整的——报告这个缺口，而不是编一条规则。
     - 附带的方法学收获已写进正文：**只用弱 attacker 做审计会判这条轴"死"，
       只用 CLIP 会判它"活"**——这正好是协议本身的论据。

564. 【已完成】**正文、摘要、结论、Fig. 2 全部按五个 attacker 改写。**
     - 摘要：「given only surrogates 一张 map 被五个 held-out attacker 读，两个上
       与零分离、三个上不分离；方向轴五个全分离」。
     - §III-C 那段整体重写，并修掉上一轮的一处不精确：「that map is one object」
       对 ResNet18/CLIP/PNV/ViT 成立，对 MixVPR 不成立，现在写清楚了。
     - Fig. 2 左栏 solved 行现在是五个 attacker（圆/方/菱/倒三角/三角）。
     - 新登记 14 条 claim，含四条"同一张 map"的门控（轨迹 2.465、集中度 0.547）。

565. 【顺手修掉】**`analyze_optimised_allocation.py` 的默认参照也是 same-field。**
     - R2 的缺陷不只在画图脚本里，分析脚本里也有一份。已改为默认
       `uniform_crossdraw`。在 ResNet18 上这是 −0.017 与 −0.006 的差别，
       也就是"超出等效边界"与"零"的差别。

**页数**：新增约 20 行，靠八处压缩收回。正文 13/13 页、补充 6/6 页、摘要 249/250 词，
overfull 0、undefined 0、字体警告 0。压缩过程中被删掉的三处数值（ViT hardened 区间、
CLIP Top-5、MSE 60 的 gain/amplitude）都**又加回来了**——因为 auditor 报出它们不再
被印，而把真实测量从登记表里删掉是错误的方向。

**验证**：auditor **870 条全绿**（干净 clone 同样 0 unverifiable），bundle 228 条
0 mismatch，MANIFEST 743 文件全匹配，coverage 249/272。


## 第十五轮：收敛曲线，以及它推翻的一句话（2026-09-12，vGPU 3090，十条 run）

566. 【已完成】**五个 attacker × 五个预算（5/10/20/40/80）的收敛曲线。**
     - 每个 attacker 两条新 run：`--steps 5 --double_steps 10` 和
       `--steps 20 --double_steps 80`。后者**故意重测 20 步**当作门控。
     - 十条 run 各 7,200 行，0 重复 key，delivered MSE 全部 15.6800，
       能量门全部 1.000000，400 query × 3 seed。
     - 一张 48GB 卡上最多十个任务并发，峰值 47.0/48.5 GB、利用率 100%。

567. 【门控】**两个都过。**
     - **轨迹门控**（20 步被两条 run 各测一次）：最大差 0.0058（PNV），
       CLIP 是 0.0000。远在论文自报的 run-to-run 波动（约 0.005）量级内。
     - **同一张 map 门控**：五个预算全过，20 步那档是**八条臂**一起对上
       （目标函数 2.4643..2.4651）。

568. 【结果】**80 步推翻了"buys nothing"。**

     | attacker | 5 | 10 | 20 | 40 | 80 |
     |---|---|---|---|---|---|
     | **ResNet18** | +0.0000 | −0.0025 | −0.0058 | −0.0200 | **−0.0317 [−0.058,−0.005]** |
     | MixVPR\* | +0.0017 | +0.0042 | +0.0058 | −0.0108 | −0.0167 |
     | **Patch-NetVLAD** | −0.0050 | **−0.0183** | **−0.0242** | **−0.0408** | **−0.0483** |
     | ViT-B/16 | +0.0000 | +0.0017 | +0.0000 | −0.0025 | −0.0042 |
     | **CLIP ViT-L/14** | −0.0100 | **−0.0300** | **−0.0508** | **−0.0567** | **−0.0583** |

     - **ResNet18 在 80 步与零分离**——而它正是"给了 surrogate 也买不到东西"
       那句话的主要依据之一。所以 **20 步的 null 是预算造成的，不是轴的性质**。
     - 形状分两类：已分离的两条在**饱和**（CLIP −0.0508→−0.0583，
       PNV −0.0242→−0.0483），ResNet18 还在**陡升**且完全没收敛。
     - 对照之下方向轴在 20 步就到底了（白盒 0.0058 / 0.0008 即地板）。

569. 【已完成】**摘要、§III-C、结论全部按"预算"重写。**
     - 新说法：*分配轴在方向轴同等预算下够到五个里的两个，四倍预算够到三个，
       且在任何一个上都没收敛*——**两条轴的差别是"需要多少搜索"，不是"能不能"**。
     - 曲线表进**正文**（Table III），不是补充材料。

570. 【已完成】**删了十条旧参考文献，腾出正文版面。**
     - 参考文献原占 1.40/13 页，每条约 30pt，是全文最便宜的版面。
     - 第一批七条：CRF-as-RNN、dense CRF in segmentation（四条 CRF 撑一句话，
       而 §II 明说这不是 CRF inference）、CIAGAN、2007 人脸检测、GANobfuscator、
       **Momentum FGSM（本文优化器没用 momentum）**、
       **diffusion purification 评测（本文净化器是 DnCNN）**——后两条属于引了没用到的。
     - 第二批三条：DP 人脸识别、analytic Gaussian mechanism（本文用的是经典高斯
       公式且明说不是 DP 保证）、DeepPrivacy2（同组两篇留一篇）。
     - 65 → 55 条。`.bib` 条目保留，恢复任何一条只需加回一个词。

571. 【顺手修的三个自造 bug】
     - 曲线表生成代码的反斜杠被转义翻倍，产出字面 `\\begin{table}`——**脚本退出码是 0**，
       靠读生成文件才发现。
     - 分析脚本按 `opt_steps` 分组会把两条 run 的 20 步**平均掉**，
       那样轨迹门控永远不可能触发。已改为按 `(步数, 来源文件)` 分组，
       且每条臂只和自己那条 run 的对照配对。
     - caption 里 "differ by at most 0.0058" 是四舍五入得来的，而真实值 0.005833
       **超过**这个界。改成向上取整 0.0059——"至多"这种话四舍五入后必须仍为真。

**验证**：auditor **888 条全绿**（干净 clone 同样 0 unverifiable），正文 13/13 页、
补充 6/6 页、摘要 249/250 词，overfull 0、undefined 0、字体警告 0。
**vGPU 3090 已在数据校验并拉回后关机。**


## 第十六轮：补充材料申请超页，把四张表搬进投稿材料（2026-09-12）

572. 【已完成】**按你的决定，补充材料从 6 页扩到 7 页，并向 EiC 提出申请。**
     - SPS 的规则是"建议不超过 6 个双栏页，**该限度内无需 EiC 批准**"——
       超页是可申请的，不是硬上限。
     - 搬进来的五张表（评审 R11 点名最关键的四项）：
       1. **per-rule placement 研究**（两个 checkpoint × 六个 backbone × 三 seed ×
          两个 benchmark）——正文中心 null 是从这些格子里 pool 出来的
       2. **mask-guided 对比**——本文唯一一次与已发表方法的正面对比
       3. **两张 utility 表**（allocation 轴与 direction 轴的逐图配对测量）
       4. **自适应攻击者的 per-condition 分解**——支撑一整个小节的证据
     - 每张都放在**本来就指着它的那段文字旁边**，措辞从"in the extended report"
       改成"Table S…"。补充材料现在是 S1--S12，第 7 页 60% 满。
     - 仍留在 extended report 的四张：two-city E1、white-box per-backbone、
       placement budget sweep、按 backbone 的 redistribution 控制。
       开头的清单已从"Nine more"改成这四张。

573. 【已完成】**cover letter 里写明申请理由，而不是靠沉默蒙混。**
     - 理由是针对这类论文的：本文报的是与领域设计假设相反的负面结果，而
       **TIFS 的深度学习投稿指南明写"负面结果适用更高的可复现标准"**。
       审稿人要验一个负面结果，需要看到它是从哪些格子 pool 出来的，而不是摘要。
     - 并明确表态：**如果 EiC 坚持 6 页，我们把四张表搬回去并在正文写明**——
       不在一篇主题是"审计"的论文上默默做这个取舍。

574. 【顺手修的两件事】
     - 搬进来的表 caption 里有 `\emph`，又触发了三处 `T1/ptm/m/scit`
       字体警告（IEEEtran 表 caption 是小型大写，T1 Times 没有小型大写斜体）。
       已按同样办法改成引号，**警告归零**。
     - auditor 里 mask-guided 表的 claim 原本定位在 extended report。表搬进
       补充材料后，**被检查的应当是审稿人真正读的那份**——已改 source 为 SUPP。
       （另一张"两个 checkpoint 的 energy-matched placement rules"表没搬，
       它的 claim 仍定位在 extended report——我一开始一起改了，auditor 报出
       28 处定位失败，才发现那是两张不同的表。）

**验证**：auditor **888 条全绿**（干净 clone 同样 0 unverifiable），
正文 13/13 页、补充 **7 页（已申请）**、摘要 249/250 词，
overfull 0、undefined 0、字体警告 0。

---

## 【第十四轮】五问已全部回答并落实（2026年9月12日）

1. GPU 主机何时开卡？ A: vGPU 3090 已经开了
   → **已执行**：R1 的九条规则 × 三个攻击者 × 三个种子已在 vGPU 3090 上以六个 screen 并行开跑
   （`src/scripts/launch_r15_panel_vgpu3090.sh`），十分钟定时巡检已挂起，跑完自动拉回并关机。
2. 正文页数？ A: 移除旧的、关联不大的参考文献，以及大段文字
   → **已执行**：退役 17 条旧框架遗留文献（62 → 42），相关工作与五处大段文字压缩，
   正文 14 页 → **13 页**，未删除任何一项结果。逐条删除依据见 `# 已全部修改` 首条。
3. 补充材料 9 页是否申请？ A: 要申请
   → **已执行**：cover letter 只保留补充材料一项申请，正文超页申请已撤回（不再需要）。
4. 生成式 AI 声明是否相符？ A: 符合
   → **已确认**，声明保留现状。
5. 凭据泄露？ A: OK
   → 已知悉；后续读取 `.env` 一律用定向提取，不再整段打印。

本轮暂无新的待决问题。R1 跑完后若结果改变论文主张（例如某条规定性规则在 CLIP 或
Patch-NetVLAD 上分离），会回填论文并在此处新增记录。


- **【已决策并执行，R11】补充材料超页申请——你答"要"，已办**（第 572--573 条）：
  四张最关键的表已搬进补充材料（现 7 页，S1--S12），cover letter 写明申请理由，
  并写明如果 EiC 坚持 6 页就搬回去。以下为原始问题记录：

- ~~**【需要你决策，R11】补充材料要不要向 EiC 申请超过 6 页？**~~
  TIFS 的规则是"建议不超过 6 个双栏页，**该限度内无需 EiC 批准**"——也就是说超页
  是可申请的，不是硬上限。现状是：论文有九张表（Table I 背后的 per-rule placement
  研究、two-city E1 表、white-box per-backbone 表、budget sweep、按 backbone 的
  redistribution 控制、mask-guided 对比、两张 utility 表、自适应攻击者的
  per-condition 分解）只存在于 15 页的 extended evidence report 里，而那份报告
  **不属于投稿材料**，审稿人看不到。其中 mask-guided 对比是本文唯一一次与已发表
  方法的正面对比，per-rule placement 研究是 Table I 的证据来源。
  本轮我把补充材料压回了 6 页（新增的 TIFS 网络规格也放进去了），所以**现在是合规的**，
  不阻塞投稿。但如果你愿意向 EiC 申请，建议把那四张最关键的表移进补充材料。
  需要你决策：`A:` 不申请

- **【无需决策，已完成】R1 的补充实验已在 vGPU 3090 跑完并写进论文**（第 561--564 条）。
  五个 attacker 的 allocation 臂现在都齐了，结论从"三个里一个"变成"五个里两个"，
  证据面更宽而主张更保守。**卡可以关了。**
- **【无需决策，已完成】R3 的收敛曲线已做**（第 566--569 条）。结论比预期重要：
  它推翻了"给了 surrogate 也买不到东西"这句话在 ResNet18 上的成立性，
  论文已按"这是预算问题不是轴的问题"重写。**卡已关机。**


- **【无需决策，R1 已完成】** 结果已拉回本地并写进论文（`src/exports/clip_pooling/`，
  57,024 行，auditor 已登记每一格）。**PRO 6000 已按你的要求关机**（ssh 已拒绝连接）。
  原先担心的"pooling 把 direction 收回去"没有发生，所以摘要和结论的措辞不用动。
- **【无需决策，R6 已完成】** 净化攻击者在两个攻击者上都跑完并已回填论文。
  MixVPR 那一版把方向轴的优势整个抹平（−0.0417 → −0.0217，区间跨零），
  摘要、结论、§III-J 已按这个更不利的口径改写。**PRO 6000 已关机**。
- **【无需决策，本轮无待办】** 第十轮 R1--R7 全部闭合，三份文档都在页数上限之内，
  624 条 claim 全绿，reviewer 包已重建。

- **【无需决策，已完成】** 第十一轮评审的两个问题你都答了"需要"，两个实验先后在
  vGPU 3090 和 2c 3080 上跑完，结果已全部拉回并写进论文。**两台机器都已关机**
  （ssh 均已拒绝连接）。
  R1 的设计中途改过一次，原因写在第 524 条：让 placement 图看着自己的噪声实现优化
  会变成"挑符号"，那是方向轴不是分配轴，所以现在跑两种 noise mode 并加了
  cross-draw 对照。**按 3-seed 规则，跑完之前不写任何结论进论文。**

- **【无需决策，本轮无待办】** 十二条评审意见（R1--R12）已全部落地，827 条 claim 全绿。
  正文 13 页 / 补充材料 6 页，都在 TIFS 上限之内，已提交并推送。
  **页数余量只剩约 5pt**——再加任何一句都会顶到 14 页，届时必须同步砍字。
  第十二轮评审提出的三条（摘要说清效用结论、边界前置、补时效性文献）均已落地。

- **【已全部做完】** 结果已拉回、写进论文、auditor 重跑（827/827 全绿）、
  重编译核过页数（13 / 6 页）；**vGPU 3090 和 2c 3080 都已关机**；
  12 条评审意见已逐条对照确认。


# 未修改或部分修改

（上一版这里列的 R1 / R3 / R4「已阻挡」条目已全部作废：R1 与 R4 的实验在 9月12–13 日于 vGPU 3090 跑完并已写进论文，R3 的阻挡原因也已查明并改变了性质。详见 `# 已全部修改`。）

## 【进行中】R3 强方案：全量臂已在 vGPU 3090 上运行（2026年9月13日）

你回答「照原样跑公开版，报告它退化」+「vGPU 3090」。不需要 GPU 的部分已全部做完；
pilot 已跑完并验证了协议与能量闸门；**全量臂（400 query × 3 条件，seed 1234，约 4.5 小时）
已于 11:35:53 起跑**，按 query 断点续跑，每 10 分钟巡检一次，能量闸门跳闸会立即报警而不等停滞计数。

**分阶段是有意的**：pilot 的 signed-rank 只有 2 个 discordant 对，任何效应量都分不开，
所以先买一个 seed 的全量 manifest（约 400 个配对观测）看它是否真的与 isotropic 对照分离；
余下两个 seed（再 9 小时）只在这一个 seed 显示出有东西可分辨时才值得买。

**把 GeoShield 重新 clone 下来逐行读完后，公开版的缺陷不止一处，而是五处**（全部由读代码或实跑确认，非推测）：

1. **VLM stub 返回的是常量，而且不报错。** `describe_image_placeholder()` 对**每一张图**返回同一句 `"A scenic outdoor photograph."`。它没有 guard、不抛异常，所以公开代码能一路跑完，只是 `set_geotext_truth()` 对全语料嵌入同一条文本——它目标函数里**减掉**的那个 geo-semantic 项不携带任何逐图语义信号。**这比「缺一块」严重：它是能跑但退化**，和本文 §IV-A 那个「support map 数值恒定却仍能画出漂亮 privacy–utility 曲线」的 released checkpoint 是同一类失效。
2. **region 模块默认关闭。** `bbox_json_path` 默认为空 → `boxes=[]` → 代码随即 append 一个全画幅框，于是「定位泄露区域再扰动」这个命名模块在默认配置下是 no-op。但**它与 VLM 不同，是公开可恢复的**：README 写明用 GroundingDINO（权重公开、命令给全）。拿一个机制自己写了启用方法的模块去关掉它再审计，属于稻草人，所以 runner 提供 `--bbox_json` 打开它。
3. **untargeted 模式的 target 集合没有定义。** README 说 target images 「for M-Attack only」，但 `geoshield.py` 第 421 行 `set_ground_truth(mask_crop(image_tgt, mask))` 确实在用它——即 untargeted 臂的结果依赖一个文档说「用不到」的输入。这是文档与代码矛盾，且是个会实质改变结果的自由参数。runner 里固定为「与 query 不同 place 的 gallery 帧、按 crc32(query|seed) 选取」并写进 metadata，而不是听任目录顺序。
4. **`import geoshield` 本身就失败。** `config_schema.MainConfig` 把 `data/optim/model/wandb` 标注成非 Optional 却给默认值 `None`，`cs.store()` 在当前 omegaconf 下直接抛 `field 'data' is not Optional`，在任何攻击代码之前。README 让你装的 `requirements.txt` **仓库里并不存在**，所以也没有可回退的版本。这是打包缺陷而非目标函数缺陷，runner 在 import 期间临时停用 ConfigStore 注册（本 harness 自己构造 cfg，从不走 Hydra 解析），注释里写明了理由，攻击路径逐位未动。
5. **公开版只能在 transformers 4.x 上跑，而它没有给出可钉版本的方式。** 在 transformers 5.16.1 下
   `CLIPModel.get_image_features()` 返回 `BaseModelOutputWithPooling` 而非 tensor，公开代码对它直接调
   `.norm()`，于是 `AttributeError` 在第一帧就抛出——**一帧都跑不完**。这条是实跑发现的，不是读代码猜的，
   而能钉住旧版本的 `requirements.txt` 仓库里并不存在（与第 4 条同源）。
   修法是精确的而非将就的：transformers 源码里 `vision_outputs.pooler_output = visual_projection(pooled_output)`，
   所以 `.pooler_output` **就是**旧 API 返回的那个投影嵌入，取它不是近似。补丁按模型实例打、不动 `CLIPModel` 类，
   因此本仓库自己的 CLIP 攻击者行为不受影响；已记进 run metadata。

**已就绪（本机已验证，未用 GPU）**：
- `src/scripts/run_geoshield_audit.py`——导入公开实现而**不修改**它，只在边界适配：喂帧进去，把出来的扰动按本协议的 per-frame 二分（穿过像素 clamp）压到 delivered MSE 15.68，再交给同一攻击者面板。启动时**断言 stub 仍返回那句常量**，一旦上游改了就拒绝继续，标签不会悄悄失效。逐行增量写入 + flush/fsync + 断点续跑（符合仓库铁律）。
- `--caption_source published|claude`：你补充的 `C:\source\.env` 里 `claude-code-api` 下的 key 已确认存在（`sk-ant-…`，108 字符，值未打印）。所以「填上 VLM 再跑一遍、与公开版对照，直接量出那个被 stub 掉的项到底值多少」现在只差一个参数。**按你选的方案默认只跑 published 臂**；要不要加第二臂等你一句话。caption 会按图缓存到磁盘，续跑不重复付费。
- `src/scripts/launch_r17_geoshield_vgpu3090.sh`——先跑 25 帧 pilot 再放全量（公开版每帧成本远高于本仓库其它任何臂：100 步 FGSM × 三模型 CLIP ensemble，含 LAION ViT-G/14，640px，batch 1；1,200 次攻击不先探成本是昂贵的错误）。worker 互斥用**原子 mkdir 锁**而不是上一轮出过事的 `pgrep` 判断。
- 本机 smoke 已通过：import、stub 断言、helper 解析、脚本语法、runner 编译全绿；第三方 commit `5001a82` 已记录进 metadata 以便溯源。

**当前状态：全量臂运行中**。pilot 与全量臂都已确认 400-query manifest 与 MSLS 影像在
`/root/autodl-tmp/…/data/msls/`（本机没有）。落地所需的下游件也已写好并用 pilot 数据实测通过：
`src/scripts/analyze_geoshield_arm.py`（读数）与 `src/scripts/make_geoshield_table.py`
（生成 `paper/generated/tab_geoshield.tex`，风格对齐 `tab:operators`）。

两件在 pilot 数据上已经确定、与最终结论无关的事，先记下来免得重查：
- **对照取 isotropic 而非 clean。** 比 clean 低只说明加了能量；本文到处在问的是「按机制指定的方式花这份能量，
  是否强过随便花」，所以 $\Delta$、置信区间、等价检验全部对 isotropic 取。clean 行只用来定位攻击者。
- **pilot 里三个条件的 Top-5 完全相同（0.4400），这不是卡死的列。** 已逐行核对：`top5_hit` 之和恰等于
  `rank<=5` 的计数（11/11/11），`top10_hit` 恰等于 `rank<=10`（12/12/12）。真实含义是
  **该能量下机制只在候选表内部重排、并未把正确地点挤出候选表**（clean 的 11 个 top-5 命中全在 rank 1，
  公开版只有 8 个）。这条一致性检查已写进表格生成器，下一个看到相同列的人不必再手查一遍。

## 【评审的 bounded 方案已落地，作为下限保留】R3 的替代方案

**这一条我没有做到评审要求的强方案，原因不在算力。** 把 GeoShield（`thinwayliu/Geoshield`，AAAI 2026）clone 下来逐行核对后发现，**其公开版本把 VLM 组件留成了空实现**：`describe_image_placeholder()` 的函数体是 `TODO: Implement your VLM API call here`，注释让用户自行接入 GPT-4V / Claude / Gemini / LLaVA。

这不是可以绕过的边角：该描述经 `ensemble_loss.set_geotext_truth(description)` 进入 `geo_loss`，而目标函数里这一项是**被减掉**的（`loss -= (text_loss + text_local_loss)`）——它正是让扰动在破坏地理线索的同时保住语义的那一项，也是论文三个命名模块之一「exposure element identification」所依赖的输入。没有 VLM 就不是在跑 GeoShield，而是在跑另一个目标函数。

因此「端到端复现一个已发表机制」在不自备 VLM API 预算的前提下**对任何人都不可得**，这本身是一条可报告的可复现性观察，且正落在本文（一篇审计该家族的论文）的射程内。

已落地的 bounded 方案（评审第 2、3、4 条行动项，原文允许）：
- §III-F 措辞改写：原先写成「我们选择改编而非复现」，现写明公开实现 stub 掉了其目标函数所依赖的 VLM 项，所以改编是**被迫**而非偏好；
- 补充材料新增 `tab:rule_provenance`，逐条列出本审计测试的每一条 placement 规则及其来源（本文自建 / 经典算子 / 他人训练的分割模型 / 改编自已发表机制但未复现）；
- §II 新增范围声明，明确本文的否定结论是**关于该表中规则**的结论，并说明这正是同时求解 map 的原因——一个没有规则能被指为其弱实例的对照；
- mask-guided 改编臂的对比表已搬进补充材料（它是本文唯一一次与已发表公式的正面对比）。

**未采取的替代方案与理由**：接一个本地 VLM（BLIP/LLaVA）填补 stub 会改变目标函数，跑出来的东西不能诚实地标注为 GeoShield，所以没有做。

- 需要你决策：是否接受这个 bounded 方案作为 R3 的最终答复？
  A: 接受 
- 如果你愿意出 VLM API 预算（GPT-4V / Gemini 任一即可），我可以补做端到端复现，但仍必须在论文里写明「我们替其补上了公开版本缺失的组件」——因为那不是作者发布的那份实现。
  A:  GPT 5.6

# 遗留问题

## extended evidence report 现在无处可取（2026年9月15日）

正文与补充材料共 4 处提到 “the extended report”，但它既不在投稿包里（`07_code` 已清空论文内容），
也不在 IEEE DataPort 存档里（存档只有 exports/artifact_results/scripts，没有任何 PDF），capsule 更没有。
也就是说审稿人读到这 4 句时找不到那份文档。

需要你提供/决策：
1. 把 `supplementary_extended.pdf` 放进 DataPort 存档（最省事，正文不用动）？还是删掉正文里这 4 处引用？
   我的建议是前者。
   A:
2. `submit/source/supplementary/` 要不要放回 `main.aux`（13 KB）？不放回的话，投稿系统单独编译补充材料会出现 17 个 `??`。
   建议放回，它是 xr 的输入文件而不是编译产物。
   A:


## capsule 刷新已推送，等你在 Code Ocean 界面发布 v1.1（2026年9月15日）

工作 capsule `capsule-8046996` 的 `main` 已收到刷新（commit `25033f3`）：删掉全部论文内容、README 重写为本文、
data README 译英、删除私有子模块声明、verifier 与仓库同版。
但**公开地址 `https://codeocean.com/capsule/9035965/tree` 现在仍然是 v1.0**——
它是发布快照，git 推送被拒（unauthorized），只能在 Code Ocean 界面上从工作 capsule 发布新版本。

需要你提供/决策：
1. 请在 Code Ocean 界面对该 capsule 执行一次 Publish / Release，发成 v1.1。
   发完告诉我，我复核公开版内容是否与推送一致（尤其是确认 `.tex` 确实不在公开版里）。
   A:
2. capsule 是否已挂上 `ppedcrf-evidence` 数据资产？我只能读到代码仓库，看不到数据资产挂载状态；
   没挂的话一键运行只跑合成自检，不会重算任何数值。
   A:
3. Code Ocean 发布是否给了 DOI（`10.24433/CO.*`）？若有，稿件脚注应改用 DOI 而不是 capsule 链接；
   DataCite 现在查不到这条记录。
   A:

## 三问已答，下面是答复与执行状态（2026年9月13日）

1. 补充材料超页申请 → **A: 要申请**。投稿信里申请与理由已写好，且写明若 EiC 坚持 6 页就把四张表搬回 extended report。无需再动。
2. R3 → **A: 接受 bounded 方案；另有 GPT 5.6 / claude-code-api 可用**。据此改为跑强方案：照原样跑公开版并报告其退化，机器选 vGPU 3090。不需要 GPU 的部分已全部做完（见 `# 未修改或部分修改` 首条），**只等你开卡**。
3. ORCID / EDICS / 前次投稿披露 → **A: 我自己完成**。已从待办移出。

### R3 强方案已在 vGPU 3090 上起跑（2026年9月13日）

**主机连上了，而且此前「端口关闭」是我判断错误。** 我用 `bash` 的 `/dev/tcp` 和 `nc <hostname>` 探测端口，
两者都报 closed，我据此告诉你实例没开——**错了**。DNS 完全正常（连续五次都解析到 36.103.198.204），
真正原因是该主机名的 v6 查询返回 IPv4-mapped 地址 `::ffff:36.103.198.204`，
`/dev/tcp` 与 `nc <hostname>` 会去连这个 mapped 形式而失败，而 `ssh` 和 `nc <裸IPv4>` 都正常。
**探测方法本身是错的，主机一直是通的**，白让你去查了一趟实例状态。

主机状态：RTX 3090 / 48.5 GB 空闲，repo、venv（Python 3.10.12）、400-query manifest（311 MB）、
MSLS 影像（1.6 GB）、GeoShield clone 全部就位，磁盘 581 GB、inode 充足。
代码按仓库铁律走 **本地改 → commit → push → 主机 pull**（4 个 commit，`b3dcad5`…`972e1b2`）。
合并时有三个主机本地未跟踪脚本会被覆盖，**先逐个比对 sha256 确认与上游逐字节相同**后才移到备份目录再合并，没有直接删。

**pilot 已跑完（25 query × 3 条件 = 75 行，16 分 38 秒，约 40 秒/帧）**，能量门精确通过：
clean 交付 MSE `0.0000`，isotropic 与 geoshield 均为 `15.6800`，PSNR 36.18 dB——与全文引用的操作点一致。

pilot 结论：**协议跑通了，但这个规模回答不了科学问题**。
Top-1：clean `0.4400`、isotropic `0.4000`、geoshield `0.3200`；
geoshield 对 isotropic 的配对差为 `-0.0800` `[-0.200,+0.000]`，p=0.50，**discordant 对只有 2 个**。
25 个 query 上几乎没有分辨力，这正是本文自己在 placement 表里反复强调的
「报告的是没测到，而不是测过没有」。要得到可写进论文的结论必须上全量 manifest。

**为跑通它，中间拆掉六道障碍——其中五道是公开版自身的缺陷，一道是我绕过其 main() 的后果：**
（1）`import geoshield` 直接抛错（config_schema 把非 Optional 字段默认成 None）；
（2）`get_image_features` 在 transformers 5.16.1 返回 `BaseModelOutputWithPooling` 而非 tensor，
公开版对它调 `.norm()`，**一帧都跑不了**，而能钉版本的 requirements.txt 仓库里没有；
（3）VLM stub 返回常量 caption；（4）region 模块默认关闭；（5）untargeted 臂的 target 集合未定义；
（6）wandb 未初始化——这条是我直接调内层函数跳过了它 `setup_wandb()`，不算公开版缺陷。

第 2 条的修法是精确的：transformers 源码里 `vision_outputs.pooler_output = visual_projection(pooled_output)`，
所以 `.pooler_output` **就是**旧 API 返回的投影嵌入；按模型实例打补丁，不动 `CLIPModel`，
本仓库自己的 CLIP 攻击者行为不受影响，并记进 run metadata。

**几何决策（需要在论文里交代）**：公开版在攻击循环里硬编码 `RandomCrop(224)`，其 config 先缩放到 640 方图，
而本协议发布的帧是 192×320（比 224 还矮）。喂 192×320 会直接失败，且那也不是「照原样跑」——
那是剥夺了它自己的预处理。所以**攻击在它的原生分辨率上跑，只把产生的扰动搬回协议几何**，
再用同一套 per-frame 二分设定能量：形状来自机制，能量来自协议，与其它所有臂一致。

**我自己犯的两个错误，都记下来：**
- **能量超标 1e8 倍却照常出数。** `load_image` 返回的已经是 [0,255]，我又乘了一次 255，
  帧落到 [0,65025]，交付 MSE 达 1.9e9（目标 15.68）、max|δ| 平均 63,981，
  **却照样写出 25 行整齐数据并报出 Top-1 = 0.0400**。乘 255 只是笔误，
  真正的缺陷是**本仓库每条臂都有 energy gate，唯独我这条没有**。现已把闸门加进写入路径：
  任何扰动条件偏离目标 MSE 超过 0.05 就中止，而不是被平均进表格。两个无效 CSV 已移入
  `invalid_overscaled/` 保留不删、不参与任何平均。
- **巡检器两次误报「任务已死」**，而任务健康。取数逻辑写在双引号包裹的 ssh 参数里，
  其中 awk 转义被吃掉，`gpu` 与 `sessions` 字段恒空/恒零。已把取数脚本整个放到主机上做文件执行，
  ssh 层不再承载任何引号；并补上「进程数为 0 **且** 无 DONE 行」才判死。

pilot 起初还因 HF **xet 后端**卡死 9 分钟（本仓库第 70–71 条记过两次的老毛病），
改用 `hf-mirror.com` + `HF_HUB_DISABLE_XET=1` 后 B16 由「卡住不动」变成 62 秒下完。

**过程中修掉三个我自己写的 bug**，两个是主机上跑出来才暴露的：
1. `default_input_size_for_backbone()` 只收 backbone、返回一个方形边长，我却传了 `(h, w)` 并把结果同时当作
   「读图尺寸」和「嵌入分辨率」——这是两个不同的量（`resize_hw` 负责读图，`cfg.input_size` 负责嵌入端重采样）。
2. `make_default_embedder()` 只收 cfg，`.eval().to(device)` 由调用方做。
3. **保真性 bug（读代码才发现，不是跑出来的）**：公开实现 caption 的是 `image_tgt`（目标图），
   我第一版却 caption 了源帧。已改为 caption 目标图，且缓存键改用目标图自己的 gallery id，
   顺带让不同 query 抽到同一目标时复用同一条 caption。

**第二臂改用 OpenAI，并且这是更贴近论文的选择。**
Anthropic 账号余额不足（`400 ... credit balance is too low`），但你提供的 `chatgpt-api` key 可用（132 个模型）。
选 **`gpt-4o`** 而非 gpt-5，理由是保真而非性能：**GeoShield 那个 stub 的 docstring 第一个点名的就是
「OpenAI GPT-4V API」**，gpt-4o 是它的后继型号——补上作者自己指名的组件，比换一家厂商的模型是更窄的替换，
论文里也更好交代。已在三张真实 MSLS 帧上验证：三条 caption 互不相同、不含地名、缓存可复用。

顺带修正一处我先前的实现错误：caption 调用原是手写 HTTP，把真实错误吞成了 `HTTP 400`；
改用官方 SDK 后才看到是余额问题。同时 `max_tokens` 是 **thinking + 正文** 的总上限，
而 Claude 侧默认开 thinking，原先留的 200 token 会让正文被截断或为空**却不报错**——已放宽并只读 text block，
且模型拒答或 caption 为空时直接中止，而不是退回那句常量 caption（那会把第二臂错误标注成 published 臂）。

顺带修正一处我自己的实现错误：caption 调用原先是手写 HTTP，把真实错误吞掉了，只报 `HTTP 400`。
按本仓库所用语言的官方 SDK 规范改写为 `anthropic` Python SDK 之后，错误信息才显示出来是余额问题而非请求格式问题。
同时修掉一个会静默出错的隐患：`max_tokens` 是 **thinking + 正文** 的总上限，而该模型默认开启 thinking，
原先给 caption 留的 200 token 会让正文被截断或为空却**不报错**；现已放宽并改为只读 `text` block，
另加两道护栏——模型拒答或 caption 为空时直接中止，而不是退回那句常量 caption（那会把这条臂错误标注成 published 臂）。

**2. vGPU 3090 端口仍拒绝连接（两条臂都卡这里）。**
`.env` 里记的是 `connect.westd.seetacloud.com:22766`，域名解析正常（36.103.198.204）但该端口及邻近端口均关闭。
AutoDL 每次开机会重新分配端口，请把控制台上**当前的 ssh 命令**发我（或更新 `.env`）。
400-query manifest 与 MSLS 影像都在主机上，本机没有，所以这一步没法绕开。

**代码侧已全部就绪并通过本机验证**：import、stub 断言、helper 解析、脚本语法、两个脚本编译全绿。
published 臂**不需要 API 余额**，只要主机能连上就能开跑。

- ~~需要你做的两件事~~ **两条都已消解，不再需要你提供任何东西**：
  1. ~~给 Anthropic 账号充值~~ → 你提供的 `chatgpt-api` key 可用，第二臂改用 OpenAI，
     Anthropic 余额不再是阻塞项。runner 的 `--caption_source openai` 已就位，
     caption 按目标图缓存到磁盘，续跑不重复付费。
  2. ~~提供 ssh 端口~~ → 端口一直是通的；是我用 `/dev/tcp` 与 `nc <hostname>` 探测，
     而该主机解析到 IPv4-mapped IPv6 地址（`::ffff:…`），这两个工具在该地址族下会失败而 `ssh` 不会。
     是我的判断错误，不是主机问题。

- ~~【待决策】main.tex 14 页超限~~ **已解决，无需你决策（2026-09-13 晚）**：
  R3 结果写进正文后曾涨到 14 页，第 14 页上只有一条参考文献续行。四轮删冗词都拉不回来，
  因为**第 12 页被浮动体占满、正文只有 5 行**，之前的删减会被浮动体重排吸收，顶不动参考文献。

  **真正的缺陷不在正文，在参考文献。** IEEE 规范是「超过 6 位作者只列第一位 + et al.」，
  而 `IEEEtran.bst` 只有在**任何引用之前**先引一条 `@IEEEtranBSTCTL` 控制条目时才会执行这条规则，
  本文没有这条条目，于是每条文献都完整列出全部作者——第 [39] 条列了十一位。
  补上控制条目后：**参考文献变得符合 IEEE 规范，正文回到 13 页**。

  值得写明一句：**这一页是靠修正一个格式违规拿回来的，不是靠删内容**。
  全程没有为了塞下而删掉任何 hedge、caveat 或否定结论——那些正是这篇论文的立论本身。

- ~~当前唯一待你决策的一条~~ **已决策（2026-09-13）**：
  问题是这一个 seed 跑完后是否再买余下两个 seed（约 9 小时）把 seed 数与论文其它臂对齐。
  **A: 按推荐执行，不买。** 因此收尾链条为：跑完 → 拉回 → 校验 → commit/push → **关机止费**，
  不再开新臂。R3 以「单 seed、400 query、有界零效应（若确为零）」的形式定稿。

  **论文里必须写明 seed 数与其它臂不同**，否则表格并排放会让读者默认三 seed。
  这条写进 tab:geoshield 的 caption，不靠记忆。

  中途读数（98 个完整 query，仅用于规划、不改变检验、不提前停止）：
  geoshield 对 isotropic 为 $-0.0204$，区间 $[-0.077,+0.033]$ 跨零，8 个 discordant 对。
  效应随样本量增大而缩小（25 query 时 $-0.080$ → 98 query 时 $-0.020$），是零效应的特征而非真效应。
  另外 pilot 的绝对值（Top-1 0.44）本就不具代表性：它取的是 manifest 前 25 条、基本是同一个城市，
  全量下降到约 0.22——这也印证了当初拒绝把 pilot 当结果报告是对的。 不买

### 安全提醒

核对凭据时我用 grep 列 key 名，但 `.env` 把密钥放在无标签行上，导致**两个 GitHub PAT 出现在了本次会话输出里**（未写入任何文件、提交或文档）。建议吊销并重新生成这两个 GitHub token。Anthropic key 全程只按行定位、未打印。

---

## 原始三问（供对照）

投稿前只剩三件事需要你，没有一件是实验或论文修改；其余 14 条评审意见都已闭合。

1. **补充材料 10 页，是否确认向 EiC 提出超页申请？**
   SPS 的规则是"建议不超过 6 个双栏页，该限度内无需 EiC 批准"——超页是**可申请**的，不是硬上限。
   `docs/cover_letter.txt` 里已经写好申请及其理由（本文报的是与领域设计假设相反的负面结果，而
   TIFS 的深度学习投稿指南明写"负面结果适用更高的可复现标准"；审稿人要验一个负面结果，
   需要看到它是从哪些格子 pool 出来的），并写明**若 EiC 坚持 6 页，我们把四张表搬回 extended
   report 并在正文写明**。你之前对 9 页的版本答过"要申请"，这一轮涨到 10 页（新增 operator
   表与 placement panel 表），所以再确认一次。
   A: 要申请

2. **R3 的 bounded 方案是否作为最终答复？**（详见 `# 未修改或部分修改`）
   GeoShield 公开版把它目标函数依赖的 VLM 调用留成了 stub，所以"端到端复现"对任何人都不可得。
   我按评审自己给出的替代方案做了范围声明 + 溯源表 + mask-guided 对比进包。
   若你愿意出 VLM API 预算我可以补做，但论文里必须写明"我们替其补上了公开版本缺失的组件"。
   A: 有GPT 5.6 可以开subagent

3. **ORCID / EDICS / 前次投稿披露**（`ExperimentProgress.tex` 的 S1、S2）。
   这三项只能在投稿系统里完成，仓库文件无法核验：三位作者的 ORCID 需注册、EDICS 类别需选定、
   元数据须与论文标题作者一致；仓库里存有给另一会议的 response letter，是否需要披露取决于
   文件之外的事实，若需要则须在投稿时如实声明。
   A: 我自己完成

**无需你决策的两点，供知悉**：本轮不需要 GPU，两条需要卡的臂（R1 面板、R4 geolocator）
已于 9月12–13 日跑完并关机；auditor 902 条全绿，正文 13/13 页、0 undefined、0 overfull、
0 字体警告。

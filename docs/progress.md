# 已全部修改

1. 已将标题、摘要和全文问题定义统一为“发布视频帧时的背景驱动 location privacy”。
修改说明：重写并收束了标题、摘要、引言、相关工作、方法、实验和结论，删除旧稿中“敏感目标类别保护”“特征空间发布重建”等混杂叙事，统一为 gallery-based retrieval attacker 与 image-space sanitization 的单一问题表述。

2. 已将方法描述与当前仓库实现对齐并补清关键公式含义。
修改说明：在 `paper/main.tex` 中重写了 DCRF、NCP 与高斯噪声标定的公式和文字说明，新增符号表与算法流程，并明确正文只主张 DP-style calibration；同时补充说明 `p_t` 负责空间 support、`\alpha_t` 负责 support 内幅度调制，双重加权是有意实现而非写法歧义。

3. 已澄清攻击评估粒度与发布对象。
修改说明：在 threat model、受控 benchmark 协议和附录中明确说明查询端使用短序列估计时序平滑 mask，但 retrieval 指标是在 sanitized middle frame 上计算，避免 frame-level 与 sequence-level 攻击定义混淆。

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

# 未修改或部分修改

1. 跨攻击骨干网络的隐私稳健性仅部分补强。
修改说明：本轮已加入 ResNet18 与 ResNet50 的 attacker-sensitivity 实验，并在正文中明确区分默认攻击者结果与跨骨干迁移结果。
未全部修改原因：当前新增实验仍是本地可获得数据上的 proxy benchmark，尚未覆盖更大规模真实 driving geo-localization 场景，因此还不能支持 attacker-agnostic 的强结论。
后续准备如何修改：补充更强 retrieval backbone、更多 gallery 规模和更贴近 driving/location retrieval 的数据设置，再决定是否继续保留更强的泛化表述。

2. DCRF 与 NCP 的长时序收益仅部分验证。
修改说明：本轮已完成 w/o temporal consistency 与 w/o NCP 消融，并在正文中如实说明它们在短监控序列上的差异有限，同时结合新增 blur baseline 指出当前 proxy benchmark 更强调空间 support 而非长时序优势。
未全部修改原因：当前代理数据序列较短、动态较弱，不足以充分拉开时序平滑与控制项在长视频上的优势。
后续准备如何修改：后续优先补长驾驶序列上的 temporal consistency、failure case 与 sensitivity analysis，并生成对应图表来支撑时序模块的价值。

3. 更强的 task-aligned baselines 仍仅部分补齐。
修改说明：本轮已补上 mask-guided blur、mask-guided mosaic，并将 full-frame baseline 统一明确为 global Gaussian noise，对照组中也保留了 random mask。
未全部修改原因：当前仍缺 segmentation-guided masking、retrieval-aware adversarial perturbation、inpainting 等更强或更接近投稿审稿要求的 baseline。
后续准备如何修改：继续在同一 benchmark 上补 1--2 个更强 baseline，并在条件允许时迁移到更接近真实 geo-localization 的数据设置。

4. 少量模板级与排版 warning 仍未完全清零。
修改说明：本轮已完成主文与附录稳定编译，主文 `paper/build/main.blg` 的 BibTeX warning 已清零，附录长路径导致的 overfull 已缓解为 underfull，剩余主要是字体替代、`IEEEtran` 附录模板与 `caption` 的兼容提示以及少量松紧度 warning。
未修改原因：这些 warning 主要来自当前 standalone appendix 模板与字号/断行策略，本轮未重构附录模板或整体版式，因此尚未全部清除。
后续准备如何修改：后续如进入终稿清理阶段，将进一步压缩附录中的长 `\texttt{}` 片段、评估是否替换或精简附录模板选项，并继续收敛局部排版 warning。

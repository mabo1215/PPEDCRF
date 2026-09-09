# 家里机器上要找的八棵导出树（2026-09-09）

这份清单是给"回家找导出树"这一步用的。**在找到或确认找不到之前，不重跑**——
重跑产出的是新数字，论文里 placement 和 operator 两节要跟着改写。

## 一条命令搜完

在家里的机器上（Git Bash / WSL / PowerShell 任选一种）：

```bash
# Git Bash / WSL：把 /c /d /e /g 换成你实际挂了的盘
find /c /d /e /g -maxdepth 8 \
  \( -name 'icme2027_placement_msls' \
  -o -name 'operator_study' \
  -o -name 'margin_oracle' \
  -o -name 'known_jacobian' \
  -o -name 'known_jacobian_operators' \
  -o -name 'd4_3seed' \
  -o -name 'sanfree_3seed' \
  -o -name 'tifs5_analysis' \
  -o -name 'results' -path '*artifact*' \) 2>/dev/null
```

```powershell
# PowerShell 等价写法
$names = 'icme2027_placement_msls','operator_study','margin_oracle',
         'known_jacobian','known_jacobian_operators','d4_3seed',
         'sanfree_3seed','tifs5_analysis'
Get-PSDrive -PSProvider FileSystem | ForEach-Object {
  Get-ChildItem $_.Root -Recurse -Directory -Depth 8 -ErrorAction SilentlyContinue |
    Where-Object { $names -contains $_.Name } | Select-Object FullName
}
```

**先搜 `src/outputs/`**：它被 gitignore，所以不会出现在 `git status` 里，
也正是上一轮判断失误的地方（`progress.md` 第 292 条说它在本机，实际不在；
本轮已更正）。

## 八棵树分别撑着什么

按重要性排序。前两棵值得单独花时间找，后面几棵找不到的代价小得多。

| 树 | 撑着论文的哪部分 | 找不到的后果 |
|---|---|---|
| **`icme2027_placement_msls`** | §"The Null Holds on Real Geographic Data"，真实地理数据上的 placement 表 | **最贵的一棵**。R1 的 allocation 家族（12 个数）和 R5 的 allocation 部分都卡在它上面。重跑要改写整节 |
| **`operator_study`** | 算子对照表（四个算子在匹配交付 MSE 下的比较） | 同上，R1 的另一半。重跑同样要改数字 |
| `margin_oracle` | margin-gradient 放置那一行 | 影响 R3 的一格；A2 落地后部分可替代 |
| `known_jacobian` | §III-H 已知 Jacobian 的合成对照模型 | 支撑"位移恒等式验证到 4% 以内"那句 |
| `known_jacobian_operators` | 同上，算子版本 | 同上 |
| `d4_3seed` | direction 家族的一组三种子运行 | 已被 D6 导出覆盖大部分，代价最小 |
| `sanfree_3seed` | gallery-free 的三种子运行 | 同上 |
| `tifs5_analysis` | 本轮分析脚本的输出目录 | 可以从原始行重新生成，**不用找** |

## 找到之后怎么带回来

**不要**用 U 盘拷贝再手工合并——直接打一个 tar，保持目录结构：

```bash
cd <家里那个 PPEDCRF>/src/outputs
tar czf ~/ppedcrf_missing_trees.tar.gz \
  icme2027_placement_msls operator_study margin_oracle \
  known_jacobian known_jacobian_operators d4_3seed sanfree_3seed
```

拿回来后在工作机上：

```bash
cd C:/source/PPEDCRF/src/outputs
tar xzf <路径>/ppedcrf_missing_trees.tar.gz
cd C:/source/PPEDCRF
python src/scripts/audit_claim_consistency.py     # 应该仍是 110/110，且不再有 no-data
python src/scripts/analyze_placement_query_level.py --help   # 然后按它的参数跑 allocation 家族
```

## 每棵树长什么样（用来确认找对了）

- `icme2027_placement_msls/`：各 backbone 一个子目录，里面是 `per_query.csv`，
  列含 `placement` / `seed` / `query_id` / `correct_rank`。
  `final/per_query.csv` 应该是 **9,600 行** = 400 query × 8 placement × 3 seed（resnet18）。
  整棵树约 **23 个文件、7.25 MB**（第 292 条记的尺寸，可用来核对）。
- `operator_study/`：按算子分的 CSV，含交付 MSE 列。
- `known_jacobian*/`：合成模型的输出，文件少、体积小。

## 如果确认找不到

告诉我，我在 PRO 6000 或 2c 上重跑。**重跑是安全的**：本轮已经确认
`manifest_all8.jsonl`（400 query / 277 place / 2000 gallery）的 query id
与已发表的 D6 导出**逐条完全相同**，所以重跑跑的是同一个基准，不是近似基准。
但产出的是新数字，届时：

- §"The Null Holds on Real Geographic Data" 的七格要换成新值
- operator 表要换
- 两节的文字结论要按新数字重新核对（结论方向大概率不变，但不能假定）

预计机时：placement + operator 一起约 6 小时（2c 双卡可以对半分）。

"""Body-text overlap between a prior (reviewed) manuscript and the current one.

The script that produced the figures in the TIFS disclosure ("94.6% new,
2,992 -> 11,461 words, 622 shared, one 25-word passage") was not preserved.
Method A below is a reconstruction fitted to those figures; method B is a
stricter prose-only variant reported as a sensitivity check.

Method A (reconstruction)
  * body = text after \\begin{document} up to the first \\bibliographystyle /
    \\bibliography (else \\end{document}); % comments and comment environments
    removed; figure/table floats and algorithm environments removed;
    IEEEkeywords removed; arguments of includegraphics/url/input/... removed;
    every other command name is deleted but its brace contents are kept, so
    section headings, math (inline and display) and cite/ref/label keys
    survive as tokens.
  * words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?"), lower-cased for matching.
  * a word is "shared" when it lies inside some run of >= K=3 consecutive
    words that also occurs verbatim anywhere in the other document.
Method B (prose only): as A, but display math, inline math and the keys of
  cite/ref/label commands are removed as well.

Reads sources with `git show` and the working-tree file; writes nothing.

The version ACM TOMM reviewed is paper commit d097f57 (main.tex plus
appendix.tex), not ca94244, against which the 94.6% was first measured. Against
d097f57 main plus appendix, method A gives 92.6% new (6,487 -> 11,460 words,
845 shared, one 25-word passage, which is the DCRF update equation); that is
the figure the cover letter, the response file and the submission form use.
"""
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PAPER = str(Path(__file__).resolve().parents[2] / "paper")
K = 3
SENTINEL = "\x00"

FLOATS = ["figure", "figure*", "table", "table*", "wrapfigure", "wraptable"]
ALGO = ["algorithm", "algorithm*", "algorithmic"]
MATH_ENVS = ["equation", "equation*", "align", "align*", "gather", "gather*",
             "multline", "multline*", "eqnarray", "eqnarray*", "displaymath", "math"]
OTHER_ARG_CMDS = ["includegraphics", "url", "input", "include", "bstctlcite", "bibliography",
                  "bibliographystyle", "hypersetup", "vspace", "hspace", "setlength",
                  "addtolength", "Description", "ccsdesc", "href"]
KEY_ARG_CMDS = ["cite", "citep", "citet", "citealp", "nocite", "ref", "eqref", "autoref",
                "cref", "Cref", "pageref", "label"]
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def git_show(rev, path):
    r = subprocess.run(["git", "-C", PAPER, "show", f"{rev}:{path}"], capture_output=True,
                       text=True, encoding="utf-8")
    if r.returncode:
        raise SystemExit(r.stderr)
    return r.stdout


def drop_env(s, name):
    e = re.escape(name)
    return re.sub(r"\\begin\{" + e + r"\}.*?\\end\{" + e + r"\}", " ", s, flags=re.S)


def drop_cmd_args(s, cmds):
    return re.sub(r"\\(?:" + "|".join(cmds) + r")\*?(?:\[[^\]]*\])*(?:\{[^{}]*\})+", " ", s)


def body_words(tex, prose_only=False):
    s = tex.replace("\r\n", "\n")
    s = "\n".join(re.split(r"(?<!\\)%", line, 1)[0] for line in s.split("\n"))
    s = drop_env(s, "comment")
    i = s.find("\\begin{document}") + len("\\begin{document}")
    m = re.search(r"\\bibliographystyle|\\bibliography\{|\\begin\{thebibliography\}", s[i:])
    j = i + m.start() if m else s.find("\\end{document}")
    s = s[i:j]
    for e in FLOATS + ALGO + ["IEEEkeywords"]:
        s = drop_env(s, e)
    if prose_only:
        for e in MATH_ENVS:
            s = drop_env(s, e)
        s = re.sub(r"\\\[.*?\\\]", " ", s, flags=re.S)
        s = re.sub(r"\$\$.*?\$\$", " ", s, flags=re.S)
        s = re.sub(r"(?<!\\)\$.*?(?<!\\)\$", " ", s, flags=re.S)
        s = re.sub(r"\\\(.*?\\\)", " ", s, flags=re.S)
        s = drop_cmd_args(s, OTHER_ARG_CMDS + KEY_ARG_CMDS)
    else:
        s = drop_cmd_args(s, OTHER_ARG_CMDS)
    s = re.sub(r"\\(begin|end)\{[^}]*\}(\[[^\]]*\])?", " ", s)
    s = re.sub(r"\\[A-Za-z@]+\*?(\[[^\]]*\])?", " ", s)
    s = re.sub(r"\\(.)", lambda m: m.group(1) if m.group(1) in "%&#_$" else " ", s)
    s = s.replace("{", " ").replace("}", " ").replace("~", " ")
    return WORD.findall(s)


def coverage(a, b, k=K):
    """cov_a[i] is True when a[i] lies in a k-gram that occurs anywhere in b."""
    gb = {tuple(b[j:j + k]) for j in range(len(b) - k + 1)}
    cov = [False] * len(a)
    for i in range(len(a) - k + 1):
        g = tuple(a[i:i + k])
        if g in gb and SENTINEL not in g:
            for t in range(i, i + k):
                cov[t] = True
    return cov


def runs(cov):
    out, start = [], None
    for i, c in enumerate(cov + [False]):
        if c and start is None:
            start = i
        elif not c and start is not None:
            out.append((i - start, start))
            start = None
    return sorted(out, reverse=True)


def maximal_common(cur, ref, k=K):
    """Longest passages contiguous in BOTH texts, non-overlapping in `cur`."""
    idx = defaultdict(list)
    for j in range(len(ref) - k + 1):
        idx[tuple(ref[j:j + k])].append(j)
    best = {}
    for i in range(len(cur) - k + 1):
        for j in idx.get(tuple(cur[i:i + k]), ()):
            L = 0
            while i + L < len(cur) and j + L < len(ref) and cur[i + L] == ref[j + L] != SENTINEL:
                L += 1
            if L > best.get(i, 0):
                best[i] = L
    taken = [False] * len(cur)
    out = []
    for i, L in sorted(best.items(), key=lambda x: -x[1]):
        if not any(taken[i:i + L]):
            for t in range(i, i + L):
                taken[t] = True
            out.append((L, i))
    return out


def compare(label, ref_words, cur_words, k=K, show=5):
    ref = [w.lower() for w in ref_words]
    cur = [w.lower() for w in cur_words]
    n_ref = sum(1 for w in ref if w != SENTINEL)
    n_cur = len(cur)
    cov_cur = coverage(cur, ref, k)
    cov_ref = coverage(ref, cur, k)
    shared_cur = sum(cov_cur)
    shared_ref = sum(cov_ref)
    mcs = maximal_common(cur, ref, k)
    cov_runs = runs(cov_cur)
    print(f"--- {label}  (K={k})")
    print(f"  body words: reference {n_ref:,}  current {n_cur:,}")
    print(f"  shared words in current: {shared_cur:,}  -> new {100 * (1 - shared_cur / n_cur):.1f}%"
          f"  (shared = {100 * shared_cur / n_cur:.1f}% of current)")
    print(f"  reference words retained in current: {shared_ref:,} = {100 * shared_ref / n_ref:.1f}%")
    print(f"  longest coverage run in current: {cov_runs[0][0]}  "
          f"(runs >= 25: {sum(1 for L, _ in cov_runs if L >= 25)})")
    for L, i in mcs[:show]:
        print(f"    {L:3d} | {' '.join(cur_words[i:i + L])}")
    return dict(n_ref=n_ref, n_cur=n_cur, shared=shared_cur, retained=shared_ref,
                longest=mcs[0][0] if mcs else 0)


def joined(*docs):
    out = []
    for d in docs:
        if out:
            out.append(SENTINEL)
        out.extend(d)
    return out


def main():
    with open(f"{PAPER}/main.tex", encoding="utf-8") as f:
        current_tex = f.read()
    src = {
        "ca94244": git_show("ca94244", "main.tex"),
        "d614b42": git_show("d614b42", "main.tex"),
        "d097f57_main": git_show("d097f57", "main.tex"),
        "d097f57_app": git_show("d097f57", "appendix.tex"),
    }
    for prose_only in (False, True):
        tag = "B (prose only)" if prose_only else "A (reconstruction)"
        w = {k: body_words(v, prose_only) for k, v in src.items()}
        cur = body_words(current_tex, prose_only)
        print(f"\n================ Method {tag} ================")
        print("Validation (claim: 2,992 -> 11,461; 622 shared; 94.6% new; 20.8% retained; one 25-word run)")
        compare("ca94244 main  vs  d614b42 main (state when claim was written)", w["ca94244"], w["d614b42"])
        compare("ca94244 main  vs  current main (working tree)", w["ca94244"], cur)
        print("Reviewed version")
        compare("d097f57 main  vs  current main", w["d097f57_main"], cur)
        compare("d097f57 main+appendix  vs  current main",
                joined(w["d097f57_main"], w["d097f57_app"]), cur)
        print(f"  d097f57 appendix alone: {len(w['d097f57_app']):,} words")
        if "--sens" in sys.argv:
            for k in (5, 8):
                compare(f"d097f57 main+appendix vs current main", joined(w["d097f57_main"], w["d097f57_app"]),
                        cur, k=k, show=1)


if __name__ == "__main__":
    main()

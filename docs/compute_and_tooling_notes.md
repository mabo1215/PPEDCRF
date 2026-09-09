# Compute host and tooling notes

These are operational rules learned by running jobs for this paper. They are
kept here, under version control, because `.claude/rules/` and `AGENTS.md` are
both gitignored (`.gitignore` lines 30--37) and so do not travel with the
repository. The same content is mirrored into `.claude/rules/src.md` locally.

## AutoDL / seetacloud network acceleration

Every seetacloud host supports AutoDL's academic network acceleration. Enable
it before any step that reaches GitHub or Hugging Face:

```bash
source /etc/network_turbo
```

Reference: <https://www.autodl.com/docs/network_turbo/>

Three things worth knowing before relying on it:

- It exports `http_proxy` / `https_proxy` / `no_proxy` **for the current shell
  only**. Source it *inside* the `setsid`/`nohup` job, not merely in the
  launching shell. Getting this wrong is not a subtle failure: four A5
  processes once hung indefinitely competing to download the same torchvision
  checkpoints without the proxy, writing nothing and logging nothing.
- It speeds up GitHub and Hugging Face and makes other traffic, notably pip
  mirrors, **slower**. Turn it on for a checkpoint download, not for a pip
  install from a domestic mirror.
- It fixes **reachability, not authentication**. A private-repo `git fetch`
  still fails with `could not read Username for 'https://github.com'`.

## Credentials do not travel to compute hosts

Host logins live outside the repository and are never committed. Do not copy
tokens, keys, or credential files onto a compute host to make a private
`git clone` work there. To move code onto a host that cannot authenticate,
stream it over the SSH connection that already exists:

```bash
tar cf - src/scripts src/eval src/tests | ssh -p <port> root@<host> \
  "cd <repo> && tar xf -"
```

That transfers data, which is fine, rather than spreading a credential, which
is not. Prefer installing a public key over reusing a password for repeated
access.

## Shared cards, and thread oversubscription

`nvidia-smi` reports the whole physical card, which may be shared with other
tenants. Check **free** memory, not total: PRO 6000 shows 97 GB while seven
other tenants hold 90 of it, and a job sized to the total dies of
`torch.OutOfMemoryError`. Size `--gallery_batch` and batch sizes to what is
actually free.

Cap threads on large-core hosts. PRO 6000 has **208 cores**; torch takes all of
them per process by default, and three concurrent jobs then spend their time in
futex contention rather than in the model — observed as ~600% CPU each with
zero rows written for 25 minutes. With the caps below the same jobs sit near
110% CPU and produce rows immediately:

```bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
```

Use `python -u` for anything long-running: a buffered log on a job that has not
yet written output is indistinguishable from a hung one.

## Detaching a long job

A plain `nohup ... &` over SSH dies when the connection closes. Use `setsid`,
redirect stdin, and then **verify in a separate call** that the process is
alive before treating the launch as successful:

```bash
setsid nohup <cmd> > logs/<job>.log 2>&1 < /dev/null & disown
```

Beware that `pkill -f <pattern>` run from an SSH command matches the SSH
command's own shell, so `pgrep -c` will report phantom survivors. Confirm with
`nvidia-smi` memory or an explicit PID list instead.

## Editing `.tex` files programmatically

**Do not edit `.tex` through a Python script embedded in a shell heredoc.** The
shell collapses `\\` to `\` inside the heredoc even when it is quoted, and the
damage is silent and two-stage:

1. LaTeX row terminators `\\` become `\`, producing
   `Extra alignment tab has been changed to \cr`.
2. Worse, `\\begin{...}` becomes `\begin{...}` in the Python source, where
   Python reads `\b` as a **backspace** and writes U+0008 into the file. LaTeX
   reports `Unicode character ^^H (U+0008)` and a cascade of unrelated errors
   pointing at the wrong line.

Use a literal, exact-string editing tool. After any programmatic edit to a
`.tex` file, verify both properties before trusting a build:

```python
s = open(path, encoding="utf-8").read()
assert not any(ord(c) < 9 or 13 < ord(c) < 32 for c in s)   # no control chars
bad = [i for i, l in enumerate(s.splitlines(), 1)
       if l.rstrip().endswith("\\") and not l.rstrip().endswith("\\\\")]
assert not bad, bad                                          # no lone backslash
```

Build with `-halt-on-error` so the first real error is the one reported;
without it, pdflatex runs on and buries the cause under its consequences.

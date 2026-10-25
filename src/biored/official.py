"""Write predictions as PubTator and score them with NCBI's run_biored_eval.py.

The script comes from biored_re_source_code.tar on the BC8 FTP folder
(https://ftp.ncbi.nlm.nih.gov/pub/lu/BC8-BioRED-track/). Save it unchanged and git-ignored as
third_party/bc8_eval/run_biored_eval.py; the tested version has SHA-256
74d71f00df3faf63d82418784f6f8a8543fb321df442237f3074ac584cc33021.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "third_party"
    / "bc8_eval"
    / "run_biored_eval.py"
)
HEADER = re.compile(r"^([^|\t]+)\|[ta]\|")
OUT_FILES = {  # the script's fixed output names -> our level names
    "pair": "biored_bin_score.txt",
    "pair_type": "biored_mul_score.txt",
    "pair_novelty": "biored_bin_novelty_score.txt",
    "pair_type_novelty": "biored_mul_novelty_score.txt",
}


def _documents(path):
    doc = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            doc.append(line)
        elif doc:
            yield doc
            doc = []
    if doc:
        yield doc


def write_pubtator(template, preds: dict, path) -> None:
    """Copy `template` (a gold PubTator file) with its relation and direction lines replaced
    by `preds` ({pmid: [Relation]}).

    Title, abstract and mention lines are kept as they are, except that ';' between IDs
    becomes ',' (the official script only splits on ','). "None" relations are never written:
    the official script would read them as real pairs. Each document ends with exactly one
    blank line, because the official script empties a document followed by two.
    The output contains abstract text, so write it under results/raw/ (git-ignored).
    """
    if Path(path).resolve() == Path(template).resolve():
        raise ValueError("write_pubtator would overwrite its own template")
    seen = set()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for lines in _documents(template):
            m = HEADER.match(lines[0])
            if not m:
                raise ValueError(
                    f"document does not start with a title line: {lines[0][:60]!r}"
                )
            pmid = m.group(1)
            seen.add(pmid)
            for line in lines:
                t = line.split("\t")
                if HEADER.match(line):
                    f.write(line + "\n")
                elif len(t) >= 6 and t[1].isdigit() and t[2].isdigit():
                    t[5] = t[5].replace(";", ",")
                    f.write("\t".join(t) + "\n")
                # every other line is a gold relation or a direction line: dropped
            for r in preds.get(pmid, []):
                if r.type != "None":
                    f.write(f"{pmid}\t{r.type}\t{r.e1}\t{r.e2}\t{r.novel}\n")
            f.write("\n")
    missing = set(preds) - seen
    if missing:
        raise KeyError(
            f"predictions for {len(missing)} PMIDs not in the template, e.g. {sorted(missing)[:3]}"
        )


def run_official(gold_path, pred_path, script=SCRIPT) -> dict[str, dict]:
    """Score `pred_path` against `gold_path` with the official script.

    Returns {level: {n_gold, tp, fp, fn, precision, recall, f1}}. `gold_path` must have no
    direction lines and no ';' ID lists, so pass a copy made with
    write_pubtator(gold_file, {d.pmid: d.relations for d in docs}, ...).
    """
    script = Path(script)
    if not script.exists():
        raise FileNotFoundError(f"{script} missing; see the module docstring")
    with tempfile.TemporaryDirectory() as tmp:
        # The script ignores --in_pred_pubtator_file and always reads ./biored_pred_mul.txt.
        # It also writes its four score files into the working directory.
        shutil.copyfile(pred_path, Path(tmp) / "biored_pred_mul.txt")
        res = subprocess.run(
            [
                sys.executable,
                str(script),
                "--exp_option",
                "biored_eval",
                "--in_gold_pubtator_file",
                str(Path(gold_path).resolve()),
                "--in_pred_pubtator_file",
                "biored_pred_mul.txt",
            ],
            cwd=tmp,
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            raise RuntimeError(
                "official scorer failed. A KeyError usually means a relation ID with no mention, "
                "a ';' ID list or a direction line in one of the files.\n"
                + res.stderr[-2000:]
            )
        out = {}
        for level, name in OUT_FILES.items():
            first = (
                (Path(tmp) / name)
                .read_text(encoding="utf-8")
                .splitlines()[0]
                .split("\t")
            )
            _, n_gold, tp, fp, fn, p, r, f = first
            out[level] = {
                "n_gold": int(n_gold),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
            }
        return out

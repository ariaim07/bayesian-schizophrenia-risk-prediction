"""
Helpers for running PRS-CS on the real PGC schizophrenia summary statistics.

  python prep.py sumstats PGC3_...vcf.tsv.gz  out_sumstats.txt  out_n_gwas.txt
  python prep.py bim      LD_DIR              out_prefix          (writes out_prefix.bim)
  python prep.py check    PRSCS_OUT_PREFIX    weights_all.txt

No target cohort exists yet, so the "bim" step builds a stand-in .bim listing every
SNP in the LD reference. PRS-CS uses the .bim only to choose which SNPs to keep.
When you get a real cohort, pass its own .bim instead.
"""
import gzip, sys, glob, re
from pathlib import Path
import numpy as np

COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}


def _open(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def sumstats(src, out, n_out, min_info=0.9):
    n_in = kept = 0
    drop = {"non_rs": 0, "non_ACGT": 0, "ambiguous": 0, "low_info": 0, "bad_se": 0, "dup": 0}
    seen, neffs, betas = set(), [], []
    with _open(src) as f, open(out, "w") as w:
        for line in f:
            if line.startswith("##"):
                continue
            header = line.lstrip("#").rstrip("\n").split("\t")
            break
        col = {c.upper(): i for i, c in enumerate(header)}
        need = ["ID", "A1", "A2", "BETA", "SE"]
        missing = [c for c in need if c not in col]
        if missing:
            sys.exit(f"Missing columns {missing}. Header was: {header}")
        has_info = "IMPINFO" in col
        has_n = "NCAS" in col and "NCON" in col
        w.write("SNP\tA1\tA2\tBETA\tSE\n")
        for line in f:
            n_in += 1
            x = line.rstrip("\n").split("\t")
            snp, a1, a2 = x[col["ID"]], x[col["A1"]].upper(), x[col["A2"]].upper()
            if not snp.startswith("rs"):
                drop["non_rs"] += 1; continue
            if a1 not in COMP or a2 not in COMP:
                drop["non_ACGT"] += 1; continue
            if COMP[a1] == a2:                 # A/T or C/G: strand can't be resolved
                drop["ambiguous"] += 1; continue
            if has_info:
                try:
                    if float(x[col["IMPINFO"]]) < min_info:
                        drop["low_info"] += 1; continue
                except ValueError:
                    pass
            try:
                beta, se = float(x[col["BETA"]]), float(x[col["SE"]])
            except ValueError:
                drop["bad_se"] += 1; continue
            if not (np.isfinite(beta) and np.isfinite(se) and se > 0):
                drop["bad_se"] += 1; continue
            if snp in seen:
                drop["dup"] += 1; continue
            seen.add(snp)
            if has_n:
                try:
                    nca, nco = float(x[col["NCAS"]]), float(x[col["NCON"]])
                    if nca > 0 and nco > 0:
                        neffs.append(4 / (1 / nca + 1 / nco))
                except ValueError:
                    pass
            if len(betas) < 200_000:
                betas.append(beta)
            w.write(f"{snp}\t{a1}\t{a2}\t{beta:.6g}\t{se:.6g}\n")
            kept += 1

    med_beta = float(np.median(np.abs(betas))) if betas else float("nan")
    if med_beta > 0.5:
        sys.exit(f"Median |BETA| = {med_beta:.3f}: this looks like an odds ratio, not log(OR). "
                 "Check the README and convert with log() before running PRS-CS.")
    if not neffs:
        sys.exit("No NCAS/NCON columns: set n_gwas by hand (effective N = 4/(1/Ncases + 1/Ncontrols)).")
    n_gwas = int(np.median(neffs))
    Path(n_out).write_text(str(n_gwas))
    print(f"read {n_in:,} variants, kept {kept:,}")
    print("dropped:", {k: f"{v:,}" for k, v in drop.items()})
    print(f"median |log OR| = {med_beta:.4f}")
    print(f"effective GWAS N (median over SNPs of 4/(1/Ncas+1/Ncon)) = {n_gwas:,}")


def bim(ld_dir, out_prefix):
    ref = next(iter(glob.glob(str(Path(ld_dir) / "snpinfo_*_hm3"))), None)
    if ref is None:
        sys.exit(f"No snpinfo_*_hm3 file in {ld_dir}")
    n = 0
    with open(ref) as f, open(f"{out_prefix}.bim", "w") as w:
        next(f)
        for line in f:
            chrom, snp, bp, a1, a2, _ = line.split()
            w.write(f"{chrom}\t{snp}\t0\t{bp}\t{a1}\t{a2}\n")
            n += 1
    print(f"wrote {n:,} reference SNPs to {out_prefix}.bim")


def check(prefix, out):
    files = sorted(glob.glob(f"{prefix}_pst_eff_*_chr*.txt"),
                   key=lambda p: int(re.search(r"_chr(\d+)\.txt$", p).group(1)))
    chroms = [int(re.search(r"_chr(\d+)\.txt$", p).group(1)) for p in files]
    missing = sorted(set(range(1, 23)) - set(chroms))
    rows = []
    for p in files:
        with open(p) as f:
            for line in f:
                c, snp, bp, a1, a2, b = line.split()
                rows.append((int(c), snp, int(bp), a1, a2, float(b)))
    if not rows:
        sys.exit("No PRS-CS output found. Check the logs/ folder.")
    with open(out, "w") as w:
        w.write("CHR\tSNP\tBP\tA1\tA2\tBETA\n")
        for r in rows:
            w.write("\t".join(map(str, r[:5])) + f"\t{r[5]:.6e}\n")

    beta = np.array([r[5] for r in rows])
    order = np.argsort(-np.abs(beta))
    top = [rows[i] for i in order[:100]]
    mhc = sum(1 for r in top if r[0] == 6 and 25_000_000 <= r[2] <= 34_000_000)
    print(f"\nPRS weights: {len(rows):,} SNPs on {len(files)} chromosomes -> {out}")
    if missing:
        print(f"WARNING: no output for chromosomes {missing}; see logs/")
    print(f"|weight| quantiles 50/90/99/max: "
          + " / ".join(f"{q:.2e}" for q in np.quantile(np.abs(beta), [.5, .9, .99, 1])))
    print(f"top-100 SNPs inside the MHC region (chr6:25-34Mb): {mhc}  "
          "(schizophrenia's strongest locus, so expect a clear share)")
    print("top 10 by |weight|:")
    for r in top[:10]:
        print(f"  chr{r[0]}:{r[2]:,}  {r[1]}  {r[3]}/{r[4]}  {r[5]:+.2e}")


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    {"sumstats": sumstats, "bim": bim, "check": check}[cmd](*args)

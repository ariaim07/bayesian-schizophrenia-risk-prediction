"""
Out-of-sample AUC of the European PRS-CS schizophrenia weights in independent
PGC3 cohorts (East Asian, African American, Latino), from summary statistics only.

  (see usage at the bottom; split into steps so each fits in a few minutes)

pop names: asian, afram, latino (each needs PGC3_SCZ_wave3.<pop>.autosome.public.v3.vcf.tsv.gz
and the matching ldblk_1kg_<eas|afr|amr>/ panel in DATA_DIR).
"""
import gzip, json, sys, time
from pathlib import Path
import numpy as np
import h5py
from scipy.stats import norm

COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}
PANEL = {"asian": "eas", "afram": "afr", "latino": "amr"}
LABEL = {"asian": "East Asian", "afram": "African American", "latino": "Latino"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def orient(a1, a2, ref1, ref2):
    """+1 if (a1,a2) matches the panel's (ref1,ref2), -1 if swapped, 0 if no match."""
    if (a1, a2) == (ref1, ref2) or (COMP.get(a1), COMP.get(a2)) == (ref1, ref2):
        return 1
    if (a1, a2) == (ref2, ref1) or (COMP.get(a1), COMP.get(a2)) == (ref2, ref1):
        return -1
    return 0


def load_weights(path):
    W = {}
    with open(path) as f:
        next(f)
        for line in f:
            c, snp, bp, a1, a2, b = line.split()
            W[snp] = (a1, a2, float(b))
    return W


def load_sumstats(path, keep, min_info=0.8, from_se=False):
    """Returns {snp: (a1, a2, z, f_controls, M)} with M = N*P*(1-P), the effective
    case-control information per SNP."""
    S, Ms = {}, []
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("##"):
                continue
            col = {c.upper(): i for i, c in enumerate(line.lstrip("#").rstrip("\n").split("\t"))}
            break
        use_counts = ("NCAS" in col and "NCON" in col) and not from_se
        for line in f:
            x = line.rstrip("\n").split("\t")
            snp = x[col["ID"]]
            if snp not in keep:
                continue
            a1, a2 = x[col["A1"]].upper(), x[col["A2"]].upper()
            if a1 not in COMP or a2 not in COMP or COMP[a1] == a2:
                continue
            try:
                if "IMPINFO" in col and float(x[col["IMPINFO"]]) < min_info:
                    continue
                b, se = float(x[col["BETA"]]), float(x[col["SE"]])
                f_ctrl = float(x[col["FCON"]])
            except ValueError:
                continue
            if not (se > 0 and 0.01 < f_ctrl < 0.99):
                continue
            if use_counts:
                try:
                    nca, nco = float(x[col["NCAS"]]), float(x[col["NCON"]])
                except ValueError:
                    continue
                if not (nca > 0 and nco > 0):
                    continue
                M = nca * nco / (nca + nco)
            else:
                # logistic regression: Var(beta) ~ 1 / (N P(1-P) 2f(1-f))  =>  M = N P(1-P)
                M = 1.0 / (se ** 2 * 2 * f_ctrl * (1 - f_ctrl))
            S[snp] = (a1, a2, b / se, f_ctrl, M)
            Ms.append(M)
    return S, float(np.median(Ms)), use_counts


def load_panel_info(ld_dir):
    info = {}
    ref = next(Path(ld_dir).glob("snpinfo_*_hm3"))
    with open(ref) as f:
        next(f)
        for line in f:
            c, snp, bp, a1, a2, maf = line.split()
            info[snp] = (a1, a2)
    return info


def prep(pop, data, W, from_se=False, tag=""):
    """Filter the validation GWAS to weighted SNPs; cache as a small TSV."""
    sst = data / f"PGC3_SCZ_wave3.{pop}.autosome.public.v3.vcf.tsv.gz"
    S, M_med, used_counts = load_sumstats(sst, W.keys(), from_se=from_se)
    with open(data / f"val_{pop}{tag}.tsv", "w") as f:
        for snp, (a1, a2, z, fr, n) in S.items():
            f.write(f"{snp}\t{a1}\t{a2}\t{z:.6g}\t{fr:.6g}\t{n:.6g}\n")
    src = "case/control counts" if used_counts else "standard errors"
    json.dump({"M_median": M_med, "n_snps": len(S), "M_source": src},
              open(data / f"val_{pop}{tag}_meta.json", "w"))
    log(f"[{pop}{tag}] {len(S):,} weighted SNPs kept; effective N*P(1-P) = {M_med:,.0f} (from {src})")


def load_val(data, pop, tag=""):
    S = {}
    with open(data / f"val_{pop}{tag}.tsv") as f:
        for line in f:
            snp, a1, a2, z, fr, n = line.split()
            S[snp] = (a1, a2, float(z), float(fr), float(n))
    return S


def chrom_part(pop, data, W, chroms, parts, tag=""):
    """Per-chromosome sums w'r and w'Rw, saved so the work can be split up."""
    ld_dir = data / f"ldblk_1kg_{PANEL[pop]}"
    S, info = load_val(data, pop, tag), load_panel_info(ld_dir)
    parts.mkdir(parents=True, exist_ok=True)
    for chrom in chroms:
        num = den = 0.0; used = 0
        with h5py.File(ld_dir / f"ldblk_1kg_chr{chrom}.hdf5", "r") as h:
            for blk in h.keys():
                snps = [s.decode() for s in h[blk]["snplist"][:]]
                idx, w, r = [], [], []
                for i, s in enumerate(snps):
                    if s not in S or s not in W or s not in info:
                        continue
                    ref1, ref2 = info[s]
                    wa1, wa2, wb = W[s]
                    sa1, sa2, z, fr, n = S[s]
                    ow, oz = orient(wa1, wa2, ref1, ref2), orient(sa1, sa2, ref1, ref2)
                    if ow == 0 or oz == 0:
                        continue
                    sd = np.sqrt(2 * fr * (1 - fr))          # target-population allele SD
                    idx.append(i); w.append(ow * wb * sd); r.append(oz * z / np.sqrt(n))
                if not idx:
                    continue
                R = h[blk]["ldblk"][:][np.ix_(idx, idx)]
                w, r = np.array(w), np.array(r)
                num += float(w @ r); den += float(w @ R @ w); used += len(idx)
        json.dump({"num": num, "den": den, "used": used}, open(parts / f"{pop}{tag}_chr{chrom}.json", "w"))
        log(f"[{pop}{tag}] chr{chrom}: {used:,} SNPs")


def combine(pop, data, parts, n_weights, tag=""):
    """d = mean PRS difference cases vs controls (within-group SD units)
         = sum_j w_j z_j / sqrt(M_j)  /  sqrt(w' R w);   AUC = Phi(d / sqrt 2)."""
    meta = json.load(open(data / f"val_{pop}{tag}_meta.json"))
    got = [json.load(open(parts / f"{pop}{tag}_chr{c}.json")) for c in range(1, 23)]
    num = sum(g["num"] for g in got); den = sum(g["den"] for g in got)
    used = sum(g["used"] for g in got)
    d = num / np.sqrt(den)
    se_d = 1 / np.sqrt(meta["M_median"])
    auc = lambda x: float(norm.cdf(x / np.sqrt(2)))
    out = {"population": LABEL[pop] + (f" ({tag.strip('_')})" if tag else ""),
           "snps_used": used, "snps_in_weights": n_weights,
           "effective_NP(1-P)": meta["M_median"], "M_source": meta["M_source"],
           "d_case_control_shift_sd": float(d),
           "auc": auc(d), "auc_ci95": [auc(d - 1.96 * se_d), auc(d + 1.96 * se_d)]}
    log(f"[{pop}{tag}] AUC {out['auc']:.3f} [{out['auc_ci95'][0]:.3f}, {out['auc_ci95'][1]:.3f}] using {used:,} SNPs")
    return out


if __name__ == "__main__":
    # python validate_real.py prep    POP DATA WEIGHTS [--from-se]
    # python validate_real.py chroms  POP DATA WEIGHTS PARTS c1 [c2 ...] [--from-se]
    # python validate_real.py combine POP DATA WEIGHTS PARTS OUT_JSON [--from-se]
    args = [a for a in sys.argv[1:] if a != "--from-se"]
    from_se = "--from-se" in sys.argv
    tag = "_se" if from_se else ""
    cmd, pop, data, weights = args[0], args[1], Path(args[2]), args[3]
    W = load_weights(weights)
    if cmd == "prep":
        prep(pop, data, W, from_se, tag)
    elif cmd == "chroms":
        chrom_part(pop, data, W, [int(c) for c in args[5:]], Path(args[4]), tag)
    elif cmd == "combine":
        out_path = Path(args[5])
        res = json.load(open(out_path)) if out_path.exists() else []
        new = combine(pop, data, Path(args[4]), len(W), tag)
        res = [r for r in res if r["population"] != new["population"]] + [new]
        json.dump(res, open(out_path, "w"), indent=1)

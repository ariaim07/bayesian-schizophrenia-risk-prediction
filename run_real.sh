#!/usr/bin/env bash
# Real-data PRS-CS run: PGC3 schizophrenia (European) + 1000 Genomes EUR LD panel.
#
#   cd ~/Downloads/scz_bayes/real_data
#   caffeinate -i bash run_real.sh          # caffeinate keeps the Mac awake
#
# Needs ~15 GB free disk, python3, curl. Safe to re-run: finished steps are skipped.
# JOBS = chromosomes run in parallel (default 4; lower it if the Mac runs out of memory).
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
DATA="$DIR/data"; OUT="$DIR/output"; LOGS="$DIR/logs"
mkdir -p "$DATA" "$OUT" "$LOGS"
JOBS="${JOBS:-4}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"          # 1 = use files already in data/ (testing)

SST_URL="https://ndownloader.figshare.com/files/34517828"
SST_GZ="$DATA/PGC3_SCZ_wave3.european.autosome.public.v3.vcf.tsv.gz"
README_URL="https://ndownloader.figshare.com/files/34865091"
LD_URL="https://www.dropbox.com/s/mt6var0z96vb6fv/ldblk_1kg_eur.tar.gz?dl=1"
LD_TGZ="$DATA/ldblk_1kg_eur.tar.gz"
LD_DIR="$DATA/ldblk_1kg_eur"
PRSCS_URL="https://github.com/getian107/PRScs/archive/refs/heads/master.zip"

step() { printf '\n==> %s\n' "$*"; }

# 0. disk space ---------------------------------------------------------------
free_gb=$(df -k "$DIR" | awk 'NR==2 {print int($4/1024/1024)}')
if [ ! -d "$LD_DIR" ] && [ "$free_gb" -lt 15 ]; then
  echo "Only ${free_gb} GB free; need ~15 GB. Free up space and re-run."; exit 1
fi

# 1. python environment -------------------------------------------------------
step "Python environment"
if [ ! -x "$DIR/.venv/bin/python" ]; then
  python3 -m venv "$DIR/.venv"
  "$DIR/.venv/bin/pip" install -q --upgrade pip
  "$DIR/.venv/bin/pip" install -q numpy scipy h5py
fi
PY="$DIR/.venv/bin/python"
"$PY" -c "import numpy, scipy, h5py; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'h5py', h5py.__version__)"

# 2. official PRS-CS ----------------------------------------------------------
step "PRS-CS code"
if [ ! -f "$DIR/PRScs/PRScs.py" ]; then
  if [ "$SKIP_DOWNLOAD" = 1 ]; then echo "PRScs/ missing"; exit 1; fi
  curl -fL --retry 3 -o "$DATA/prscs.zip" "$PRSCS_URL"
  (cd "$DIR" && unzip -q -o "$DATA/prscs.zip" && rm -rf PRScs && mv PRScs-master PRScs)
fi
# NumPy 2 refuses float() / "%f" on 1-element arrays, which PRS-CS relies on.
# Small compatibility patches; the maths is unchanged.
"$PY" - "$DIR/PRScs" <<'EOF'
import sys, pathlib
root = pathlib.Path(sys.argv[1])
patches = [
    ("gigrnd.py", "p = float(p); a = float(a); b = float(b)",
     "p, a, b = (float(x.item()) if hasattr(x, 'item') else float(x) for x in (p, a, b))  # numpy-2 safe"),
    ("mcmc_gtb.py", "sst_dict['A2'], beta_est):", "sst_dict['A2'], beta_est.ravel()):  # numpy-2 safe"),
    ("mcmc_gtb.py", "zip(sst_dict['SNP'], psi_est):", "zip(sst_dict['SNP'], psi_est.ravel()):  # numpy-2 safe"),
    ("mcmc_gtb.py", "% phi_est )", "% float(np.ravel(phi_est)[0]) )  # numpy-2 safe"),
]
for fname, old, new in patches:
    f = root / fname; s = f.read_text()
    if old in s:
        f.write_text(s.replace(old, new)); print("patched", fname)
EOF
echo "ok: $DIR/PRScs"

# 3. downloads ----------------------------------------------------------------
if [ "$SKIP_DOWNLOAD" != 1 ]; then
  step "PGC3 schizophrenia summary statistics (~240 MB)"
  [ -f "$DATA/PGC3_SCZ_README.pdf" ] || curl -fL --retry 3 -o "$DATA/PGC3_SCZ_README.pdf" "$README_URL"
  if [ ! -f "$SST_GZ" ]; then
    curl -fL --retry 3 -C - -o "$SST_GZ.part" "$SST_URL"
    mv "$SST_GZ.part" "$SST_GZ"
  fi

  step "1000 Genomes EUR LD panel (~4.6 GB; the slow part)"
  if [ ! -d "$LD_DIR" ]; then
    [ -f "$LD_TGZ" ] || { curl -fL --retry 3 -C - -o "$LD_TGZ.part" "$LD_URL" && mv "$LD_TGZ.part" "$LD_TGZ"; }
    tar -xzf "$LD_TGZ" -C "$DATA"
    rm -f "$LD_TGZ"
  fi
fi
[ -d "$LD_DIR" ] || { echo "LD panel not found at $LD_DIR"; exit 1; }
echo "LD panel: $(ls "$LD_DIR" | wc -l | tr -d ' ') files"

# 4. prepare inputs -----------------------------------------------------------
step "Preparing summary statistics"
if [ ! -s "$DATA/sumstats_prscs.txt" ] || [ ! -s "$DATA/n_gwas.txt" ]; then
  "$PY" "$DIR/prep.py" sumstats "$SST_GZ" "$DATA/sumstats_prscs.txt" "$DATA/n_gwas.txt"
fi
N_GWAS=$(cat "$DATA/n_gwas.txt"); echo "n_gwas = $N_GWAS"
[ -f "$DATA/ref_snps.bim" ] || "$PY" "$DIR/prep.py" bim "$LD_DIR" "$DATA/ref_snps"

# 5. PRS-CS per chromosome, in parallel ---------------------------------------
step "Running PRS-CS on 22 chromosomes, $JOBS at a time (roughly 1-3 hours total)"
export PY DIR LD_DIR DATA OUT LOGS N_GWAS
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
CHROMS="${CHROMS:-$(seq 1 22)}"
for c in $CHROMS; do echo "$c"; done | xargs -n 1 -P "$JOBS" bash "$DIR/run_chr.sh"

# 6. combine + sanity check ---------------------------------------------------
step "Combining and checking"
"$PY" "$DIR/prep.py" check "$OUT/scz" "$OUT/scz_prscs_weights_all.txt" | tee "$OUT/check.txt"
echo; echo "Done. Weights: $OUT/scz_prscs_weights_all.txt"

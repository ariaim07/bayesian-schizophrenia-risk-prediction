#!/usr/bin/env bash
# Run PRS-CS on one chromosome. Called by run_real.sh (environment variables set there).
c="$1"
if ls "$OUT"/scz_pst_eff_*_chr"$c".txt >/dev/null 2>&1; then
  echo "chr$c already done"; exit 0
fi
echo "chr$c started $(date +%H:%M)"
if "$PY" "$DIR/PRScs/PRScs.py" --ref_dir="$LD_DIR" --bim_prefix="$DATA/ref_snps" \
     --sst_file="$DATA/sumstats_prscs.txt" --n_gwas="$N_GWAS" --out_dir="$OUT/scz" \
     --chrom="$c" --seed=7 > "$LOGS/chr$c.log" 2>&1; then
  echo "chr$c finished $(date +%H:%M)"
else
  echo "chr$c FAILED, see logs/chr$c.log"
fi

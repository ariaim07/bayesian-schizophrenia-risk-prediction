#!/usr/bin/env bash
# Downloads independent (non-European) PGC3 schizophrenia GWAS + matching LD panels,
# used to validate the European PRS-CS weights out of sample.
#   cd ~/Downloads/scz_bayes/real_data && bash fetch_validation.sh
# ~13 GB download; safe to re-run (finished files are skipped).
set -euo pipefail
DATA="$(cd "$(dirname "$0")" && pwd)/data"; mkdir -p "$DATA"

get() {  # url, output file
  if [ -s "$2" ]; then echo "have $(basename "$2")"; return; fi
  echo "downloading $(basename "$2")"
  curl -fL --retry 3 -C - -o "$2.part" "$1" && mv "$2.part" "$2"
}

get https://ndownloader.figshare.com/files/34517804 "$DATA/PGC3_SCZ_wave3.asian.autosome.public.v3.vcf.tsv.gz"
get https://ndownloader.figshare.com/files/34517801 "$DATA/PGC3_SCZ_wave3.afram.autosome.public.v3.vcf.tsv.gz"
get https://ndownloader.figshare.com/files/34517855 "$DATA/PGC3_SCZ_wave3.latino.autosome.public.v3.vcf.tsv.gz"

for pop in eas:7ek4lwwf2b7f749 afr:mq94h1q9uuhun1h amr:uv5ydr4uv528lca; do
  name=${pop%%:*}; key=${pop##*:}
  dir="$DATA/ldblk_1kg_$name"
  if [ -f "$dir/ldblk_1kg_chr22.hdf5" ]; then echo "have LD panel $name"; continue; fi
  get "https://www.dropbox.com/s/$key/ldblk_1kg_$name.tar.gz?dl=1" "$DATA/ldblk_1kg_$name.tar.gz"
  echo "extracting $name"
  tar -xzf "$DATA/ldblk_1kg_$name.tar.gz" -C "$DATA" && rm -f "$DATA/ldblk_1kg_$name.tar.gz"
done
echo "All validation data downloaded."

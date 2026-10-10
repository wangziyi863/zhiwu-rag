#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
ABO_ROOT="${1:-${ABO_ROOT:-$ZHIWU_ROOT/data/ABO}}"
mkdir -p "$ABO_ROOT/_downloads"
for name in abo-listings.tar abo-images-small.tar; do
  archive="$ABO_ROOT/_downloads/$name"
  if [[ ! -f "$archive" ]]; then
    curl --fail --location --retry 3 --continue-at - --output "$archive.part" \
      "https://amazon-berkeley-objects.s3.amazonaws.com/archives/$name"
    tar -tf "$archive.part" >/dev/null
    mv -- "$archive.part" "$archive"
  fi
  tar -tf "$archive" >/dev/null
  tar -xf "$archive" -C "$ABO_ROOT" --no-same-owner
done
echo "ABO files extracted to $ABO_ROOT"
echo 'Next: bash tools/prepare_abo_rag.sh /absolute/path/to/ABO'

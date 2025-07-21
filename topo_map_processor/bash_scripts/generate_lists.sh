#!/bin/bash

prog_name=${COMMNAND_NAME:-$(basename "$0")}
release_base=$1
ext=$2

if [ -z "$release_base" ] || [ -z "$ext" ]; then
  echo "Usage: $prog_name <release_base> <extension>"
  exit 1
fi

#owner=$(gh repo view --json owner -q .owner.login)
#name=$(gh repo view --json name -q .name)
#repo="$owner/$name"

echo "getting file list"
echo "name,size,url" > listing_files.csv

# Get all releases, then filter for the base release and any -extra releases.
releases_to_process=$(gh release list --repo "$repo" --json tagName -q '.[].tagName' | grep -E "^${release_base}(-extra[0-9]+)?$")
releases_to_process=$(echo "$releases_to_process" | sort -u)

echo "Will process releases: $releases_to_process"

for release in $releases_to_process; do
  echo "Processing release: $release"
  gh release view "$release" --json assets -q ".assets[] | select(.name | endswith(\"$ext\")) | \"\\(.name),\\(.size),\\(.url)\"" >> listing_files.csv
done

gh release upload "$release_base" listing_files.csv --clobber

rm listing_files.csv


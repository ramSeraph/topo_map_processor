
import argparse
import csv
import glob
import shutil
import subprocess
import sys
from pathlib import Path
from multiprocessing import set_start_method

from pmtiles_mosaic.partition import partition_main

from .retile import retile_main


def run_command(cmd, check=True, cwd=None):
    """Runs a command and prints its output."""
    print(f"Running command: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=True,
            text=True,
            cwd=cwd,
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}", file=sys.stderr)
        if e.stdout:
            print("STDOUT:", file=sys.stderr)
            print(e.stdout, file=sys.stderr)
        if e.stderr:
            print("STDERR:", file=sys.stderr)
            print(e.stderr, file=sys.stderr)
        raise


def cli():
    """Main function to execute the end-to-end retiling process."""

    if sys.platform == 'darwin':
        set_start_method('fork')

    parser = argparse.ArgumentParser(description="End-to-end retiling script.")
    parser.add_argument("-p", "--pmtiles-release", required=True, help="PMTiles release tag")
    parser.add_argument("-g", "--gtiffs-release", required=True, help="GeoTIFFs release tag")
    parser.add_argument("-x", "--pmtiles-prefix", required=True, help="Prefix for PMTiles files")
    parser.add_argument("-l", "--listing-files-tiled", required=True, help="Name of the tiled listing file")
    args = parser.parse_args()

    # Define paths and filenames
    sheets_to_pull_list_outfile = Path("sheets_to_pull_list.txt")
    tiles_dir = Path("staging/tiles/")
    tiffs_dir = Path("staging/gtiffs/")
    from_pmtiles_dir = Path("staging/pmtiles")
    to_pmtiles_dir = Path("export/pmtiles")
    retile_list_file = Path("to_retile.txt")

    from_pmtiles_prefix = from_pmtiles_dir / args.pmtiles_prefix
    to_pmtiles_prefix = to_pmtiles_dir / args.pmtiles_prefix

    # Create directories
    for d in [tiles_dir, tiffs_dir, from_pmtiles_dir, to_pmtiles_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Download listing files
    print("Downloading listing files...")
    run_command(["gh", "release", "download", args.gtiffs_release, "-p", "listing_files.csv", "--clobber"])
    run_command(["gh", "release", "download", args.pmtiles_release, "-p", args.listing_files_tiled, "--clobber", "-O", "listing_files_tiled.csv"])

    # Determine sheets to retile
    print("Determining sheets to retile...")
    with open("listing_files.csv") as f:
        gtiffs_sheets = {row['name'] for row in csv.DictReader(f)}
    with open("listing_files_tiled.csv") as f:
        pmtiles_sheets = {row['name'] for row in csv.DictReader(f)}

    sheets_to_retile = sorted(list(gtiffs_sheets - pmtiles_sheets))
    
    with open(retile_list_file, "w") as f:
        for sheet in sheets_to_retile:
            f.write(f"{sheet}\n")

    print(f"Found {len(sheets_to_retile)} sheets to retile.")
    Path("listing_files_tiled.csv").unlink()

    if not sheets_to_retile:
        print("No sheets to retile found. Cleaning up and exiting.")
        Path("listing_files.csv").unlink()
        retile_list_file.unlink()
        sys.exit(0)

    # Download bounds and original PMTiles
    print("Downloading bounds file...")
    run_command(["gh", "release", "download", args.gtiffs_release, "-p", "bounds.geojson"])
    print("Getting original PMTiles files...")
    run_command(["gh", "release", "download", args.pmtiles_release, "-D", str(from_pmtiles_dir), "-p", f"{args.pmtiles_prefix}*"])

    # Get list of sheets to pull
    print("Getting list of sheets to pull...")
    retile_main([
        "--retile-list-file", str(retile_list_file),
        "--bounds-file", "bounds.geojson",
        "--sheets-to-pull-list-outfile", str(sheets_to_pull_list_outfile),
        "--from-source", f"{from_pmtiles_prefix}*.pmtiles",
    ])

    print("Sheets to pull:")
    print(sheets_to_pull_list_outfile.read_text())

    # Download TIFFs
    print("Downloading TIFFs...")
    with open("listing_files.csv") as f:
        url_map = {row[0]: row[2] for row in csv.reader(f)}

    with open(sheets_to_pull_list_outfile) as f:
        for line in f:
            fname = line.strip()
            if fname in url_map:
                url = url_map[fname]
                print(f"Pulling {fname}")
                run_command(["wget", "-q", "-P", str(tiffs_dir), url])

    sheets_to_pull_list_outfile.unlink()

    # Retile sheets
    print("Retiling the sheets...")
    retile_main([
        "--retile-list-file", str(retile_list_file),
        "--bounds-file", "bounds.geojson",
        "--from-source", f"{from_pmtiles_prefix}*.pmtiles",
        "--tiles-dir", str(tiles_dir),
        "--tiffs-dir", str(tiffs_dir),
    ])

    retile_list_file.unlink()
    Path("bounds.geojson").unlink()

    # Create new PMTiles files
    print("Creating new pmtiles files...")
    partition_main([
        "--from-source", str(tiles_dir),
        "--from-source", f"{from_pmtiles_prefix}*.pmtiles",
        "--to-pmtiles", f"{to_pmtiles_prefix}.pmtiles",
    ])

    # Upload new PMTiles and manage release assets
    print("Deleting old PMTiles files from the release...")
    old_pmtiles = glob.glob(f"{from_pmtiles_dir}/{args.pmtiles_prefix}*")
    for pmtile in old_pmtiles:
        run_command(["gh", "release", "delete-asset", args.pmtiles_release, Path(pmtile).name, "-y"])

    print("Uploading new PMTiles files...")
    new_pmtiles = glob.glob(f"{to_pmtiles_prefix}*")
    run_command(["gh", "release", "upload", args.pmtiles_release, "--clobber"] + new_pmtiles)

    # Handle listing files
    if args.listing_files_tiled != "listing_files.csv":
        print(f"Renaming listing_files.csv to {args.listing_files_tiled}")
        Path("listing_files.csv").rename(args.listing_files_tiled)
    
    print("Uploading new listing file...")
    run_command(["gh", "release", "upload", args.pmtiles_release, args.listing_files_tiled, "--clobber"])
    Path(args.listing_files_tiled).unlink()

    # Cleanup
    print("Cleaning up staging directories...")
    shutil.rmtree(tiles_dir, ignore_errors=True)
    shutil.rmtree(tiffs_dir, ignore_errors=True)
    shutil.rmtree(from_pmtiles_dir, ignore_errors=True)
    shutil.rmtree(to_pmtiles_dir, ignore_errors=True)

    print("Done!")


if __name__ == "__main__":
    cli()

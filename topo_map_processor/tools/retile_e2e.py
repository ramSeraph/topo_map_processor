import argparse
import glob
import json
import shutil
import subprocess
import sys
from pathlib import Path
from multiprocessing import set_start_method

from pmtiles_mosaic.partition import partition_main
from release_tools.download_from_release import main as download_from_release_main

from .retile import retile_main
from .create_force_redo_bounds import main as create_force_redo_bounds_main


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
    parser.add_argument("--bounds-file-tiled", required=True, help="Name of the bounds file in the pmtiles release (the old bounds file)")
    parser.add_argument("--no-cache", action='store_true', required=False, help="Don't cache files locally during pmtiles partitioning")
    args = parser.parse_args()

    # Define paths and filenames
    sheets_to_pull_list_outfile = Path("sheets_to_pull_list.txt")
    tiles_dir = Path("staging/tiles/")
    tiffs_dir = Path("staging/gtiffs/")
    from_pmtiles_dir = Path("staging/pmtiles")
    to_pmtiles_dir = Path("export/pmtiles")
    
    old_bounds_file = Path("old_bounds.geojson")
    new_bounds_file = Path("new_bounds.geojson")
    force_redo_bounds_file = Path("force_redo_bounds.geojson")

    from_pmtiles_prefix = from_pmtiles_dir / args.pmtiles_prefix
    to_pmtiles_prefix = to_pmtiles_dir / args.pmtiles_prefix

    # Create directories
    for d in [tiles_dir, tiffs_dir, from_pmtiles_dir, to_pmtiles_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Download bounds files
    print("Downloading new bounds file from gtiffs release...")
    run_command(["gh", "release", "download", args.gtiffs_release, "-p", "bounds.geojson", "--clobber", "-O", str(new_bounds_file)])
    
    print("Downloading old bounds file from pmtiles release...")
    run_command(["gh", "release", "download", args.pmtiles_release, "-p", args.bounds_file_tiled, "--clobber", "-O", str(old_bounds_file)])

    # Create force redo bounds file
    print("Creating force redo bounds file...")
    create_force_redo_bounds_main([
        "--old-bounds-file", str(old_bounds_file),
        "--new-bounds-file", str(new_bounds_file),
        "--output-file", str(force_redo_bounds_file)
    ])

    # Check if there are any changes
    with open(force_redo_bounds_file) as f:
        force_redo_data = json.load(f)
    
    if not force_redo_data.get("features"):
        print("No changes detected between bounds files. Exiting.")
        sys.exit(0)

    # Download original PMTiles
    print("Getting original PMTiles files...")
    run_command(["gh", "release", "download", args.pmtiles_release, "-D", str(from_pmtiles_dir), "-p", f"{args.pmtiles_prefix}*"])

    # Get list of sheets to pull
    print("Getting list of sheets to pull...")
    retile_main([
        "--bounds-file", str(new_bounds_file),
        "--force-redo-bounds-file", str(force_redo_bounds_file),
        "--sheets-to-pull-list-outfile", str(sheets_to_pull_list_outfile),
        "--from-source", f"{from_pmtiles_prefix}*.pmtiles",
    ])

    print("Sheets to pull:")
    print(sheets_to_pull_list_outfile.read_text())

    # Download TIFFs
    print("Downloading TIFFs...")
    # The download script will report if files are not found, which is expected for dummy sheets.
    if download_from_release_main([
        "--release", args.gtiffs_release,
        "--output-dir", str(tiffs_dir),
        "--file-list", str(sheets_to_pull_list_outfile),
    ]) != 0:
        print("Some TIFFs failed to download. This may be expected for dummy sheets from force redo. Continuing...", file=sys.stderr)

    sheets_to_pull_list_outfile.unlink()

    # Retile sheets
    print("Retiling the sheets...")
    retile_main([
        "--bounds-file", str(new_bounds_file),
        "--force-redo-bounds-file", str(force_redo_bounds_file),
        "--from-source", f"{from_pmtiles_prefix}*.pmtiles",
        "--tiles-dir", str(tiles_dir),
        "--tiffs-dir", str(tiffs_dir),
    ])

    old_bounds_file.unlink()
    force_redo_bounds_file.unlink()

    # Create new PMTiles files
    print("Creating new pmtiles files...")
    partition_main([
        "--from-source", str(tiles_dir),
        "--from-source", f"{from_pmtiles_prefix}*.pmtiles",
        "--to-pmtiles", f"{to_pmtiles_prefix}.pmtiles",
        "--exclude-transparent"
    ] + (["--no-cache"] if args.no_cache else []))

    # Upload new PMTiles and manage release assets
    print("Deleting old PMTiles files from the release...")
    old_pmtiles = glob.glob(f"{from_pmtiles_dir}/{args.pmtiles_prefix}*")
    for pmtile in old_pmtiles:
        run_command(["gh", "release", "delete-asset", args.pmtiles_release, Path(pmtile).name, "-y"])

    print("Uploading new PMTiles files...")
    new_pmtiles = glob.glob(f"{to_pmtiles_prefix}*")
    run_command(["gh", "release", "upload", args.pmtiles_release, "--clobber"] + new_pmtiles)

    # Handle bounds file
    print("Uploading new bounds file to pmtiles release...")
    upload_bounds_file = Path(args.bounds_file_tiled)
    new_bounds_file.rename(upload_bounds_file)
    
    run_command(["gh", "release", "upload", args.pmtiles_release, str(upload_bounds_file), "--clobber"])
    
    upload_bounds_file.unlink()

    # Cleanup
    print("Cleaning up staging directories...")
    shutil.rmtree(tiles_dir, ignore_errors=True)
    shutil.rmtree(tiffs_dir, ignore_errors=True)
    shutil.rmtree(from_pmtiles_dir, ignore_errors=True)
    shutil.rmtree(to_pmtiles_dir, ignore_errors=True)

    print("Done!")


if __name__ == "__main__":
    cli()


import json
import argparse
from pathlib import Path

def cli():
    parser = argparse.ArgumentParser(description="Update bounds in a GeoJSON file.")
    parser.add_argument("bounds-file", help="Input GeoJSON file with bounds.")
    parser.add_argument("bounds-dir", help="Directory containing the individual bounds files")
    parser.add_argument("retile-list-file", help="File containing the list of sheets to update.")

    args = parser.parse_args()

    bounds_file = Path(args.bounds_file)
    bounds_dir = Path(args.bounds_dir)
    retile_list_file = Path(args.retile_list_file)

    existing_map = {}
    if not bounds_file.exists():
        raise FileNotFoundError(f"Bounds file does not exist: {bounds_file}")

    # Read the existing bounds file
    existing_data = json.loads(bounds_file.read_text())
    for feature in existing_data.get('features', []):
        if 'properties' in feature and 'id' in feature['properties']:
            sheet_name = feature['properties']['id']
            existing_map[sheet_name] = feature

    sheets = retile_list_file.read_text().strip().splitlines()
    sheets = [s.strip() for s in sheets if s.strip()]
    for s in sheets:
        bfile = bounds_dir / f'{s}.geojsonl'
        if not bfile.exists():
            raise FileNotFoundError(f"Bounds file for sheet '{s}' does not exist: {bfile}")

        feature = json.loads(bfile.read_text())
        existing_data[s] = feature


    all_feats = list(existing_data.values())

    with open(bounds_file, 'w') as f:
        f.write('{ "type": "FeatureCollection", "features": [\n')
        for i,feat in enumerate(all_feats):
            line = json.dumps(feat)
            if i != len(all_feats) - 1:
                line += ','
            line += '\n'
            f.write(line)
        f.write(']}\n')

if __name__ == "__main__":
    cli()

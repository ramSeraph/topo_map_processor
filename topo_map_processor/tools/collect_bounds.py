
import json
import argparse
from pathlib import Path

def cli():
    parser = argparse.ArgumentParser(description='Collect GeoJSONL files into a single GeoJSON file.')
    parser.add_argument('-b', '--bounds-dir', required=True, help='Directory containing GeoJSONL files')
    parser.add_argument('-o', '--output-file', required=True, help='Output GeoJSON file path')
    parser.add_argument('-u', '--update', action='store_true', help='Update the output file in place')
    parser.add_argument('-d', '--delete', nargs='+', help='List of ids to delete from the output file (only in update mode)')

    args = parser.parse_args()

    if args.delete and not args.update:
        raise parser.error("--delete can only be used with --update")

    feat_map = {}
    if args.update:
        output_file = Path(args.output_file)
        if not output_file.exists():
            raise FileNotFoundError(f"Output file does not exist for updating: {output_file}")

        existing_data = json.loads(output_file.read_text())
        for feature in existing_data.get('features', []):
            if 'properties' in feature and 'id' in feature['properties']:
                sheet_name = feature['properties']['id']
                feat_map[sheet_name] = feature
    
    if args.delete:
        for sheet_id in args.delete:
            if sheet_id in feat_map:
                del feat_map[sheet_id]

    bounds_dir = Path(args.bounds_dir)

    paths = list(bounds_dir.glob('*.geojsonl'))
    for p in paths:
        feat = json.loads(p.read_text())
        if 'properties' in feat and 'id' in feat['properties']:
            sheet_name = feat['properties']['id']
            feat_map[sheet_name] = feat

    with open(args.output_file, 'w') as f:
        f.write('{ "type": "FeatureCollection", "features": [\n')

        feats = list(feat_map.values())
        for i,feat in enumerate(feats):
            line = json.dumps(feat)

            if i != len(feats) - 1:
                line += ','
            line += '\n'
            f.write(line)

        f.write(']}\n')

if __name__ == "__main__":
    cli()


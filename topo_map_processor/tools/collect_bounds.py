
import json
import argparse
from pathlib import Path

def cli():
    parser = argparse.ArgumentParser(description='Collect GeoJSONL files into a single GeoJSON file.')
    parser.add_argument('--bounds-dir', required=True, help='Directory containing GeoJSONL files')
    parser.add_argument('--output-file', required=True, help='Output GeoJSON file path')
    parser.add_argument('--preexisting-file', help='preexisting GeoJSON file to base things on', default=None)

    args = parser.parse_args()

    feat_map = {}
    if args.preexisting_file:
        preexisting_file = Path(args.preexisting_file)
        if not preexisting_file.exists():
            raise FileNotFoundError(f"Preexisting file does not exist: {preexisting_file}")

        existing_data = json.loads(preexisting_file.read_text())
        for feature in existing_data.get('features', []):
            if 'properties' in feature and 'id' in feature['properties']:
                sheet_name = feature['properties']['id']
                feat_map[sheet_name] = feature

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


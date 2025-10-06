import argparse
import json
import sys
from pathlib import Path

def get_sheet_map(bounds_file):
    """Reads a bounds file and returns a map of sheet ID to feature."""
    with open(bounds_file, 'r') as f:
        data = json.load(f)
    
    sheet_map = {}
    for feature in data.get('features', []):
        if 'properties' in feature and 'id' in feature['properties']:
            sheet_id = feature['properties']['id']
            sheet_map[sheet_id] = feature
    return sheet_map

def main(argv):
    parser = argparse.ArgumentParser(description='Create a force redo bounds file by comparing old and new bounds.')
    parser.add_argument('--old-bounds-file', required=True, help='Path to the old GeoJSON bounds file.')
    parser.add_argument('--new-bounds-file', required=True, help='Path to the new GeoJSON bounds file.')
    parser.add_argument('--output-file', required=True, help='Path to write the output GeoJSON file with force redo bounds.')

    args = parser.parse_args(argv)

    old_sheets = get_sheet_map(args.old_bounds_file)
    new_sheets = get_sheet_map(args.new_bounds_file)

    old_ids = set(old_sheets.keys())
    new_ids = set(new_sheets.keys())

    added_ids = new_ids - old_ids
    deleted_ids = old_ids - new_ids
    common_ids = old_ids.intersection(new_ids)

    output_features = []

    # Added sheets
    for sheet_id in added_ids:
        feature = new_sheets[sheet_id]
        output_features.append({
            "type": "Feature",
            "geometry": feature['geometry'],
            "properties": { "reason": f"Sheet {sheet_id} added" }
        })

    # Deleted sheets
    for sheet_id in deleted_ids:
        feature = old_sheets[sheet_id]
        output_features.append({
            "type": "Feature",
            "geometry": feature['geometry'],
            "properties": { "reason": f"Sheet {sheet_id} deleted" }
        })

    # Common sheets (check for modifications)
    for sheet_id in common_ids:
        old_feature = old_sheets[sheet_id]
        new_feature = new_sheets[sheet_id]
        old_geom = old_feature.get('geometry')
        new_geom = new_feature.get('geometry')
        geom_changed = json.dumps(old_geom, sort_keys=True) != json.dumps(new_geom, sort_keys=True)
        old_digest = old_feature.get('properties', {}).get('digest')
        new_digest = new_feature.get('properties', {}).get('digest')
        digest_changed = old_digest != new_digest
        if geom_changed:
            output_features.append({
                "type": "Feature",
                "geometry": old_geom,
                "properties": { "reason": f"Sheet {sheet_id} geometry changed (old)" }
            })
            output_features.append({
                "type": "Feature",
                "geometry": new_geom,
                "properties": { "reason": f"Sheet {sheet_id} geometry changed (new)" }
            })
        elif digest_changed:
            output_features.append({
                "type": "Feature",
                "geometry": new_geom,
                "properties": { "reason": f"Sheet {sheet_id} digest changed" }
            })

    with open(args.output_file, 'w') as f:
        f.write('{ "type": "FeatureCollection", "features": [\n')
        for i, feature in enumerate(output_features):
            line = json.dumps(feature)
            if i < len(output_features) - 1:
                line += ','
            line += '\n'
            f.write(line)
        f.write(']}\n')

    print(f"Created force redo bounds file at {args.output_file} with {len(output_features)} features.")

def cli():
    main(sys.argv[1:])

if __name__ == "__main__":
    cli()
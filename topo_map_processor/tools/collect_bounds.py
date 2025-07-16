
import argparse

def cli():
    parser = argparse.ArgumentParser(description='Collect GeoJSONL files into a single GeoJSON file.')
    parser.add_argument('--bounds-dir', required=True, help='Directory containing GeoJSONL files')
    parser.add_argument('--output-file', required=True, help='Output GeoJSON file path')

    args = parser.parse_args()

    paths = list(args.bounds_dir.glob('*.geojsonl'))
    with open(args.output_file, 'w') as f:
        f.write('{ "type": "FeatureCollection", "features": [\n')
        for i,p in enumerate(paths):
            line = p.read_text().strip('\n')

            if i != len(paths) - 1:
                line += ','
            line += '\n'
            f.write(line)
        f.write(']}\n')

if __name__ == "__main__":
    cli()


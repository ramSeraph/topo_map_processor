
import sys
from pathlib import Path

def cli():
    bounds_dir = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    paths = list(bounds_dir.glob('*.geojsonl'))
    with open(output_file, 'w') as f:
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


import re
import os
import sys
import glob
import json
import argparse
import time
import subprocess

from pathlib import Path
from multiprocessing import set_start_method, cpu_count
from pprint import pprint

from osgeo_utils.gdal2tiles import submain as gdal2tiles_main

from .utils import relax_open_file_limit

SUPPORTED_FORMATS = ['webp', 'jpeg', 'png']

def run_external(cmd):
    print(f'running cmd - {cmd}')
    start = time.time()
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    end = time.time()
    print(f'STDOUT: {res.stdout}')
    print(f'STDERR: {res.stderr}')
    print(f'command took {end - start} secs to run')
    if res.returncode != 0:
        raise Exception(f'command {cmd} failed')

def convert_paths_in_vrt(vrt_file):
    vrt_dirname = str(vrt_file.resolve().parent)
    vrt_text = vrt_file.read_text()
    replaced = re.sub(
        r'<SourceFilename relativeToVRT="1">(.*)</SourceFilename>',
        rf'<SourceFilename relativeToVRT="0">{vrt_dirname}/\1</SourceFilename>',
        vrt_text
    )
    vrt_file.write_text(replaced)

def get_tiler_cmd_params(tile_extension, tile_quality):

    if tile_extension == 'webp':
        return [
            '--tiledriver', 'WEBP',
            '--webp-quality', str(tile_quality),
        ]

    if tile_extension == 'jpeg':
        return [
            '--tiledriver', 'JPEG',
            '--jpeg_quality', str(tile_quality),
        ]

    if tile_extension == 'png':
        return [
            '--tiledriver', 'PNG',
        ]

    raise ValueError(f'Unsupported tile extension: {tile_extension}, {SUPPORTED_FORMATS=}')

def get_format(tile_extension):
    if tile_extension == 'webp':
        return 'webp'
    if tile_extension == 'jpeg':
        return 'jpeg'
    if tile_extension == 'png':
        return 'png'
    raise ValueError(f'Unsupported tile extension: {tile_extension}, {SUPPORTED_FORMATS=}')


def cli():
    if sys.platform == 'darwin':
        set_start_method('fork')

    parser = argparse.ArgumentParser(description='Tile the GTiffs and create a TileJSON file')
    parser.add_argument('--tiles-dir', required=True, help='Directory to store the tiles')
    parser.add_argument('--tiffs-dir', required=True, help='Directory with GTiffs to tile')
    parser.add_argument('--max-zoom', type=int, required=True, help='Maximum zoom level for tiling')
    parser.add_argument('--min-zoom', type=int, default=0, help='Minimum zoom level for tiling')
    parser.add_argument('--tile-extension', type=str, default='webp', choices=SUPPORTED_FORMATS, help='Tile file extension (default: webp)')
    parser.add_argument('--tile-quality', type=int, default=75, help='Compression quality of the tiles (default: 75)')
    parser.add_argument('--num-parallel', type=int, default=cpu_count(), help='Number of parallel processes to use for tiling (default: number of CPU cores)')
    parser.add_argument('--name', required=True, help='Name of the mosaic.')
    parser.add_argument('--description', required=True, help='Description of the mosaic.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--attribution', help='Attribution text for the mosaic.')
    group.add_argument('--attribution-file', help='File containing attribution text for the mosaic.')

    args = parser.parse_args()

    if args.attribution_file:
        attribution_file = Path(args.attribution_file)
        if not attribution_file.exists():
            parser.error(f'Attribution file {args.attribution_file} does not exist')
        attribution_text = attribution_file.read_text().strip()
    else:
        attribution_text = args.attribution

    relax_open_file_limit()

    tiles_dir = Path(args.tiles_dir)
    tiles_dir.mkdir(parents=True, exist_ok=True)

    file_names = list(glob.glob(f'{args.tiffs_dir}/*.tif'))
    file_names = [ str(Path(f).resolve()) for f in file_names ]
    print(f'total files: {len(file_names)}')

    file_list_file = Path('files_to_tile.txt')
    file_list_file.write_text('\n'.join(file_names))

    vrt_file = Path('files_to_tile.vrt')
    if vrt_file.exists():
        print(f'vrt file {vrt_file} already exists, deleting it')
        vrt_file.unlink()

    # create vrt file
    run_external(f'gdalbuildvrt -input_file_list {str(file_list_file)} {str(vrt_file)}')
    convert_paths_in_vrt(vrt_file)
    file_list_file.unlink()

    print('start tiling')
    os.environ['GDAL_CACHEMAX'] = '2048'
    os.environ['GDAL_MAX_DATASET_POOL_SIZE'] = '5000'
    os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'TRUE'
    #os.environ['VRT_SHARED_SOURCE'] = '1'
    #os.environ['GTIFF_VIRTUAL_MEM_IO'] = 'TRUE'
    cmd = [
        'gdal2tiles.py',
        '-r', 'antialias',
        '--verbose',
        '--exclude', 
        '--resume', 
        '--xyz', 
        '--processes', f'{args.num_parallel}', 
        '-z', f'{args.min_zoom}-{args.max_zoom}',
    ] + get_tiler_cmd_params(args.tile_extension, args.tile_quality) + [
        str(vrt_file), str(tiles_dir)
    ]
    gdal2tiles_main(cmd, called_from_main=True)

    vrt_file.unlink()

    metadata_file = tiles_dir / 'tiles.json'

    fmt = get_format(args.tile_extension)

    # it is not really a proper tilejson spec.. it is the basic things needed to populate a pmtiles metadata
    metadata = {
        'type': 'baselayer',
        'format': fmt,
        'attribution': attribution_text,
        'description': args.description,
        'name': args.name,
        'version': '1',
        'maxzoom': args.max_zoom,
        'minzoom': args.min_zoom,
    }

    print(f'writing metadata to {metadata_file}')
    pprint(metadata)
    metadata_file.write_text(json.dumps(metadata, indent=2))

    print('All Done!!')

if __name__ == '__main__':
    cli()

import re
import os
import glob
import argparse

import time
import subprocess

from pathlib import Path
from multiprocessing import set_start_method, cpu_count

from osgeo_utils.gdal2tiles import main as gdal2tiles_main

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

def cli():
    if sys.platform == 'darwin':
        set_start_method('fork')

    parser = argparse.ArgumentParser(description='Tile the GTiffs')
    parser.add_argument('--tiles-dir', required=True, help='Directory to store the tiles')
    parser.add_argument('--tiffs-dir', required=True, help='Directory with GTiffs to tile')
    parser.add_argument('--max-zoom', type=int, required=True, help='Maximum zoom level for tiling')
    parser.add_argument('--min-zoom', type=int, default=0, help='Minimum zoom level for tiling')
    parser.add_argument('--num-parallel', type=int, default=cpu_count(), help='Number of parallel processes to use for tiling (default: number of CPU cores)')

    args = parser.parse_args()

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
        '--processes=8', 
        '-z', f'{args.min_zoom}-{args.max_zoom}',
        '--tiledriver', 'WEBP',
        '--webp-quality', '75',
        str(vrt_file), str(tiles_dir)
    ]
    print(cmd)
    gdal2tiles_main(cmd, calling_from_main=True)

    vrt_file.unlink()

    print('All Done!!')

if __name__ == '__main__':
    cli()



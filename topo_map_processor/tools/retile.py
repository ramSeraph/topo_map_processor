# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mercantile",
#     "pmtiles",
# ]
# ///

import os
import re
import json
import argparse

import time
import subprocess

from pathlib import Path


import mercantile
from shapely.geometry import mapping

from osgeo_utils.gdal2tiles import main as gdal2tiles_main
from osgeo_utils.gdal2tiles import create_overview_tile, TileJobInfo, GDAL2Tiles


from tile_sources import PartitionedPMTilesSource, MissingTileError

WEBP_QUALITY = 75



class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


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
    # <SourceFilename relativeToVRT="1">40M_15.tif</SourceFilename>
    vrt_dirname = str(vrt_file.resolve().parent)
    vrt_text = vrt_file.read_text()
    replaced = re.sub(
        r'<SourceFilename relativeToVRT="1">(.*)</SourceFilename>',
        rf'<SourceFilename relativeToVRT="0">{vrt_dirname}/\1</SourceFilename>',
        vrt_text
    )
    vrt_file.write_text(replaced)

    
def create_base_tiles(inp_file, output_dir, zoom_levels):
    print('start tiling')
    os.environ['GDAL_CACHEMAX'] = '2048'
    os.environ['GDAL_MAX_DATASET_POOL_SIZE'] = '5000'
    os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'TRUE'
    #os.environ['VRT_SHARED_SOURCE'] = '1'
    #os.environ['GTIFF_VIRTUAL_MEM_IO'] = 'TRUE'
    gdal2tiles_main(['gdal2tiles.py',
                     '-r', 'antialias',
                     '--verbose',
                     '-w', 'none',
                     '--exclude', 
                     '--resume', 
                     '--xyz', 
                     '--processes=8', 
                     '-z', zoom_levels,
                     '--tiledriver', 'WEBP',
                     '--webp-quality', f'{WEBP_QUALITY}',
                     inp_file, output_dir])



pmtiles_reader = None
def get_pmtiles_reader(from_pmtiles_prefix):
    global pmtiles_reader
    if pmtiles_reader is None:
        pmtiles_reader = PartitionedPMTilesSource(from_pmtiles_prefix)
    return pmtiles_reader


def pull_from_pmtiles(file, from_pmtiles_prefix):
    reader = get_pmtiles_reader(from_pmtiles_prefix)
    fname = str(file)
    pieces = fname.split('/')
    tile = mercantile.Tile(x=int(pieces[-2]),
                           y=int(pieces[-1].replace('.webp', '')),
                           z=int(pieces[-3]))
    try:
        t_data = reader.get_tile_data(tile)
    except MissingTileError:
        return
    file.parent.mkdir(exist_ok=True, parents=True)
    file.write_bytes(t_data)


def get_tile_file(tile, tiles_dir):
    return f'{tiles_dir}/{tile.z}/{tile.x}/{tile.y}.webp'

def check_sheets(sheets_to_pull, tiffs_dir):
    for sheet_no in sheets_to_pull:
        to = tiffs_dir.joinpath(f'{sheet_no}.tif')
        if not to.exists():
            raise Exception(f'missing file {to}')

def copy_tiles_over(tiles_to_pull, tiles_dir, from_pmtiles_prefix):
    for tile in tiles_to_pull:
        to = Path(get_tile_file(tile, tiles_dir))
        if to.exists():
            continue
        pull_from_pmtiles(to, from_pmtiles_prefix)


def create_upper_tiles(tiles_to_create, tiles_dir):
    options = AttrDict({
        'resume': True,
        'verbose': False,
        'quiet': False,
        'xyz': True,
        'exclude_transparent': True,
        'profile': 'mercator',
        'resampling': 'antialias',
        'tiledriver': 'WEBP',
        'webp_quality': WEBP_QUALITY,
        'webp_lossless': False
    })
    tile_job_info = TileJobInfo(
        tile_driver='WEBP',
        nb_data_bands=3,
        tile_size=256,
        tile_extension='webp',
        kml=False
    )

    for tile in tiles_to_create:
        ctiles = mercantile.children(tile)
        tiles_dir.joinpath(str(tile.z), str(tile.x)).mkdir(parents=True, exist_ok=True)
        base_tile_group = [ (t.x, GDAL2Tiles.getYTile(t.y, t.z, options)) for t in ctiles ]
        #print(f'{tile=}, {base_tile_group=}')
        create_overview_tile(z + 1, output_folder=str(tiles_dir), tile_job_info=tile_job_info, options=options, base_tiles=base_tile_group)
 

def get_sheet_data(bounds_fname):

    index_file = Path(bounds_fname)
    if not index_file.exists():
        raise Exception(F'missing index file at {bounds_fname}')

    index_data = json.loads(index_file.read_text())

    sheets_to_box = {}
    for f in index_data['features']:
        sheet_no = f['properties']['id']
        geom = mapping(f['geometry'])
        xmin, ymin, xmax, ymax = geom.bounds
        box = mercantile.LngLatBbox(xmin, ymin, xmax, ymax)
        sheets_to_box[sheet_no] = box

    return sheets_to_box


def get_base_tile_sheet_mappings(sheets_to_box, base_zoom):
    sheets_to_base_tiles = {}
    base_tiles_to_sheets = {}

    for sheet_no, box in sheets_to_box.items():
        tiles = set(mercantile.tiles(box.west, box.south, box.east, box.north, [base_zoom])) 
        sheets_to_base_tiles[sheet_no] = tiles
        for tile in tiles:
            if tile not in base_tiles_to_sheets:
                base_tiles_to_sheets[tile] = set()
            base_tiles_to_sheets[tile].add(sheet_no)

    return sheets_to_base_tiles, base_tiles_to_sheets


def delete_unwanted_tiles(tiles_to_keep, z, tiles_dir):
    delete_count = 0
    for file in tiles_dir.glob(f'{z}/*/*.webp'):
        y = int(file.name[:-5])
        x = int(file.parent.name)
            
        disk_tile = mercantile.Tile(z=z, x=x, y=y)
        if disk_tile not in tiles_to_keep:
            #print(f'deleting {disk_tile}')
            file.unlink()
            delete_count += 1
    print(f'deleted {delete_count} files at level {z}')


def create_vrt_file(sheets, tiffs_dir):
    vrt_file = Path(f'{tiffs_dir}/combined.vrt')
    if vrt_file.exists():
        return vrt_file
    tiff_list = [ tiffs_dir.joinpath(f'{p_sheet}.tif').resolve() for p_sheet in sheets ]
    tiff_list = [ str(f) for f in tiff_list if f.exists() ]

    tiff_list_str = ' '.join(tiff_list)

    run_external(f'gdalbuildvrt {vrt_file} {tiff_list_str}')
    convert_paths_in_vrt(vrt_file)

    return vrt_file

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--retile-list-file', required=True, help='File containing list of sheets to retile')
    parser.add_argument('--bounds-file', required=True, help='Geojson file containing list of available sheets and their georaphic bounds')
    parser.add_argument('--min-zoom', default=0, type=int)
    parser.add_argument('--max-zoom', required=True, type=int)
    parser.add_argument('--sheets-to-pull-list-outfile', default=None,
                        help='Output into which we write the list of sheet that need to be pulled, if set the script ends after created the list file')
    parser.add_argument('--from-pmtiles-prefix')
    parser.add_argument('--tiles-dir', required=True)
    parser.add_argument('--tiffs-dir', required=True)

    args = parser.parse_args()
    if not args.sheets_to_pull_list_outfile:
        missing = []
        if not args.from_pmtiles_prefix:
            missing.append('--from-pmtiles-prefix')
        
        if missing:
            parser.error(f"The following arguments are required when --sheets-to-pull-list-outfile is not provided: {', '.join(missing)}")

    retile_sheets = Path(args.retile_list_file).read_text().split('\n')
    retile_sheets = set([ r.strip() for r in retile_sheets if r.strip() != '' ])

    print('getting base tiles to sheet mapping')
    sheets_to_box = get_sheet_data(args.bounds_file)
    sheets_to_base_tiles, base_tiles_to_sheets = get_base_tile_sheet_mappings(sheets_to_box, args.max_zoom)

    all_affected_tiles = set()

    print('calculating sheets to pull')
    affected_base_tiles = set()
    for sheet_no in retile_sheets:
        affected_base_tiles.update(sheets_to_base_tiles[sheet_no])
    all_affected_tiles.update(affected_base_tiles)

    sheets_to_pull = set()
    for tile in affected_base_tiles:
        to_add = base_tiles_to_sheets[tile]
        for sheet in to_add:
            sheets_to_pull.add(sheet)

    Path(args.tiffs_dir).mkdir(exist_ok=True, parents=True)
    Path(args.tiles_dir).mkdir(exist_ok=True, parents=True)
    print(f'{sheets_to_pull=}')
    if args.sheets_to_pull_list_outfile is not None:
        Path(args.sheets_to_pull_list_outfile).write_text('\n'.join(sheets_to_pull) + '\n')
        exit(0)


    print('check the sheets availability')
    check_sheets(sheets_to_pull, args.tiffs_dir)

    print('creating vrt file from sheets involved')
    vrt_file = create_vrt_file(sheets_to_pull, args.tiffs_dir)

    print('creating tiles for base zoom with a vrt')
    create_base_tiles(f'{vrt_file}', str(args.tiles_dir), f'{args.max_zoom}')

    print('deleting unwanted base tiles')
    delete_unwanted_tiles(affected_base_tiles, args.max_zoom, args.tiles_dir)

    prev_affected_tiles = affected_base_tiles
    for z in range(args.max_zoom-1, args.min_zoom-1, -1):
        print(f'handling level {z}')

        curr_affected_tiles = set()
        for tile in prev_affected_tiles:
            curr_affected_tiles.add(mercantile.parent(tile))
        all_affected_tiles.update(curr_affected_tiles)

        child_tiles_to_pull = set()
        for ptile in curr_affected_tiles:
            for ctile in mercantile.children(ptile):
                if ctile not in prev_affected_tiles:
                    child_tiles_to_pull.add(ctile)

        print('copying additional child tiles required for curr level')
        copy_tiles_over(child_tiles_to_pull, args.tiles_dir, args.from_pmtiles_prefix)

        print('creating tiles for current level')
        create_upper_tiles(curr_affected_tiles, args.tiles_dir)

        print('removing unwanted child tiles')
        delete_unwanted_tiles(prev_affected_tiles, z+1, args.tiles_dir)

        prev_affected_tiles = curr_affected_tiles

    print('All Done!!!')


if __name__ == '__main__':
    cli()




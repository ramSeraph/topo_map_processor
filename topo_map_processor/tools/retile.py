
import os
import re
import sys
import json
import argparse

import time
import subprocess

from pathlib import Path
from functools import partial
from multiprocessing import cpu_count, Pool, set_start_method


import mercantile
from shapely.geometry import shape

from osgeo_utils.gdal2tiles import (
    submain as gdal2tiles_main,
    create_overview_tile,
    TileJobInfo, 
    GDAL2Tiles, 
    DividedCache,
)

from pmtiles_mosaic.tile_sources import create_source_from_paths, MissingTileError

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


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class DiskTilesHandler:
    def __init__(self, tiles_dir, tile_extension):
        self.dir = Path(tiles_dir)
        self.ext = tile_extension

    def init(self):
        self.dir.mkdir(parents=True, exist_ok=True)

    def preprare_dir(self, tile):
        tile_dir = self.dir.joinpath(f'{tile.z}/{tile.x}')
        tile_dir.mkdir(parents=True, exist_ok=True)

    def get_tile_file(self, tile):
        return self.dir.joinpath(f'{tile.z}/{tile.x}/{tile.y}.{self.ext}')

    def all_z_tiles(self, z):
        z_dir = self.dir.joinpath(str(z))
        for f in z_dir.glob(f'*/*.{self.ext}'):
            parts = f.parts
            x = int(parts[-2])
            y = int(parts[-1].replace(f'.{self.ext}', ''))
            yield mercantile.Tile(x=x, y=y, z=z)

    def del_tile(self, tile):
        tile_file = self.get_tile_file(tile)
        if tile_file.exists():
            tile_file.unlink()

    def delete_unwanted_tiles(self, tiles_to_keep, z):
        delete_count = 0
    
        for tile in self.all_z_tiles(z):
            if tile not in tiles_to_keep:
                self.del_tile(tile)
                delete_count += 1
    
        print(f'deleted {delete_count} files at level {z}')

    def save_metadata(self, metadata):
        metadata_file = self.dir.joinpath('tiles.json')
        metadata_file.write_text(json.dumps(metadata, indent=2))

 
class Tiler:
    def __init__(self, tiles_dir, tiffs_dir,
                 orig_tile_source, tile_extension, tile_quality, pool):

        self.disk_handler = DiskTilesHandler(tiles_dir, tile_extension)
        self.orig_tile_source = orig_tile_source
        self.tiffs_dir = tiffs_dir
        self.tile_extension = tile_extension
        self.tile_quality = tile_quality
        self.pool = pool

    def convert_paths_in_vrt(self, vrt_file):
        # <SourceFilename relativeToVRT="1">40M_15.tif</SourceFilename>
        vrt_dirname = str(vrt_file.resolve().parent)
        vrt_text = vrt_file.read_text()
        replaced = re.sub(
            r'<SourceFilename relativeToVRT="1">(.*)</SourceFilename>',
            rf'<SourceFilename relativeToVRT="0">{vrt_dirname}/\1</SourceFilename>',
            vrt_text
        )
        vrt_file.write_text(replaced)


    def create_vrt_file(self, sheets):
        vrt_file = self.tiffs_dir.joinpath('combined.vrt')
        if vrt_file.exists():
            return vrt_file
        tiff_list = [ self.tiffs_dir.joinpath(f'{sheet}').resolve() for sheet in sheets ]
        tiff_list = [ str(f) for f in tiff_list if f.exists() ]

        tiff_list_str = ' '.join(tiff_list)

        run_external(f'gdalbuildvrt {vrt_file} {tiff_list_str}')
        self.convert_paths_in_vrt(vrt_file)

        return vrt_file

    def get_tiler_cmd_params(self):
        params = [
            '--tiledriver', self.get_tile_driver()
        ]

        if self.tile_extension == 'webp':
            return params + [
                '--webp-quality', str(self.tile_quality),
            ]

        if self.tile_extension == 'jpeg':
            return params + [
                '--jpeg_quality', str(self.tile_quality),
            ]

        if self.tile_extension == 'png':
            return params

        raise ValueError(f'Unsupported tile extension: {self.tile_extension}')

    def get_tile_driver(self):
        if self.tile_extension == 'webp':
            return 'WEBP'

        if self.tile_extension == 'jpeg':
            return 'JPEG'

        if self.tile_extension == 'png':
            return 'PNG'

        raise ValueError(f'Unsupported tile extension: {self.tile_extension}')

    def get_tiler_options(self):
        if self.tile_extension == 'webp':
            return {
                'webp_quality': self.tile_quality,
                'webp_lossless': False,
            }

        if self.tile_extension == 'jpeg':
            return {
                'jpeg_quality': self.tile_quality,
            }

        if self.tile_extension == 'png':
            return {}

        raise ValueError(f'Unsupported tile extension: {self.tile_extension}')

    def create_base_tiles(self, sheets, max_zoom):
        print('creating vrt file from sheets involved')
        vrt_file = self.create_vrt_file(sheets)

        print('start creating base tiles')
        cmd = [
            'gdal2tiles.py',
            '-r', 'antialias',
            '--verbose',
            '-w', 'none',
            '--exclude', 
            '--resume', 
            '--xyz', 
            '--processes', f'{self.pool._processes}', 
            '-z', f'{max_zoom}'
        ] + self.get_tiler_cmd_params() + [
            str(vrt_file), str(self.disk_handler.dir)
        ]

        gdal2tiles_main(cmd, pool=self.pool, called_from_main=True)
        vrt_file.unlink()

    def create_upper_tiles(self, z, tiles_to_create):
        tile_driver = self.get_tile_driver()
    
        options_dict = {
            'resume': True,
            'verbose': False,
            'quiet': False,
            'xyz': True,
            'exclude_transparent': True,
            'profile': 'mercator',
            'resampling': 'antialias',
            'tiledriver': tile_driver,
        }
        options_dict.update(self.get_tiler_options())

        options = AttrDict(options_dict)

        tile_job_info = TileJobInfo(
            tile_driver=tile_driver,
            nb_data_bands=3,
            tile_size=256,
            tile_extension=self.tile_extension,
            kml=False
        )
    
        nb_processes = self.pool._processes
    
        base_tile_groups = []
        for tile in tiles_to_create:
            self.disk_handler.preprare_dir(tile)
            ctiles = mercantile.children(tile)
            base_tile_group = [ (t.x, GDAL2Tiles.getYTile(t.y, t.z, options)) for t in ctiles ]
            base_tile_groups.append(base_tile_group)
    
        chunksize = max(1, min(128, len(base_tile_groups) // nb_processes))
    
        for _ in self.pool.imap_unordered(
            partial(
                create_overview_tile,
                z+1,
                output_folder=str(self.disk_handler.dir),
                tile_job_info=tile_job_info,
                options=options,
            ),
            base_tile_groups,
            chunksize=chunksize,
        ):
            pass
    
        print(f'Done creating {len(tiles_to_create)} tiles for zoom level {z}')
 

    def copy_tiles_over(self, tiles_to_pull):
        for tile in tiles_to_pull:
            tile_file = self.disk_handler.get_tile_file(tile)
            # TODO: check probably not required?
            if tile_file.exists():
                continue

            try:
                t_data = self.orig_tile_source.get_tile_data(tile)
            except MissingTileError:
                continue

            self.disk_handler.preprare_dir(tile)
            tile_file.write_bytes(t_data)


    def retile(self, sheet_list, affected_base_tiles):
    
        max_zoom = self.orig_tile_source.max_zoom
        min_zoom = self.orig_tile_source.min_zoom
    
        self.disk_handler.init()
    
        self.create_base_tiles(sheet_list, max_zoom)
    
        print('deleting unwanted base tiles')
        self.disk_handler.delete_unwanted_tiles(affected_base_tiles, max_zoom)
    
        prev_affected_tiles = affected_base_tiles
        for z in range(max_zoom - 1, min_zoom - 1, -1):
            print(f'handling level {z}')
    
            curr_affected_tiles = set()
            for tile in prev_affected_tiles:
                curr_affected_tiles.add(mercantile.parent(tile))
    
            child_tiles_to_pull = set()
            for ptile in curr_affected_tiles:
                for ctile in mercantile.children(ptile):
                    if ctile not in prev_affected_tiles:
                        child_tiles_to_pull.add(ctile)
    
            print('copying additional child tiles required for curr level')
            self.copy_tiles_over(child_tiles_to_pull)
    
            print('creating tiles for current level')
            self.create_upper_tiles(z, curr_affected_tiles)
    
            print('removing unwanted child tiles')
            self.disk_handler.delete_unwanted_tiles(prev_affected_tiles, z+1)
    
            prev_affected_tiles = curr_affected_tiles
    
        metadata = self.orig_tile_source.get_metadata()
        metadata['minzoom'] = min_zoom
        metadata['maxzoom'] = max_zoom
        self.disk_handler.save_metadata(metadata)
        print('All Done!!!')


def check_sheets(sheets_to_pull, tiffs_dir):
    missing = []
    for sheet_no in sheets_to_pull:
        file = tiffs_dir.joinpath(f'{sheet_no}')
        if not file.exists():
            missing.append(file)
    if len(missing):
        raise Exception(f'missing files {missing}')


def get_sheet_data(bounds_fname):

    index_file = Path(bounds_fname)
    if not index_file.exists():
        raise Exception(F'missing index file at {bounds_fname}')

    index_data = json.loads(index_file.read_text())

    sheets_to_box = {}
    for f in index_data['features']:
        sheet_no = f['properties']['id']
        geom = shape(f['geometry'])
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

   

def assess_sheet_requirements(retile_list_file, bounds_file, max_zoom):

    retile_sheets = retile_list_file.read_text().split('\n')
    retile_sheets = set([ r.strip().replace('.tif', '') for r in retile_sheets if r.strip() != '' ])

    print('getting base tiles to sheet mapping')
    sheets_to_box = get_sheet_data(bounds_file)
    sheets_to_base_tiles, base_tiles_to_sheets = get_base_tile_sheet_mappings(sheets_to_box, max_zoom)

    print('calculating sheets to pull')
    affected_base_tiles = set()
    for sheet_no in retile_sheets:
        affected_base_tiles.update(sheets_to_base_tiles[sheet_no])

    sheets_to_pull = set()
    for tile in affected_base_tiles:
        to_add = base_tiles_to_sheets[tile]
        for sheet in to_add:
            sheets_to_pull.add(sheet + '.tif')

    return sheets_to_pull, affected_base_tiles

def retile_main(args):

    parser = argparse.ArgumentParser()
    parser.add_argument('--retile-list-file', required=True, help='File containing list of sheets to retile')
    parser.add_argument('--bounds-file', required=True, help='Geojson file containing list of available sheets and their georaphic bounds')
    parser.add_argument('--max-zoom', type=int, help='Maximum zoom level to create tiles for, needed only when --from-source is not provided')
    parser.add_argument('--sheets-to-pull-list-outfile', default=None,
                        help='Output into which we write the list of sheet that need to be pulled, if set, the script ends after it created the list file')
    parser.add_argument('--from-source', help='location of the source from which we pull tiles, can be a glob pattern to match a group of pmtiles, it needs to be set when --sheets-to-pull-list-outfile is not set')
    parser.add_argument('--tiles-dir', help='Directory where the tiles will be created')
    parser.add_argument('--tiffs-dir', help='Directory where the tiffs are present')
    parser.add_argument('--num-parallel', type=int, default=cpu_count(), help='Number of parallel processes to use for tiling (default: number of CPU cores)')
    parser.add_argument('--tile-quality', default=75, help='quality of compression for webp and jpeg (default: 75)')

    args = parser.parse_args(args)

    retile_list_file = Path(args.retile_list_file)
    if not retile_list_file.exists():
        parser.error(f'Retile list file {args.retile_list_file} does not exist')

    bounds_file = Path(args.bounds_file)
    if not bounds_file.exists():
        parser.error(f'Bounds file {args.bounds_file} does not exist')


    if not args.sheets_to_pull_list_outfile:
        missing = []
        if not args.from_source:
            missing.append('--from-source')
        if not args.tiles_dir:
            missing.append('--tiles-dir')
        if not args.tiffs_dir:
            missing.append('--tiffs-dir')
        
        if missing:
            parser.error(f"The following arguments are required when --sheets-to-pull-list-outfile is not provided: {', '.join(missing)}")

    if args.from_source is None:
        if not args.max_zoom:
            parser.error('--max-zoom is required when --from-source is not set')

        orig_source = None
        max_zoom = args.max_zoom
    else:
        try:
            orig_source = create_source_from_paths([args.from_source])
        except ValueError as e:
            parser.error(f'Error creating tile source from {args.from_source}: {e}')

        max_zoom = orig_source.max_zoom
        if args.max_zoom is not None:
            print(f'--max-zoom argument is ignored when --from-source is set, using max zoom from original source: {max_zoom}')

    # calculate the sheets to pull and affected base tiles
    sheets_to_pull, affected_base_tiles = assess_sheet_requirements(retile_list_file, bounds_file, max_zoom)

    if args.sheets_to_pull_list_outfile is not None:
        Path(args.sheets_to_pull_list_outfile).write_text('\n'.join(sheets_to_pull) + '\n')
        print(f'Wrote sheets to pull to {args.sheets_to_pull_list_outfile}.. exiting')
        return

    tiffs_dir = Path(args.tiffs_dir)
    tiles_dir = Path(args.tiles_dir)

    print('check the sheets availability')
    check_sheets(sheets_to_pull, tiffs_dir)

    metadata = orig_source.get_metadata()
    tile_extension = metadata['format']
    if tile_extension == 'jpg':
        tile_extension = 'jpeg'

    os.environ['GDAL_CACHEMAX'] = '2048'
    os.environ['GDAL_MAX_DATASET_POOL_SIZE'] = '5000'
    os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'TRUE'
    #os.environ['VRT_SHARED_SOURCE'] = '1'
    #os.environ['GTIFF_VIRTUAL_MEM_IO'] = 'TRUE'
    with DividedCache(args.num_parallel), Pool(processes=args.num_parallel) as pool:

        tiler = Tiler(tiles_dir,
                      tiffs_dir, 
                      orig_source, 
                      tile_extension,
                      args.tile_quality,
                      pool)
        tiler.retile(sheets_to_pull, affected_base_tiles)

    return 0

def cli():
    if sys.platform == 'darwin':
        set_start_method('fork')

    retile_main(sys.argv[1:])

if __name__ == '__main__':
    cli()

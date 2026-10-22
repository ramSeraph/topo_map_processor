
import os
import re
import sys
import json
import argparse
import tempfile
import shlex
import shutil
from collections import defaultdict
from contextlib import contextmanager

import time
import subprocess

from pathlib import Path
from functools import partial
from multiprocessing import cpu_count, Pool, set_start_method


import mercantile
from shapely.geometry import shape
from osgeo import gdal

from osgeo_utils.gdal2tiles import (
    create_base_tile,
    create_overview_tile,
    process_args as gdal2tiles_process_args,
    TileDetail,
    TileJobInfo, 
    GDAL2Tiles, 
    DividedCache,
)

from pmtiles_mosaic.tile_sources import create_source_from_paths
from pmtiles_mosaic.tiles_common import MissingTileError

from .utils import relax_open_file_limit

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


def base_tile_batches(tiles, batch_size=32):
    """Assign each XYZ tile to one bounded geographic batch."""
    if batch_size < 1:
        raise ValueError('batch size must be positive')
    batches = defaultdict(set)
    for tile in tiles:
        batches[(tile.z, tile.x // batch_size, tile.y // batch_size)].add(tile)
    return [batches[key] for key in sorted(batches)]


@contextmanager
def gdal_exceptions():
    """Restore the caller's error handling, including on GDAL 3.8."""
    previously_enabled = gdal.GetUseExceptions()
    gdal.UseExceptions()
    try:
        yield
    finally:
        if not previously_enabled:
            gdal.DontUseExceptions()


def render_base_tile(conf, detail):
    # Spawned workers do not inherit the parent's GDAL exception setting.
    with gdal_exceptions():
        create_base_tile(conf, detail)


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
        metadata_file = self.dir.joinpath('metadata.json')
        metadata_file.write_text(json.dumps(metadata, indent=2))

 
class Tiler:
    def __init__(self, tiles_dir, tiffs_dir,
                 orig_tile_source, tile_extension, tile_quality, pool, batch_size=32):

        self.disk_handler = DiskTilesHandler(tiles_dir, tile_extension)
        self.orig_tile_source = orig_tile_source
        self.tiffs_dir = tiffs_dir
        self.tile_extension = tile_extension
        self.tile_quality = tile_quality
        self.pool = pool
        self.batch_size = batch_size

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


    def create_vrt_file(self, sheets, background_sheets=()):
        vrt_file = self.tiffs_dir.joinpath('combined.vrt')
        # Use the same source order and native pixel grid for every batch.
        # Do not mix coarse dummy rasters into the mosaic's average resolution.
        # A dummy provides an empty source only when all real sheets were removed.
        ordered_sheets = sorted(sheets) or sorted(background_sheets)
        tiff_list = [str(self.tiffs_dir.joinpath(sheet).resolve()) for sheet in ordered_sheets]
        run_external(shlex.join(['gdalbuildvrt', str(vrt_file), *tiff_list]))
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

    def create_base_tiles(self, sheets, max_zoom, affected_tiles, background_sheets=()):
        batches = base_tile_batches(affected_tiles, self.batch_size)
        if not batches:
            return
        print('creating vrt file from sheets involved')
        vrt_file = self.create_vrt_file(sheets, background_sheets)
        print(f'Creating {len(affected_tiles)} base tiles in {len(batches)} geographic batches', flush=True)

        # The common VRT preserves overlap priority, the pixel grid, and all
        # neighboring source pixels. Only the output job bounds change; cropping
        # the input raster separately for each batch could introduce seams.
        with (
            tempfile.TemporaryDirectory(
                prefix='.retile-batches-', dir=self.disk_handler.dir.parent,
            ) as workdir,
            gdal_exceptions(),
        ):
            input_file, output_folder, options = gdal2tiles_process_args([
                '-r', 'lanczos', '--quiet', '-w', 'none', '--xyz',
                '-z', str(max_zoom),
            ] + self.get_tiler_cmd_params() + [str(vrt_file), workdir])
            tiler = GDAL2Tiles(input_file, output_folder, options)
            try:
                tiler.open_input()
                source_minx, source_miny, source_maxx, source_maxy = tiler.tminmax[max_zoom]
                completed = 0
                for index, batch in enumerate(batches, 1):
                    batch_dir = Path(workdir) / str(index)
                    tiler.output_folder = str(batch_dir)
                    # GDAL stores planning bounds in TMS coordinates, while
                    # TileDetail.ty and our ownership sets use XYZ coordinates.
                    minx, maxx = min(t.x for t in batch), max(t.x for t in batch)
                    xyz_miny, xyz_maxy = min(t.y for t in batch), max(t.y for t in batch)
                    miny = (1 << max_zoom) - 1 - xyz_maxy
                    maxy = (1 << max_zoom) - 1 - xyz_miny
                    # Cleared areas can lie beyond the remaining source mosaic.
                    # Plan reads only inside it, then explicitly write transparent
                    # replacement tiles outside it instead of retaining old tiles.
                    tiler.tminmax[max_zoom] = (
                        max(minx, source_minx), max(miny, source_miny),
                        min(maxx, source_maxx), min(maxy, source_maxy),
                    )
                    conf, details = tiler.generate_base_tiles()
                    details = [d for d in details if mercantile.Tile(d.tx, d.ty, d.tz) in batch]
                    covered = {mercantile.Tile(d.tx, d.ty, d.tz) for d in details}
                    expected_covered = {
                        t for t in batch
                        if source_minx <= t.x <= source_maxx and
                        source_miny <= (1 << max_zoom) - 1 - t.y <= source_maxy
                    }
                    if covered != expected_covered:
                        raise RuntimeError('GDAL did not plan every affected tile inside the source mosaic')
                    for tile in sorted(batch - covered):
                        batch_dir.joinpath(str(tile.z), str(tile.x)).mkdir(parents=True, exist_ok=True)
                        details.append(TileDetail(
                            tx=tile.x, ty=tile.y, tz=tile.z,
                            rx=0, ry=0, rxsize=0, rysize=0,
                            wx=0, wy=0, wxsize=0, wysize=0,
                            querysize=tiler.querysize,
                        ))
                    if len(details) != len(batch):
                        raise RuntimeError('GDAL did not plan every tile assigned to the batch')

                    print(
                        f'Base batch {index}/{len(batches)}: {len(details)} tiles, '
                        f'XYZ x={minx}..{maxx}, y={xyz_miny}..{xyz_maxy}',
                        flush=True,
                    )
                    chunksize = max(1, min(128, len(details) // self.pool._processes))
                    for _ in self.pool.imap_unordered(
                        partial(render_base_tile, conf), details, chunksize=chunksize,
                    ):
                        pass

                    # A batch never reads partially rendered output from another
                    # batch. Publish only its owned tiles after every job succeeds.
                    # Keep transparent tiles: they replace old content in PMTiles.
                    output = DiskTilesHandler(batch_dir, self.tile_extension)
                    for tile in batch:
                        path = output.get_tile_file(tile)
                        if not path.is_file() or path.stat().st_size == 0:
                            raise RuntimeError(f'Base batch did not produce tile {tile}')
                    for tile in batch:
                        self.disk_handler.preprare_dir(tile)
                        output.get_tile_file(tile).replace(self.disk_handler.get_tile_file(tile))
                    shutil.rmtree(batch_dir)
                    completed += len(batch)
                    print(
                        f'Completed base batch {index}/{len(batches)} '
                        f'({completed}/{len(affected_tiles)} tiles)',
                        flush=True,
                    )
            finally:
                tiler.warped_input_dataset = None
                shutil.rmtree(tiler.tmp_dir, ignore_errors=True)
                vrt_file.unlink(missing_ok=True)

    def create_upper_tiles(self, z, tiles_to_create):
        tile_driver = self.get_tile_driver()
    
        options_dict = {
            'resume': False,
            'verbose': False,
            'quiet': False,
            'xyz': True,
            # N.B. if transperent tiles are excluded, the old data might show through
            #'exclude_transparent': True,
            'profile': 'mercator',
            'resampling': 'lanczos',
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


    def retile(self, sheet_list, affected_base_tiles, background_sheets=()):
    
        max_zoom = self.orig_tile_source.max_zoom
        min_zoom = self.orig_tile_source.min_zoom
    
        self.disk_handler.init()
    
        self.create_base_tiles(sheet_list, max_zoom, affected_base_tiles, background_sheets)
    
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


def get_sheet_data(bounds_fname, dummy=False):

    index_file = Path(bounds_fname)
    if not index_file.exists():
        raise Exception(F'missing index file at {bounds_fname}')

    index_data = json.loads(index_file.read_text())

    dummy_count = 0
    sheets_to_box = {}
    for f in index_data['features']:
        if not dummy:
            sheet_no = f['properties']['id']
        else:
            sheet_no = f'dummy_{dummy_count}'
            dummy_count += 1
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

   

def assess_sheet_requirements(retile_list_file, bounds_file, force_redo_bounds_file, max_zoom):

    retile_sheets = set()
    if retile_list_file:
        retile_sheets = retile_list_file.read_text().split('\n')
        retile_sheets = set([ r.strip().replace('.tif', '') for r in retile_sheets if r.strip() != '' ])

    print('getting base tiles to sheet mapping')
    sheets_to_box = get_sheet_data(bounds_file)
    force_redo_sheets_to_box = {}
    if force_redo_bounds_file is not None:
        force_redo_sheets_to_box = get_sheet_data(force_redo_bounds_file, dummy=True)
        sheets_to_box.update(force_redo_sheets_to_box)

    sheets_to_base_tiles, base_tiles_to_sheets = get_base_tile_sheet_mappings(sheets_to_box, max_zoom)

    print('calculating sheets to pull')
    affected_base_tiles = set()
    for sheet_no in retile_sheets:
        affected_base_tiles.update(sheets_to_base_tiles[sheet_no])
    for sheet_no in force_redo_sheets_to_box.keys():
        affected_base_tiles.update(sheets_to_base_tiles[sheet_no])

    sheets_to_pull = set()
    for tile in affected_base_tiles:
        to_add = base_tiles_to_sheets[tile]
        for sheet in to_add:
            if sheet not in force_redo_sheets_to_box:
                sheets_to_pull.add(sheet + '.tif')

    return sheets_to_pull, affected_base_tiles

def create_dummy_sheets(dummy_bounds_file, tiffs_dir):
    print('creating dummy sheets for force redo areas')
    dummy_sheets_to_box = get_sheet_data(dummy_bounds_file, dummy=True)

    sheets = set()
    for sheet_no, box in dummy_sheets_to_box.items():
        file = tiffs_dir.joinpath(f'{sheet_no}.tif')
        sheets.add(file.name)
        if file.exists():
            continue

        west, south, east, north = box.west, box.south, box.east, box.north
        temp_file = Path(tempfile.mktemp(suffix='.tif'))
        cmd = f'gdal_create -of GTiff -a_srs EPSG:4326 -a_ullr {west} {north} {east} {south} -a_nodata 0 -ot Byte -co COMPRESS=LZW -bands 3 -burn 0 0 0 -outsize 256 256 {temp_file}'
        run_external(cmd)
        temp_file_1 = Path(tempfile.mktemp(suffix='.tif'))
        cmd = f'gdalwarp -dstalpha -t_srs EPSG:3857 -co COMPRESS=LZW {temp_file} {temp_file_1}'
        run_external(cmd)
        cmd = f'gdal_translate -co TILING_SCHEME=GoogleMapsCompatible -co COMPRESS=JPEG -co QUALITY=100 --config GDAL_TIFF_INTERNAL_MASK YES  -b 1 -b 2 -b 3 -mask 4 --config GDAL_CACHEMAX 512 -of COG {temp_file_1} {file}'
        run_external(cmd)

    return sheets

def retile_main(args):

    parser = argparse.ArgumentParser()
    parser.add_argument('--retile-list-file', required=False, default=None, help='File containing list of sheets to retile')
    parser.add_argument('--bounds-file', required=True, help='Geojson file containing list of available sheets and their georaphic bounds')
    parser.add_argument('--force-redo-bounds-file', required=False, help='Geojson file containing georaphic bounds of areas that need to be updated, these areas might not have any new sheets')
    parser.add_argument('--max-zoom', type=int, help='Maximum zoom level to create tiles for, needed only when --from-source is not provided')
    parser.add_argument('--sheets-to-pull-list-outfile', default=None,
                        help='Output into which we write the list of sheet that need to be pulled, if set, the script ends after it created the list file')
    parser.add_argument('--from-source', help='location of the source from which we pull tiles, can be a glob pattern to match a group of pmtiles, it needs to be set when --sheets-to-pull-list-outfile is not set')
    parser.add_argument('--tiles-dir', help='Directory where the tiles will be created')
    parser.add_argument('--tiffs-dir', help='Directory where the tiffs are present')
    parser.add_argument('--num-parallel', type=int, default=cpu_count(), help='Number of parallel processes to use for tiling (default: number of CPU cores)')
    parser.add_argument('--tile-quality', default=75, help='quality of compression for webp and jpeg (default: 75)')
    parser.add_argument('--batch-size', type=int, default=32, help='Maximum width and height of a base-tile batch in tiles (default: 32)')

    args = parser.parse_args(args)
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')

    if not args.retile_list_file and not args.force_redo_bounds_file:
        parser.error('At least one of --retile-list-file or --force-redo-bounds-file must be provided.')

    retile_list_file = Path(args.retile_list_file) if args.retile_list_file else None
    if retile_list_file and not retile_list_file.exists():
        parser.error(f'Retile list file {args.retile_list_file} does not exist')

    bounds_file = Path(args.bounds_file)
    if not bounds_file.exists():
        parser.error(f'Bounds file {args.bounds_file} does not exist')

    force_redo_bounds_file = None
    if args.force_redo_bounds_file:
        force_redo_bounds_file = Path(args.force_redo_bounds_file) 
        if not force_redo_bounds_file.exists():
            parser.error(f'Force redo bounds file {args.force_redo_bounds_file} does not exist')


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
    sheets_to_pull, affected_base_tiles = assess_sheet_requirements(retile_list_file, bounds_file, force_redo_bounds_file, max_zoom)

    if args.sheets_to_pull_list_outfile is not None:
        Path(args.sheets_to_pull_list_outfile).write_text('\n'.join(sheets_to_pull) + '\n')
        print(f'Wrote sheets to pull to {args.sheets_to_pull_list_outfile}.. exiting')
        return

    tiffs_dir = Path(args.tiffs_dir)
    tiles_dir = Path(args.tiles_dir)

    print('check the sheets availability')
    check_sheets(sheets_to_pull, tiffs_dir)

    background_sheets = set()
    # Ordinary batches read the real mosaic. Empty replacement jobs handle
    # removed areas outside it; a dummy source is needed only if none remain.
    if force_redo_bounds_file is not None and not sheets_to_pull:
        background_sheets = create_dummy_sheets(force_redo_bounds_file, tiffs_dir)

    metadata = orig_source.get_metadata()
    tile_extension = metadata['format']
    if tile_extension == 'jpg':
        tile_extension = 'jpeg'

    relax_open_file_limit()

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
                      pool,
                      batch_size=args.batch_size)
        tiler.retile(sheets_to_pull, affected_base_tiles, background_sheets)

    return 0

def cli():
    if sys.platform == 'darwin':
        set_start_method('fork')

    retile_main(sys.argv[1:])

if __name__ == '__main__':
    cli()

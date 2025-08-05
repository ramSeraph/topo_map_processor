# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mercantile",
#     "pmtiles",
# ]
# ///

import json
import gzip
import copy
import argparse
import re
import sys
from pathlib import Path

import mercantile
from pprint import pprint
from pmtiles.tile import zxy_to_tileid, TileType, Compression
from pmtiles.writer import Writer as PMTilesWriter

from .tile_sources import create_source_from_paths

BASE_SIZE_FOR_DELTA = 2 * 1024 * 1024 * 1024 # 2GB
BASE_DELTA = 5 * 1024 * 1024 # 5MB

SLICE_HEADER_EXPORT_KEYS = [
    'min_lon_e7',
    'min_lat_e7',
    'max_lon_e7',
    'max_lat_e7',
    'min_zoom',
    'max_zoom',
]

HEADER_EXPORT_KEYS = SLICE_HEADER_EXPORT_KEYS + [
    'center_zoom',
    'center_lon_e7',
    'center_lat_e7',
    'tile_type',
    'tile_compression',
]

def parse_size(size_str):
    size_str = str(size_str).strip().upper()

    intended_size = None

    # Handle preset values
    if size_str == 'GITHUB_RELEASE':
        intended_size = 2 * 1024 * 1024 * 1024
    elif size_str == 'GITHUB_FILE':
        intended_size = 100 * 1024 * 1024
    elif size_str == 'CLOUDFLARE_OBJECT':
        intended_size = 512 * 1024 * 1024
    else:
        # Handle numeric values with optional units
        match = re.match(r'^(\d+)([GKM]?)$', size_str)
        if not match:
            raise argparse.ArgumentTypeError(f"Invalid size format: {size_str}")

        value, unit = match.groups()
        value = int(value)

        if unit == 'G':
            intended_size = value * 1024 * 1024 * 1024
        elif unit == 'M':
            intended_size = value * 1024 * 1024
        elif unit == 'K':
            intended_size = value * 1024
        else:
            intended_size = value

    # Calculate scaled DELTA
    if intended_size < 1024 * 1024:  # less than 1MB
        raise argparse.ArgumentTypeError(f"Size must be at least 1MB, got {size_str}")

    scaled_delta = int(BASE_DELTA * (intended_size / BASE_SIZE_FOR_DELTA))
    final_size = intended_size - scaled_delta
    print(f"Calculated size: {final_size} bytes (intended: {intended_size} bytes, scaled delta: {scaled_delta} bytes)")
    if final_size < 0:
        raise argparse.ArgumentTypeError(f"Calculated size is negative: {final_size} for {size_str}")

    return final_size

def convert_header(header, filter_keys):
    new_header = copy.copy(header)
    new_header['tile_compression'] = header['tile_compression'].value
    new_header['tile_type'] = header['tile_type'].value
    header_keys = list(new_header.keys())
    for k in header_keys:
        if k not in filter_keys:
            del new_header[k]
    return new_header


def get_pmtiles_file_name(to_pmtiles_prefix, suffix):
    if suffix == '':
        out_pmtiles_file = f'{to_pmtiles_prefix}.pmtiles'
    else:
        out_pmtiles_file = f'{to_pmtiles_prefix}-{suffix}.pmtiles'

    return out_pmtiles_file

class Partitioner:
    def __init__(self, reader, to_pmtiles_prefix, size_limit_bytes):
        self.reader = reader

        self.max_zoom_level = reader.max_zoom
        self.min_zoom_level = reader.min_zoom

        self.to_pmtiles_prefix = to_pmtiles_prefix
        self.size_limit_bytes = size_limit_bytes

        self.tiles_to_slice_idx = {}

        self.slices = []

    def get_layer_tiles_and_sizes(self, zoom_level):
        tiles = []
        size = 0

        for tile, tsize in self.reader.for_all_z(zoom_level):
            tiles.append(tile)
            size += tsize

        return tiles, size

    def get_partition_name_from_levels(self, frm, to):

        if frm == to:
            partition_name = f'z{frm}'
        else:
            partition_name = f'z{frm}-{to}'

        return partition_name

    def add_to_current_slice(self, tiles):
        curr_idx = len(self.slices)
        for t in tiles:
            self.tiles_to_slice_idx[t] = curr_idx


    def create_top_slice(self):
    
        print('getting top slice')

        size_till_now = 0

        tiles = []
        for zoom_level in range(self.min_zoom_level, self.max_zoom_level + 1):

            curr_level_tiles, curr_level_size = self.get_layer_tiles_and_sizes(zoom_level)

            size_till_now += curr_level_size

            print(f'{zoom_level=}, {curr_level_size=}, {size_till_now=}')

            if size_till_now > self.size_limit_bytes:
                self.add_to_current_slice(tiles)
                return zoom_level - 1

            tiles.extend(curr_level_tiles)

        self.add_to_current_slice(tiles)
        return self.max_zoom_level

    def get_x_stripes(self, min_stripe_level):
        tiles_by_x = {}
        sizes_by_x = {}
    
        for t, tsize in self.reader.all_sizes():
            if t.z < min_stripe_level:
                continue
    
            if t.z == min_stripe_level:
                pt = t
            else:
                pt = mercantile.parent(t, zoom=min_stripe_level)
    
            if pt.x not in tiles_by_x:
                tiles_by_x[pt.x] = []
            tiles_by_x[pt.x].append(t)
    
            if pt.x not in sizes_by_x:
                sizes_by_x[pt.x] = 0
            sizes_by_x[pt.x] += tsize
    
        return sizes_by_x, tiles_by_x


    def get_buckets(self, sizes_by_x, tiles_by_x):
        buckets = []
        all_bucket_tiles = []
        expected_bucket_sizes = []
    
        x_coords = sorted(sizes_by_x.keys())
        if not x_coords:
            return [], []
    
        current_bucket_tiles = []
        current_bucket_size = 0
        current_bucket_range = None
    
        for x in x_coords:
            x_size = sizes_by_x[x]
            x_tiles = tiles_by_x[x]
    
            # if the current bucket has tiles and adding the next stripe would exceed the size limit,
            # then finalize the current bucket and start a new one.
            if current_bucket_range is not None and (current_bucket_size + x_size > self.size_limit_bytes):
                if current_bucket_size > self.size_limit_bytes:
                    raise Exception(f'Current bucket size {current_bucket_size} exceeds size limit {self.size_limit_bytes}, the striping algorithm failed')
                buckets.append(current_bucket_range)
                all_bucket_tiles.append(current_bucket_tiles)
                expected_bucket_sizes.append(current_bucket_size)
    
                current_bucket_tiles = []
                current_bucket_size = 0
                current_bucket_range = None
    
            if current_bucket_range is None:  # Starting a new bucket
                current_bucket_range = (x, x)
    
            current_bucket_tiles.extend(x_tiles)
            current_bucket_size += x_size
            current_bucket_range = (current_bucket_range[0], x)
    
        # Add the last bucket if it has any tiles.
        if current_bucket_tiles:
            if current_bucket_size > self.size_limit_bytes:
                raise Exception(f'Current bucket size {current_bucket_size} exceeds size limit {self.size_limit_bytes}, the striping algorithm failed')
            buckets.append(current_bucket_range)
            all_bucket_tiles.append(current_bucket_tiles)
            expected_bucket_sizes.append(current_bucket_size)
    
        pprint(f'{expected_bucket_sizes=}')
    
        return buckets, all_bucket_tiles


    def partition(self):

        min_zoom_level = self.reader.min_zoom
        max_zoom_level = self.reader.max_zoom
    
        top_slice_max_level = self.create_top_slice()

        print(f'top slice max zoom level: {top_slice_max_level}')

        if top_slice_max_level >= min_zoom_level:
            partition_name = self.get_partition_name_from_levels(min_zoom_level, top_slice_max_level)
    
            if top_slice_max_level == max_zoom_level:
                partition_name = ''
    
            self.slices.append(partition_name)
            if top_slice_max_level == max_zoom_level:
                print('no more slicing required')
                return


        from_level = top_slice_max_level + 1

        print('getting x stripes from zoom level', from_level)
        x_stripe_sizes, x_stripe_tiles = self.get_x_stripes(from_level)
        print('getting buckets')
        buckets, bucket_tiles = self.get_buckets(x_stripe_sizes, x_stripe_tiles)
    
        num_buckets = len(buckets)
        for i,bucket in enumerate(buckets):
            partition_name = self.get_partition_name_from_levels(from_level, max_zoom_level)
    
            if num_buckets > 1:
                partition_name += f'-part{i}'

            self.add_to_current_slice(bucket_tiles[i])
            self.slices.append(partition_name)

    def get_bounds(self, tiles):
    
        bounds = [ mercantile.bounds(t) for t in tiles ]
    
        max_x = bounds[0].east
        min_x = bounds[0].west
        max_y = bounds[0].north
        min_y = bounds[0].south
        for b in bounds:
            if b.east > max_x:
                max_x = b.east
    
            if b.west < min_x:
                min_x = b.west
    
            if b.north > max_y:
                max_y = b.north
    
            if b.south < min_y:
                min_y = b.south
    
        return (min_y, min_x, max_y, max_x)


    def get_info(self, tiles):
        tiles_by_z = {}

        for t in tiles:
            if t.z not in tiles_by_z:
                tiles_by_z[t.z] = []

            tiles_by_z[t.z].append(t)

        min_z = min(tiles_by_z.keys())
        max_z = max(tiles_by_z.keys())

        higher_bounds = self.get_bounds(tiles_by_z[max_z])
        lower_bounds = self.get_bounds(tiles_by_z[min_z])

        return min_z, max_z, lower_bounds, higher_bounds
 

    def get_header(self, tiles):

        min_zoom, max_zoom, lower_bounds, higher_bounds = self.get_info(tiles)

        lower_min_lat, lower_min_lon, lower_max_lat, lower_max_lon = lower_bounds
        higher_min_lat, higher_min_lon, higher_max_lat, higher_max_lon = higher_bounds

        src_metadata = self.reader.get_metadata()

        format_string = src_metadata['format']
        if format_string == "mvt":
            format_string = "pbf"

        if format_string == "jpg":
            format_string = "jpeg"

        is_vector_tiles = (format_string == "pbf")

        if is_vector_tiles or src_metadata.get('compression', None) == "gzip":
            tile_compression = Compression.GZIP
        else:
            tile_compression = Compression.NONE

        header = {
            "tile_compression": tile_compression,
            "min_lon_e7": int(lower_min_lon * 10000000),
            "min_lat_e7": int(lower_min_lat * 10000000),
            "max_lon_e7": int(lower_max_lon * 10000000),
            "max_lat_e7": int(lower_max_lat * 10000000),
            "min_zoom": min_zoom,
            "max_zoom": max_zoom,
            "center_zoom": (max_zoom + min_zoom) // 2,
            "center_lon_e7": int(10000000 * (higher_min_lon + higher_max_lon)/2),
            "center_lat_e7": int(10000000 * (higher_min_lat + higher_max_lat)/2),
        }
        if format_string in "pbf":
            header["tile_type"] = TileType.MVT
        elif format_string == "png":
            header["tile_type"] = TileType.PNG
        elif format_string == "jpeg":
            header["tile_type"] = TileType.JPEG
        elif format_string == "webp":
            header["tile_type"] = TileType.WEBP
        elif format_string == "avif":
            header["tile_type"] = TileType.AVIF
        else:
            header["tile_type"] = TileType.UNKNOWN

        return header
         
    def get_header_and_metadata(self, tiles):
        header = self.get_header(tiles)
        src_metadata = self.reader.get_metadata()

        metadata = copy.deepcopy(src_metadata)

        min_zoom = header['min_zoom']
        max_zoom = header['max_zoom']
        if 'vector_layers' in metadata:
            for layer in metadata['vector_layers']:
                if layer['maxzoom'] > max_zoom:
                    layer['maxzoom'] = max_zoom
                if layer['minzoom'] < min_zoom:
                    layer['minzoom'] = min_zoom

        return header, metadata
    
    def write_mosaic_file(self, mosaic_data):
        if len(self.slices) == 0:
            print('No slices to write mosaic file')
            return

        out_mosaic_file = f'{self.to_pmtiles_prefix}.mosaic.json'
        with open(out_mosaic_file, 'w') as f:
            json.dump(mosaic_data, f, indent=2)
        print(f'Wrote mosaic file to {out_mosaic_file}')

    def write_partitions(self):

        # go through all tiles and create a list of all tiles and a mapping from tile to slice index
        # could have calculate bounds here, but easier/more readable to do it later
        all_tiles = []
        tiles_by_idx = {}

        for t, idx in self.tiles_to_slice_idx.items():
            if idx not in tiles_by_idx:
                tiles_by_idx[idx] = []

            tiles_by_idx[idx].append(t)

            all_tiles.append(t)

        headers = []
        metadatas = []
        writers = []
        for i,slice in enumerate(self.slices):
            tiles = tiles_by_idx[i]

            header, metadata = self.get_header_and_metadata(tiles)
            headers.append(header)
            metadatas.append(metadata)

            out_pmtiles_file = get_pmtiles_file_name(self.to_pmtiles_prefix, slice)
            Path(out_pmtiles_file).parent.mkdir(exist_ok=True, parents=True)

            writer = PMTilesWriter(open(out_pmtiles_file, 'wb'))
            writers.append(writer)

        full_header = self.get_header(all_tiles)
        full_metadata = self.reader.get_metadata()

        done = set()
        for tile, tdata in self.reader.all():
            if tile in done:
                continue

            idx = self.tiles_to_slice_idx[tile]
            writer = writers[idx]
            header = headers[idx]
            should_be_compressed = (header['tile_compression'] == Compression.GZIP)
            if should_be_compressed and tdata[0:2] != b"\x1f\x8b":
                tdata = gzip.compress(tdata)
            
            tile_id = zxy_to_tileid(tile.z, tile.x, tile.y)
            writer.write_tile(tile_id, tdata)
            done.add(tile)

        for i, slice in enumerate(self.slices):
            print(f'writing partition {slice} to {out_pmtiles_file}')
            writer = writers[i]
            header = headers[i]
            metadata = metadatas[i]
            writer.finalize(header, metadata)

        mosaic_data = {
            'version': 1,
            'metadata': full_metadata,
            'header': convert_header(full_header, HEADER_EXPORT_KEYS),
            'slices': {}
        }

        for i,slice in enumerate(self.slices):
            out_pmtiles_file = get_pmtiles_file_name(self.to_pmtiles_prefix, slice)
            key = Path(out_pmtiles_file).name
            header = headers[i]
            mosaic_data['slices'][key] = {
                'header': convert_header(header, SLICE_HEADER_EXPORT_KEYS)
            }

        self.write_mosaic_file(mosaic_data)

# TODO: this code doesn't account for tile redundency that may happen when writing to pmtiles 
# TODO: maybe keep track of delta size based on the number of tiles being added into each partition instead fixing it ahead of time.
# TODO: handle cases where vertical striping fails because some single x stripe is too large
#       may be slice that x strip by y, if that also doesn't fix it and we have a single x,y area whcih is too big, drill down by z
def partition_main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-source', action='append', required=True, help='Path to a source file or directory. Can be repeated.')
    parser.add_argument('--to-pmtiles', required=True, help='Output PMTiles file.')
    parser.add_argument('--size-limit', default='github_release', type=parse_size, help='Maximum size of each partition. Can be a number in bytes or a preset: github_release (2G), github_file (100M), cloudflare_object (512M). Can also be a number with optional units (K, M, G). Default is github_release (2G).')
    args = parser.parse_args(args)

    if not args.to_pmtiles.endswith('.pmtiles'):
        parser.error("Output PMTiles file must end with '.pmtiles'")
    to_pmtiles_prefix = args.to_pmtiles.removesuffix('.pmtiles')

    reader = create_source_from_paths(args.from_source)

    print(f'size limit: {args.size_limit} bytes')

    partitioner = Partitioner(reader, to_pmtiles_prefix, args.size_limit)

    partitioner.partition()

    partitioner.write_partitions()

def cli():
    partition_main(sys.argv[1:])

if __name__ == '__main__':
    cli()



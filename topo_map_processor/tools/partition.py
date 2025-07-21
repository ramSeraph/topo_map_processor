# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mercantile",
#     "pmtiles",
# ]
# ///

import json
import copy
import argparse
import re
from pathlib import Path

import mercantile
from pprint import pprint
from pmtiles.tile import zxy_to_tileid, TileType, Compression
from pmtiles.writer import Writer as PMTilesWriter

from .tile_sources import create_source_from_paths

BASE_SIZE_FOR_DELTA = 2 * 1024 * 1024 * 1024 # 2GB
BASE_DELTA = 5 * 1024 * 1024 # 5MB

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


def get_layer_info(level, reader):
    tiles = {}
    total_size = 0
    for tile, size in reader.for_all_z(level):
        tiles[tile] = size
        total_size += size
    return total_size, tiles


def get_buckets(sizes_by_x, tiles_by_x, size_limit_bytes):
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
        if current_bucket_range is not None and (current_bucket_size + x_size > size_limit_bytes):
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
        buckets.append(current_bucket_range)
        all_bucket_tiles.append(current_bucket_tiles)
        expected_bucket_sizes.append(current_bucket_size)

    pprint(f'{expected_bucket_sizes=}')

    return buckets, all_bucket_tiles


def get_x_stripes(min_stripe_level, reader, max_zoom_level):
    tiles_by_x = {}
    sizes_by_x = {}

    print(f'striping from level {min_stripe_level}')
    for t, tsize in reader.all_sizes():
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


def get_top_slice(reader, size_limit_bytes):
    min_zoom_level = reader.min_zoom
    max_zoom_level = reader.max_zoom

    print('getting top slice')
    size_till_now = 0
    tiles = {}
    prev_size = 0
    for level in range(min_zoom_level, max_zoom_level + 1):
        lsize, ltiles = get_layer_info(level, reader)
        prev_size = size_till_now
        size_till_now += lsize
        print(f'{level=}, {lsize=}, {size_till_now=}, {size_limit_bytes=}')
        tiles.update(ltiles)
        if size_till_now > size_limit_bytes:
            print(f'Expected top slice size is {prev_size} bytes')
            return level - 1, tiles
    return max_zoom_level, tiles

def save_partition_info(inp_p_info, partition_file):
    p_info = copy.deepcopy(inp_p_info)
    for p_name, p_data in p_info.items():
        tiles_new = [ f'{t.z},{t.x},{t.y}' for t in p_data['tiles'] ]
        p_data['tiles'] = tiles_new

    partition_file.parent.mkdir(exist_ok=True, parents=True)
    partition_file.write_text(json.dumps(p_info))

def load_partition_info(partition_file):
    partition_info = json.loads(partition_file.read_text())

    for suffix, data in partition_info.items():
        tdata = []
        for k in data['tiles']:
            kps = k.split(',')
            tile = mercantile.Tile(
                x=int(kps[1]),
                y=int(kps[2]),
                z=int(kps[0]),
            )
            tdata.append(tile)
        data['tiles'] = tdata

    return partition_info

def get_bounds(tiles):

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


def get_partition_info(reader, to_partition_file, size_limit_bytes):
    if to_partition_file.exists():
        return load_partition_info(to_partition_file)

    min_zoom_level = reader.min_zoom
    max_zoom_level = reader.max_zoom

    partition_info = {}
    top_slice_max_level, top_slice_tiles = get_top_slice(reader, size_limit_bytes)

    top_slice_bounds = get_bounds(top_slice_tiles.keys())

    if top_slice_max_level == min_zoom_level:
        partition_name = f'z{min_zoom_level}'
    else:
        partition_name = f'z{min_zoom_level}-{top_slice_max_level}'

    if top_slice_max_level == max_zoom_level:
        partition_name = ''

    partition_info[partition_name] = {
        "tiles": top_slice_tiles,
        "min_zoom": min_zoom_level,
        "max_zoom": top_slice_max_level,
        "bounds": top_slice_bounds
    }
    if top_slice_max_level == max_zoom_level:
        print('no more slicing required')
        save_partition_info(partition_info, to_partition_file)
        return partition_info


    from_level = top_slice_max_level + 1
    x_stripe_sizes, x_stripe_tiles = get_x_stripes(from_level, reader, max_zoom_level)
    buckets, bucket_tiles = get_buckets(x_stripe_sizes, x_stripe_tiles, size_limit_bytes)

    num_buckets = len(buckets)
    for i,bucket in enumerate(buckets):

        if from_level == max_zoom_level:
            partition_name = f'z{max_zoom_level}'
        else:
            partition_name = f'z{from_level}-{max_zoom_level}'

        if num_buckets > 1:
            partition_name += f'-part{i}'

        partition_info[partition_name] = {
            'tiles': bucket_tiles[i],
            "min_zoom": from_level,
            "max_zoom": max_zoom_level,
            "bounds": get_bounds(bucket_tiles[i]),
        }

    save_partition_info(partition_info, to_partition_file)
    return partition_info

def get_pmtiles_file_name(to_pmtiles_prefix, suffix):
    if suffix == '':
        out_pmtiles_file = f'{to_pmtiles_prefix}.pmtiles'
    else:
        out_pmtiles_file = f'{to_pmtiles_prefix}-{suffix}.pmtiles'

    return out_pmtiles_file

def create_pmtiles(partition_info, reader, to_pmtiles_prefix):
    writers = {}
    suffix_arr = []
    mosaic_data = {}
    tiles_to_suffix_idx = {}

    i = 0
    for suffix, data in partition_info.items():
        out_pmtiles_file = get_pmtiles_file_name(to_pmtiles_prefix, suffix)

        Path(out_pmtiles_file).parent.mkdir(exist_ok=True, parents=True)

        writers[suffix] = PMTilesWriter(open(out_pmtiles_file, 'wb'))

        suffix_arr.append(suffix)
        for t in data['tiles']:
            tiles_to_suffix_idx[t] = i
        i += 1

    curr_zs = {}
    max_lats = {}
    min_lats = {}
    max_lons = {}
    min_lons = {}
    min_zooms = {}
    max_zooms = {}
    for suffix in suffix_arr:
        curr_zs[suffix] = None
        max_lats[suffix] = min_lats[suffix] = max_lons[suffix] = min_lons[suffix] = None
        min_zooms[suffix] = max_zooms[suffix] = None

    done = set()

    for t, t_data in reader.all():
        if t in done:
            continue

        suffix = suffix_arr[tiles_to_suffix_idx[t]]
        writer = writers[suffix]

        if curr_zs[suffix] is None or curr_zs[suffix] < t.z:
            max_lats[suffix] = min_lats[suffix] = max_lons[suffix] = min_lons[suffix] = None
            curr_zs[suffix] = t.z

        t_bounds = mercantile.bounds(t)
        if curr_zs[suffix] == t.z and (max_lats[suffix] is None or t_bounds.north > max_lats[suffix]):
            max_lats[suffix] = t_bounds.north

        if curr_zs[suffix] == t.z and (min_lats[suffix] is None or t_bounds.south < min_lats[suffix]):
            min_lats[suffix] = t_bounds.south

        if curr_zs[suffix] == t.z and (max_lons[suffix] is None or t_bounds.east > max_lons[suffix]):
            max_lons[suffix] = t_bounds.east

        if curr_zs[suffix] == t.z and (min_lons[suffix] is None or t_bounds.west < min_lons[suffix]):
            min_lons[suffix] = t_bounds.west
                        
        if min_zooms[suffix] is None or min_zooms[suffix] > t.z:
            min_zooms[suffix] = t.z

        if max_zooms[suffix] is None or max_zooms[suffix] < t.z:
            max_zooms[suffix] = t.z

        t_id = zxy_to_tileid(t.z, t.x, t.y)
        writer.write_tile(t_id, t_data)
        done.add(t)

    for suffix in suffix_arr:
        out_pmtiles_file = get_pmtiles_file_name(to_pmtiles_prefix, suffix)

        src_metadata = reader.get_metadata()

        format_string = src_metadata['format']
        if format_string == "mvt":
            format_string = "pbf"
        if format_string == "jpg":
            format_string = "jpeg"

        is_vector_tiles = format_string == "pbf"

        header = {
            "tile_compression": Compression.GZIP if is_vector_tiles else Compression.NONE,
            "min_lon_e7": int(min_lons[suffix] * 10000000),
            "min_lat_e7": int(min_lats[suffix] * 10000000),
            "max_lon_e7": int(max_lons[suffix] * 10000000),
            "max_lat_e7": int(max_lats[suffix] * 10000000),
            "min_zoom": min_zooms[suffix],
            "max_zoom": max_zooms[suffix],
            "center_zoom": (max_zooms[suffix] + min_zooms[suffix]) // 2,
            "center_lon_e7": int(10000000 * (min_lons[suffix] + max_lons[suffix])/2),
            "center_lat_e7": int(10000000 * (min_lats[suffix] + max_lats[suffix])/2),
        }
        if format_string == "pbf":
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
            
        m_header = copy.copy(header)
        m_key = f'{Path(out_pmtiles_file).name}'
        m_header['tile_type'] = header['tile_type'].value
        m_header['tile_compression'] = header['tile_compression'].value
        mosaic_data[m_key] = { 'header': m_header, 'metadata': src_metadata }

        writer = writers[suffix]
        print(f'finalizing writing {suffix}')
        writer.finalize(header, src_metadata)

    return mosaic_data

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-source', action='append', required=True, help='Path to a source file or directory. Can be repeated.')
    parser.add_argument('--to-pmtiles', required=True, help='Output PMTiles file.')
    parser.add_argument('--size-limit', default='github_release', type=parse_size, help='Maximum size of each partition. Can be a number in bytes or a preset: github_release (2GB), github_file (100MB), cloudflare_object (512MB).')
    args = parser.parse_args()

    reader = create_source_from_paths(args.from_source)

    size_limit_bytes = args.size_limit
    print(f'size limit: {size_limit_bytes} bytes')

    if not args.to_pmtiles.endswith('.pmtiles'):
        parser.error("Output PMTiles file must end with '.pmtiles'")

    to_pmtiles_prefix = args.to_pmtiles.removesuffix('.pmtiles')

    print('getting partition info')
    to_partition_file = Path(f'{to_pmtiles_prefix}.partition_info.json')
    partition_info = get_partition_info(reader, to_partition_file, size_limit_bytes)


    print('creating pmtiles')
    mosaic_data = create_pmtiles(partition_info, reader, to_pmtiles_prefix)
    pprint(mosaic_data)
    if len(mosaic_data) > 1:
        Path(f'{to_pmtiles_prefix}.mosaic.json').write_text(json.dumps(mosaic_data))
    
    to_partition_file.unlink()


if __name__ == '__main__':
    cli()



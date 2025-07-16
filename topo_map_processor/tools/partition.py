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
from pathlib import Path

import mercantile
from pprint import pprint
from pmtiles.tile import zxy_to_tileid, TileType, Compression
from pmtiles.writer import Writer as PMTilesWriter

from tile_sources import (
    DiskSource,
    DiskAndPartitionedPMTilesSource,
    MissingTileError
)

# account for some directory overhead
# TODO: this should be dependant on the file size
DELTA = 5 * 1024 * 1024
# github release limit
SIZE_LIMIT_BYTES = (2 * 1024 * 1024 * 1024) - DELTA
# github git file size limit
#SIZE_LIMIT_BYTES = (100 * 1024 * 1024) - DELTA
# cloudflare cache size limit
#SIZE_LIMIT_BYTES = (512 * 1024 * 1024) - DELTA


def get_layer_info(level, reader):
    tiles = {}
    total_size = 0
    for tile, size in reader.for_all_z(level):
        tiles[tile] = size
        total_size += size
    return total_size, tiles


def get_buckets(sizes, tiles):
    buckets = []
    bucket_tiles = []

    max_x = max(sizes.keys())
    min_x = min(sizes.keys())

    cb = (min_x, min_x)
    cs = 0
    bts = {}
    for i in range(min_x, max_x + 1):
        if i not in sizes:
            continue
        if cs > SIZE_LIMIT_BYTES:
            buckets.append(cb)
            bucket_tiles.append(bts)
            cb = (i,i)
            cs = 0
            bts = {}
        cs += sizes[i]
        bts.update(tiles[i])
        cb = (cb[0], i)
    buckets.append(cb)
    bucket_tiles.append(bts)
    return buckets, bucket_tiles

missing_tiles = set()

# TODO: can do better than just vertical slices
def get_stripes(level, reader, max_zoom_level):
    tiles = {}
    sizes = {}
    #max_x = min_x = max_y = min_y = None
    #missed_count = 0
    print(f'striping from level {level}')
    count = 0
    for tile, size in reader.for_all_z(level):
        count += 1
        #if count % 1000 == 0:
        #    print(f'handled {count} tiles')
        to_add = [(tile, size)]
        for clevel in range(level + 1, max_zoom_level + 1):
            children = mercantile.children(tile, zoom=clevel)
            for ctile in children:
                try:
                    csize = reader.get_tile_size(ctile)
                    to_add.append((ctile, csize))
                except MissingTileError:
                    missing_tiles.add(ctile)
                    continue

        for t, s in to_add:
            if tile.x not in tiles:
                tiles[tile.x] = {}
            tiles[tile.x][t] = s
            if tile.x not in sizes:
                sizes[tile.x] = 0
            sizes[tile.x] += s
    print(f'{len(missing_tiles)=}')
    return sizes, tiles


def get_top_slice(reader, min_zoom_level, max_zoom_level):
    print('getting top slice')
    size_till_now = 0
    tiles = {}
    for level in range(min_zoom_level, max_zoom_level + 1):
        lsize, ltiles = get_layer_info(level, reader)
        size_till_now += lsize
        print(f'{level=}, {lsize=}, {size_till_now=}, {SIZE_LIMIT_BYTES=}')
        tiles.update(ltiles)
        if size_till_now > SIZE_LIMIT_BYTES:
            return level - 1, tiles
    return max_zoom_level, tiles

def save_partition_info(inp_p_info, partition_file):
    p_info = copy.deepcopy(inp_p_info)
    for p_name, p_data in p_info.items():
        tiles_new = { f'{t.z},{t.x},{t.y}':size for t, size in p_data['tiles'].items() }
        p_data['tiles'] = tiles_new

    partition_file.parent.mkdir(exist_ok=True, parents=True)
    partition_file.write_text(json.dumps(p_info))


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


def get_partition_info(reader, min_zoom_level, max_zoom_level, to_partition_file):
    if to_partition_file.exists():
        partition_info = json.loads(to_partition_file.read_text())
        for suffix, data in partition_info.items():
            tdata = {}
            for k, size in data['tiles'].items():
                kps = k.split(',')
                tile = mercantile.Tile(
                    x=int(kps[1]),
                    y=int(kps[2]),
                    z=int(kps[0]),
                )
                tdata[tile] = size
            data['tiles'] = tdata
        return partition_info

    partition_info = {}
    top_slice_max_level, top_slice_tiles = get_top_slice(reader, min_zoom_level, max_zoom_level)

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
    stripe_sizes, stripe_tiles = get_stripes(from_level, reader, max_zoom_level)
    buckets, bucket_tiles = get_buckets(stripe_sizes, stripe_tiles)

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
            "bounds": get_bounds(bucket_tiles[i].keys()),
        }

    save_partition_info(partition_info, to_partition_file)
    return partition_info

def get_pmtiles_file_name(to_pmtiles_prefix, suffix):
    if suffix == '':
        out_pmtiles_file = f'{to_pmtiles_prefix}.pmtiles'
    else:
        out_pmtiles_file = f'{to_pmtiles_prefix}-{suffix}.pmtiles'

    return out_pmtiles_file

def create_pmtiles(partition_info, reader, to_pmtiles_prefix, name, description, attribution):
    mosaic_data = {}
    writers = {}
    suffix_arr = []
    tiles_to_suffix = {}
    i = 0
    for suffix, data in partition_info.items():
        out_pmtiles_file = get_pmtiles_file_name(to_pmtiles_prefix, suffix)
        Path(out_pmtiles_file).parent.mkdir(exist_ok=True, parents=True)
        writer = PMTilesWriter(open(out_pmtiles_file, 'wb'))
        writers[suffix] = writer
        suffix_arr.append(suffix)
        for t in data['tiles'].keys():
            tiles_to_suffix[t] = i
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
        suffix = suffix_arr[tiles_to_suffix[t]]
        writer = writers[suffix]
        if curr_zs[suffix] is None or curr_zs[suffix] < t.z:
            max_lats[suffix] = min_lats[suffix] = max_lons[suffix] = min_lons[suffix] = None
            curr_zs[suffix] = t.z
        t_bounds = mercantile.bounds(t)
        if max_lats[suffix] is None or t_bounds.north > max_lats[suffix]:
            max_lats[suffix] = t_bounds.north
        if min_lats[suffix] is None or t_bounds.south < min_lats[suffix]:
            min_lats[suffix] = t_bounds.south
        if max_lons[suffix] is None or t_bounds.east > max_lons[suffix]:
            max_lons[suffix] = t_bounds.east
        if min_lons[suffix] is None or t_bounds.west < min_lons[suffix]:
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
        metadata = {
            "type": "baselayer",
            "format": "webp",
            "version": "1",
            "name": name,
            "description": description,
            'attribution': attribution
        }

        header = {
            "tile_type": TileType.WEBP,
            "tile_compression": Compression.NONE,
            "min_lon_e7": int(min_lons[suffix] * 10000000),
            "min_lat_e7": int(min_lats[suffix] * 10000000),
            "max_lon_e7": int(max_lons[suffix] * 10000000),
            "max_lat_e7": int(max_lats[suffix] * 10000000),
            "min_zoom": min_zooms[suffix],
            "max_zoom": max_zooms[suffix],
            "center_zoom": 0,
            "center_lon_e7": int(10000000 * (min_lons[suffix] + max_lons[suffix])/2),
            "center_lat_e7": int(10000000 * (min_lats[suffix] + max_lats[suffix])/2),
        }
        m_header = copy.copy(header)
        m_key = f'../{Path(out_pmtiles_file).name}'
        m_header['tile_type'] = header['tile_type'].value
        m_header['tile_compression'] = header['tile_compression'].value
        writer = writers[suffix]
        print(f'finalizing writing {suffix}')
        writer.finalize(header, metadata)
        mosaic_data[m_key] = { 'header': m_header, 'metadata': metadata }
    return mosaic_data

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only-disk', action='store_true', help='Only read tiles from disk.')
    parser.add_argument('--to-pmtiles-prefix', required=True, help='Prefix for the output PMTiles files.')
    parser.add_argument('--from-pmtiles-prefix', help='Prefix for the input PMTiles files. Required if not using --only-disk.')
    parser.add_argument('--from-tiles-dir', required=True, help='Directory containing the input tiles')
    parser.add_argument('--name', required=True, help='Name of the mosaic.')
    parser.add_argument('--description', required=True, help='Description of the mosaic.')
    parser.add_argument('--min-zoom', default=0, type=int, help='Minimum zoom level for the mosaic.')
    parser.add_argument('--max-zoom', required=True, type=int, help='Maximum zoom level for the mosaic.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--attribution', help='Attribution text for the mosaic.')
    group.add_argument('--attribution-file', help='File containing attribution text for the mosaic.')
    args = parser.parse_args()

    if args.attribution_file:
        with open(args.attribution_file, 'r') as f:
            attribution_text = f.read().strip()
    else:
        attribution_text = args.attribution

    if not args.only_disk and not args.from_pmtiles_prefix:
        parser.error("--from-pmtiles-prefix is required unless --only-disk is set.")


    if args.only_disk:
        reader = DiskSource(args.from_tiles_dir)
    else:
        reader = DiskAndPartitionedPMTilesSource(args.from_tiles_dir, args.from_pmtiles_prefix)

    print('getting partition info')
    to_partition_file = Path(f'{args.to_pmtiles_prefix}.partition_info.json')
    partition_info = get_partition_info(reader, args.min_zoom, args.max_zoom, to_partition_file)

    print('creating pmtiles')
    mosaic_data = create_pmtiles(partition_info, reader, args.to_pmtiles_prefix,
                                 args.name, args.description, attribution_text)
    pprint(mosaic_data)
    if len(mosaic_data) > 1:
        Path(f'{args.to_pmtiles_prefix}.mosaic.json').write_text(json.dumps(mosaic_data))


if __name__ == '__main__':
    cli()



#!/usr/bin/env -S uv run 
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mercantile",
#     "pmtiles",
#     "requests",
# ]
# ///

import json
import sqlite3

from pathlib import Path, PurePosixPath
from urllib.parse import urlparse, urljoin, unquote

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from pmtiles.reader import Reader, MmapSource, all_tiles
from pmtiles.writer import Writer
from pmtiles.tile import Compression

# heavily copied from https://github.com/protomaps/PMTiles/blob/main/python/pmtiles/convert.py
# and https://github.com/mapbox/mbutil/blob/master/mbutil/util.py

# TODO: double check the compression related stuff

session = None
timeout = None

def download_file(url, fname):
    global session, timeout
    print(f'downloading {url} to {fname}')
    resp = session.get(url, stream=True, timeout=timeout)
    with open(fname, 'wb') as f:
        for data in resp.iter_content(chunk_size=4096):
            f.write(data)


def get_mosaic(mosaic_file, mosaic_url):
    if not mosaic_file.exists():
        download_file(mosaic_url, mosaic_file)

    return json.loads(mosaic_file.read_text())

def enhance_metadata(cm, ch):

    cm['maxzoom'] = ch['max_zoom']
    cm['minzoom'] = ch['min_zoom']

    max_lat = ch['max_lat_e7'] / 10000000 
    min_lat = ch['min_lat_e7'] / 10000000
    max_lon = ch['max_lon_e7'] / 10000000
    min_lon = ch['min_lon_e7'] / 10000000
    cm['bounds'] = f"{min_lon},{min_lat},{max_lon},{max_lat}"

    center_lat = ch['center_lat_e7'] / 10000000
    center_lon = ch['center_lon_e7'] / 10000000
    center_zoom = ch['center_zoom']
    cm['center'] = f"{center_lon},{center_lat},{center_zoom}"

    compression = ch['compression']
    cm['compression'] = compression.name.lower() if isinstance(compression, Compression) else compression


def collect_header(items):
    ch = {}
    for item in items:
        h = item['header']
        if 'max_zoom' not in ch or ch['max_zoom'] < h['max_zoom']:
            ch['max_zoom'] = h['max_zoom']
        if 'min_zoom' not in ch or ch['min_zoom'] > h['min_zoom']:
            ch['min_zoom'] = h['min_zoom']

        if 'max_lat_e7' not in ch or ch['max_lat_e7'] < h['max_lat_e7']:
            ch['max_lat_e7'] = h['max_lat_e7']
        if 'min_lat_e7' not in ch or ch['min_lat_e7'] > h['min_lat_e7']:
            ch['min_lat_e7'] = h['min_lat_e7']
        if 'max_lon_e7' not in ch or ch['max_lon_e7'] < h['max_lon_e7']:
            ch['max_lon_e7'] = h['max_lon_e7']
        if 'min_lon_e7' not in ch or ch['min_lon_e7'] > h['min_lon_e7']:
            ch['min_lon_e7'] = h['min_lon_e7']

        ch['compression'] = h.get('compression', Compression.NONE)

    ch['center_lat_e7'] = (ch['max_lat_e7'] + ch['min_lat_e7']) // 2
    ch['center_lon_e7'] = (ch['max_lon_e7'] + ch['min_lon_e7']) // 2
    ch['center_zoom'] = (ch['max_zoom'] + ch['min_zoom']) // 2

    return ch

def get_metadata_and_header(mosaic_data, mosaic_version, archive_type):
    if mosaic_version != 0:
        metadata = mosaic_data['metadata']
        header = mosaic_data['header']
    else:
        metadata = list(mosaic_data.values())[0]['metadata']
        header = collect_header(mosaic_data.values())

    if archive_type == 'mbtiles':
        enhance_metadata(metadata, header)

    return metadata, header


def finalize_mbtiles(conn, cursor):
    cursor.execute(
        "CREATE UNIQUE INDEX tile_index on tiles (zoom_level, tile_column, tile_row);"
    )
    conn.commit()
    cursor.execute("""ANALYZE;""")

    conn.close()


def add_to_mbtiles(pmtiles_fname, cursor, conn):
    print(f'adding {pmtiles_fname} to mbtiles')
    with open(pmtiles_fname, "r+b") as f:
        source = MmapSource(f)
        reader = Reader(source)

        for zxy, tile_data in all_tiles(reader.get_bytes):
            flipped_y = (1 << zxy[0]) - 1 - zxy[2]
            cursor.execute(
                "INSERT INTO tiles VALUES(?,?,?,?)",
                (zxy[0], zxy[1], flipped_y, tile_data),
            )
        conn.commit()

def add_to_pmtiles(pmtiles_fname, writer):
    print(f'adding {pmtiles_fname} to pmtiles')
    with open(pmtiles_fname, "r+b") as f:
        source = MmapSource(f)
        reader = Reader(source)
        for zxy, tile_data in all_tiles(reader.get_bytes):
            writer.write_tile(zxy[0], zxy[1], zxy[2], tile_data)

def optimize_cursor(cursor):
    cursor.execute("""PRAGMA synchronous=0""")
    cursor.execute("""PRAGMA locking_mode=EXCLUSIVE""")
    cursor.execute("""PRAGMA journal_mode=DELETE""")


def initialize_tables(cursor, metadata):
    cursor.execute("CREATE TABLE metadata (name text, value text);")
    cursor.execute(
        "CREATE TABLE tiles (zoom_level integer, tile_column integer, tile_row integer, tile_data blob);"
    )

    json_metadata = {}
    for k, v in metadata.items():
        if k == "vector_layers":
            json_metadata["vector_layers"] = v
            continue
        elif k == "tilestats":
            json_metadata["tilestats"] = v
            continue
        elif not isinstance(v, str):
            v = json.dumps(v, ensure_ascii=False)
        cursor.execute("INSERT INTO metadata VALUES(?,?)", (k, v))

    if len(json_metadata) > 0:
        cursor.execute(
            "INSERT INTO metadata VALUES(?,?)",
            ("json", json.dumps(json_metadata, ensure_ascii=False)),
        )



def get_mbtiles_conn(output_file):
    conn = sqlite3.connect(output_file)
    cursor = conn.cursor()
    optimize_cursor(cursor)
    return conn, cursor


def get_pmtiles_url(mosaic_url, k, version):
    if version == 0 and k.startswith('../'):
        k = k[3:]
    return urljoin(mosaic_url, k)


def init_tracker(tracker_file):
    if not tracker_file.exists():
        tracker_file.write_text("")

def stage_done(tracker_file, stage):
    txt = tracker_file.read_text()

    done_stages = txt.split('\n')

    if stage in done_stages:
        return True

    return False


def mark_done(tracker_file, stage):
    with open(tracker_file, 'a') as f:
        f.write(stage)
        f.write('\n')

def get_filename_from_url(url):
    url_parsed = urlparse(url)
    return unquote(PurePosixPath(url_parsed.path).name)

def cli():
    import argparse
    parser = argparse.ArgumentParser(description='Download a mosaic and convert it to MBTiles or PMTiles format.')
    parser.add_argument('--mosaic-url',  '-u', required=True, type=str, help='URL of the mosaic JSON file')
    parser.add_argument('--output-file', '-o', type=str, help='Output MBTiles/PMTiles file name. The format is inferred from the extension.')
    parser.add_argument('--archive-type', choices=['mbtiles', 'pmtiles'], help='Type of archive to create. Required if --output-file is not provided.')
    parser.add_argument('--request-timeout-secs', '-t', type=int, default=60, help='Timeout for HTTP requests in seconds')
    parser.add_argument('--num-http-retries', '-r', type=int, default=3, help='Number of retries for HTTP requests')
    args = parser.parse_args()

    mosaic_url = args.mosaic_url
    mosaic_fname = get_filename_from_url(mosaic_url)

    archive_type = args.archive_type
    output_file = None

    if args.output_file:
        output_file = Path(args.output_file)
        if output_file.suffix == '.mbtiles':
            if archive_type and archive_type != 'mbtiles':
                raise ValueError("Output file extension is .mbtiles but --archive-type is not 'mbtiles'")
            archive_type = 'mbtiles'
        elif output_file.suffix == '.pmtiles':
            if archive_type and archive_type != 'pmtiles':
                raise ValueError("Output file extension is .pmtiles but --archive-type is not 'pmtiles'")
            archive_type = 'pmtiles'
        else:
            raise ValueError("If --output-file is provided, it must have a .mbtiles or .pmtiles extension.")

    if not archive_type:
        raise ValueError("You must specify either --output-file (with .mbtiles or .pmtiles extension) or --archive-type.")

    if not output_file:
        if not mosaic_fname.endswith('.mosaic.json'):
            raise ValueError('Input mosaic URL must point to a file ending with .mosaic.json if --output-file is not specified.')
        
        out_fname = mosaic_fname[:-len('.mosaic.json')] + '.' + archive_type
        output_file = Path(out_fname)

    global session, timeout
    session = requests.session()
    retries = args.num_http_retries
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
    )
    session.mount('http://', HTTPAdapter(max_retries=retry))
    session.mount('https://', HTTPAdapter(max_retries=retry))
    timeout = args.request_timeout_secs

    mosaic_file = Path(mosaic_fname)
    mosaic_data = get_mosaic(mosaic_file, mosaic_url)
    mosaic_version = mosaic_data.get('version', 0)
    metadata, header = get_metadata_and_header(mosaic_data, mosaic_version, archive_type)

    tracker_file = Path('tracker.txt')

    if output_file.exists() and not tracker_file.exists():
        print(f'Output file {output_file} already exists, and no tracker file found. Exiting to avoid overwriting.')
        return

    init_tracker(tracker_file)

    print(f'Output file: {output_file}')

    writer = None
    conn = None
    cursor = None

    if archive_type == 'mbtiles':
        conn, cursor = get_mbtiles_conn(output_file)
        if not stage_done(tracker_file, 'table_init'):
            print('Initializing tables...')
            initialize_tables(cursor, metadata)
            mark_done(tracker_file, 'table_init')
    elif archive_type == 'pmtiles':
        writer = Writer(open(output_file, 'wb'))

    slice_data = mosaic_data if mosaic_version == 0 else mosaic_data.get('slices', {})
    for k in slice_data.keys():
        pmtiles_url = get_pmtiles_url(mosaic_url, k, mosaic_version)
        if stage_done(tracker_file, k):
            continue

        pmtiles_fname = get_filename_from_url(pmtiles_url)
        if Path(pmtiles_fname).exists():
            raise Exception(f'{pmtiles_fname} already exists, delete existing file to continue')
        download_file(pmtiles_url, pmtiles_fname)
        
        if archive_type == 'mbtiles':
            add_to_mbtiles(pmtiles_fname, cursor, conn)
        else:
            add_to_pmtiles(pmtiles_fname, writer)

        mark_done(tracker_file, k)
        Path(pmtiles_fname).unlink()

    if archive_type == 'mbtiles':
        finalize_mbtiles(conn, cursor)
    else:
        print("Finalizing pmtiles archive...")
        writer.finalize(header, metadata)

    tracker_file.unlink()
    mosaic_file.unlink()
    print('Done!!!')
    

if __name__ == '__main__':
    cli()


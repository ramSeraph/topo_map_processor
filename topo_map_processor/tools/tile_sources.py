import glob
import json
import sqlite3

from pathlib import Path

import mercantile

from pmtiles.reader import (
    MmapSource, 
    Reader as PMTilesReader, 
    all_tiles,
)

from pmtiles.tile import (
    deserialize_header,
    deserialize_directory,
    tileid_to_zxy,
)


class MissingTileError(Exception):
    pass

class DiskTilesSource:
    def __init__(self, directory):
        self.dir = Path(directory)
        metadata = self.get_metadata()
        self.ext = metadata['format']

    def _get_tile_from_file(self, file):
        parts = file.parts
        tile = mercantile.Tile(z=int(parts[-3]),
                               x=int(parts[-2]),
                               y=int(parts[-1].replace(f'.{self.ext}', '')))
        return tile

    def _file_from_tile(self, tile):
        return self.dir / f'{tile.z}' / f'{tile.x}' / f'{tile.y}.{self.ext}'

    def get_tile_data(self, tile):
        file = self._file_from_tile(tile)
        if not file.exists():
            raise MissingTileError()

        return file.read_bytes()

    def get_tile_size(self, tile):
        file = self._file_from_tile(tile)
        if not file.exists():
            raise MissingTileError()

        return file.stat().st_size
        
    def for_all_z(self, z):
        for file in self.dir.glob(f'{z}/*/*.{self.ext}'):
            tile = self._get_tile_from_file(file)
            fstats = file.stat()
            yield (tile, fstats.st_size)

    def all(self):
        for file in self.dir.glob(f'*/*/*.{self.ext}'):
            tile = self._get_tile_from_file(file)
            t_data = file.read_bytes()
            yield (tile, t_data)

    def all_sizes(self):
        for file in self.dir.glob(f'*/*/*.{self.ext}'):
            tile = self._get_tile_from_file(file)
            fstats = file.stat()
            yield (tile, fstats.st_size)


    def cleanup(self):
        pass

    def _get_zoom_levels(self):
        zoom_levels = set()
        for file in self.dir.glob(f'*/*/*.{self.ext}'):
            try:
                zoom = int(file.parts[-3])
                zoom_levels.add(zoom)
            except (ValueError, IndexError):
                continue
        return list(zoom_levels)

    def get_min_zoom(self):
        zoom_levels = self._get_zoom_levels()
        if len(zoom_levels) == 0:
            raise ValueError("No zoom directories found in the disk source.")

        return min(zoom_levels)

    def get_max_zoom(self):
        zoom_levels = self._get_zoom_levels()
        if len(zoom_levels) == 0:
            raise ValueError("No zoom directories found in the disk source.")

        return max(zoom_levels)

    def get_tilejson_file(self):
        return self.dir / 'tiles.json'

    def get_metadata(self):
        tilejson_file = self.get_tilejson_file()

        if not tilejson_file.exists():
            raise ValueError("TileJSON file not found in the disk source.")

        return json.loads(tilejson_file.read_text())

    @property
    def min_zoom(self):
        try:
            metadata = self.get_metadata()
            return metadata['minzoom']
        except ValueError:
            return self.get_min_zoom()
 
    @property
    def max_zoom(self):
        try:
            metadata = self.get_metadata()
            return metadata['maxzoom']
        except ValueError:
            return self.get_max_zoom()


class MBTilesSource:
    def __init__(self, fname):
        self.con = sqlite3.connect(fname)
        self._metadata = None

    def _to_xyz(self, x, y, z):
        y = (1 << z) - 1 - y
        return x, y, z

    def get_tile_data(self, tile):
        x, y, z = self._to_xyz(tile.x, tile.y, tile.z)
        res = self.con.execute(f'select tile_data from tiles where zoom_level={z} and tile_column={x} and tile_row={y};')
        out = res.fetchone()
        if not out:
            raise MissingTileError()
        return out[0]

    def get_tile_size(self, tile):
        data = self.get_tile_data(tile)
        return len(data)
 
    def for_all_z(self, z):
        res = self.con.execute(f'select tile_column, tile_row, length(tile_data) from tiles where zoom_level={z};')
        while True:
            t = res.fetchone()
            if not t:
                break
            x, y, z = self._to_xyz(t[0], t[1], z)
            tile_size = t[2]
            tile = mercantile.Tile(x=x, y=y, z=z)
            yield (tile, tile_size)

    def all(self):
        res = self.con.execute('select zoom_level, tile_column, tile_row, tile_data from tiles;')
        while True:
            t = res.fetchone()
            if not t:
                break
            x, y, z = self._to_xyz(t[1], t[2], t[0])
            data = t[3]
            tile = mercantile.Tile(x=x, y=y, z=z)
            yield (tile, data)

    def all_sizes(self):
        res = self.con.execute('select zoom_level, tile_column, tile_row, length(tile_data) from tiles;')
        while True:
            t = res.fetchone()
            if not t:
                break
            x, y, z = self._to_xyz(t[1], t[2], t[0])
            tile_size = t[3]
            tile = mercantile.Tile(x=x, y=y, z=z)
            yield (tile, tile_size)

    def cleanup(self):
        self.con.close()

    def get_metadata(self):
        if self._metadata is not None:
            return self._metadata

        all_metadata = {}
        for row in self.con.execute("SELECT name,value FROM metadata"):
            k = row[0]
            v = row[1]
            if k == 'json':
                json_data = json.loads(v)
                for k, v in json_data.items():
                    all_metadata[k] = v
                continue
            all_metadata[k] = v

        metadata = {}
        for k in ['type', 'format', 'attribution', 'description', 'name', 'version', 'vector_layers', 'maxzoom', 'minzoom', 'compression']:
            if k not in all_metadata:
                continue
            metadata[k] = all_metadata[k]

        self._metadata = metadata
        return self._metadata
            
    def _get_meta_prop(self, prop_name):
        metadata = self.get_metadata()
        if prop_name in metadata:
            return metadata[prop_name]

        raise ValueError(f"Source does not have a {prop_name} property.")

    @property
    def min_zoom(self):
        return int(self._get_meta_prop('minzoom'))

    @property
    def max_zoom(self):
        return int(self._get_meta_prop('maxzoom'))


# pmtiles sources
def traverse_sizes(get_bytes, header, dir_offset, dir_length):
    entries = deserialize_directory(get_bytes(dir_offset, dir_length))
    for entry in entries:
        if entry.run_length > 0:
            for i in range(entry.run_length):
                yield tileid_to_zxy(entry.tile_id + i), entry.length
        else:
            for t in traverse_sizes(
                get_bytes,
                header,
                header["leaf_directory_offset"] + entry.offset,
                entry.length,
            ):
                yield t

def all_tile_sizes(get_bytes):
    header = deserialize_header(get_bytes(0, 127))
    return traverse_sizes(get_bytes, header, header["root_offset"], header["root_length"])

class PMTilesSource:
    def __init__(self, fname):
        self.file = open(fname, 'rb')
        self.src = MmapSource(self.file)
        self.reader = PMTilesReader(self.src)

    def get_tile_data(self, tile):
        data = self.reader.get(tile.z, tile.x, tile.y)
        if data is None:
            raise MissingTileError()
        return data

    def get_tile_size(self, tile):
        data = self.get_tile_data(tile)
        return len(data)
 
    def for_all_z(self, z):
        for t, size in all_tile_sizes(self.reader.get_bytes):
            if t[0] == z:
                tile = mercantile.Tile(x=t[1], y=t[2], z=t[0])
                yield (tile, size)

    def all(self):
        for t, data in all_tiles(self.reader.get_bytes):
            tile = mercantile.Tile(x=t[1], y=t[2], z=t[0])
            yield (tile, data)

    def all_sizes(self):
        for t, size in all_tile_sizes(self.reader.get_bytes):
            tile = mercantile.Tile(x=t[1], y=t[2], z=t[0])
            yield (tile, size)

    def cleanup(self):
        self.file.close()

    @property
    def min_zoom(self):
        return int(self.reader.header()['min_zoom'])

    @property
    def max_zoom(self):
        return int(self.reader.header()['max_zoom'])

    def get_metadata(self):
        return self.reader.metadata()

    def _get_meta_prop(self, prop_name):
        metadata = self.reader.metadata()
        if prop_name in metadata:
            return metadata[prop_name]
        raise ValueError(f"Source does not have a {prop_name} property.")

 
# hybrid source that combines multiple tile sources in order
class StackedTileSource:
    def __init__(self, srcs):
        self.srcs = srcs

    def get_tile_data(self, tile):
        for src in self.srcs:
            try:
                return src.get_tile_data(tile)
            except MissingTileError:
                continue
        raise MissingTileError()

    def get_tile_size(self, tile):
        for src in self.srcs:
            try:
                return src.get_tile_size(tile)
            except MissingTileError:
                continue
        raise MissingTileError()

    def for_all_z(self, z):
        seen = set()
        for i, src in enumerate(self.srcs):
            #print(f'iterating over source {i} for {z}')
            for (tile, size) in src.for_all_z(z):
                if tile in seen:
                    continue
                seen.add(tile)
                yield (tile, size)

    def all(self):
        seen = set()
        for i, src in enumerate(self.srcs):
            #print(f'iterating over source {i} for all levels')
            for (tile, data) in src.all():
                if tile in seen:
                    continue
                seen.add(tile)
                yield (tile, data)

    def all_sizes(self):
        seen = set()
        for i, src in enumerate(self.srcs):
            #print(f'iterating over source {i} for all levels')
            for (tile, size) in src.all_sizes():
                if tile in seen:
                    continue
                seen.add(tile)
                yield (tile, size)

    def cleanup(self):
        for src in self.srcs:
            src.cleanup()

    @property
    def min_zoom(self):
        min_zooms = []
        for src in self.srcs:
            try:
                min_zooms.append(src.min_zoom)
            except ValueError:
                continue

        if len(min_zooms) == 0:
            raise ValueError("No zoom levels found in any of the sources.")

        return min(min_zooms)

    @property
    def max_zoom(self):
        max_zooms = []
        for src in self.srcs:
            try:
                max_zooms.append(src.max_zoom)
            except ValueError:
                continue

        if len(max_zooms) == 0:
            raise ValueError("No zoom levels found in any of the sources.")

        return max(max_zooms)

    def get_metadata(self):
        metadata = None
        for src in self.srcs:
            try:
                metadata = src.get_metadata()
                return metadata
            except ValueError:
                continue
        if metadata is None:
            raise ValueError("No metadata found in any of the sources.")
        return metadata

        
def create_source_from_paths(source_paths):
    sources = []
    for source_path in source_paths:
        if source_path.endswith('.mbtiles'):
            sources.append(MBTilesSource(source_path))
        elif source_path.endswith('.pmtiles'):
            pmtiles_files = glob.glob(source_path)
            if not pmtiles_files:
                raise ValueError(f"No PMTiles files found for pattern: {source_path}")
            for pmtiles_file in pmtiles_files:
                sources.append(PMTilesSource(pmtiles_file))
        else:
            if Path(source_path).is_dir():
                sources.append(DiskTilesSource(source_path))
            else:
                raise ValueError(f"Invalid source: {source_path}")

    if not sources:
        raise ValueError("No valid sources provided.")

    return StackedTileSource(sources)



import glob

from pathlib import Path

import mercantile

from pmtiles.reader import MmapSource, Reader as PMTilesReader, all_tiles

class MissingTileError(Exception):
    pass


class DiskSource:
    def __init__(self, directory):
        self.dir = Path(directory)

    def get_tile_from_file(self, file):
        parts = file.parts
        tile = mercantile.Tile(z=int(parts[-3]),
                               x=int(parts[-2]),
                               y=int(parts[-1].replace('.webp', '')))
        return tile

    def file_from_tile(self, tile):
        return self.dir / f'{tile.z}' / f'{tile.x}' / f'{tile.y}.webp'

    def get_tile_data(self, tile):
        file = self.file_from_tile(tile)
        if not file.exists():
            raise MissingTileError()

        return file.read_bytes()

    def get_tile_size(self, tile):
        file = self.file_from_tile(tile)
        if not file.exists():
            raise MissingTileError()

        return file.stat().st_size
        
    def for_all_z(self, z):
        for file in self.dir.glob(f'{z}/*/*.webp'):
            tile = self.get_tile_from_file(file)
            fstats = file.stat()
            yield (tile, fstats.st_size)

    def all(self):
        for file in self.dir.glob('*/*/*.webp'):
            tile = self.get_tile_from_file(file)
            t_data = file.read_bytes()
            yield (tile, t_data)

    def cleanup(self):
        pass

    def get_zomm_levels(self):
        zoom_levels = []
        for dir in self.dir.glob('*'):
            if not dir.is_dir():
                continue

            try:
                zoom = int(dir.name)
            except ValueError:
                continue
            zoom_levels.append(zoom)
        return zoom_levels

    @property
    def min_zoom(self):
        zoom_levels = self.get_zomm_levels()
        if len(zoom_levels) == 0:
            raise ValueError("No zoom directories found in the disk source.")

        return min(zoom_levels)

    @property
    def max_zoom(self):
        zoom_levels = self.get_zomm_levels()
        if len(zoom_levels) == 0:
            raise ValueError("No zoom directories found in the disk source.")

        return max(zoom_levels)

    @property
    def name(self):
        raise ValueError("DiskSource does not have a name property.")
        
    @property
    def description(self):
        raise ValueError("DiskSource does not have a description property.")

    @property
    def attribution(self):
        raise ValueError("DiskSource does not have a attribution property.")
        
class PartitionedPMTilesSource:
    def __init__(self, pmtiles_prefix):
        pmtiles_files = glob.glob(f'{pmtiles_prefix}*.pmtiles')
        self.files = []
        self.readers = {}
        for fname in pmtiles_files:
            file = open(fname, 'rb')
            self.files.append(file)
            src = MmapSource(file)
            reader = PMTilesReader(src)
            self.readers[fname] = reader

        self.fnames = list(pmtiles_files)
        fname_to_index = { fname:i for i,fname in enumerate(self.fnames) }
        self.tiles_to_findex = {}
        self.tiles_to_size = {}
        for fname in self.fnames:
            print(f'collecting tile sizes from {fname}')
            reader = self.readers[fname]
            for t, t_data in all_tiles(reader.get_bytes):
                tile = mercantile.Tile(x=t[1], y=t[2], z=t[0])
                self.tiles_to_findex[tile] = fname_to_index[fname]
                self.tiles_to_size[tile] = len(t_data)

    def get_reader_from_tile(self, tile):
        if tile not in self.tiles_to_findex:
            raise MissingTileError()
        s = self.tiles_to_findex[tile]
        fname = self.fnames[s]
        reader = self.readers[fname]
        return reader


    def get_tile_data(self, tile):
        reader = self.get_reader_from_tile(tile)
        data = reader.get(tile.z, tile.x, tile.y)
        if data is None:
            raise MissingTileError()
        return data

    def get_tile_size(self, tile):
        if tile not in self.tiles_to_size:
            raise MissingTileError()
        return self.tiles_to_size[tile]

    def for_all_z(self, z):
        for tile, size in self.tiles_to_size.items():
            if tile.z == z:
                yield (tile, size)

    def all(self):
        for suffix, reader in self.readers.items():
            print(f'yielding from {suffix}')
            for t, t_data in all_tiles(reader.get_bytes):
                tile = mercantile.Tile(x=t[1], y=t[2], z=t[0])
                yield tile, t_data

    def cleanup(self):
        for f in self.files:
            f.close()

    @property
    def min_zoom(self):
        return min([int(reader.header()['min_zoom']) for reader in self.readers.values()])

    @property
    def max_zoom(self):
        return max([int(reader.header()['max_zoom']) for reader in self.readers.values()])

    def get_meta_prop(self, prop_name):
        for reader in self.readers.values():
            metadata = reader.metadata()
            if prop_name in metadata:
                return metadata[prop_name]
        raise ValueError(f"Source does not have a {prop_name} property.")

    @property
    def name(self):
        return self.get_meta_prop('name')
        
    @property
    def description(self):
        return self.get_meta_prop('description')

    @property
    def attribution(self):
        return self.get_meta_prop('attribution')
        

 

class DiskAndPartitionedPMTilesSource:
    def __init__(self, directory, pmtiles_prefix):
        self.dsrc = DiskSource(directory)
        self.psrc = PartitionedPMTilesSource(pmtiles_prefix)
        
    def get_tile_data(self, tile):
        try:
            return self.dsrc.get_tile_data(tile)
        except MissingTileError:
            return self.psrc.get_tile_data(tile)

    def get_tile_size(self, tile):
        try:
            return self.dsrc.get_tile_size(tile)
        except MissingTileError:
            return self.psrc.get_tile_size(tile)

    def for_all_z(self, z):
        print(f'iterating over all {z}')
        seen = set()
        for (tile, size) in self.dsrc.for_all_z(z):
            seen.add(tile)
            yield (tile, size)

        print(f'iterated over all {z} disk')
        for (tile, size) in self.psrc.for_all_z(z):
            if tile in seen:
                continue
            yield (tile, size)
        print(f'iterated over all {z} pmtiles')

    def all(self):
        print('yielding from disk')
        for res in self.dsrc.all():
            yield res
        print('yielding from pmtiles files')
        for res in self.psrc.all():
            yield res

    def cleanup(self):
        self.dsrc.cleanup()
        self.psrc.cleanup()

    @property
    def min_zoom(self):
        min_zooms = []

        try:
            min_disk_zoom = self.dsrc.min_zoom
            min_zooms.append(min_disk_zoom)
        except ValueError:
            pass

        try:
            min_pmtiles_zoom = self.psrc.min_zoom
            min_zooms.append(min_pmtiles_zoom)
        except ValueError:
            pass

        if len(min_zooms) == 0:
            raise ValueError("No zoom directories found in the disk source or PMTiles source.")

        return min(min_zooms)

    @property
    def max_zoom(self):
        max_zooms = []

        try:
            max_disk_zoom = self.dsrc.max_zoom
            max_zooms.append(max_disk_zoom)
        except ValueError:
            pass

        try:
            max_pmtiles_zoom = self.psrc.max_zoom
            max_zooms.append(max_pmtiles_zoom)
        except ValueError:
            pass

        if len(max_zooms) == 0:
            raise ValueError("No zoom directories found in the disk source or PMTiles source.")

        return max(max_zooms)

    @property
    def name(self):
        return self.psrc.name
        
    @property
    def description(self):
        return self.psrc.description

    @property
    def attribution(self):
        return self.psrc.attribution
        



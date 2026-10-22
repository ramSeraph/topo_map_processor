import contextlib
import io
import json
import multiprocessing
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mercantile
import numpy as np
from PIL import Image
from osgeo import gdal, osr
from osgeo_utils.gdal2tiles import submain as gdal2tiles_main

from topo_map_processor.tools.retile import (
    Tiler, base_tile_batches, create_dummy_sheets,
)


class SerialPool:
    _processes = 1

    def imap_unordered(self, function, items, chunksize=1):
        return map(function, items)


class FailingPool(SerialPool):
    def __init__(self):
        self.calls = 0

    def imap_unordered(self, function, items, chunksize=1):
        self.calls += 1
        for index, item in enumerate(items):
            if self.calls == 2 and index == 1:
                raise RuntimeError('simulated worker failure')
            yield function(item)


class OldTileSource:
    min_zoom = 4
    max_zoom = 6

    def __init__(self, tile_extension='png'):
        self.tile_extension = tile_extension
        image = Image.new('RGBA', (256, 256), (19, 39, 59, 255))
        buffer = io.BytesIO()
        image.save(buffer, format=tile_extension.upper())
        self.tile = buffer.getvalue()

    def get_tile_data(self, tile):
        return self.tile

    def get_metadata(self):
        return {'format': self.tile_extension, 'minzoom': self.min_zoom, 'maxzoom': self.max_zoom}


class SinglePassTiler(Tiler):
    """Reference: render the complete rectangle using unmodified GDAL2Tiles."""

    def create_base_tiles(self, sheets, max_zoom, affected_tiles, background_sheets=()):
        vrt = self.create_vrt_file(sheets, background_sheets)
        gdal2tiles_main([
            'gdal2tiles.py', '-r', 'lanczos', '--quiet', '-w', 'none',
            '--xyz', '-z', str(max_zoom),
        ] + self.get_tiler_cmd_params() + [str(vrt), str(self.disk_handler.dir)])
        vrt.unlink()


class RetileBatchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.tiffs = self.root / 'tiffs'
        self.tiffs.mkdir()
        self.old_source = OldTileSource()
        gdal.UseExceptions()

    def write_sheet(self, name, left, top, width, height, phase, transparent=False):
        # Fractional origins and pixel sizes exercise resampling and overlap
        # alignment, not just copying a solid, tile-aligned raster.
        zoom = self.old_source.max_zoom
        upper_left = mercantile.xy_bounds(mercantile.Tile(0, 0, zoom))
        tile_span = upper_left.right - upper_left.left
        resolution = tile_span / 320
        xsize, ysize = round(width * 320), round(height * 320)
        dataset = gdal.GetDriverByName('GTiff').Create(
            str(self.tiffs / name), xsize, ysize, 3, gdal.GDT_Byte)
        dataset.SetGeoTransform((
            upper_left.left + left * tile_span, resolution, 0,
            upper_left.top - top * tile_span, 0, -resolution,
        ))
        projection = osr.SpatialReference()
        projection.ImportFromEPSG(3857)
        dataset.SetProjection(projection.ExportToWkt())
        yy, xx = np.indices((ysize, xsize))
        for band, color in enumerate((gdal.GCI_RedBand, gdal.GCI_GreenBand, gdal.GCI_BlueBand), 1):
            dataset.GetRasterBand(band).SetColorInterpretation(color)
            values = ((xx * (band + 1) + yy * band + phase) % 240 + 15).astype(np.uint8)
            dataset.GetRasterBand(band).WriteArray(values)
        dataset.CreateMaskBand(gdal.GMF_PER_DATASET)
        # Irregular transparent holes expose lower-priority sheets in overlaps.
        mask = np.where((xx // 43 + yy // 37 + phase) % 7 == 0, 0, 255).astype(np.uint8)
        if transparent:
            mask[:] = 0
        dataset.GetRasterBand(1).GetMaskBand().WriteArray(mask)
        dataset = None

    def overlapping_sheets(self):
        self.write_sheet('a.tif', 30.1, 24.1, 3.2, 2.3, 17)
        self.write_sheet('b.tif', 31.45, 24.2, 2.8, 2.4, 83)
        self.write_sheet('c.tif', 30.05, 25.65, 4.5, 1.4, 129)
        # Two holes in the affected region require unchanged siblings from the
        # old tile source when their common parents are regenerated.
        return {mercantile.Tile(x, y, 6) for x in range(30, 35) for y in range(24, 28)} - {
            mercantile.Tile(31, 25, 6), mercantile.Tile(32, 26, 6),
        }

    def make_tiler(self, name, batch_size=2, pool=None, cls=Tiler):
        return cls(self.root / name, self.tiffs, self.old_source,
                   self.old_source.tile_extension, 75, pool or SerialPool(), batch_size=batch_size)

    def tile_pixels(self, directory):
        return {str(path.relative_to(directory)): np.array(Image.open(path).convert('RGBA'))
                for path in directory.glob('*/*/*') if path.suffix in ('.png', '.webp')}

    def assert_same_tiles(self, expected, actual):
        left, right = self.tile_pixels(expected), self.tile_pixels(actual)
        self.assertEqual(set(left), set(right))
        for tile in left:
            np.testing.assert_array_equal(left[tile], right[tile], err_msg=tile)

    def test_sparse_batches_have_exclusive_bounded_ownership(self):
        tiles = {mercantile.Tile(11000 + x, 7000 + y, 14) for x in range(3) for y in range(4)}
        tiles |= {mercantile.Tile(14000, 1000, 14)}
        batches = base_tile_batches(tiles)
        self.assertEqual(set().union(*batches), tiles)
        self.assertEqual(sum(map(len, batches)), len(tiles))
        self.assertEqual(batches, base_tile_batches(reversed(sorted(tiles))))
        for batch in batches:
            self.assertLess(max(t.x for t in batch) - min(t.x for t in batch), 32)
            self.assertLess(max(t.y for t in batch) - min(t.y for t in batch), 32)
        self.assertEqual(base_tile_batches([]), [])
        with self.assertRaises(ValueError):
            base_tile_batches(tiles, 0)

    def test_overlaps_and_shared_parents_match_single_pass(self):
        affected = self.overlapping_sheets()
        reference = self.make_tiler('reference', cls=SinglePassTiler)
        with contextlib.redirect_stdout(io.StringIO()):
            reference.retile(['a.tif', 'b.tif', 'c.tif'], affected)
            for batch_size in (1, 2, 32):
                actual = self.make_tiler(f'batch-{batch_size}', batch_size)
                # A stale parent must not survive merely because its path exists.
                stale_parent = actual.disk_handler.dir / '5/15/12.png'
                stale_parent.parent.mkdir(parents=True)
                stale_parent.write_bytes(self.old_source.tile)
                actual.retile(['c.tif', 'b.tif', 'a.tif'], affected)
                self.assert_same_tiles(reference.disk_handler.dir, actual.disk_handler.dir)
            # Batch execution order must not decide which pixels win an overlap.
            batches = list(reversed(base_tile_batches(affected, 2)))
            with patch('topo_map_processor.tools.retile.base_tile_batches', return_value=batches):
                actual = self.make_tiler('reversed')
                actual.retile(['b.tif', 'a.tif', 'c.tif'], affected)
                self.assert_same_tiles(reference.disk_handler.dir, actual.disk_handler.dir)
        parent = self.tile_pixels(reference.disk_handler.dir)['5/15/12.png']
        np.testing.assert_array_equal(parent[200, 200], [19, 39, 59, 255])
        self.assertFalse(np.array_equal(parent[50, 50], [19, 39, 59, 255]))

    def test_webp_overlaps_and_parents_match_single_pass(self):
        self.old_source = OldTileSource('webp')
        affected = self.overlapping_sheets()
        reference = self.make_tiler('reference', cls=SinglePassTiler)
        actual = self.make_tiler('batched')
        with contextlib.redirect_stdout(io.StringIO()):
            reference.retile(['a.tif', 'b.tif', 'c.tif'], affected)
            actual.retile(['c.tif', 'b.tif', 'a.tif'], affected)
        self.assert_same_tiles(reference.disk_handler.dir, actual.disk_handler.dir)

    def test_process_workers_match_single_pass(self):
        affected = self.overlapping_sheets()
        reference = self.make_tiler('reference', cls=SinglePassTiler)
        with contextlib.redirect_stdout(io.StringIO()):
            reference.retile(['a.tif', 'b.tif', 'c.tif'], affected)
            with multiprocessing.get_context('spawn').Pool(2) as pool:
                actual = self.make_tiler('parallel', pool=pool)
                actual.retile(['a.tif', 'b.tif', 'c.tif'], affected)
        self.assert_same_tiles(reference.disk_handler.dir, actual.disk_handler.dir)

    def test_failed_batch_does_not_publish_partial_tiles(self):
        self.write_sheet('a.tif', 30, 24, 4, 2, 17)
        affected = {mercantile.Tile(x, 24, 6) for x in range(30, 34)}
        actual = self.make_tiler('failed', pool=FailingPool())
        actual.disk_handler.init()
        for tile in affected:
            actual.disk_handler.preprare_dir(tile)
            actual.disk_handler.get_tile_file(tile).write_bytes(self.old_source.tile)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'simulated worker failure'):
            actual.create_base_tiles(['a.tif'], 6, affected)
        for tile in affected:
            contents = actual.disk_handler.get_tile_file(tile).read_bytes()
            if tile.x < 32:
                self.assertNotEqual(contents, self.old_source.tile)
            else:
                self.assertEqual(contents, self.old_source.tile)
        self.assertFalse(list(self.root.glob('.retile-batches-*')))

    def test_cleared_tiles_outside_source_are_transparent_replacements(self):
        self.write_sheet('a.tif', 30, 24, 1, 1, 17)
        removed = mercantile.Tile(40, 34, 6)
        actual = self.make_tiler('cleared')
        affected = {mercantile.Tile(30, 24, 6), removed}
        with contextlib.redirect_stdout(io.StringIO()):
            actual.retile(['a.tif'], affected)
        pixels = self.tile_pixels(actual.disk_handler.dir)
        self.assertEqual(set(t for t in pixels if t.startswith('6/')), {'6/30/24.png', '6/40/34.png'})
        self.assertFalse(pixels['6/40/34.png'][:, :, 3].any())
        # Other children of the removed tile's parent still come from the old source.
        self.assertEqual(pixels['5/20/17.png'][200, 200, 3], 255)

    def test_all_sheets_removed_uses_transparent_background(self):
        tile = mercantile.Tile(30, 24, 6)
        bounds = mercantile.bounds(tile)
        force_file = self.root / 'force.geojson'
        force_file.write_text(json.dumps({'type': 'FeatureCollection', 'features': [{
            'type': 'Feature', 'properties': {}, 'geometry': {
                'type': 'Polygon', 'coordinates': [[
                    [bounds.west, bounds.north], [bounds.east, bounds.north],
                    [bounds.east, bounds.south], [bounds.west, bounds.south],
                    [bounds.west, bounds.north],
                ]],
            },
        }]}))
        actual = self.make_tiler('empty')
        with contextlib.redirect_stdout(io.StringIO()):
            background = create_dummy_sheets(force_file, self.tiffs)
            actual.retile([], {tile}, background)
        self.assertFalse(self.tile_pixels(actual.disk_handler.dir)['6/30/24.png'][:, :, 3].any())


if __name__ == '__main__':
    unittest.main()

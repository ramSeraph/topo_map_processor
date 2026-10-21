import contextlib
import io
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from topo_map_processor.processor import TopoMapProcessor


class RotationTests(unittest.TestCase):
    def rotate_rectangle(self, rectangle, threshold=0):
        with tempfile.TemporaryDirectory() as directory:
            workdir = Path(directory)
            source = np.full((100, 60, 3), 255, dtype=np.uint8)
            source[10:30, 10:30] = (0, 0, 255)
            source[70:90, 30:50] = (255, 0, 0)
            filepath = workdir / 'full.jpg'
            cv2.imwrite(str(filepath), source)
            processor = TopoMapProcessor(filepath, {'auto_rotate_thresh': threshold}, None)

            # Exercise the normal cache read as well as rotation. Cached rectangles
            # can have either angle convention, regardless of installed OpenCV.
            contour = cv2.boxPoints(rectangle)
            with (workdir / 'maparea_info.pkl').open('wb') as cache:
                pickle.dump((cv2.boundingRect(contour), rectangle, contour), cache)
            (workdir / 'small.jpg').touch()
            with patch.object(processor, 'get_workdir', return_value=workdir), \
                    patch.object(processor, 'rotate_image_bound', wraps=processor.rotate_image_bound) as rotate, \
                    contextlib.redirect_stdout(io.StringIO()):
                processor.rotate()
                angle, status = (workdir / 'rotated_info.txt').read_text().split(', ')
                reversal = processor.get_rotation_reversal_params()
                output_path = workdir / 'full.rotated.jpg'
                output = cv2.imread(str(output_path)) if output_path.exists() else None
                applied_angle = rotate.call_args.args[1] if rotate.called else None
            return float(angle), status, applied_angle, reversal, output

    def test_upright_rectangles_stay_upright(self):
        for raw_angle in (-90., 0., 90.):
            with self.subTest(raw_angle=raw_angle):
                angle, status, applied, reversal, output = self.rotate_rectangle(
                    ((100., 100.), (60., 100.), raw_angle))
                self.assertAlmostEqual(angle, 0.)
                self.assertAlmostEqual(applied, 0.)
                self.assertEqual(output.shape, (100, 60, 3))
                self.assertGreater(int(output[20, 20, 2]), 200)
                self.assertGreater(int(output[80, 40, 0]), 200)
                self.assertAlmostEqual(reversal.angle, 0.)

    def test_small_skews_keep_their_direction_in_both_conventions(self):
        cases = [(-89., 1.), (-80., 10.), (-10., -10.), (-1., -1.),
                 (1., 1.), (10., 10.), (80., -10.), (89., -1.)]
        for raw_angle, expected in cases:
            with self.subTest(raw_angle=raw_angle):
                angle, status, applied, reversal, output = self.rotate_rectangle(
                    ((100., 100.), (60., 100.), raw_angle))
                self.assertAlmostEqual(angle, expected)
                self.assertAlmostEqual(applied, -expected)
                self.assertEqual(status, 'rotated')
                self.assertTrue(reversal.rotated)
                self.assertAlmostEqual(reversal.angle, -expected)

    def test_upright_rectangle_respects_rotation_threshold(self):
        angle, status, applied, reversal, output = self.rotate_rectangle(
            ((100., 100.), (60., 100.), -90.), threshold=0.5)
        self.assertAlmostEqual(angle, 0.)
        self.assertEqual(status, 'not_rotated')
        self.assertIsNone(applied)
        self.assertIsNone(output)
        self.assertFalse(reversal.rotated)

    def test_rectangles_from_installed_opencv(self):
        for tilt in (-10., -1., 0., 1., 10.):
            with self.subTest(tilt=tilt):
                corners = cv2.boxPoints(((100., 100.), (60., 100.), tilt))
                rectangle = cv2.minAreaRect(corners)
                angle, status, applied, reversal, output = self.rotate_rectangle(rectangle)
                self.assertAlmostEqual(angle, tilt, places=4)
                self.assertAlmostEqual(applied, -tilt, places=4)


if __name__ == '__main__':
    unittest.main()

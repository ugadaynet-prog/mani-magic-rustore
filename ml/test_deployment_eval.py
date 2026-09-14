"""Regression tests for the offline native/JS reference (no phone required).

Run: python -m unittest discover -s ml -p test_deployment_eval.py
These tests validate discrete operations, not Pillow versus Skia raster parity.
"""
import json
import pathlib
import shutil
import subprocess
import unittest

import numpy as np
from PIL import Image

import deployment_eval as D


def skin_input(side):
    return np.full((side, side, 3), [200, 150, 120], dtype=np.uint8)


class InputGeometryTests(unittest.TestCase):
    def test_resize_then_integer_padding_matches_native_geometry(self):
        im = Image.new('RGB', (7, 3), (200, 100, 50))
        got = np.asarray(D.prepare_input(im, 10))
        self.assertEqual(got.shape, (10, 10, 3))
        self.assertTrue((got[:3] == 0).all())
        self.assertTrue((got[3:7] == [200, 100, 50]).all())
        self.assertTrue((got[7:] == 0).all())

    def test_fractional_ui_geometry_returns_original_shape(self):
        prepared = D.prepare_input(Image.new('RGB', (31, 17), (200, 150, 120)), 20)
        prob = np.zeros((20, 20), dtype=np.float32)
        prob[7:13, 7:13] = .8
        got = D.postprocess(prob, prepared, (31, 17))
        self.assertEqual(got.shape, (17, 31))
        self.assertEqual(got.dtype, bool)
        self.assertTrue(got[8, 15])
        self.assertFalse(got[0, 0])


class NativeFilterTests(unittest.TestCase):
    def test_tiny_skin_surrounded_component_survives(self):
        prob = np.zeros((30, 30), dtype=np.float32)
        prob[15, 15] = .9
        self.assertAlmostEqual(float(D.suppress_stray(prob, skin_input(30))[15, 15]), .9, places=6)

    def test_rejection_preserves_low_probability_halo(self):
        prob = np.zeros((30, 30), dtype=np.float32)
        prob[13:18, 13:18] = .45
        prob[14:17, 14:17] = .9
        got = D.suppress_stray(prob, np.zeros((30, 30, 3), np.uint8))
        self.assertLess(got[15, 15], 1e-10)
        self.assertAlmostEqual(float(got[13, 13]), .45, places=6)

    def test_ring_under_twenty_is_not_rejected(self):
        prob = np.ones((10, 10), dtype=np.float32) * .9
        prob[0, 0] = 0
        got = D.suppress_stray(prob, np.zeros((10, 10, 3), np.uint8))
        np.testing.assert_array_equal(got, prob)

    def test_square_ring_includes_corner_and_excludes_other_foreground(self):
        # Single foreground component in a 15x15 square. All background in
        # its radius-7 square is skin; this must retain the component.
        side = 15
        prob = np.ones((side, side), np.float32) * .9
        prob[0, :] = 0
        prob[-1, :] = 0
        prob[:, 0] = 0
        prob[:, -1] = 0
        rgb = np.zeros((side, side, 3), np.uint8)
        rgb[prob == 0] = [200, 150, 120]
        np.testing.assert_array_equal(D.suppress_stray(prob, rgb), prob)

    def test_normalized_float_input_equals_png_rgb_input(self):
        prob = np.zeros((30, 30), np.float32)
        prob[10:20, 10:20] = .9
        rgb = skin_input(30)
        np.testing.assert_array_equal(D.suppress_stray(prob, rgb),
                                      D.suppress_stray(prob, rgb.astype(np.float32)/255))


class FrontendTests(unittest.TestCase):
    def test_only_enclosed_small_holes_filled(self):
        mask = np.zeros((30, 30), bool)
        mask[2:28, 2:28] = True
        mask[4:8, 4:8] = False        # area16 => fill
        mask[10:18, 10:18] = False    # area64 > max(40, area*.04)
        mask[0:8, 22:25] = False     # connected to exterior => leave
        got = D.fill_holes(mask)
        self.assertTrue(got[5, 5])
        self.assertFalse(got[14, 14])
        self.assertFalse(got[5, 23])

    def test_png_float32_boundary_matches_js(self):
        # Kotlin floor(.4*255)=102; Float32(102/255) is greater than JS .4,
        # so the JS bin mask becomes foreground and maskCanvas makes it255.
        prob = np.full((10, 10), np.float32(.4))
        got = D.postprocess(prob, skin_input(10), (10, 10))
        self.assertTrue(got.all())
        below = np.full((10, 10), np.float32(101/255))
        self.assertFalse(D.postprocess(below, skin_input(10), (10, 10)).any())

    def test_actual_js_fill_holes_matches_reference(self):
        node = shutil.which('node')
        if not node:
            self.skipTest('Node unavailable; cannot execute source JS parity check')
        source_path = pathlib.Path(__file__).resolve().parents[1] / 'app-addons/tryon/tryon.js'
        source = source_path.read_text(encoding='utf-8')
        start = source.index('  function fillHoles(')
        end = source.index('  function maskCanvas(', start)
        script = ('const fs=require("fs"); const HOLE_MAX_FRAC=.04;\n' + source[start:end]
                  + '\nconst cases=JSON.parse(fs.readFileSync(0,"utf8"));'
                  + 'process.stdout.write(JSON.stringify(cases.map(c=>'
                  + 'Array.from(fillHoles(Uint8Array.from(c.mask),c.side)))));')
        rng = np.random.default_rng(11)
        cases = []
        for side in (9, 15, 31):
            for fraction in (.1, .5, .9):
                mask = rng.random((side, side)) < fraction
                cases.append({'side': side, 'mask': mask.astype(int).ravel().tolist()})
        result = subprocess.run([node, '-e', script], input=json.dumps(cases),
                                text=True, capture_output=True, check=True, timeout=20)
        actual = json.loads(result.stdout)
        for case, js in zip(cases, actual):
            shape = (case['side'], case['side'])
            expected = D.fill_holes(np.array(case['mask'], dtype=bool).reshape(shape))
            np.testing.assert_array_equal(np.array(js, dtype=bool).reshape(shape), expected)


class MetricsTests(unittest.TestCase):
    def test_attached_skin_paint_is_counted_in_fp_not_historical_stray(self):
        gt = np.zeros((100, 100), np.uint8)
        gt[20:40, 20:40] = 1
        pred = gt > 0
        pred[20:40, 40:70] = True
        row = D.score_frame(gt, pred)
        self.assertEqual((row['nails'], row['found']), (1, 1))
        self.assertEqual((row['stray'], row['stray_px']), (0, 0))
        self.assertEqual((row['tp'], row['fp'], row['fn']), (400, 600, 0))
        self.assertAlmostEqual(row['precision'], .4)
        self.assertAlmostEqual(row['fp_rate'], 1.5)
        self.assertAlmostEqual(row['iou'], .4)
        self.assertLess(row['boundary_fscore'], 1)

    def test_component_iou_is_separate_for_each_nail(self):
        gt = np.zeros((80, 80), np.uint8)
        gt[10:20, 10:20] = 1
        gt[40:50, 40:50] = 2
        pred = gt > 0
        pred[60:70, 60:70] = True
        row = D.score_frame(gt, pred)
        self.assertEqual(row['per_nail'], {'1': 1.0, '2': 1.0})
        self.assertEqual((row['stray'], row['stray_px']), (1, 100))
        self.assertEqual(row['fp'], 100)

    def test_empty_gt_is_finite_and_macro_aggregation_is_explicit(self):
        gt = np.zeros((20, 20), np.uint8)
        empty = D.score_frame(gt, gt > 0)
        self.assertIsNone(empty['fp_rate'])
        self.assertEqual(empty['fp_frame_rate'], 0)
        self.assertEqual(empty['boundary_fscore'], 1)
        positive_gt = gt.copy()
        positive_gt[5:10, 5:10] = 1
        row = D.score_frame(positive_gt, positive_gt > 0)
        agg = D.aggregate([row, empty])
        self.assertEqual(agg['recall'], 1)
        self.assertEqual(agg['fp_rate'], 0)
        self.assertEqual(agg['frames'], 2)
        self.assertIsNone(D.aggregate([])['recall'])
        # Fully negative images contribute false paint to aggregate FP.
        false = D.score_frame(gt, np.ones_like(gt, dtype=bool))
        self.assertEqual(false['precision'], 0)
        self.assertEqual(false['boundary_fscore'], 0)
        self.assertEqual(D.aggregate([row, false])['fp_rate'], 16)

    def test_sigmoid_handles_extreme_logits(self):
        result = D.sigmoid(np.array([-1e10, 0, 1e10], np.float32))
        np.testing.assert_array_equal(result, [0, .5, 1])


if __name__ == '__main__':
    unittest.main()

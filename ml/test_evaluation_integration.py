"""Regression tests for historical compatibility, guard and reproducibility."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import benchmark
import exam
import finetune
import train


class EvaluationIntegrationTest(unittest.TestCase):
    def test_attached_paint_is_measured_without_redefining_old_stray(self):
        gt = np.zeros((80, 80), dtype=np.uint8)
        gt[20:40, 20:40] = 1
        pred = gt > 0
        pred[20:40, 40:70] = True
        row = exam.score_frame(gt, pred)
        self.assertEqual(row['found'], 1)
        self.assertEqual(row['stray_px'], 0)
        self.assertEqual(row['fp'], 600)
        self.assertAlmostEqual(row['iou'], .4)

    def test_guard_rejects_skin_spill_even_when_recall_improves(self):
        base = {'parts': {'own': {'found': 8, 'fp': 30}}}
        candidate = {'parts': {'own': {'found': 9, 'fp': 31}}}
        self.assertFalse(finetune.passes_guard(candidate, base))
        candidate['parts']['own']['fp'] = 29
        self.assertTrue(finetune.passes_guard(candidate, base))

    def test_compare_distinguishes_gains_and_losses_on_same_image(self):
        base = [dict(key='labels/x', per_nail={'1': .9, '2': None, '3': .8})]
        candidate = [dict(key='labels/x', per_nail={'1': None, '2': .7, '3': .85})]
        result = benchmark.compare(base, candidate)
        self.assertEqual(result['gained'], ['labels/x/2'])
        self.assertEqual(result['lost'], ['labels/x/1'])
        self.assertAlmostEqual(result['common_iou_candidate'], .85)
        self.assertAlmostEqual(result['common_iou_baseline'], .8)

    def test_legacy_color_synthesis_obeys_numpy_seed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images, masks = root / 'images', root / 'masks'
            images.mkdir()
            masks.mkdir()
            rgb = np.full((32, 32, 3), (180, 130, 110), dtype=np.uint8)
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[8:24, 8:24] = 255
            Image.fromarray(rgb).save(images / '1.jpg')
            Image.fromarray(mask).save(masks / '1.png')
            dataset = train.NailDataset(str(images), str(masks), size=32,
                                        files=['1.jpg'], augment=False, dark_p=1)
            np.random.seed(123)
            a = dataset[0][0].numpy()
            np.random.seed(123)
            b = dataset[0][0].numpy()
            self.assertTrue(np.array_equal(a, b))


if __name__ == '__main__':
    unittest.main()

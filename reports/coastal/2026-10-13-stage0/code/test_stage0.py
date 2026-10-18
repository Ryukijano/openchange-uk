import unittest

import numpy as np
import pandas as pd
from run_stage0 import contour, match_surveys, retain, scene_quantile, utc


class Stage0Tests(unittest.TestCase):
    def test_contour(self):
        value, direct = contour([0, 10, 20], [1, 0.2, -0.8])
        self.assertAlmostEqual(value, 12)
        self.assertAlmostEqual(direct, 12)

    def test_invalid_contours(self):
        self.assertIsNone(contour([0, 1, 2], [2, 1, 0.1]))
        self.assertIsNone(contour([0, 1, 2, 3, 4], [2, -1, 2, -1, -2]))

    def test_exact_hours_not_truncated_days(self):
        survey = pd.DataFrame({"profile": ["PF1"], "survey_date": utc(["2020-01-03"]), "truth": [12]})
        frame = pd.DataFrame({"profile": ["PF1"] * 3,
                              "date": utc(["2020-01-01", "2019-12-31 23:59", "2020-01-05 00:01"]),
                              "position": [15, 15, 15]})
        matched = match_surveys(frame, survey)
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched.iloc[0].gap_hours, 48)
        self.assertEqual(matched.iloc[0].error, 3)

    def test_retains_whole_scenes(self):
        frame = pd.DataFrame({"scene": ["a", "a", "b", "b", "c", "c"]})
        keep = retain(frame, np.array([1, 1, 2, 2, 3, 3]), 0.5)
        self.assertEqual(keep.tolist(), [True, True, True, True, False, False])

    def test_calibration_is_scene_max(self):
        frame = pd.DataFrame({"scene": ["a", "a", "b", "b", "c", "c"]})
        self.assertEqual(scene_quantile(frame, [1, 7, 2, 6, 3, 5], 0.5), 6)


if __name__ == "__main__":
    unittest.main()

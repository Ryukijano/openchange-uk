import unittest

import numpy as np
import pandas as pd
from audit_surveys import (
    load_submission,
    match_surveys,
    projected_chainage,
    unique_crossing,
)


class ContourTests(unittest.TestCase):
    def test_linear(self):
        self.assertAlmostEqual(unique_crossing([0, 10, 20], [1, .5, -.5]), 15)

    def test_datum_shift(self):
        self.assertAlmostEqual(unique_crossing([0, 10, 20], [2, 1, 0], .774), 12.26)

    def test_missing_cells_not_bridged(self):
        self.assertIsNone(unique_crossing([0, 10, 20], [1, np.nan, -1]))

    def test_multiple_crossings(self):
        self.assertIsNone(unique_crossing([0, 10, 20, 30], [1, -1, 1, -1]))

    def test_upward_then_downward_is_ambiguous(self):
        self.assertIsNone(unique_crossing([0, 10, 20], [-1, 1, -1]))

    def test_reverse_order(self):
        self.assertAlmostEqual(unique_crossing([20, 10, 0], [-1, 0, 1]), 10)

    def test_projection_origin(self):
        class Identity:
            @staticmethod
            def transform(x, y):
                return x, y
        chain, offaxis = projected_chainage([20, 30, 40], [0, 0, 0], Identity(), (np.array([20, 0]), np.array([1, 0])))
        np.testing.assert_allclose(chain, [0, 10, 20])
        np.testing.assert_allclose(offaxis, 0)

    def test_exact_hours(self):
        surveys = pd.DataFrame({'profile': ['A'], 'survey_date': pd.to_datetime(['2020-01-01'], utc=True), 'truth': [1.]})
        frame = pd.DataFrame({'profile': ['A', 'A'], 'date': pd.to_datetime(['2020-01-03T00:00:00Z', '2020-01-03T00:00:01Z']), 'position': [2., 2.]})
        result = match_surveys(frame, surveys)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.gap_hours.iloc[0], 48.)

    def test_mixed_timestamp_resolutions(self):
        surveys = pd.DataFrame({'profile': ['A'], 'survey_date': pd.to_datetime(['2020-01-01'], utc=True).astype('datetime64[us, UTC]'), 'truth': [1.]})
        frame = pd.DataFrame({'profile': ['A'], 'date': pd.to_datetime(['2020-01-02'], utc=True).astype('datetime64[ns, UTC]'), 'position': [2.]})
        self.assertEqual(len(match_surveys(frame, surveys)), 1)

    def test_missing_submission_preserves_datetime_dtype(self):
        empty = load_submission('NARRABEEN', 'team_SHOREX', 'nonexistent', True)
        full = load_submission('NARRABEEN', 'team_CoastSat', 'PF1')
        merged = pd.concat([empty, full], ignore_index=True)
        self.assertEqual(str(merged.date.dtype), 'datetime64[ns, UTC]')


if __name__ == '__main__':
    unittest.main()

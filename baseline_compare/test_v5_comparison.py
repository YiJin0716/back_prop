"""Paired ROC cohorts must keep the same labels, misses and exclusions."""
import unittest
import numpy as np
import pandas as pd

from .v5_comparison import paired_rows, draw_comparison


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.volumes = pd.DataFrame([
            dict(case_id=k, patient_id=k, mode=mode, any_reader_target=y, predicted_risk=p)
            for mode in ('exclude_3', 'ge_3', 'gt_3')
            for k, y, p in [('negative', 0, .1), ('positive', 1, .9),
                            ('ambiguous', np.nan if mode == 'exclude_3' else int(mode == 'ge_3'), .4)]])
        self.scores = dict(deeplung=dict(negative=0., positive=.8, ambiguous=.3),
                           sybil=dict(negative=.7, positive=.7, ambiguous=.7))

    def test_common_exclusions_and_tied_scores(self):
        rows = paired_rows(self.volumes, self.scores)
        fig, metrics = draw_comparison(rows)
        try:
            excluded = [r for r in metrics if r['mode'] == 'exclude_3']
            self.assertEqual([r['n'] for r in excluded], [2, 2, 2])
            self.assertEqual([r['excluded'] for r in excluded], [1, 1, 1])
            self.assertEqual([r['auc'] for r in excluded], [1., 1., .5])
            self.assertEqual(len(rows), 9)  # Zero detections remain as zero-risk CTs.
            self.assertEqual(len(fig.axes[0].lines), 4)  # Three models plus diagonal.
        finally:
            import matplotlib.pyplot as plt
            plt.close(fig)

    def test_missing_prediction_rejected_instead_of_dropping_case(self):
        del self.scores['sybil']['ambiguous']
        with self.assertRaises(AssertionError):
            paired_rows(self.volumes, self.scores)

    def test_nonfinite_or_out_of_range_prediction_rejected(self):
        for invalid in (float('nan'), -1., 1.1):
            self.scores['sybil']['positive'] = invalid
            with self.assertRaises(AssertionError):
                paired_rows(self.volumes, self.scores)

    def test_duplicate_mode_cannot_replace_missing_label_rule(self):
        self.volumes.loc[self.volumes['mode'] == 'gt_3', 'mode'] = 'ge_3'
        with self.assertRaises(AssertionError):
            paired_rows(self.volumes, self.scores)


if __name__ == '__main__':
    unittest.main()

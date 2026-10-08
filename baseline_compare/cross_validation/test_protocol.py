"""Small independent tests of patient pairing and fold/cohort handoff."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from .build import HERE_CV
from .statistics import summarize, save_results
from ..cohort import load


class ProtocolTest(unittest.TestCase):
    def records(self):
        return [{'key': f'f{f}c{i}', 'patient_id': f'p{f}_{i}', 'fold': f,
                 'included': True, 'label': i % 2,
                 'scores': {'v3': float(i % 2), 'sybil': .5, 'deeplung': float(1 - i % 2)}}
                for f in range(5) for i in range(20)]

    def test_known_aucs_and_paired_differences(self):
        rows = self.records(); keys = [r['key'] for r in rows]
        report, _ = summarize(rows, keys, repetitions=40)
        self.assertEqual([report['models'][m]['auc'] for m in ('v3', 'sybil', 'deeplung')], [1., .5, 0.])
        self.assertEqual(report['paired_auc_differences']['sybil']['v3_minus_baseline'], .5)
        self.assertEqual(report['paired_auc_differences']['deeplung']['ci95'], [1., 1.])

    def test_exclusions_and_duplicate_rejection(self):
        rows = self.records()
        rows.append({'key': 'excluded', 'patient_id': 'extra', 'fold': 0, 'included': False, 'label': 0, 'scores': {}})
        report, _ = summarize(rows, [r['key'] for r in rows], repetitions=40)
        self.assertEqual(report['evaluated_scans'], 100)
        with self.assertRaises(AssertionError):
            summarize(rows + rows[:1], [r['key'] for r in rows] + [rows[0]['key']], repetitions=40)
        rows[20]['patient_id'] = rows[0]['patient_id']
        with self.assertRaisesRegex(AssertionError, 'crosses test folds'):
            summarize(rows, [r['key'] for r in rows], repetitions=40)

    def test_frozen_folds(self):
        plan = json.loads((HERE_CV / 'plan.json').read_text()); heldout = set()
        for fold in plan['folds']:
            cohort = load(fold['cohort'])
            train = {r['patient_id'] for r in cohort['splits']['training']}
            test = {r['patient_id'] for r in cohort['splits']['testing']}
            validation = {r['patient_id'] for r in cohort['splits']['validation']}
            self.assertFalse(test & (train | validation | heldout)); heldout |= test
        self.assertEqual(len(heldout), plan['eligible_patients'])


if __name__ == '__main__':
    unittest.main()

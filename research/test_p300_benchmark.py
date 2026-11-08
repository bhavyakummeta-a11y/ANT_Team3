"""Regression checks for stimulus parsing, held-out separation and replay provenance."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
import numpy as np
import p300_benchmark as b


class PublicP300Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((b.OUT / 'p300_report.json').read_text())

    def test_marker_pulses_count_once(self):
        values = np.repeat([0, 1, 0, 7, 0], 32)
        np.testing.assert_array_equal(b.flash_onsets(values), [32, 96])

    def test_mapping_follows_explicit_file_metadata(self):
        groups = b.group_cells([f'Row{i}' for i in range(1, 7)] + [f'Col{i}' for i in range(1, 7)])
        self.assertEqual(groups[1], [0, 1, 2, 3, 4, 5])
        self.assertEqual(groups[7], [0, 6, 12, 18, 24, 30])
        self.assertEqual(b.MATRIX[b.expected_cell(np.array([5, 10]), np.array([2, 2]))], '2')

    def test_no_train_test_overlap_and_all_trials_accounted(self):
        r = self.report
        self.assertEqual([p['id'] for p in r['metrics']['participants']], ['A01', 'A02', 'A03'])
        self.assertEqual(len(r['trials']), 60)
        for p in r['metrics']['participants']:
            self.assertFalse(set(p['train_ids']) & set(p['test_ids']))
            self.assertEqual(len(p['test_ids']), 20)
            self.assertEqual(p['train_trials'] + p['excluded_train_trials'], 15)
            self.assertTrue(all(int(t.split('-')[1]) <= 15 for t in p['train_ids']))
            self.assertTrue(all(int(t.split('-')[1]) >= 16 for t in p['test_ids']))
        for t in r['trials']:
            self.assertEqual(len(t['sequence']), 120)
            self.assertEqual(sorted(e['group'] for e in t['sequence']), sorted(list(range(1, 13)) * 10))

    def test_metrics_and_scores_match_actual_predictions(self):
        r = self.report
        self.assertEqual(r['metrics']['correct'], sum(t['predicted_cell'] == t['expected_cell'] for t in r['trials']))
        self.assertEqual(r['metrics']['abstained'], sum(t['predicted_cell'] is None for t in r['trials']))
        self.assertAlmostEqual(r['metrics']['accuracy'], r['metrics']['correct']/60)
        for t in r['trials']:
            if t['quality_rejected']:
                self.assertIsNone(t['predicted_cell'])
                continue
            last = t['scores'][-1]
            self.assertEqual(t['predicted_cell'], last['predicted_cell'])
            rows, columns = np.array(last['row_scores']), np.array(last['column_scores'])
            margin = min(np.sort(rows)[-1] - np.sort(rows)[-2], np.sort(columns)[-1] - np.sort(columns)[-2])
            self.assertAlmostEqual(margin, last['margin'])
            if margin >= .05:
                self.assertEqual(t['predicted_cell'], int(rows.argmax()*6 + columns.argmax()))
            else:
                self.assertIsNone(t['predicted_cell'])

    def test_provenance_matches_scripts_and_ant_inventory(self):
        r = self.report
        self.assertEqual(r['provenance']['pipeline_sha256'], b.sha(Path(b.__file__)))
        self.assertEqual(r['provenance']['shared_decoder_sha256'], b.sha(b.ROOT/'bridge'/'p300.py'))
        ant = b.OUT/'ant_report.json'
        self.assertEqual(r['provenance']['ant_report_sha256'], b.sha(ant))
        for recording in json.loads(ant.read_text())['recordings']:
            self.assertTrue(set(b.COMMON).issubset(recording['channels']))

    def test_heldout_target_labels_do_not_change_predictions(self):
        path = b.DATA/'A01.mat'
        if not path.exists(): self.skipTest('Download dataset to run the real-data leakage regression.')
        trials, _ = b.extract_subject(path)
        _, first = b.evaluate_subject(trials)
        changed = copy.deepcopy(trials)
        for trial in changed[15:]:
            trial['labels'] = 1 - trial['labels']
            trial['expected_cell'] = (trial['expected_cell'] + 1) % 36
        _, second = b.evaluate_subject(changed)
        self.assertEqual([t['predicted_cell'] for t in first], [t['predicted_cell'] for t in second])
        self.assertEqual([t['scores'] for t in first], [t['scores'] for t in second])
        expected = [t for t in self.report['trials'] if t['subject'] == 'A01']
        self.assertEqual([t['predicted_cell'] for t in first], [t['predicted_cell'] for t in expected])
        self.assertEqual(b.sha(path), self.report['provenance']['sources'][0]['sha256'])


if __name__ == '__main__': unittest.main()

"""Temporal and numerical checks; synthetic signals are not EEG accuracy evidence."""
import hashlib
import json
from pathlib import Path
import sys
import unittest

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ant


class EventWindowTests(unittest.TestCase):
    def test_full_window_and_touching_boundaries_are_valid(self):
        task = {'onset': 10.0, 'duration': 2.0}
        prior = {'onset': 4.0, 'duration': 4.0}
        following = {'onset': 12.0, 'duration': 2.0}
        self.assertEqual(ant.window_reasons(10.0, 2.0, [prior, task, following], 30), [])

    def test_short_annotation_is_not_silently_extended(self):
        task = {'onset': 10.0, 'duration': 1.635}
        reasons = ant.window_reasons(10.0, 1.635, [task], 30)
        self.assertTrue(any('shorter' in reason.lower() for reason in reasons))

    def test_actual_right_real_baseline_overlap_is_rejected(self):
        prior = {'onset': 105.484, 'duration': 7.182}
        task = {'onset': 114.436, 'duration': 4.606}
        reasons = ant.window_reasons(task['onset'], task['duration'], [prior, task], 143.672)
        self.assertTrue(any('baseline overlaps' in reason.lower() for reason in reasons))

    def test_actual_imagery_overlapping_labels_are_rejected(self):
        prior = {'onset': 60.942, 'duration': 10.014}
        task = {'onset': 68.316, 'duration': 6.830}
        reasons = ant.window_reasons(task['onset'], task['duration'], [prior, task], 75.148)
        self.assertTrue(any('baseline overlaps' in reason.lower() for reason in reasons))
        self.assertTrue(any('active window overlaps' in reason.lower() for reason in reasons))

    def test_segment_guard_rejects_nearby_windows(self):
        task={'onset':10.,'duration':3.}
        reasons=ant.window_reasons(10.,3.,[task],30,boundaries=[9.])
        self.assertTrue(any('segment boundary' in reason for reason in reasons))

    def test_recording_boundaries_are_respected(self):
        self.assertTrue(ant.window_reasons(1, 3, [{'onset': 1, 'duration': 3}], 30))
        self.assertTrue(ant.window_reasons(29, 3, [{'onset': 29, 'duration': 3}], 30))


class SpectralChecks(unittest.TestCase):
    def test_known_sines_preserve_power_and_band_identity(self):
        time = np.arange(ant.RATE * 3) / ant.RATE
        amplitudes = np.array([10e-6, 6e-6])
        signal = amplitudes[:, None] * np.sin(2 * np.pi * np.array([10, 20])[:, None] * time)
        powers = ant.bandpower(signal)
        np.testing.assert_allclose(powers['mu'][0], amplitudes[0]**2 / 2, rtol=0.01)
        np.testing.assert_allclose(powers['beta'][1], amplitudes[1]**2 / 2, rtol=0.01)
        self.assertLess(powers['beta'][0], powers['mu'][0] * 1e-6)
        self.assertLess(powers['mu'][1], powers['beta'][1] * 1e-6)

    def test_preprocess_resamples_and_references_without_swapping_bands(self):
        rate = 500
        time = np.arange(rate * 12) / rate
        data = np.zeros((len(ant.COMMON), len(time)))
        data[0] = 10e-6 * np.sin(2 * np.pi * 10 * time)
        data[1] = 6e-6 * np.sin(2 * np.pi * 20 * time)
        raw = mne.io.RawArray(data, mne.create_info(ant.COMMON, rate, 'eeg'), verbose='ERROR')
        processed = ant.preprocess(raw)
        self.assertEqual(processed.shape, (len(ant.COMMON), ant.RATE * 12))
        self.assertTrue(np.isfinite(processed).all())
        np.testing.assert_allclose(processed.mean(axis=0), 0, atol=1e-15)
        powers = ant.bandpower(processed[:, ant.RATE * 3:ant.RATE * 9])
        self.assertGreater(powers['mu'][0], 50 * powers['beta'][0])
        self.assertGreater(powers['beta'][1], 10 * powers['mu'][1])


class GeneratedReportChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parent / 'outputs' / 'ant_report.json'
        if not path.exists():
            raise unittest.SkipTest('Generate the ANT report with research/ant.py first.')
        cls.report = json.loads(path.read_text())

    def test_all_recordings_and_events_are_accounted_for(self):
        report = self.report
        self.assertEqual(len(report['recordings']), 10)
        self.assertEqual(len(report['events']), 24)
        self.assertEqual(sum(event['valid'] for event in report['events']), 18)
        self.assertEqual(len(report['comparisons']), 3)
        self.assertTrue(all(len(recording['channels']) == 24 for recording in report['recordings']))
        for event in report['events']:
            if event['valid']:
                self.assertTrue(event['features'])
                self.assertTrue(all(np.isfinite(v) for v in event['features'].values()))
            else:
                self.assertTrue(event['reason'])
                self.assertFalse(event['features'])

    def test_labels_and_timing_are_not_silently_overwritten(self):
        unknown = [r for r in self.report['recordings'] if r['id'].startswith('ImaginaryHand_')]
        self.assertEqual(len(unknown), 1)
        self.assertEqual(unknown[0]['hand'], 'unconfirmed')
        self.assertEqual(len(self.report['timing_audit']), 24)
        self.assertTrue(all(abs(row['difference_seconds']) > 0.001 for row in self.report['timing_audit']))
        self.assertTrue(all(pair['hand'] == 'left' for pair in self.report['comparisons']))

    def test_report_matches_the_analysis_version(self):
        digest = hashlib.sha256(Path(ant.__file__).read_bytes()).hexdigest()
        self.assertEqual(self.report['provenance']['analysis_sha256'], digest,
                         'ANT report is stale: regenerate after changing analysis code.')


if __name__ == '__main__':
    unittest.main()

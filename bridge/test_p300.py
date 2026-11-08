import unittest
import numpy as np
from p300 import P300Session, SignalQualityError

CHANNELS = [dict(label=label, unit='microvolts') for label in ['Pz', 'C3', 'C4']]


class P300Tests(unittest.TestCase):
    def session(self):
        return P300Session(100, CHANNELS)

    def trial(self, target, seed, signal=True):
        rng = np.random.default_rng(seed)
        markers = [{'cell': int(i), 'onset': j * .35, 'uncertainty': .001}
                   for j, i in enumerate(np.tile(np.arange(9), 3))]
        features = rng.normal(0, .1, (27, 48))
        if signal:
            for i, m in enumerate(markers):
                if m['cell'] == target:
                    features[i, 12:24] += 3
        return dict(target=target, markers=markers, features_uv=features.tolist())

    def test_holdout_frozen_and_no_target_input_needed(self):
        s = self.session()
        s.trials = [self.trial(i % 9, i) for i in range(18)]
        result = s.evaluate()
        self.assertTrue(result['enabled'])
        self.assertEqual(result['correct'], 9)
        probe = self.trial(4, 91)
        probe['target'] = None
        self.assertEqual(s.predict(probe)['cell'], 4)
        original = s.model[-1].coef_.copy()
        for trial in s.trials[9:]:
            trial['features_uv'] = np.zeros((27, 48)).tolist()
        self.assertFalse(s.evaluate()['enabled'])
        np.testing.assert_array_equal(original, s.model[-1].coef_)

    def test_noise_does_not_unlock(self):
        s = self.session()
        s.trials = [self.trial(i % 9, i, signal=False) for i in range(18)]
        self.assertFalse(s.evaluate()['enabled'])

    def test_gates(self):
        s = self.session()
        with self.assertRaises(ValueError): s.begin()
        with self.assertRaises(ValueError): s.evaluate()
        s.begin(2)
        with self.assertRaises(ValueError): s.marker(2, 10, .1, 10)
        with self.assertRaises(ValueError): s.marker(2, 10, .001, 11)
        s.marker(2, 10, .001, 10.01)
        with self.assertRaises(ValueError): s.marker(2, 10.1, .001, 10.11)
        with self.assertRaises(ValueError): s.finish()
        self.assertIsNone(s.round)

    def test_epoch_extraction_and_gap_rejection(self):
        for gap in [False, True]:
            s = self.session()
            rng = np.random.default_rng(25)
            times = np.arange(0, 14, .01)
            values = rng.normal(0, 1, (len(times), 3))
            if gap:
                keep = ~((times > 7) & (times < 7.1))
                times, values = times[keep], values[keep]
            s.feed(values.tolist(), times.tolist())
            s.begin(0)
            for j, cell in enumerate(np.tile(np.arange(9), 3)):
                onset = 3 + j * .35
                s.marker(int(cell), onset, .001, onset+.01)
            if gap:
                with self.assertRaisesRegex(ValueError, 'gap'): s.finish()
            else:
                self.assertEqual(s.finish()['calibration_rounds'], 1)
                self.assertEqual(np.array(s.trials[0]['features_uv']).shape, (27, 48))
                self.assertTrue(np.isfinite(s.trials[0]['features_uv']).all())

    def measured_session(self, unit='microvolts', extra_flat=False, high=False, dc=0):
        channels = [dict(label=c['label'], unit=unit) for c in CHANNELS]
        if extra_flat:
            channels.append(dict(label='Cz', unit=unit))
        s = P300Session(500, channels)
        times = np.arange(0, 14, 1 / 500)
        rng = np.random.default_rng(123)
        values = rng.normal(0, 2, (len(times), len(channels))) + dc
        if extra_flat:
            values[:, -1] = 0
        if high:
            # Large out-of-band noise must still be rejected by raw checks.
            values[:, 1] += 200 * np.sin(2 * np.pi * 50 * times)
        if unit.strip() == 'Volt':
            values /= 1e6
        s.feed(values.tolist(), times.tolist())
        return s

    def finish_measured(self, s):
        s.begin(0)
        for j, cell in enumerate(np.tile(np.arange(9), 3)):
            onset = 3 + j * .35
            s.marker(int(cell), onset, .001, onset + .01)
        return s.finish()

    def test_flat_optional_channel_named_and_rejected_evidence_retained(self):
        s = self.measured_session(extra_flat=True)
        with self.assertRaises(SignalQualityError) as caught:
            self.finish_measured(s)
        self.assertIn('CZ: flat', str(caught.exception))
        report = caught.exception.quality
        bad = [c for c in report['channels'] if c['reasons']]
        self.assertEqual([c['label'] for c in bad], ['CZ'])
        self.assertEqual(bad[0]['raw_std_uv'], 0)
        self.assertEqual(s.trials, [])
        self.assertEqual(len(s.export()['rejected_rounds']), 1)
        self.assertEqual(s.export()['rejected_rounds'][0]['quality'], report)

    def test_raw_noise_remains_rejected_even_if_filter_removes_it(self):
        s = self.measured_session(high=True)
        with self.assertRaises(SignalQualityError) as caught:
            self.finish_measured(s)
        self.assertIn('C3: raw peak-to-peak', str(caught.exception))
        c3 = next(c for c in caught.exception.quality['channels'] if c['label'] == 'C3')
        self.assertGreater(c3['raw_ptp_uv'], 300)
        self.assertLess(c3['filtered_ptp_uv'], 300)
        self.assertFalse(s.enabled)

    def test_volts_microvolts_and_dc_offset_equivalence(self):
        uv, volts, offset = (self.measured_session(), self.measured_session(unit=' Volt '),
                             self.measured_session(dc=10000))
        for s in [uv, volts, offset]:
            self.assertEqual(self.finish_measured(s)['calibration_rounds'], 1)
        expected = uv.trials[0]['features_uv']
        np.testing.assert_allclose(expected, volts.trials[0]['features_uv'], atol=1e-9)
        np.testing.assert_allclose(expected, offset.trials[0]['features_uv'], atol=1e-7)
        self.assertEqual(volts.selected[0]['scale_to_uv'], 1e6)

    def test_explicit_exclusion_before_calibration_and_frozen_after(self):
        s = self.measured_session(extra_flat=True)
        s.configure([0, 1, 2])
        self.assertEqual(s.labels, ['PZ', 'C3', 'C4'])
        self.assertEqual(len(s.buffer), 0)
        source = self.measured_session()
        s.feed([row for _, row in source.buffer], [t for t, _ in source.buffer])
        self.finish_measured(s)
        with self.assertRaisesRegex(ValueError, 'locked'):
            s.configure([0, 1, 2])
        fresh = self.measured_session(extra_flat=True)
        with self.assertRaises(ValueError): fresh.configure([1, 2, 3])
        with self.assertRaises(ValueError): fresh.configure([0, 1])
        with self.assertRaises(ValueError): fresh.configure([0, 1, 99])
        self.assertEqual(len(fresh.selected), 4)

    def test_preflight_waits_then_identifies_flat_channel(self):
        self.assertEqual(self.session().quality()['status'], 'waiting')
        report = self.measured_session(extra_flat=True).quality()
        self.assertEqual(report['status'], 'rejected')
        self.assertEqual(report['epochs_checked'], 1)
        self.assertEqual(report['channels'][-1]['reasons'], ['flat raw signal'])

    def test_mislabeled_microvolts_require_explicit_override(self):
        channels = [dict(label=c['label'], unit='Volt') for c in CHANNELS]
        s = P300Session(500, channels)
        times = np.arange(0, 14, 1 / 500)
        values = np.column_stack([-1400 + a * np.sin(2 * np.pi * 5 * times) for a in [20, 30, 40]])
        s.feed(values.tolist(), times.tolist())
        first = s.quality()
        self.assertEqual(first['status'], 'rejected')
        self.assertIn('Possible input-unit mismatch', first['unit_warning'])
        self.assertEqual(s.unit_mode, 'metadata')
        self.assertEqual(s.scales, [1e6, 1e6, 1e6])
        original_min = first['channels'][0]['source_min']
        s.configure([0, 1, 2], 'microvolts')
        self.assertEqual(len(s.buffer), 0)
        self.assertIsNone(s.last_quality)
        s.feed(values.tolist(), times.tolist())
        second = s.quality()
        self.assertEqual(second['status'], 'ok')
        self.assertAlmostEqual(second['channels'][0]['raw_ptp_uv'], 40)
        self.assertAlmostEqual(second['channels'][0]['source_min'], original_min)
        self.assertEqual(second['channels'][0]['source_unit'], 'Volt')
        self.assertEqual(second['channels'][0]['effective_unit'], 'microvolts')
        self.assertEqual(second['channels'][0]['scale_to_uv'], 1)
        self.assertEqual(second['channels'][0]['declared_scale_to_uv'], 1e6)
        self.finish_measured(s)
        with self.assertRaisesRegex(ValueError, 'locked'):
            s.configure([0, 1, 2], 'metadata')
        self.assertEqual(s.export()['unit_mode'], 'microvolts')
        self.assertEqual(P300Session(500, channels).unit_mode, 'metadata')

    def test_unit_override_does_not_bypass_artifact_checks(self):
        s = P300Session(500, [dict(label=c['label'], unit='Volt') for c in CHANNELS])
        s.configure([0, 1, 2], 'microvolts')
        raw = self.measured_session(high=True)
        s.feed([row for _, row in raw.buffer], [t for t, _ in raw.buffer])
        with self.assertRaises(SignalQualityError): self.finish_measured(s)
        self.assertEqual(s.trials, [])
        self.assertFalse(s.enabled)

    def test_invalid_units_are_atomic_and_channel_changes_preserve_override(self):
        s = self.measured_session()
        before = list(s.buffer)
        with self.assertRaises(ValueError): s.configure([0, 1, 2], 'guess')
        self.assertEqual(s.unit_mode, 'metadata')
        self.assertEqual(list(s.buffer), before)
        s.configure([0, 1, 2], 'microvolts')
        s.configure([0, 1, 2])
        self.assertEqual(s.unit_mode, 'microvolts')
        s.configure([0, 1, 2], 'volts')
        self.assertEqual(s.scales, [1e6, 1e6, 1e6])
        s.configure([0, 1, 2], 'metadata')
        self.assertEqual(s.scales, [1, 1, 1])

    def test_channels_and_units(self):
        with self.assertRaises(ValueError): P300Session(100, [dict(label='C3', unit='Volt')])
        with self.assertRaises(ValueError): P300Session(50, CHANNELS)
        s = P300Session(100, [dict(label=c['label'], unit='Volt') for c in CHANNELS])
        s.feed([[1e-6, 2e-6, 3e-6]], [1.0])
        np.testing.assert_allclose(s.buffer[0][1], [1, 2, 3])


if __name__ == '__main__':
    unittest.main()

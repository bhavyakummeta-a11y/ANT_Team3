import unittest
import numpy as np
from p300 import P300Session

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

    def test_channels_and_units(self):
        with self.assertRaises(ValueError): P300Session(100, [dict(label='C3', unit='Volt')])
        with self.assertRaises(ValueError): P300Session(50, CHANNELS)
        s = P300Session(100, [dict(label=c['label'], unit='Volt') for c in CHANNELS])
        s.feed([[1e-6, 2e-6, 3e-6]], [1.0])
        np.testing.assert_allclose(s.buffer[0][1], [1, 2, 3])


if __name__ == '__main__':
    unittest.main()

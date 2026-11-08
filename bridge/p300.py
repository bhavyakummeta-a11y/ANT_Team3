"""Session-local, supervised ERP grid prototype. No pretrained intention decoder."""
from collections import deque
import math
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


class P300Session:
    REPEATS = 3
    MIN_MARGIN = 0.05

    def __init__(self, rate, channels):
        if not 100 <= rate <= 2000:
            raise ValueError('P300 requires a regular EEG stream at 100–2000 Hz.')
        preferred = {'PZ', 'P3', 'P4', 'C3', 'C4', 'FZ', 'CZ'}
        self.indices, self.labels, self.scales = [], [], []
        for i, channel in enumerate(channels):
            label = channel['label'].upper().replace('-REF', '').strip()
            unit = channel['unit'].lower().replace('μ', 'u').replace('µ', 'u')
            if label in preferred and unit in ('v', 'volt', 'volts', 'uv', 'microvolt', 'microvolts'):
                self.indices.append(i)
                self.labels.append(label)
                self.scales.append(1e6 if unit in ('v', 'volt', 'volts') else 1)
        if len(self.indices) < 3 or not set(self.labels) & {'PZ', 'P3', 'P4'}:
            raise ValueError('Need at least three named EEG channels including Pz, P3 or P4, with Volt or microvolt units.')
        self.rate = rate
        self.buffer = deque(maxlen=int(rate * 90))
        self.round = None
        self.trials = []
        self.model = None
        self.report = None
        self.enabled = False
        self.sos = butter(3, [1, 15], btype='bandpass', fs=rate, output='sos')

    def feed(self, rows, times):
        for row, stamp in zip(rows, times):
            values = [float(row[i]) * scale if row[i] is not None else float('nan')
                      for i, scale in zip(self.indices, self.scales)]
            self.buffer.append((float(stamp), values))

    def begin(self, target=None):
        if self.round is not None:
            raise ValueError('A round is already running.')
        if target is not None and (type(target) is not int or not 0 <= target < 9):
            raise ValueError('Invalid calibration target.')
        if target is None and not self.enabled:
            raise ValueError('Live selection is locked until held-out calibration passes.')
        if target is not None and len(self.trials) >= 18:
            raise ValueError('Calibration is complete. Reconnect for a new calibration.')
        self.round = {'target': target, 'markers': []}

    def marker(self, cell, onset, uncertainty, now):
        if self.round is None:
            raise ValueError('No active round.')
        if type(cell) is not int or not 0 <= cell < 9:
            raise ValueError('Invalid grid cell.')
        if not all(math.isfinite(x) for x in [onset, uncertainty]) or not 0 <= uncertainty <= .020:
            raise ValueError('Clock uncertainty exceeds 20 ms. Retry on a stable local connection.')
        if not -.02 <= now - onset <= .25:
            raise ValueError('Highlight timestamp arrived too late or in the wrong clock domain.')
        markers = self.round['markers']
        if len(markers) >= 27 or (markers and not .25 <= onset - markers[-1]['onset'] <= .55):
            raise ValueError('Highlight timing was interrupted. Repeat this round.')
        markers.append(dict(cell=cell, onset=onset, uncertainty=uncertainty))

    def finish(self):
        trial, self.round = self.round, None
        if trial is None:
            raise ValueError('No active round.')
        markers = trial['markers']
        if len(markers) != 27 or any(sum(m['cell'] == i for m in markers) != 3 for i in range(9)):
            raise ValueError('Incomplete highlights; round discarded.')
        raw = list(self.buffer)
        times = np.array([x[0] for x in raw])
        data = np.array([x[1] for x in raw])
        start, end = markers[0]['onset'] - 2, markers[-1]['onset'] + .85
        mask = (times >= start) & (times <= end)
        times, data = times[mask], data[mask]
        if len(times) < self.rate or times[0] > start + .05 or times[-1] < end - .05:
            raise ValueError('Not enough EEG before/after highlights. Wait for samples and retry.')
        if not np.isfinite(data).all() or np.any(np.diff(times) <= 0) or np.max(np.diff(times)) > max(.03, 3 / self.rate):
            raise ValueError('EEG contains invalid samples or a timing gap. Round discarded.')
        filtered = sosfiltfilt(self.sos, data, axis=0)
        features = []
        for marker in markers:
            onset = marker['onset']
            epoch_mask = (times >= onset - .2) & (times <= onset + .8)
            epoch = data[epoch_mask]
            if np.any(np.ptp(epoch, axis=0) > 300) or np.any(np.std(epoch, axis=0) < .01):
                raise ValueError('Flat or large-amplitude EEG (>300 µV peak-to-peak). Check contact/movement and repeat.')
            baseline = filtered[(times >= onset - .2) & (times < onset)].mean(axis=0)
            # Nonoverlapping 50 ms means provide low-pass ERP features, 0–800 ms.
            bins = []
            for offset in np.arange(0, .8, .05):
                segment = filtered[(times >= onset + offset) & (times < onset + offset + .05)]
                if not len(segment):
                    raise ValueError('Missing samples inside an ERP feature bin.')
                bins.extend((segment.mean(axis=0) - baseline).tolist())
            features.append(bins)
        trial['features_uv'] = features
        if trial['target'] is not None:
            self.trials.append(trial)
            return {'calibration_rounds': len(self.trials)}
        return self.predict(trial)

    @staticmethod
    def fit(trials):
        x = np.array([f for t in trials for f in t['features_uv']])
        y = np.array([int(m['cell'] == t['target']) for t in trials for m in t['markers']])
        model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, class_weight='balanced', max_iter=1000, random_state=42))
        model.fit(x, y)
        return model

    def predict(self, trial):
        probabilities = self.model.predict_proba(trial['features_uv'])[:, 1]
        scores = [float(np.mean([p for p, m in zip(probabilities, trial['markers']) if m['cell'] == i])) for i in range(9)]
        order = np.argsort(scores)
        margin = scores[order[-1]] - scores[order[-2]]
        return dict(cell=int(order[-1]) if margin >= self.MIN_MARGIN else None,
                    margin=float(margin), scores=scores)

    def evaluate(self):
        if len(self.trials) != 18:
            raise ValueError('Complete 18 calibration rounds before evaluation.')
        # Freeze model trained only on first block. Never refit using the holdout.
        if any(sorted(t['target'] for t in block) != list(range(9)) for block in (self.trials[:9], self.trials[9:])):
            raise ValueError('Each calibration block must cover every grid position once.')
        self.model = self.fit(self.trials[:9])
        predictions = [self.predict(t) for t in self.trials[9:]]
        correct = sum(p['cell'] == t['target'] for p, t in zip(predictions, self.trials[9:]))
        self.enabled = correct >= 7
        self.report = dict(correct=correct, total=9, enabled=self.enabled, chance=1/9,
                           predictions=predictions, minimum_correct=7, minimum_margin=self.MIN_MARGIN,
                           note='Small same-session holdout; not proof of generalization, intention reading, or assistive reliability.')
        return self.report

    def export(self):
        return dict(schema='neurobind-p300-v1', channels=self.labels, sample_rate=self.rate,
                    trials=self.trials, evaluation=self.report,
                    features='Baseline-corrected 1–15 Hz ERP; 16 bins of 50 ms × channels, µV.',
                    timing='Software frame timestamps mapped to receiver LSL clock; physical display latency unmeasured.',
                    recording='Epoched features and markers only, not a raw EEG recording. Use LabRecorder for raw XDF.')

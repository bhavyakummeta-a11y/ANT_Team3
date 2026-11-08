"""Session-local, supervised ERP grid prototype. No pretrained intention decoder."""
from collections import deque
import math
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


class SignalQualityError(ValueError):
    def __init__(self, message, quality):
        super().__init__(message)
        self.quality = quality


class P300Session:
    REPEATS = 3
    MIN_MARGIN = 0.05
    MAX_PTP_UV = 300.0
    MIN_STD_UV = 0.01

    def __init__(self, rate, channels):
        if not 100 <= rate <= 2000:
            raise ValueError('P300 requires a regular EEG stream at 100–2000 Hz.')
        preferred = {'PZ', 'P3', 'P4', 'C3', 'C4', 'FZ', 'CZ'}
        self.candidates = []
        for i, channel in enumerate(channels):
            label = channel['label'].upper().replace('-REF', '').strip()
            unit = channel['unit'].lower().replace('μ', 'u').replace('µ', 'u').strip()
            if label in preferred and unit in ('v', 'volt', 'volts', 'uv', 'microvolt', 'microvolts'):
                self.candidates.append(dict(index=i, label=label, source_unit=channel['unit'],
                                            scale_to_uv=1e6 if unit in ('v', 'volt', 'volts') else 1))
        self._select_channels([c['index'] for c in self.candidates])
        if len(self.indices) < 3 or not set(self.labels) & {'PZ', 'P3', 'P4'}:
            raise ValueError('Need at least three named EEG channels including Pz, P3 or P4, with Volt or microvolt units.')
        self.rate = rate
        self.buffer = deque(maxlen=int(rate * 90))
        self.round = None
        self.trials = []
        self.model = None
        self.report = None
        self.enabled = False
        self.last_quality = None
        self.rejected_rounds = []
        self.sos = butter(3, [1, 15], btype='bandpass', fs=rate, output='sos')

    def _select_channels(self, indices):
        selected = [c for c in self.candidates if c['index'] in indices]
        if len(selected) < 3 or not {c['label'] for c in selected} & {'PZ', 'P3', 'P4'}:
            raise ValueError('Keep at least three supported EEG channels including Pz, P3 or P4.')
        self.selected = selected
        self.indices = [c['index'] for c in selected]
        self.labels = [c['label'] for c in selected]
        self.scales = [c['scale_to_uv'] for c in selected]

    def configure(self, indices):
        if self.round is not None or self.trials or self.model is not None:
            raise ValueError('Channel selection is locked after calibration starts. Reconnect to change it.')
        if (not isinstance(indices, list) or any(type(i) is not int for i in indices)
                or len(set(indices)) != len(indices)
                or not set(indices).issubset({c['index'] for c in self.candidates})):
            raise ValueError('Choose channels from this stream.')
        self._select_channels(indices)
        self.buffer.clear()
        self.last_quality = None
        return {'channels': self.selected, 'message': 'Channels applied. Wait three seconds, then Check signal.'}

    @classmethod
    def quality_policy(cls):
        return dict(max_raw_ptp_uv=cls.MAX_PTP_UV, min_raw_std_uv=cls.MIN_STD_UV,
                    checked_signal='Unfiltered EEG converted from declared source units to microvolts.',
                    filtered_signal='1–15 Hz; shown for comparison, not substituted for raw quality checks.')

    def _quality_report(self, data, filtered, masks, window_seconds=1.0):
        channels = []
        for i, metadata in enumerate(self.selected):
            raw_ptp = [float(np.ptp(data[mask, i])) for mask in masks]
            raw_std = [float(np.std(data[mask, i])) for mask in masks]
            filtered_ptp = [float(np.ptp(filtered[mask, i])) for mask in masks]
            worst_ptp, worst_std = max(raw_ptp), min(raw_std)
            reasons = []
            if worst_ptp > self.MAX_PTP_UV:
                reasons.append('high raw amplitude')
            if worst_std < self.MIN_STD_UV:
                reasons.append('flat raw signal')
            span = np.logical_or.reduce(masks)
            channels.append(dict(**metadata, raw_ptp_uv=worst_ptp, raw_std_uv=worst_std,
                                 filtered_ptp_uv=max(filtered_ptp),
                                 source_min=float(np.min(data[span, i]) / metadata['scale_to_uv']),
                                 source_max=float(np.max(data[span, i]) / metadata['scale_to_uv']),
                                 reasons=reasons, worst_ptp_epoch=int(np.argmax(raw_ptp)) + 1,
                                 worst_flat_epoch=int(np.argmin(raw_std)) + 1))
        bad = [c for c in channels if c['reasons']]
        details = []
        for c in bad:
            reasons = []
            if 'flat raw signal' in c['reasons']:
                reasons.append(f"flat (SD {c['raw_std_uv']:.4g} µV < {self.MIN_STD_UV:g})")
            if 'high raw amplitude' in c['reasons']:
                reasons.append(f"raw peak-to-peak {c['raw_ptp_uv']:.4g} µV > {self.MAX_PTP_UV:g}; 1–15 Hz {c['filtered_ptp_uv']:.4g} µV")
            details.append(c['label'] + ': ' + ', '.join(reasons))
        message = ('Signal check failed — ' + '; '.join(details) +
                   '. Inspect channel measurements and declared units below. No calibration round was saved.'
                   if bad else 'Raw signal checks passed for the measured window; this does not establish decoding accuracy.')
        self.last_quality = dict(status='rejected' if bad else 'ok', message=message,
                                 channels=channels, window_seconds=window_seconds,
                                 epochs_checked=len(masks), policy=self.quality_policy(),
                                 note='Peak-to-peak values are maxima across checked epochs; SD is the minimum. Source min/max span checked epochs.')
        return self.last_quality

    def quality(self):
        raw = list(self.buffer)
        waiting = dict(status='waiting', message='Wait for at least three seconds of continuous EEG, then check again.',
                       channels=[], window_seconds=0)
        if not raw or raw[-1][0] - raw[0][0] < 2.95:
            return waiting
        times = np.array([x[0] for x in raw])
        data = np.array([x[1] for x in raw])
        keep = times >= times[-1] - 3
        times, data = times[keep], data[keep]
        if not np.isfinite(data).all() or np.any(np.diff(times) <= 0) or np.max(np.diff(times)) > max(.03, 3 / self.rate):
            return dict(status='rejected', message='Signal check needs continuous, finite samples. A gap or invalid sample was detected.',
                        channels=[], window_seconds=0)
        filtered = sosfiltfilt(self.sos, data, axis=0)
        # Use a central one-second window to keep this diagnostic away from filter edges.
        mask = (times >= times[-1] - 1.5) & (times < times[-1] - .5)
        return self._quality_report(data, filtered, [mask])

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
        trial = self.round
        try:
            return self._finish()
        except ValueError as exc:
            if trial is not None:
                self.rejected_rounds.append(dict(target=trial['target'], markers=trial['markers'],
                                                labels=trial.get('labels'), purpose=trial.get('purpose'),
                                                channels=self.selected.copy(), error=str(exc),
                                                quality=getattr(exc, 'quality', None)))
                self.rejected_rounds = self.rejected_rounds[-20:]
            raise

    def _finish(self):
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
        masks = [(times >= m['onset'] - .2) & (times <= m['onset'] + .8) for m in markers]
        quality = self._quality_report(data, filtered, masks)
        if quality['status'] == 'rejected':
            raise SignalQualityError(quality['message'], quality)
        trial['quality'] = quality
        features = []
        for marker in markers:
            onset = marker['onset']
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
                    selected_channels=self.selected, quality_policy=self.quality_policy(),
                    last_quality=self.last_quality, rejected_rounds=self.rejected_rounds,
                    features='Baseline-corrected 1–15 Hz ERP; 16 bins of 50 ms × channels, µV.',
                    timing='Software frame timestamps mapped to receiver LSL clock; physical display latency unmeasured.',
                    recording='Epoched features and markers only, not a raw EEG recording. Use LabRecorder for raw XDF.')

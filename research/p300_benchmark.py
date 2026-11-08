"""Public P300 calibration and held-out letter replay; never calibrates the ANT cap.

Predeclared sample: A01–A03, first 15 selections train, last 20 test per person.
The public 6x6 row/column protocol differs from the live 3x3 single-item grid.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from urllib.request import urlopen

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bridge'))
from p300 import P300Session, erp_features, make_erp_classifier

DATA = ROOT / 'data' / 'external' / 'bnci2014_008'
OUT = ROOT / 'research' / 'outputs'
SOURCE = 'https://lampx.tugraz.at/~bci/database/008-2014/'
SUBJECTS = ['A01', 'A02', 'A03']
COMMON = ['Fz', 'Pz', 'P3', 'P4']
RATE = 256
MATRIX = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789_')
TRAIN_N = 15


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(subject):
    path = DATA / (subject + '.mat')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urlopen(SOURCE + path.name, timeout=90) as response:
            content = response.read()
        temp = path.with_suffix('.part')
        temp.write_bytes(content)
        temp.replace(path)
    return path


def group_cells(class_names):
    """Use MAT's explicit Row/Col labels, not the contradictory PDF figure."""
    result = {}
    for i, name in enumerate(class_names, 1):
        number = int(name[-1]) - 1
        if not 0 <= number < 6:
            raise ValueError('Invalid row/column number.')
        if name.startswith('Row'):
            result[i] = [number * 6 + j for j in range(6)]
        elif name.startswith('Col'):
            result[i] = [j * 6 + number for j in range(6)]
        else:
            raise ValueError('Unknown stimulus group.')
    return result


def flash_onsets(stimuli):
    # Marker pulses last 32 samples: use rising edges, not every nonzero sample.
    previous = np.r_[0, stimuli[:-1]]
    return np.flatnonzero((stimuli > 0) & (stimuli != previous))


def expected_cell(groups, targets):
    target_groups = np.unique(groups[targets == 2])
    if len(target_groups) != 2:
        raise ValueError('Expected one target row and one target column.')
    rows = [int(g) - 1 for g in target_groups if 1 <= g <= 6]
    columns = [int(g) - 7 for g in target_groups if 7 <= g <= 12]
    if len(rows) != 1 or len(columns) != 1:
        raise ValueError('Target labels do not identify a row and column.')
    return rows[0] * 6 + columns[0]


def extract_subject(path):
    data = loadmat(path, simplify_cells=True)['data']
    channels = list(data['channels'])
    classes = list(data['classes_stim'])
    if classes != [f'Row{i}' for i in range(1, 7)] + [f'Col{i}' for i in range(1, 7)]:
        raise ValueError('Unexpected metadata: verify row/column mapping before analysis.')
    x = np.asarray(data['X'][:, [channels.index(c) for c in COMMON]], dtype=float)
    # Source X is already µV, confirmed by the maintainer's MOABB loader.
    y, stimulus = data['y'], data['y_stim']
    starts = np.asarray(data['trial'], dtype=int) - 1
    if len(starts) != 35:
        raise ValueError('Expected exactly 35 selections.')
    onsets = flash_onsets(stimulus)
    sos = butter(3, [1, 15], btype='bandpass', fs=RATE, output='sos')
    result = []
    for trial_i, start in enumerate(starts):
        stop = starts[trial_i + 1] if trial_i + 1 < len(starts) else len(x)
        events = onsets[(onsets >= start) & (onsets < stop)]
        groups, targets = stimulus[events].astype(int), y[events].astype(int)
        if len(events) != 120 or any(np.count_nonzero(groups == i) != 10 for i in range(1, 13)):
            raise ValueError('Expected 120 flashes and ten of each group per selection.')
        # Sequence boundaries in these files are twelve consecutive groups.
        if any(sorted(block.tolist()) != list(range(1, 13)) for block in groups.reshape(10, 12)):
            raise ValueError('Unexpected repetition structure.')
        first, last = int(events[0] - 2 * RATE), int(events[-1] + np.ceil(.85 * RATE))
        times = np.arange(first, last + 1) / RATE
        raw = x[first:last + 1]
        if first < 0 or len(raw) != len(times) or not np.isfinite(raw).all():
            raise ValueError('Missing/nonfinite source EEG.')
        filtered = sosfiltfilt(sos, raw, axis=0)
        features, bad = [], []
        for event in events:
            onset = event / RATE
            mask = (times >= onset - .2) & (times <= onset + .8)
            ptp, sd = np.ptp(raw[mask], axis=0), np.std(raw[mask], axis=0)
            failed = (ptp > P300Session.MAX_PTP_UV) | (sd < P300Session.MIN_STD_UV)
            bad.append([COMMON[i] for i in np.flatnonzero(failed)])
            features.append(erp_features(times, filtered, onset))
        cell = expected_cell(groups, targets)
        result.append(dict(id=f'{path.stem}-{trial_i + 1:02d}', subject=path.stem,
                           run=trial_i // 5 + 1, trial_index=trial_i + 1,
                           expected_cell=cell, expected_character=MATRIX[cell],
                           onsets=events, groups=groups, labels=(targets == 2).astype(int),
                           features=np.asarray(features), bad=bad,
                           valid=not any(bad), split='train' if trial_i < TRAIN_N else 'test'))
    return result, group_cells(classes)


def score_selection(model, features, groups):
    """Inference receives only features and flashed group IDs, never target labels."""
    probabilities = model.predict_proba(features)[:, 1]
    scores = []
    for repetition in range(1, 11):
        n = repetition * 12
        means = np.array([probabilities[:n][groups[:n] == group].mean() for group in range(1, 13)])
        rows, cols = means[:6], means[6:]
        row_order, col_order = np.argsort(rows), np.argsort(cols)
        margin = min(rows[row_order[-1]] - rows[row_order[-2]], cols[col_order[-1]] - cols[col_order[-2]])
        cell = int(row_order[-1] * 6 + col_order[-1]) if margin >= P300Session.MIN_MARGIN else None
        scores.append(dict(repetition=repetition, row_scores=rows.tolist(), column_scores=cols.tolist(),
                           predicted_cell=cell, margin=float(margin)))
    return scores


def evaluate_subject(trials):
    train = [t for t in trials[:TRAIN_N] if t['valid']]
    if len(train) < 3:
        raise ValueError('Fewer than three usable training selections; do not fit a model.')
    model = make_erp_classifier()
    model.fit(np.concatenate([t['features'] for t in train]), np.concatenate([t['labels'] for t in train]))
    heldout = []
    for trial in trials[TRAIN_N:]:
        scores = score_selection(model, trial['features'], trial['groups']) if trial['valid'] else []
        prediction = scores[-1]['predicted_cell'] if scores else None
        reason = ('Raw signal quality rejected this selection.' if not trial['valid'] else
                  'Row or column score margin below 0.05; no selection.' if prediction is None else
                  'Predicted row and column from ten repetitions of recorded EEG.')
        heldout.append({k: trial[k] for k in ['id', 'subject', 'run', 'trial_index', 'expected_cell', 'expected_character']} |
                       dict(predicted_cell=prediction, predicted_character=MATRIX[prediction] if prediction is not None else None,
                            correct=prediction == trial['expected_cell'], reason=reason, scores=scores,
                            sequence=[dict(group=int(g), onset_seconds=float((onset - trial['onsets'][0]) / RATE))
                                      for g, onset in zip(trial['groups'], trial['onsets'])],
                            quality_rejected=not trial['valid'], bad_epochs=sum(bool(b) for b in trial['bad'])))
    correct = sum(t['correct'] for t in heldout)
    stats = dict(id=trials[0]['subject'], correct=correct, total=len(heldout), accuracy=correct / len(heldout),
                 train_trials=len(train), training_candidates=TRAIN_N,
                 excluded_train_trials=TRAIN_N - len(train), excluded_test_trials=sum(not t['valid'] for t in trials[TRAIN_N:]),
                 excluded_train_epochs=120 * (TRAIN_N - len(train)),
                 excluded_test_epochs=120 * sum(not t['valid'] for t in trials[TRAIN_N:]),
                 abstained=sum(t['predicted_cell'] is None for t in heldout),
                 train_ids=[t['id'] for t in train], test_ids=[t['id'] for t in trials[TRAIN_N:]])
    return stats, heldout


def figures(subject_trials, participants):
    # Training data only, so display exploration does not influence held-out choices.
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True, layout='constrained')
    for channel, ax in zip(range(4), axes):
        for target, color, label in [(1, '#138f96', 'Attended'), (0, '#a85e41', 'Unattended')]:
            means = []
            for trials in subject_trials:
                data = np.concatenate([t['features'][t['labels'] == target] for t in trials[:TRAIN_N] if t['valid']])
                means.append(data.reshape(-1, 16, 4)[:, :, channel].mean(axis=0))
            ax.plot(np.arange(16) * .05 + .025, np.mean(means, axis=0), color=color, label=label)
        ax.axhline(0, color='gray', linewidth=.6)
        ax.set_title(COMMON[channel]); ax.set_xlabel('Seconds after flash'); ax.grid(alpha=.15)
    axes[0].set_ylabel('Baseline-relative amplitude (µV)'); axes[0].legend(fontsize=8)
    fig.suptitle('Public P300 calibration responses · equal weight per participant · training trials only')
    fig.savefig(OUT / 'p300_erp.svg'); fig.savefig(OUT / 'p300_erp.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
    ax.bar([p['id'] for p in participants], [100 * p['accuracy'] for p in participants], color='#138f96')
    ax.axhline(100/36, color='#a85e41', linestyle='--', label='Uniform random 1 of 36: 2.8%')
    for i, p in enumerate(participants): ax.text(i, 100 * p['accuracy'] + 2, f"{p['correct']}/{p['total']}", ha='center')
    ax.set_ylim(0, 110);ax.set_ylabel('Held-out correct selections (%)');ax.set_title('Final 20 selections per participant · abstentions count as incorrect');ax.legend(fontsize=8)
    fig.savefig(OUT / 'p300_accuracy.svg');fig.savefig(OUT / 'p300_accuracy.png', dpi=160);plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    with ThreadPoolExecutor(max_workers=3) as pool: paths = list(pool.map(download, SUBJECTS))
    all_trials, participants, replay = [], [], []
    for path in paths:
        trials, groups = extract_subject(path)
        stats, predictions = evaluate_subject(trials)
        all_trials.append(trials);participants.append(stats);replay.extend(predictions)
        print(f"{path.stem}: {stats['correct']}/{stats['total']} correct; {stats['abstained']} abstained; {stats['train_trials']} training selections", flush=True)
    ant_path = OUT / 'ant_report.json'
    ant = json.loads(ant_path.read_text())
    for recording in ant['recordings']:
        if not set(COMMON).issubset(recording['channels']): raise ValueError('ANT channel compatibility claim failed.')
    rates = sorted({r['sampling_hz'] for r in ant['recordings']})
    figures(all_trials, participants)
    correct, total = sum(p['correct'] for p in participants), len(replay)
    report = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        dataset=dict(name='BNCI 008-2014 · public P300 speller', url='https://bnci-horizon-2020.eu/database/data-sets',
                     description_url=SOURCE + 'description.pdf', publication_url='https://doi.org/10.3389/fnhum.2013.00732',
                     license='CC BY-NC-ND 4.0 · Santa Lucia Neuroelectrical Imaging and BCI Laboratory',
                     sample_rate=RATE, paradigm='6×6 row/column highlights; ten repetitions; 250 ms stimulus onset interval'),
        matrix=MATRIX, groups=[dict(id=k, cells=v) for k, v in groups.items()],
        metrics=dict(train_trials=sum(p['train_trials'] for p in participants), test_trials=total,
                     correct=correct, abstained=sum(p['abstained'] for p in participants), accuracy=correct / total,
                     incorrect=sum(t['predicted_cell'] is not None and not t['correct'] for t in replay),
                     participants=participants),
        method=dict(channels=COMMON,
                    training='First three participants A01–A03 fixed before fitting. Separate model per participant: first 15 selections train, final 20 test. No test-label fitting or tuning.',
                    features='Same feature extractor and estimator as the live prototype: 3rd-order Butterworth 1–15 Hz, -200–0 ms baseline, sixteen 50 ms bins over 0–800 ms, training-only scaler and class-balanced logistic regression C=0.1. Trials filtered independently.',
                    quality='Same raw limits as live: 300 µV peak-to-peak maximum and 0.01 µV minimum SD. Any failed epoch rejects the whole selection. Rejected/abstained test selections stay in the accuracy denominator.',
                    selection='Average target-response probabilities separately across six rows and six columns. Both winning score margins must reach 0.05. This row/column adapter differs from the live single-item protocol.'),
        ant_fit=dict(shared_channels=COMMON, ant_rates=rates,
                     findings=[f'All {len(ant["recordings"])} ANT recordings contain Fz, Pz, P3 and P4.',
                               'Shared electrode names and ERP feature definitions support software compatibility testing.',
                               'Existing ANT movement/imagery analyses remain separate from this letter-selection benchmark.'],
                     unsupported=['ANT recordings have movement annotations, not letter-flash targets; P300 decoding accuracy cannot be calculated from them.',
                                  'Electrode overlap does not establish waveform correlation, cross-person model transfer, or successful calibration of your cap.',
                                  'Public data use 256 Hz, a right-earlobe reference and different acquisition hardware; ANT recordings use 500/1000 Hz.',
                                  'Public six-by-six row/column flashes differ from the live three-by-three individual flashes.']),
        trials=replay,
        figures=[dict(path='research/outputs/p300_erp.svg',title='Measured calibration responses',description='Attended and unattended responses at four shared electrodes; training data only.'),
                 dict(path='research/outputs/p300_accuracy.svg',title='Held-out letter selection',description='All 20 later trials per participant; errors and abstentions retained.')],
        limitations=['Results describe these three recorded participants only; they are not accuracy on your ANT setup.',
                     'The replay shows stored classifier predictions from real held-out EEG. It does not react to a viewer’s thoughts.',
                     'This benchmark never unlocks the live EEG calibration gate or installs its models in the cap connection.',
                     'A small, fixed subset is used. No claim of across-participant generalization is made.',
                     'Known target labels are used to score predictions and show the answer, not as inputs to the inference function.'],
        provenance=dict(sources=[dict(subject=p.stem,url=SOURCE+p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in paths],
                        subjects_rule='First three numeric subject IDs, fixed before evaluating.',
                        mapping='MAT classes_stim explicitly identifies codes 1–6 as Row1–Row6 and 7–12 as Col1–Col6. This overrides the transposed illustration in the description PDF. Matrix alphabet from that figure.',
                        marker_rule='Rising edges of 32-sample marker pulses, not every nonzero sample; MATLAB trial indices converted to zero-based.',
                        ant_report_sha256=sha(ant_path),pipeline_sha256=sha(Path(__file__)),shared_decoder_sha256=sha(ROOT/'bridge'/'p300.py'),
                        versions={name:importlib.metadata.version(name) for name in ['numpy','scipy','scikit-learn','matplotlib']}))
    (OUT / 'p300_report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    lines=['# Public P300 offline calibration benchmark','',f'{correct}/{total} held-out selections correct ({correct/total:.1%}).',
           '',report['method']['training'],report['method']['quality'],'',*report['limitations'],'','## Attribution',
           'Riccio et al. (2013), Attention and P300-based BCI performance in people with amyotrophic lateral sclerosis.',
           'https://doi.org/10.3389/fnhum.2013.00732',report['dataset']['license'],report['dataset']['url'],
           '',report['provenance']['mapping'],'','## Reproduce','`.venv/bin/python research/p300_benchmark.py`']
    (OUT / 'p300_report.md').write_text('\n'.join(lines)+'\n')
    print(f'Total {correct}/{total} = {correct/total:.1%}. No live model updated.', flush=True)


if __name__ == '__main__': main()

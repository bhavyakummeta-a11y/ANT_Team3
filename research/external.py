"""Reproducible exploratory executed-vs-imagined EEG benchmark.

Run from the repository root:
    .venv/bin/python research/external.py

The selection and analysis are fixed before evaluation: first ten subjects,
all six unilateral fist runs, one non-overlapping active window per event,
subject-separated validation. No live decoder or ANT accuracy is produced.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from urllib.request import urlopen
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch, resample_poly
from fractions import Fraction
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'external' / 'physionet'
OUT = ROOT / 'research' / 'outputs'
SOURCE = 'https://physionet.org/files/eegmmidb/1.0.0/'
SUBJECTS = list(range(1, 11))
RUNS = [3, 4, 7, 8, 11, 12]
COMMON = ['C3', 'C4', 'F3', 'F4', 'Fz', 'P3', 'P4', 'Pz', 'O1', 'O2']
BANDS = {'mu_alpha': (8., 13.), 'beta': (13., 30.)}
SFREQ = 160.
SEED = 20261107
BASELINE = (-2., -.5)
ACTIVE = (.5, 2.)
MAX_PTP_V = 500e-6


def download_one(subject, run):
    name = f'S{subject:03d}R{run:02d}.edf'
    url = SOURCE + f'S{subject:03d}/' + name
    path = DATA / f'S{subject:03d}' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        # Atomic rename avoids treating an interrupted download as a valid EDF.
        with urlopen(url, timeout=90) as response:
            content = response.read()
        temp = path.with_suffix('.edf.part')
        temp.write_bytes(content)
        temp.replace(path)
    return path, url


def get_files(download=True):
    files, failures = [], []
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = {}
        for subject in SUBJECTS:
            for run in RUNS:
                path = DATA / f'S{subject:03d}' / f'S{subject:03d}R{run:02d}.edf'
                if download:
                    jobs[pool.submit(download_one, subject, run)] = (subject, run)
                elif path.exists():
                    files.append((subject, run, path, SOURCE + f'S{subject:03d}/' + path.name))
                else:
                    failures.append({'subject': subject, 'run': run, 'reason': 'Missing cached EDF; download disabled.'})
        for future in as_completed(jobs):
            subject, run = jobs[future]
            try:
                path, url = future.result()
                files.append((subject, run, path, url))
                print(f'Cached {path.name}', flush=True)
            except Exception as exc:
                failures.append({'subject': subject, 'run': run, 'reason': str(exc)})
    return sorted(files), failures


def bandpower(segment):
    """10 channels × 2 bands, volts squared, fixed Welch parameters."""
    f, psd = welch(segment, fs=SFREQ, nperseg=160, noverlap=80,
                   window='hann', detrend='constant', axis=-1)
    powers = []
    for low, high in BANDS.values():
        use = (f >= low) & (f <= high)
        powers.append(np.trapezoid(psd[:, use], f[use], axis=-1))
    return np.stack(powers, axis=-1)


def read_record(subject, run, path, url):
    raw = mne.io.read_raw_edf(path, preload=True, verbose='ERROR')
    standardized = {ch: ch.strip('.').upper() for ch in raw.ch_names}
    raw.rename_channels(standardized)
    raw.pick([c.upper() for c in COMMON])
    original_sfreq = float(raw.info['sfreq'])
    x = raw.get_data()  # MNE returns EEG in volts.
    # Shared-channel mean reference, identical definition for all datasets.
    x = x - x.mean(axis=0, keepdims=True)
    if original_sfreq != SFREQ:
        factor = Fraction(SFREQ / original_sfreq).limit_denominator(10000)
        x = resample_poly(x, factor.numerator, factor.denominator, axis=-1)
    x = sosfiltfilt(butter(4, [1., 35.], fs=SFREQ, btype='bandpass', output='sos'), x, axis=-1)
    annotations = list(zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description))
    epochs, rejected = [], []
    condition = 'executed' if run in (3, 7, 11) else 'imagined'
    for onset, duration, description in annotations:
        if description not in ('T1', 'T2'):
            continue
        onset, duration = float(onset), float(duration)
        common = {'subject': subject, 'run': run, 'condition': condition,
                  'hand': 'left' if description == 'T1' else 'right',
                  'onset_seconds': onset, 'duration_seconds': duration,
                  'recording': path.name}
        if duration < ACTIVE[1] or onset + ACTIVE[1] > x.shape[-1] / SFREQ:
            rejected.append({**common, 'reason': 'Active window outside annotated task/recording.'})
            continue
        a, b = [round((onset + t) * SFREQ) for t in ACTIVE]
        active = x[:, a:b]
        if active.shape[-1] != round((ACTIVE[1] - ACTIVE[0]) * SFREQ):
            rejected.append({**common, 'reason': 'Incorrect active sample count.'})
            continue
        if not np.isfinite(active).all() or np.ptp(active, axis=-1).max() > MAX_PTP_V:
            rejected.append({**common, 'reason': 'Non-finite data or >500 microvolt peak-to-peak in a selected channel.'})
            continue
        power = bandpower(active)
        if np.any(power <= 0) or not np.isfinite(power).all():
            rejected.append({**common, 'reason': 'Non-positive or non-finite bandpower.'})
            continue
        baseline_ok = any(d == 'T0' and float(t) <= onset + BASELINE[0] + 1e-6
                          and float(t) + float(length) >= onset + BASELINE[1] - 1e-6
                          for t, length, d in annotations)
        erd_db = None
        if baseline_ok:
            a, b = [round((onset + t) * SFREQ) for t in BASELINE]
            rest = x[:, a:b]
            baseline_ok = (rest.shape[-1] == 240 and np.isfinite(rest).all()
                           and np.ptp(rest, axis=-1).max() <= MAX_PTP_V)
            if baseline_ok:
                base_power = bandpower(rest)
                if np.all(base_power > 0) and np.isfinite(base_power).all():
                    erd_db = 10 * np.log10(power / base_power)
        epochs.append({**common, 'features': np.log10(power).ravel().tolist(),
                       'active_power_v2': power.tolist(),
                       'baseline_change_db': erd_db.tolist() if erd_db is not None else None,
                       'max_peak_to_peak_uv': float(np.ptp(active, axis=-1).max() * 1e6)})
    metadata = {'subject': subject, 'run': run, 'condition': condition,
                'file': path.relative_to(ROOT).as_posix(), 'url': url,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'bytes': path.stat().st_size, 'original_sampling_hz': original_sfreq,
                'duration_seconds': float(raw.n_times / original_sfreq),
                'accepted_task_epochs': len(epochs), 'rejected_task_epochs': len(rejected)}
    return epochs, rejected, metadata


def balance_epochs(epochs):
    """Balance condition AND hand within each subject using a fixed RNG."""
    retained, removed = [], []
    rng = np.random.default_rng(SEED)
    for subject in SUBJECTS:
        groups = [[e for e in epochs if e['subject'] == subject and e['condition'] == condition and e['hand'] == hand]
                  for condition in ('executed', 'imagined') for hand in ('left', 'right')]
        n = min(map(len, groups))
        for group in groups:
            order = rng.permutation(len(group))
            retained.extend(group[i] for i in order[:n])
            removed.extend({k: v for k, v in group[i].items() if k not in ('features', 'active_power_v2', 'baseline_change_db')}
                           for i in order[n:])
    return sorted(retained, key=lambda e: (e['subject'], e['run'], e['onset_seconds'])), removed


def evaluate(epochs):
    X = np.array([e['features'] for e in epochs])
    y = np.array([int(e['condition'] == 'imagined') for e in epochs])
    subjects = np.array([e['subject'] for e in epochs])
    unique = np.unique(subjects)
    if len(unique) < 3 or len(np.unique(y)) < 2:
        raise ValueError('Need at least three usable subjects and both conditions for this benchmark.')
    prediction = np.zeros_like(y)
    dummy_prediction = np.zeros_like(y)
    scores = np.zeros(len(y))
    subject_scores = []
    for subject in unique:
        test = subjects == subject
        train = ~test
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=2000, random_state=SEED))
        model.fit(X[train], y[train])
        prediction[test] = model.predict(X[test])
        scores[test] = model.predict_proba(X[test])[:, 1]
        dummy = DummyClassifier(strategy='most_frequent').fit(X[train], y[train])
        dummy_prediction[test] = dummy.predict(X[test])
        subject_scores.append({'subject': int(subject), 'epochs': int(test.sum()),
                               'balanced_accuracy': float(balanced_accuracy_score(y[test], prediction[test])),
                               'dummy_balanced_accuracy': float(balanced_accuracy_score(y[test], dummy_prediction[test]))})
    participant_scores = np.array([x['balanced_accuracy'] for x in subject_scores])
    rng = np.random.default_rng(SEED)
    boot = rng.choice(participant_scores, size=(5000, len(participant_scores)), replace=True).mean(axis=1)
    metrics = {'balanced_accuracy': float(balanced_accuracy_score(y, prediction)),
               'dummy_balanced_accuracy': float(balanced_accuracy_score(y, dummy_prediction)),
               'roc_auc': float(roc_auc_score(y, scores)),
               'subject_mean_balanced_accuracy': float(participant_scores.mean()),
               'subject_sd_balanced_accuracy': float(participant_scores.std(ddof=1)),
               'subject_bootstrap_95ci': np.quantile(boot, [.025, .975]).tolist(),
               'confusion_matrix': confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
               'class_labels': ['executed', 'imagined'],
               'hand_scores': {hand: float(balanced_accuracy_score(y[[e['hand'] == hand for e in epochs]], prediction[[e['hand'] == hand for e in epochs]])) for hand in ['left', 'right']}}
    predictions = [{k: e[k] for k in ('subject', 'run', 'condition', 'hand', 'onset_seconds', 'recording')} |
                   {'prediction': 'imagined' if int(prediction[i]) else 'executed', 'imagined_score': float(scores[i])}
                   for i, e in enumerate(epochs)]
    return metrics, subject_scores, predictions


def summarize_bands(epochs):
    rows, participant_rows = [], []
    for subject in sorted({e['subject'] for e in epochs}):
        for condition in ('executed', 'imagined'):
            values = [e['baseline_change_db'] for e in epochs if e['subject'] == subject and e['condition'] == condition and e['baseline_change_db'] is not None]
            if not values:
                continue
            average = np.mean(values, axis=0)
            for channel_i, channel in enumerate(COMMON):
                for band_i, band in enumerate(BANDS):
                    participant_rows.append({'subject': subject, 'condition': condition, 'channel': channel,
                                             'band': band, 'mean_change_db': float(average[channel_i, band_i]),
                                             'epochs': len(values)})
    for channel in COMMON:
        for band in BANDS:
            paired = []
            for subject in sorted({e['subject'] for e in epochs}):
                pair = {r['condition']: r['mean_change_db'] for r in participant_rows if r['subject'] == subject and r['channel'] == channel and r['band'] == band}
                if len(pair) == 2:
                    paired.append([pair['executed'], pair['imagined']])
            if paired:
                means = np.mean(paired, axis=0)
                rows.append({'band': band, 'channel': channel, 'executed_mean_db': float(means[0]),
                             'imagined_mean_db': float(means[1]), 'difference_db': float(means[1] - means[0]),
                             'subjects': len(paired)})
    return rows, participant_rows


def figures(metrics, scores, bands, participant_rows):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'font.family': 'DejaVu Sans'})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={'width_ratios': [1.45, 1]}, constrained_layout=True)
    ax = axes[0]
    ids = [f'S{s["subject"]:03d}' for s in scores]
    ax.bar(ids, [100*s['balanced_accuracy'] for s in scores], color='#267f94')
    ax.axhline(50, linestyle='--', color='#9b542b', label='Majority-class baseline: 50%')
    ax.set(ylim=(0, 100), ylabel='Balanced accuracy (%)', title='Executed vs imagined: held-out participant')
    ax.tick_params(axis='x', labelrotation=45); ax.legend(loc='upper left', fontsize=8)
    cm = np.array(metrics['confusion_matrix'])
    ax = axes[1]; ax.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center', color='white' if cm[i,j] > (cm.min()+cm.max())/2 else '#182c38', fontsize=18)
    ax.set(xticks=[0,1], xticklabels=['Executed','Imagined'], yticks=[0,1], yticklabels=['Executed','Imagined'], xlabel='Predicted condition', ylabel='Recorded condition', title='Pooled held-out predictions')
    fig.suptitle('Public-data benchmark · PhysioNet EEGMMIDB · S001–S010', fontsize=14)
    for suffix in ('svg','png'):
        fig.savefig(OUT/f'external_evaluation.{suffix}', dpi=160, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True, constrained_layout=True)
    for ax, band in zip(axes, BANDS):
        # Average C3/C4 per subject; do not treat electrodes or trials as independent people.
        for subject in sorted({r['subject'] for r in participant_rows}):
            vals=[]
            for condition in ('executed', 'imagined'):
                numbers=[r['mean_change_db'] for r in participant_rows if r['subject']==subject and r['condition']==condition and r['band']==band and r['channel'] in ('C3','C4')]
                vals.append(float(np.mean(numbers)) if numbers else np.nan)
            ax.plot([0,1],vals,'o-',color='#9baab2',alpha=.7,linewidth=1,markersize=4)
        means=[]
        for condition in ('executed','imagined'):
            means.append(float(np.mean([r[f'{condition}_mean_db'] for r in bands if r['band']==band and r['channel'] in ('C3','C4')])))
        ax.plot([0,1],means,'o-',color='#0c7285',linewidth=3,markersize=7,label='Mean of participant means')
        ax.axhline(0,color='#666',linewidth=.8,linestyle='--')
        ax.set(xticks=[0,1],xticklabels=['Executed','Imagined'],xlim=(-.3,1.3),title=f'{band.replace("_", " / ").title()} ({BANDS[band][0]:g}–{BANDS[band][1]:g} Hz)')
    axes[0].set_ylabel('Task / pre-task rest power (dB), C3/C4 mean')
    axes[1].legend(fontsize=8)
    fig.suptitle('Observed power changes · one line per participant · negative = lower task power', fontsize=12)
    for suffix in ('svg','png'):
        fig.savefig(OUT/f'external_bandpower.{suffix}', dpi=160, bbox_inches='tight')
    plt.close(fig)
    return [
        {'path':'research/outputs/external_evaluation.svg','title':'Public-data evaluation',
         'description':'Each bar is a participant excluded from model training. Counts combine held-out predictions; these are not ANT results.'},
        {'path':'research/outputs/external_bandpower.svg','title':'Task-related power changes in public recordings',
         'description':'C3/C4 baseline-relative power, with paired participant means. Negative values indicate lower task-window power; they do not by themselves identify intention.'}
    ]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--no-download',action='store_true',help='Use cached EDF files only; list missing recordings.')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    files, failures=get_files(not args.no_download)
    epochs, rejected, recordings=[],[],[]
    for subject, run, path, url in files:
        try:
            accepted, excluded, metadata=read_record(subject,run,path,url)
            epochs.extend(accepted); rejected.extend(excluded); recordings.append(metadata)
            print(f'Processed {path.name}: {len(accepted)} epochs, {len(excluded)} excluded',flush=True)
        except Exception as exc:
            failures.append({'subject':subject,'run':run,'reason':f'Processing failed: {exc}'})
    # Incomplete run sets are excluded at subject level; all exclusions are retained in the report.
    complete_subjects={s for s in SUBJECTS if {r['run'] for r in recordings if r['subject']==s}==set(RUNS)}
    incomplete=[e for e in epochs if e['subject'] not in complete_subjects]
    epochs=[e for e in epochs if e['subject'] in complete_subjects]
    balanced, balance_removed=balance_epochs(epochs)
    metrics, subject_scores, predictions=evaluate(balanced)
    bands, participant_rows=summarize_bands(balanced)
    figure_list=figures(metrics,subject_scores,bands,participant_rows)
    n_conditions=Counter(e['condition'] for e in balanced)
    n_subjects=len({e['subject'] for e in balanced})
    findings=[
        f'Analyzed {len(balanced)} balanced task epochs from {n_subjects} participants and {len(recordings)} public recordings.',
        f'Executed-versus-imagined condition classification achieved {metrics["balanced_accuracy"]:.1%} pooled leave-one-participant-out balanced accuracy; the majority-class baseline was {metrics["dummy_balanced_accuracy"]:.1%}.',
        f'Participant balanced accuracy ranged from {min(s["balanced_accuracy"] for s in subject_scores):.1%} to {max(s["balanced_accuracy"] for s in subject_scores):.1%}; performance is heterogeneous.',
        'This simple fixed model is near the 50% baseline; these results do not justify using it for software control.',
        'This benchmark uses movement conditions as experimental labels. It does not evaluate software-command selection, free-form thoughts, or ANT participants.'
    ]
    report={
        'schema_version':1,'status':'complete' if not failures else 'complete_with_exclusions',
        'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'source':{'name':'PhysioNet EEG Motor Movement/Imagery Dataset','version':'1.0.0',
                  'url':'https://physionet.org/content/eegmmidb/1.0.0/',
                  'doi':'10.13026/C28G6P','citation':'Schalk, G. (2009). EEG Motor Movement/Imagery Dataset (version 1.0.0). PhysioNet. Also cite Schalk et al., BCI2000, IEEE TBME 51(6):1034–1043 (2004), and PhysioNet.',
                  'license':'Open Data Commons Attribution License v1.0',
                  'license_url':'https://physionet.org/content/eegmmidb/view-license/1.0.0/'},
        'selection':{'planned_subjects':[f'S{s:03d}' for s in SUBJECTS], 'runs':RUNS,
                     'rationale':'First ten numbered subjects, selected before inspecting outcomes. Three repeated executed and three imagined unilateral-fist runs per participant.',
                     'executed_runs':[3,7,11],'imagined_runs':[4,8,12],
                     'hand_labels':{'T1':'left fist','T2':'right fist'},
                     'balance':'Within each participant, fixed-seed subsampling gives equal counts in executed-left, executed-right, imagined-left, imagined-right.',
                     'seed':SEED},
        'counts':{'subjects':n_subjects,'recordings':len(recordings),'epochs':len(balanced),
                  'executed_epochs':n_conditions['executed'],'imagined_epochs':n_conditions['imagined'],
                  'baseline_usable_epochs':sum(e['baseline_change_db'] is not None for e in balanced),
                  'quality_or_window_exclusions':len(rejected),'balancing_exclusions':len(balance_removed),
                  'incomplete_subject_epochs_excluded':len(incomplete)},
        'methods':{'channels':COMMON,'sampling_hz':SFREQ,'reference':'Mean of the same ten selected channels',
                   'filter':'Fourth-order Butterworth 1–35 Hz, forward/backward offline filtering on each recording independently.',
                   'resampling':'Polyphase anti-alias resampling to 160 Hz if the source rate differs.',
                   'task_window_seconds':list(ACTIVE),'baseline_window_seconds':list(BASELINE),
                   'baseline_rule':'Window must be fully contained in an annotated T0 rest interval and pass quality checks.',
                   'quality_rule':'Reject active windows with non-finite samples, non-positive bandpower or >500 µV peak-to-peak in any selected channel. Same amplitude rule for baseline inclusion. This is only a coarse artifact screen.',
                   'bands_hz':{k:list(v) for k,v in BANDS.items()},
                   'features':'20 features: log10 absolute active-window bandpower in volts², two bands per channel.',
                   'power_estimator':'Welch PSD, 160-sample Hann windows, 80-sample overlap, constant detrend; trapezoidal integration including band endpoints.',
                   'descriptive_change':'10 log10(task power / prior-rest power), in dB; average trials within participant before averaging participants.',
                   'classifier':'StandardScaler + logistic regression (C=1, max_iter=2000), fixed settings, no hyperparameter search.',
                   'validation':'Leave one entire participant out; standardization fitted on training participants only.',
                   'baseline_classifier':'Most-frequent-class dummy fitted on each training fold.',
                   'uncertainty':'95% percentile bootstrap interval over ten participant scores, 5,000 draws; small-sample descriptive uncertainty, not a generalization guarantee.'},
        'recordings':recordings,'download_or_processing_failures':failures,
        'excluded_epochs':rejected,'balancing_exclusions':balance_removed,
        'metrics':metrics,'subject_scores':subject_scores,
        'band_summary':bands,'participant_band_summary':participant_rows,
        'epoch_features':balanced,'held_out_predictions':predictions,
        'findings':findings,
        'limitations':[
            'Only a predeclared ten-participant subset was analyzed; this is not a full-dataset benchmark.',
            'Executed and imagined tasks occur in separate runs. Run effects, cues, muscle/eye activity and session conditions can contribute to classification.',
            'Coarse amplitude screening does not establish that signals are artifact-free or identify motor intent.',
            'Offline zero-phase filtering uses future samples; this processing is not a causal live decoder.',
            'Baseline-relative comparisons use annotated rest in PhysioNet. Unannotated gaps in ANT recordings are less certain baselines.',
            'Only one window is used per task annotation. Overlapping windows are not counted as independent trials.',
            'Public-data accuracy is not ANT accuracy. No model transfer, live control, clinical benefit, or arbitrary-thought decoding is established.',
            'Physical-versus-imagined condition classification does not itself solve intentional activation versus rest for an inclusive interface.'
        ],
        'figures':figure_list,
        'software':{name:importlib.metadata.version(name) for name in ('mne','numpy','scipy','scikit-learn','matplotlib')} | {'python':platform.python_version()},
        'analysis_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'reproduce':'.venv/bin/python research/external.py --no-download'
    }
    output=OUT/'external_report.json'
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(output),'counts':report['counts'],'metrics':metrics},indent=2),flush=True)


if __name__=='__main__':
    main()

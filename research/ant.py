"""Reproducible, descriptive ANT analysis; never dispatches application commands."""
from __future__ import annotations
import ast
import hashlib
import json
import platform
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly, welch
from fractions import Fraction

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research' / 'outputs'
COMMON = ['C3','C4','F3','F4','Fz','P3','P4','Pz','O1','O2']
BANDS = {'mu': (8,13), 'beta': (13,30)}
RATE = 160
TASKS = ['reach','one_finger','two_finger','pickup','putdown','letgo']
BASELINE = (-2., -.5)
ACTIVE = (.5, 2.)
COLORS = {'performed':'#2580a0','imagined':'#b56c32'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_name(label):
    label = label.split('/')[-1].lower().replace('imagine_', '').replace('image_', '').replace('_','')
    return {'onefinger':'one_finger','twofinger':'two_finger'}.get(label, label)


def overlap(a,b):
    return a[0] < b[1] and b[0] < a[1]


def window_reasons(onset, duration, tasks, length, boundaries=()):
    baseline=(onset+BASELINE[0],onset+BASELINE[1])
    active=(onset+ACTIVE[0],onset+ACTIVE[1])
    reasons=[]
    if baseline[0]<0 or active[1]>length: reasons.append('Window outside recording')
    if duration < ACTIVE[1]: reasons.append('Annotation shorter than fixed 2-second active endpoint')
    for boundary in boundaries:
        if overlap(baseline,(boundary-3,boundary+3)) or overlap(active,(boundary-3,boundary+3)):
            reasons.append('Window within 3 seconds of an acquisition-segment boundary')
    for other in tasks:
        interval=(other['onset'],other['onset']+other['duration'])
        if overlap(baseline,interval): reasons.append('Baseline overlaps an annotated task')
        if other['onset'] != onset and overlap(active,interval): reasons.append('Active window overlaps another task')
    return list(dict.fromkeys(reasons))


def bandpower(x):
    freq, psd=welch(x, fs=RATE, nperseg=RATE, noverlap=RATE//2, axis=-1)
    return {name: np.trapezoid(psd[:,(freq>=lo)&(freq<=hi)],freq[(freq>=lo)&(freq<=hi)],axis=-1)
            for name,(lo,hi) in BANDS.items()}


def segment_metadata(path, n_samples, sfreq):
    sidecar=path.with_suffix('.seg')
    if not sidecar.exists(): return {'segments':1,'boundaries_seconds':[],'source':None}
    lines=sidecar.read_text().strip().splitlines()
    declared=int(lines[0].split('=')[1])
    later=[int(line.split()[-1]) for line in lines[1:]]
    first=n_samples-sum(later)
    if len(later)!=declared-1 or first<=0 or any(n<=0 for n in later):
        raise ValueError(f'Unrecognized segmentation metadata in {sidecar}')
    return {'segments':declared,'samples_per_segment':[first,*later],
            'boundaries_seconds':(np.cumsum([first,*later])[:-1]/sfreq).tolist(),
            'source':str(sidecar.relative_to(ROOT)),
            'interpretation':'Later segment sample counts from SEG; first inferred from total CNT samples; boundary markers cross-checked against impedance annotations.'}


def preprocess(raw, boundaries=()):
    data=raw.get_data()
    ref=[raw.ch_names.index(c) for c in COMMON]
    data=data-data[ref].mean(axis=0,keepdims=True)
    ratio=Fraction(RATE/int(raw.info['sfreq'])).limit_denominator(10000)
    data=resample_poly(data,ratio.numerator,ratio.denominator,axis=-1)
    sos=butter(4,[1,35],btype='bandpass',fs=RATE,output='sos')
    edges=[0,*[round(t*RATE) for t in boundaries],data.shape[1]]
    for start,stop in zip(edges[:-1],edges[1:]):
        data[:,start:stop]=sosfiltfilt(sos,data[:,start:stop],axis=-1)
    return data


def savefig(fig, name, title, description, figures):
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight')
    fig.savefig(OUT/(name+'.png'),dpi=150,bbox_inches='tight')
    plt.close(fig)
    figures.append({'path':f'research/outputs/{name}.svg','title':title,'description':description,
                    'png':f'research/outputs/{name}.png'})


def json_clean(value):
    if isinstance(value,dict): return {k:json_clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [json_clean(v) for v in value]
    if isinstance(value,np.ndarray): return json_clean(value.tolist())
    if isinstance(value,(np.floating,float)): return float(value) if np.isfinite(value) else None
    if isinstance(value,np.integer): return int(value)
    return value


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    recordings=[];events=[];manifest=[];figures=[];timing=[];matrices={};processed={};spectra=[]
    for p in sorted((ROOT/'data').glob('rotation_*/*')):
        if p.is_file():manifest.append({'file':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)})
    legacy={}
    tree=ast.parse((ROOT/'data/analysis.py').read_text())
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DATASETS' for t in node.targets):
            legacy=ast.literal_eval(node.value)
    for path in sorted((ROOT/'data').glob('rotation_*/*.cnt')):
        print('ANT:',path.name,flush=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            raw=mne.io.read_raw_ant(path,preload=True,verbose='ERROR')
        duration=raw.n_times/raw.info['sfreq']
        anns=[{'onset':float(t),'duration':float(d),'label':str(l)} for t,d,l in zip(raw.annotations.onset,raw.annotations.duration,raw.annotations.description)]
        motor=path.parent.name=='rotation_two'
        condition='imagined' if ('imaginary' in path.name.lower()) else ('performed' if motor else 'initial')
        hand='left' if 'left' in path.name.lower() else ('right' if 'right' in path.name.lower() else 'unconfirmed')
        tasks=[a for a in anns if task_name(a['label']) in TASKS] if motor else []
        segmentation=segment_metadata(path,raw.n_times,raw.info['sfreq'])
        quality={}; warn=[str(w.message) for w in caught]
        values=raw.get_data()
        quality['nonfinite_samples']=int((~np.isfinite(values)).sum())
        quality['flat_channels']=[c for c,v in zip(raw.ch_names,np.std(values,axis=1)) if v<1e-12]
        quality['raw_peak_to_peak_uv']={c:float(v) for c,v in zip(raw.ch_names,np.ptp(values,axis=1)*1e6)}
        if hand=='unconfirmed' and motor:warn.append('Imagery hand is unconfirmed; legacy code calls it right, but filename and annotations do not establish this.')
        if not motor:warn.append('Initial recording is not a labeled performed-versus-imagined training example.')
        if any('EC' in a['label'] for a in anns):warn.append('EC meaning and eye-state durations require experiment notes.')
        rec={'id':path.stem,'file':str(path.relative_to(ROOT)),'duration_seconds':duration,'sampling_hz':raw.info['sfreq'],'channels':raw.ch_names,
             'role':'performed versus imagined comparison' if motor else 'initial signal and annotation inspection',
             'condition':condition,'hand':hand,'segmentation':segmentation,'annotations':anns,'quality':quality,'warnings':warn,'task_intervals':len(tasks)}
        recordings.append(rec)
        if quality['nonfinite_samples'] or quality['flat_channels']:
            warn.append('Nonfinite or flat-channel data prevents this pipeline from making valid spectral comparisons.');continue
        data=preprocess(raw,segmentation['boundaries_seconds']); common_idx=[raw.ch_names.index(c) for c in COMMON]
        processed[path.stem]=(rec,data,raw.ch_names)
        f,psd=welch(data[common_idx],fs=RATE,nperseg=RATE*2,axis=-1)
        spectra.append((rec,f,psd))
        for a in tasks:
            reasons=window_reasons(a['onset'],a['duration'],tasks,duration,segmentation['boundaries_seconds'])
            event={'recording':path.stem,'condition':condition,'hand':hand,'task':task_name(a['label']),**a,
                   'valid':not reasons,'reason':'; '.join(reasons) if reasons else 'Fixed windows available; baseline is unlabelled, not confirmed rest.',
                   'features':{},'quality_flags':[]}
            if not reasons:
                b=data[:,round((a['onset']+BASELINE[0])*RATE):round((a['onset']+BASELINE[1])*RATE)]
                x=data[:,round((a['onset']+ACTIVE[0])*RATE):round((a['onset']+ACTIVE[1])*RATE)]
                pb,px=bandpower(b),bandpower(x)
                if any(np.any(pb[n]<=0)|np.any(px[n]<=0) for n in BANDS):
                    event['valid']=False;event['reason']='Nonpositive band power prevents a finite power ratio.'
                else:
                    for name in BANDS:
                        db=10*np.log10(px[name]/pb[name])
                        for i,c in enumerate(raw.ch_names):
                            event['features'][f'{name}_{c}_db']=float(db[i])
                            if c in COMMON:event['features'][f'{name}_{c}_log10_v2']=float(np.log10(px[name][i]))
                        event['features'][name+'_common_mean_db']=float(db[common_idx].mean())
                    peak=float(max(np.ptp(b[common_idx],axis=1).max(),np.ptp(x[common_idx],axis=1).max())*1e6)
                    event['peak_to_peak_uv']=peak
                    if peak>300:event['quality_flags'].append('Processed common-channel peak-to-peak exceeds 300 µV; artifact screening flag, not a diagnosis or automatic exclusion.')
            events.append(event)
        for name,old in legacy.items():
            if Path(old['file']).name==path.name:
                for t,a in zip(old['times'],tasks):
                    timing.append({'recording':path.stem,'task':task_name(a['label']),'legacy_seconds':t,'annotation_seconds':a['onset'],
                                   'difference_seconds':a['onset']-t})
    valid=[e for e in events if e['valid']]
    comparisons=[]
    for task in TASKS:
        matched=[e for e in valid if e['hand']=='left' and e['task']==task]
        a=next((e for e in matched if e['condition']=='performed'),None)
        b=next((e for e in matched if e['condition']=='imagined'),None)
        if a and b:
            comparisons.append({'task':task,'hand':'left','performed_recording':a['recording'],'imagined_recording':b['recording'],
              'mu_difference_db':b['features']['mu_common_mean_db']-a['features']['mu_common_mean_db'],
              'beta_difference_db':b['features']['beta_common_mean_db']-a['features']['beta_common_mean_db'],
              'interpretation':'Imagined minus performed baseline-relative change; one interval per condition, no uncertainty estimate.'})
    # All recordings contribute to the spectral inventory, including unlabelled initial tests.
    fig,axes=plt.subplots(2,1,figsize=(11,8),sharex=True)
    for rec,f,psd in spectra:
        ax=axes[1 if rec['condition']!='initial' else 0]
        label=(rec['condition']+' / '+rec['hand']) if rec['condition']!='initial' else rec['id'].split('_')[-1]
        ax.plot(f,10*np.log10(np.maximum(psd.mean(axis=0)*1e12,1e-20)),label=label)
    for ax in axes:
        ax.set_xlim(1,35);ax.set_ylabel('Mean PSD (dB µV²/Hz)');ax.legend(fontsize=8);ax.grid(alpha=.2)
    axes[0].set_title('Initial recordings: whole-record spectral inventory')
    axes[1].set_title('Movement experiments: whole-record spectra (includes tasks and unlabelled time)')
    axes[1].set_xlabel('Frequency (Hz)');fig.tight_layout()
    savefig(fig,'ant_spectra','Spectral inventory of all 10 recordings','Whole-record common10 mean spectra are descriptive; recording duration and task occupancy differ.',figures)
    motor_recs=[v for v in processed.values() if v[0]['condition']!='initial']
    fig,axes=plt.subplots(len(motor_recs),1,figsize=(13,10))
    for ax,(rec,data,ch) in zip(np.atleast_1d(axes),motor_recs):
        ts=np.arange(data.shape[1])/RATE
        for c in ['C3','C4']:
            ax.plot(ts[::4],data[ch.index(c),::4]*1e6,lw=.65,label=c,alpha=.8)
        for a in rec['annotations']:
            if task_name(a['label']) in TASKS:
                ax.axvspan(a['onset'],a['onset']+a['duration'],color='#e4ac66',alpha=.25)
                ax.text(a['onset'],.95,task_name(a['label']),rotation=30,fontsize=7,va='top',transform=ax.get_xaxis_transform())
        ax.set_ylabel('µV');ax.set_title(f"{rec['condition'].title()} · {rec['hand']} hand · {rec['sampling_hz']:g} Hz source",loc='left');ax.legend(loc='lower right',fontsize=8)
    axes[-1].set_xlabel('Seconds from recording start');fig.tight_layout()
    savefig(fig,'ant_timeline','EEG with actual task annotations','Measured C3/C4 signals, common10 reference and offline 1–35 Hz filter, downsampled for display. Shading is annotation timing, not model output.',figures)
    if valid:
        fig,axes=plt.subplots(1,2,figsize=(14,max(5,len(valid)*.33)))
        labels=[f"{e['condition'][:4]} / {e['hand']} / {e['task']}" for e in valid]
        for ax,band in zip(axes,BANDS):
            matrix=np.array([[e['features'][f'{band}_{c}_db'] for c in COMMON] for e in valid])
            limit=max(1,float(np.max(np.abs(matrix))))
            im=ax.imshow(matrix,aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit)
            ax.set_xticks(range(len(COMMON)),COMMON,rotation=45);ax.set_yticks(range(len(valid)),labels,fontsize=8)
            ax.set_title(f'{band.title()} change relative to unlabelled pre-task interval')
            fig.colorbar(im,ax=ax,label='10 log10(active / baseline) · dB',shrink=.8)
        fig.tight_layout()
        savefig(fig,'ant_band_changes','Task-level band-power changes','Each row is one annotated interval. Negative values indicate reduced band power; differences do not establish decoded intent. Missing intervals are listed with exclusions.',figures)
        # Scalp positions use a template, never imply source localisation.
        info=mne.create_info(COMMON,RATE,ch_types='eeg');info.set_montage(mne.channels.make_standard_montage('standard_1020'))
        fig,axes=plt.subplots(2,2,figsize=(9,8))
        for row,condition in enumerate(['performed','imagined']):
            subset=[e for e in valid if e['hand']=='left' and e['condition']==condition and e['task'] in {c['task'] for c in comparisons}]
            for col,band in enumerate(BANDS):
                ax=axes[row,col]
                if subset:
                    v=np.mean([[e['features'][f'{band}_{c}_db'] for c in COMMON] for e in subset],axis=0)
                    im,_=mne.viz.plot_topomap(v,info,axes=ax,show=False,cmap='RdBu_r',vlim=(-20,20),contours=0)
                    fig.colorbar(im,ax=ax,shrink=.7,label='dB')
                ax.set_title(f'Left {condition}: {band}, n={len(subset)} intervals')
        fig.suptitle('Exploratory sensor maps · template positions · not source localisation')
        fig.tight_layout()
        savefig(fig,'ant_topography','Exploratory electrode maps','Averages of the same three eligible left-hand task types in each condition; one observation per task/condition. Fixed ±20 dB color scale; 10 shared scalp electrodes, template geometry.',figures)
    if comparisons:
        fig, axes = plt.subplots(1,2,figsize=(11,4))
        for ax,band in zip(axes,BANDS):
            ax.axvline(0,color='gray',lw=1)
            for i,comparison in enumerate(comparisons):
                vals=[]
                for condition in ['performed','imagined']:
                    e=next(e for e in valid if e['hand']=='left' and e['task']==comparison['task'] and e['condition']==condition)
                    vals.append(e['features'][band+'_common_mean_db'])
                ax.plot(vals,[i,i],color='#999999',lw=1)
                for v,c in zip(vals,['performed','imagined']):ax.scatter(v,i,color=COLORS[c],label=c if i==0 else None,zorder=3)
            ax.set_yticks(range(len(comparisons)),[c['task'] for c in comparisons])
            ax.set_xlabel('Change from unlabelled baseline (dB)');ax.set_title(band.title());ax.legend();ax.grid(alpha=.15)
        fig.suptitle('Matched left-hand tasks · one interval per condition · common10 mean')
        fig.tight_layout()
        savefig(fig,'ant_matched_pairs','Matched performed-versus-imagined observations','Each line connects the same annotated action in the two left-hand recordings. These three pairs do not establish reproducibility.',figures)
    excluded=len(events)-len(valid)
    findings=[f'Inspected {len(recordings)} ANT recordings containing {sum(len(r["annotations"]) for r in recordings)} annotations; {len(events)} are annotated movement/imagery intervals.',
              f'{len(valid)} of {len(events)} task intervals satisfy the predefined timing and numeric checks; {excluded} are excluded from fixed-window band-power comparison.',
              f'{len(comparisons)} left-hand task pairs have usable windows in both conditions. These are descriptive paired observations, not repeated independent trials.',
              f'All {len(timing)} legacy action timestamps differ from the exported annotations; the new analysis uses annotation timing.',
              'No ANT classifier accuracy is reported: each condition/hand has one recording and each action has one interval, so independent replication is missing.']
    if comparisons:
        for band in BANDS:
            diffs=[x[f'{band}_difference_db'] for x in comparisons]
            findings.append(f'Across {len(diffs)} eligible left-task pairs, imagined-minus-performed {band} baseline-relative change spans {min(diffs):.2f} to {max(diffs):.2f} dB. This range describes these intervals only.')
    limitations=['The annotations label instructions/epochs, not verified cognition or precise movement onset.',
      'Pre-task intervals are unlabelled; rest is assumed for descriptive normalization and needs experimental confirmation.',
      'The ImaginaryHand recording has unconfirmed laterality and is excluded from matched-hand comparisons.',
      'Some task annotations overlap or are shorter than the analysis window; exclusions are retained in the report.',
      'Eye and muscle activity can affect these signals. Bandpass filtering and a scalp map do not remove or prove absence of artifacts.',
      'Only one recording per hand/condition is available; run effects, task order, and recording setup can explain differences.',
      'No p-values, population confidence intervals, live-decoder accuracy, or clinical conclusions are claimed.',
      'Offline zero-phase filtering uses future samples; a future live decoder requires separately validated causal preprocessing.',
      '300 µV flags are a screening heuristic; no automatic artifact repair, ICA, or manual adjudication has been performed.',
      'Physical vs imagined classification is a research comparison; successful separation would not establish an intentional software-selection command.']
    report={'schema_version':1,'status':'Measured ANT observations — exploratory','source':{'name':'ANT_Team3 local eego exports','root':'data/rotation_one and data/rotation_two'},
      'methods':{'common_channels':COMMON,'target_hz':RATE,'filter':'Anti-alias polyphase resampling to 160 Hz, then 4th-order Butterworth 1–35 Hz zero-phase SOS (same filter at the same rate as external analysis)',
      'reference':'Mean of the same 10 common EEG channels, applied to all ANT electrodes','bands_hz':BANDS,'baseline_seconds':BASELINE,'active_seconds':ACTIVE,
      'spectrum':'Welch Hann, 1-second windows, 50% overlap; trapezoidal band integration','feature':'10 log10(active power / pre-task power), dB; absolute log10(V²) features retained for comparison',
      'quality':'Nonfinite/flat recordings excluded; >300 µV processed peak-to-peak flagged without automatic exclusion','labels':'MNE ANT reader annotations; no hardcoded analysis timestamps','segmentation':'SEG sample counts identify discontinuities. Resample to 160 Hz, then filter each segment independently; exclude windows within 3 s of boundaries to guard filter/resampling transients.'},
      'recordings':recordings,'events':events,'comparisons':comparisons,'timing_audit':timing,'findings':findings,'limitations':limitations,'figures':figures,
      'counts':{'recordings':len(recordings),'task_intervals':len(events),'eligible_intervals':len(valid),'excluded_intervals':excluded,'matched_left_pairs':len(comparisons)},
      'provenance':{'files':manifest,'python':platform.python_version(),'mne':mne.__version__,'numpy':np.__version__,'analysis_sha256':sha(Path(__file__))}}
    (OUT/'ant_report.json').write_text(json.dumps(json_clean(report),indent=2,allow_nan=False)+'\n')
    lines=['# ANT recording analysis','',*['- '+f for f in findings],'','## Methods','',json.dumps(report['methods'],indent=2),'','## Limitations','',*['- '+s for s in limitations],
           '','## File inventory','',*['- '+r['file']+f' — {r["duration_seconds"]:.3f}s, {r["sampling_hz"]:g}Hz, {len(r["channels"])} channels' for r in recordings],
           '','## Event eligibility','',*['- '+e['recording']+' / '+e['task']+': '+('included' if e['valid'] else 'excluded')+' — '+e['reason'] for e in events]]
    (OUT/'ant_report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(findings),flush=True)


if __name__=='__main__':main()

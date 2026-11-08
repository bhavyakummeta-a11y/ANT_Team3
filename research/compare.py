"""Compare declared features across datasets without pooling them as one trial set."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/outputs'


def main():
    ant=json.loads((OUT/'ant_report.json').read_text())
    external=json.loads((OUT/'external_report.json').read_text())
    channels=ant['methods']['common_channels']
    tasks={c['task'] for c in ant['comparisons']}
    local=[e for e in ant['events'] if e['valid'] and e['hand']=='left' and e['task'] in tasks]
    public=[e for e in external['epoch_features'] if e['hand']=='left' and e['baseline_change_db'] is not None]
    rows=[];subject_diffs={};local_diffs={}
    for band_i,band in enumerate(['mu','beta']):
        local_diffs[band]=[]
        for task in sorted(tasks):
            pair={e['condition']:np.array([e['features'][f'{band}_{c}_db'] for c in channels]) for e in local if e['task']==task}
            local_diffs[band].append(pair['imagined']-pair['performed'])
        subject_diffs[band]=[]
        for subject in sorted({e['subject'] for e in public}):
            pair={c:np.mean([np.array(e['baseline_change_db'])[:,band_i] for e in public if e['subject']==subject and e['condition']==c],axis=0) for c in ['executed','imagined']}
            subject_diffs[band].append(pair['imagined']-pair['executed'])
        local_mean=np.mean(local_diffs[band],axis=0)
        external_mean=np.mean(subject_diffs[band],axis=0)
        for i,ch in enumerate(channels):
            rows.append({'band':band,'channel':ch,'ant_difference_db':float(local_mean[i]),'external_difference_db':float(external_mean[i]),
                         'direction_agrees':bool(np.sign(local_mean[i])==np.sign(external_mean[i]))})
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),sharey=True,layout='constrained')
    for ax,band in zip(axes,['mu','beta']):
        sub=[r for r in rows if r['band']==band]
        ax.axhline(0,color='gray',linestyle='--',linewidth=.8)
        ax.plot(channels,[r['ant_difference_db'] for r in sub],'o-',label='ANT: mean of 3 matched task pairs',color='#b56c32')
        ax.plot(channels,[r['external_difference_db'] for r in sub],'s-',label='PhysioNet: mean of 10 participant differences',color='#2580a0')
        ax.set_title(band.title()+' band');ax.tick_params(axis='x',rotation=45);ax.grid(alpha=.15);ax.legend(fontsize=8)
    axes[0].set_ylabel('Imagined − performed baseline-relative change (dB)')
    fig.suptitle('Left-hand experimental conditions · shared electrodes · different tasks and baselines',fontsize=12)
    for suffix in ['svg','png']:fig.savefig(OUT/f'comparison_profiles.{suffix}',dpi=160,bbox_inches='tight')
    plt.close(fig)
    findings=[
      'Matched features were compared at the same ten electrodes using 160 Hz, a 1–35 Hz passband, 8–13/13–30 Hz power, common-channel mean reference, and fixed event windows.',
      'ANT contributes three matched left-hand task pairs from two recordings; PhysioNet contributes left-fist conditions from ten participants. These are different experimental units and different actions.',
      f'The public classifier reached {external["metrics"]["balanced_accuracy"]:.1%} balanced accuracy versus a {external["metrics"]["dummy_balanced_accuracy"]:.1%} dummy baseline. This simple approach is near chance and is not ready to drive software selection.',
      'This comparison is descriptive: matching or opposing feature directions does not establish model transfer or validate cognition. No ANT accuracy or raw-waveform correlation is computed.'
    ]
    limitations=[
      'ANT baselines are unlabelled gaps; PhysioNet baselines are explicitly annotated rest.',
      'The ANT reach/two-finger/let-go labels do not correspond exactly to repeated fist opening/closing in PhysioNet.',
      'ANT task pairs come from one participant/session context; they are not three independent participants.',
      'ANT amplitude excursions are flagged at 300 µV; external epochs above 500 µV are excluded. Quality-control policy differs and is disclosed in each report.',
      'Electrode measurements share a reference and are statistically dependent; no electrode-wise significance test is offered.',
      'No consistent, validated software-command signal has been established. A new randomized, repeated-task ANT recording with an independent held-out run is needed.'
    ]
    report={'schema_version':1,'status':'Exploratory cross-dataset comparison','findings':findings,'limitations':limitations,'channel_comparisons':rows,
            'counts':{'ant_task_pairs':len(tasks),'external_participants':len({e['subject'] for e in public})},
            'figures':[{'path':'research/outputs/comparison_profiles.svg','title':'Compare feature directions across datasets','description':'Imagined-minus-performed change in baseline-relative power. Left-hand conditions only; different tasks, baselines and experimental units prevent a direct replication claim.'}],
            'provenance':{'ant_report_sha256':hashlib.sha256((OUT/'ant_report.json').read_bytes()).hexdigest(),'external_report_sha256':hashlib.sha256((OUT/'external_report.json').read_bytes()).hexdigest()},
            'next_experiment':[
              'Choose one or two precisely defined mental tasks after reviewing signal quality; do not map them to hand directions by default.',
              'Randomize repeated task and explicit-rest trials within each run, saving cue onset and end markers.',
              'Keep hardware settings, references, and sampling rates consistent; record observations of movement/eye artifacts.',
              'Collect a separate run that remains untouched during feature, threshold, and model selection.',
              'Measure false activations during rest, missed trials, latency, and participant variability before connecting a command detector.'
            ]}
    (OUT/'comparison_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (OUT/'comparison_report.md').write_text('# Cross-dataset comparison\n\n'+'\n\n'.join(findings)+'\n\n## Limitations\n\n'+'\n'.join('- '+l for l in limitations)+'\n\n## Next experiment\n\n'+'\n'.join('- '+l for l in report['next_experiment'])+'\n')
    print('\n'.join(findings))


if __name__=='__main__':main()

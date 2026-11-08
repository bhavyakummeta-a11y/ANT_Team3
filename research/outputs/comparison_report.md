# Cross-dataset comparison

Matched features were compared at the same ten electrodes using 160 Hz, a 1–35 Hz passband, 8–13/13–30 Hz power, common-channel mean reference, and fixed event windows.

ANT contributes three matched left-hand task pairs from two recordings; PhysioNet contributes left-fist conditions from ten participants. These are different experimental units and different actions.

The public classifier reached 53.9% balanced accuracy versus a 50.0% dummy baseline. This simple approach is near chance and is not ready to drive software selection.

This comparison is descriptive: matching or opposing feature directions does not establish model transfer or validate cognition. No ANT accuracy or raw-waveform correlation is computed.

## Limitations

- ANT baselines are unlabelled gaps; PhysioNet baselines are explicitly annotated rest.
- The ANT reach/two-finger/let-go labels do not correspond exactly to repeated fist opening/closing in PhysioNet.
- ANT task pairs come from one participant/session context; they are not three independent participants.
- ANT amplitude excursions are flagged at 300 µV; external epochs above 500 µV are excluded. Quality-control policy differs and is disclosed in each report.
- Electrode measurements share a reference and are statistically dependent; no electrode-wise significance test is offered.
- No consistent, validated software-command signal has been established. A new randomized, repeated-task ANT recording with an independent held-out run is needed.

## Next experiment

- Choose one or two precisely defined mental tasks after reviewing signal quality; do not map them to hand directions by default.
- Randomize repeated task and explicit-rest trials within each run, saving cue onset and end markers.
- Keep hardware settings, references, and sampling rates consistent; record observations of movement/eye artifacts.
- Collect a separate run that remains untouched during feature, threshold, and model selection.
- Measure false activations during rest, missed trials, latency, and participant variability before connecting a command detector.

# Public P300 offline calibration benchmark

26/60 held-out selections correct (43.3%).

First three participants A01–A03 fixed before fitting. Separate model per participant: first 15 selections train, final 20 test. No test-label fitting or tuning.
Same raw limits as live: 300 µV peak-to-peak maximum and 0.01 µV minimum SD. Any failed epoch rejects the whole selection. Rejected/abstained test selections stay in the accuracy denominator.

Results describe these three recorded participants only; they are not accuracy on your ANT setup.
The replay shows stored classifier predictions from real held-out EEG. It does not react to a viewer’s thoughts.
This benchmark never unlocks the live EEG calibration gate or installs its models in the cap connection.
A small, fixed subset is used. No claim of across-participant generalization is made.
Known target labels are used to score predictions and show the answer, not as inputs to the inference function.

## Attribution
Riccio et al. (2013), Attention and P300-based BCI performance in people with amyotrophic lateral sclerosis.
https://doi.org/10.3389/fnhum.2013.00732
CC BY-NC-ND 4.0 · Santa Lucia Neuroelectrical Imaging and BCI Laboratory
https://bnci-horizon-2020.eu/database/data-sets

MAT classes_stim explicitly identifies codes 1–6 as Row1–Row6 and 7–12 as Col1–Col6. This overrides the transposed illustration in the description PDF. Matrix alphabet from that figure.

## Reproduce
`.venv/bin/python research/p300_benchmark.py`

# ANT recording analysis

- Inspected 10 ANT recordings containing 57 annotations; 24 are annotated movement/imagery intervals.
- 18 of 24 task intervals satisfy the predefined timing and numeric checks; 6 are excluded from fixed-window band-power comparison.
- 3 left-hand task pairs have usable windows in both conditions. These are descriptive paired observations, not repeated independent trials.
- All 24 legacy action timestamps differ from the exported annotations; the new analysis uses annotation timing.
- No ANT classifier accuracy is reported: each condition/hand has one recording and each action has one interval, so independent replication is missing.
- Across 3 eligible left-task pairs, imagined-minus-performed mu baseline-relative change spans -1.38 to 1.66 dB. This range describes these intervals only.
- Across 3 eligible left-task pairs, imagined-minus-performed beta baseline-relative change spans -1.45 to 3.12 dB. This range describes these intervals only.

## Methods

{
  "common_channels": [
    "C3",
    "C4",
    "F3",
    "F4",
    "Fz",
    "P3",
    "P4",
    "Pz",
    "O1",
    "O2"
  ],
  "target_hz": 160,
  "filter": "Anti-alias polyphase resampling to 160 Hz, then 4th-order Butterworth 1\u201335 Hz zero-phase SOS (same filter at the same rate as external analysis)",
  "reference": "Mean of the same 10 common EEG channels, applied to all ANT electrodes",
  "bands_hz": {
    "mu": [
      8,
      13
    ],
    "beta": [
      13,
      30
    ]
  },
  "baseline_seconds": [
    -2.0,
    -0.5
  ],
  "active_seconds": [
    0.5,
    2.0
  ],
  "spectrum": "Welch Hann, 1-second windows, 50% overlap; trapezoidal band integration",
  "feature": "10 log10(active power / pre-task power), dB; absolute log10(V\u00b2) features retained for comparison",
  "quality": "Nonfinite/flat recordings excluded; >300 \u00b5V processed peak-to-peak flagged without automatic exclusion",
  "labels": "MNE ANT reader annotations; no hardcoded analysis timestamps",
  "segmentation": "SEG sample counts identify discontinuities. Resample to 160 Hz, then filter each segment independently; exclude windows within 3 s of boundaries to guard filter/resampling transients."
}

## Limitations

- The annotations label instructions/epochs, not verified cognition or precise movement onset.
- Pre-task intervals are unlabelled; rest is assumed for descriptive normalization and needs experimental confirmation.
- The ImaginaryHand recording has unconfirmed laterality and is excluded from matched-hand comparisons.
- Some task annotations overlap or are shorter than the analysis window; exclusions are retained in the report.
- Eye and muscle activity can affect these signals. Bandpass filtering and a scalp map do not remove or prove absence of artifacts.
- Only one recording per hand/condition is available; run effects, task order, and recording setup can explain differences.
- No p-values, population confidence intervals, live-decoder accuracy, or clinical conclusions are claimed.
- Offline zero-phase filtering uses future samples; a future live decoder requires separately validated causal preprocessing.
- 300 µV flags are a screening heuristic; no automatic artifact repair, ICA, or manual adjudication has been performed.
- Physical vs imagined classification is a research comparison; successful separation would not establish an intentional software-selection command.

## File inventory

- data/rotation_one/1_Test_2026-11-07_13-52-30.cnt — 67.644s, 1000Hz, 24 channels
- data/rotation_one/1_Test_2026-11-07_13-58-53.cnt — 20.044s, 500Hz, 24 channels
- data/rotation_one/1_Test_2026-11-07_14-00-26.cnt — 29.506s, 500Hz, 24 channels
- data/rotation_one/1_Test_2026-11-07_14-03-18.cnt — 110.740s, 500Hz, 24 channels
- data/rotation_one/1_Test_2026-11-07_14-09-11.cnt — 18.812s, 500Hz, 24 channels
- data/rotation_one/1_Test_2026-11-07_14-20-52.cnt — 578.374s, 1000Hz, 24 channels
- data/rotation_two/EthanImaginaryLeft_Team 3_2026-11-07_16-24-16.cnt — 43.790s, 1000Hz, 24 channels
- data/rotation_two/EthanLeftReal_Team 3_2026-11-07_16-20-38.cnt — 45.022s, 1000Hz, 24 channels
- data/rotation_two/ImaginaryHand_Ethan_Team 3_2026-11-07_16-15-59.cnt — 75.148s, 500Hz, 24 channels
- data/rotation_two/RightHandReal_Ethan_Team 3_2026-11-07_16-09-21.cnt — 143.672s, 500Hz, 24 channels

## Event eligibility

- EthanImaginaryLeft_Team 3_2026-11-07_16-24-16 / reach: included — Fixed windows available; baseline is unlabelled, not confirmed rest.
- EthanImaginaryLeft_Team 3_2026-11-07_16-24-16 / one_finger: excluded — Annotation shorter than fixed 2-second active endpoint
- EthanImaginaryLeft_Team 3_2026-11-07_16-24-16 / two_finger: included — Fixed windows available; baseline is unlabelled, not confirmed rest.
- EthanImaginaryLeft_Team 3_2026-11-07_16-24-16 / pickup: excluded — Annotation shorter than fixed 2-second active endpoint
- EthanImaginaryLeft_Team 3_2026-11-07_16-24-16 / putdown: excluded — Annotation shorter than fixed 2-second active endpoint
- EthanImaginaryLeft_Team 3_2026-11-07_16-24-16 / letgo: included — Fixed windows available; baseline is unlabelled, not confirmed rest.
- EthanLeftReal_Team 3_2026-11-07_16-20-38 / reach: included — Fixed windows available; baseline is unlabelled, not confirmed rest.
- EthanLeftReal_Team 3_2026-11-07_16-20-38 / one_finger: included — Fixed windows available; baseline is unlabelled, not confirmed rest.
- EthanLeftReal_Team 3_2026-11-07_16-20-38 / two_finger: included — Fixed windows available; baseline is unlabelled, not confirmed rest.
- EthanLeftReal_Team 3_2026-11-07_16-20-38 / pickup: included — Fixed windows available; baseline is unlabelled, not confirmed rest.
- EthanLeftReal_Team 3_2026-11-07_16-20-38 / putdown: included — Fixed windows available; baseline is unlabelled, not confirmed rest.
- EthanLeftReal_Team 3_2026-11-07_16-20-38 / letgo: included — Fixed windows available; baseline is unlabelled, not confirmed rest.
- ImaginaryHand_Ethan_Team 3_2026-11-07_16-15-59 / reach: included — Fixed windows available; baseline is unlabelled, not confirmed rest.
- ImaginaryHand_Ethan_Team 3_2026-11-07_16-15-59 / one_finger: excluded — Baseline overlaps an annotated task; Active window overlaps another task
- ImaginaryHand_Ethan_Team 3_2026-11-07_16-15-59 / two_finger: included — Fixed windows available; baseline is unlabelled, not confirmed rest.
- ImaginaryHand_Ethan_Team 3_2026-11-07_16-15-59 / pickup: included — Fixed windows available; baseline is unlabelled, not confirmed rest.
- ImaginaryHand_Ethan_Team 3_2026-11-07_16-15-59 / putdown: included — Fixed windows available; baseline is unlabelled, not confirmed rest.
- ImaginaryHand_Ethan_Team 3_2026-11-07_16-15-59 / letgo: excluded — Baseline overlaps an annotated task; Active window overlaps another task
- RightHandReal_Ethan_Team 3_2026-11-07_16-09-21 / reach: included — Fixed windows available; baseline is unlabelled, not confirmed rest.
- RightHandReal_Ethan_Team 3_2026-11-07_16-09-21 / one_finger: included — Fixed windows available; baseline is unlabelled, not confirmed rest.
- RightHandReal_Ethan_Team 3_2026-11-07_16-09-21 / two_finger: excluded — Baseline overlaps an annotated task
- RightHandReal_Ethan_Team 3_2026-11-07_16-09-21 / pickup: included — Fixed windows available; baseline is unlabelled, not confirmed rest.
- RightHandReal_Ethan_Team 3_2026-11-07_16-09-21 / putdown: included — Fixed windows available; baseline is unlabelled, not confirmed rest.
- RightHandReal_Ethan_Team 3_2026-11-07_16-09-21 / letgo: included — Fixed windows available; baseline is unlabelled, not confirmed rest.

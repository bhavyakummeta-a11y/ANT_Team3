# ANT_Team3

NEUROBIND website and EEG recording analysis.

## Attention grid: typing and assistant prototype

Run the bridge **on your Mac** while eego streams LSL on Windows over the same Wi-Fi:

```sh
.venv/bin/python -m pip install -r bridge/requirements.txt
.venv/bin/python bridge/server.py --port 8765
```

Open **http://localhost:8765** (or `/speller.html`). The grid is now the bridge's
home page; the existing reports remain at `/research.html` and the waveform
monitor at `/research.html#live`. Windows does not need Python for this setup.

1. Find EEG streams, choose the eego stream, and Connect.
2. Click **Check signal** before a round. It shows each decoder electrode's raw
   and 1–15 Hz peak-to-peak amplitude, raw standard deviation, declared source
   unit and conversion factor. A rejected round shows the same diagnostic for
   all 27 epochs, naming each failing electrode. A flat channel and excessive
   raw amplitude are reported separately; thresholds remain 300 µV raw
   peak-to-peak and 0.01 µV raw standard deviation.
   If the lab operator verifies that an electrode is unused, deselect it under
   **Calibration channels**, then **Apply channels** and check again. At least three
   supported channels including Pz/P3/P4 are required. Choices freeze after the
   first accepted round and remain fixed for training, evaluation and selection.
   Do not exclude a channel just to improve the held-out score.
   Then start calibration round 1. Attend to the named tile and count its highlights.
3. Complete 18 rounds, each started when ready (roughly four minutes of stimuli,
   plus breaks). The first nine train; the final nine evaluate the frozen model.
4. At least 7/9 correct held-out selections unlock **Start EEG selection**.
   Otherwise the decoder remains locked; reconnect to start fresh calibration.
5. Attend to a letter or command tile during a selection round. The decoder
   selects a tile only when the top two mean scores differ by at least 0.05.
   This score difference is not calibrated confidence. Use Stop/Escape anytime.

Letters are paged across five nine-tile grids with Next letters, Space and Delete.
Assistant mode proposes an action, then requires Confirm or Cancel; all robot
execution is explicitly simulated and sends no hardware commands. Conventional
mouse/touch/keyboard controls work between rounds. Preview highlights produces
no EEG selection. The older amplitude-trigger navigation is a separate experiment
on the research/legacy pages and is not used by the attention grid.

The decoder uses three passes of nine individually highlighted cells, 100 ms
highlights approximately 350 ms apart, and named central/parietal channels.
It needs at least three supported EEG channels including Pz/P3/P4, recognized
voltage units, and a regular 100–2000 Hz stream. Features are baseline-corrected
1–15 Hz responses in sixteen 50 ms bins over 0–800 ms. A training-only scaler
and class-balanced logistic regression classify target versus non-target events;
held-out evaluation is by whole selection round, never by randomly split flashes.
The model is not refit on the holdout. Nine test rounds are a small same-session
check, not evidence of generalizable assistive reliability. Calibration on letters
does not separately validate robot-command use.

The inlet applies LSL clock correction; browser frame timestamps are mapped to
the receiver's LSL clock using round-trip measurements each round. Timing gaps,
invalid samples, flat channels, >300 µV peak-to-peak epochs, or software timing
uncertainty >20 ms reject a round. Physical display latency remains unmeasured;
use a photodiode/hardware timing validation for rigorous ERP experiments.

Download session evidence before disconnecting: JSON includes grid labels,
targets, onset markers, processed ERP features and held-out predictions, plus
the last 20 rejected-round diagnostics. Download is available even at 0/18 rounds. This is
**not continuous raw EEG**; use LabRecorder/XDF separately for raw recordings.
Sessions/models are memory-only and reconnecting clears calibration. Existing
movement recordings/public motor benchmarks are not P300 training data; no
pretrained P300 model or successful cap-based selection is claimed.

Method references: [BCI2000 P3SpellerTask](https://www.bci2000.org/mediawiki/index.php/User_Reference:P3SpellerTask)
and [LSL time synchronization](https://labstreaminglayer.readthedocs.io/info/time_synchronization.html).

Tests: `.venv/bin/python -m unittest discover -s bridge -p 'test_*.py'`.

## Local development

From this repository, run:

```sh
python3 -m http.server 8000 --bind 127.0.0.1
```

Open http://localhost:8000 in your browser. Edit `index.html`, save, and
refresh the browser to see changes. Stop the server with Ctrl+C.
No frontend dependencies or build step are required.

## EEG integration status

The original demo charts and commands are simulated. The separate Live EEG
connection panel can receive real samples through the Python bridge described below;
a physical device connection must be established explicitly.
Recorded EEG files are in `data/rotation_one` and `data/rotation_two`;
`data/analysis.py` contains a separate offline analysis pipeline.

The bridge supports eego LSL EEG streaming. Movement classification and
separate motion-sensor integration are not implemented.

## Live eego connection (Windows acquisition computer)

The live EEG panel is available on both `index.html` and `index2.html`.
Run the bridge on the Windows computer where eego displays the EEG traces.
In eego, enable **Enable LSL EEG streaming** and keep acquisition running.

In PowerShell, from this repository:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r bridge\requirements.txt
.\.venv\Scripts\python.exe bridge\server.py --host 0.0.0.0 --port 8765
```

Open `http://localhost:8765/research.html#live` on Windows. Click **Find EEG streams**, select
the eego stream, then **Connect**. Choose a channel to view its received samples.
For the alternate page, open `http://localhost:8765/index2.html`.

On your Mac, open `http://WINDOWS-IP:8765` using the Windows computer's active
IPv4 address from `ipconfig`. Both devices must have local network access.
If Windows Firewall prompts, allow Python on the trusted private network.
This development bridge has no authentication: use it only on your trusted
local network and do not forward its port to the internet.

For a bridge running on the Mac instead:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r bridge/requirements.txt
.venv/bin/python bridge/server.py
```

The bridge serves the website pages, their UI assets, and approved generated
research reports/figures; it does not expose raw recordings or model binaries. The old port-8000
static server cannot provide the EEG API. Open the bridge URL on port 8765.

### What the panel does

- Discovers numeric LSL streams whose type is EEG and displays their source host.
- Receives samples over a WebSocket and plots the last five seconds of one channel.
- Preserves source channel labels, units, and LSL sample timestamps.
- Shows connection errors, stale data, and explicit disconnect status.
- Does not classify movement or dispatch commands. Existing demo graphs remain simulated.

Values use the units advertised by the source, with no implicit conversion.
The display auto-scales each channel; its amplitude is not a calibrated fixed scale.
LSL timestamps are source-clock values, not browser wall-clock timestamps.
This panel is a live preview, not a recording tool; continue recording in eego.

### Troubleshooting

- **Bridge unavailable:** start the Python bridge and open its port-8765 URL.
- **No EEG streams:** check acquisition and the EEG-streaming option, then search
  again. Running the bridge on the eego computer avoids cross-computer LSL discovery.
- **Mac cannot open Windows URL:** check the IPv4 address, Python's private-network
  firewall access, and whether the network isolates devices.
- **No samples / connection lost:** restart acquisition as needed, disconnect,
  find streams again, and reconnect. Reconnection is deliberately manual.
- **Missing liblsl:** follow the platform installation instructions at
  https://github.com/labstreaminglayer/pylsl#installation.

Bridge integration checks (uses a clearly named synthetic LSL test source):

```sh
.venv/bin/python -m unittest discover -s bridge -p 'test_*.py'
```

## Research workspace: ANT + PhysioNet

Open **http://localhost:8765/research.html** (also the bridge's default `/` page).
The earlier pages link to **Research & interaction**. For static-only browsing,
run the port-8000 server above and open `http://localhost:8000/research.html`.
The reports and figures work without a connected EEG device.

The research workspace includes:

- All ten ANT recordings, source-file hashes, annotations, segment metadata,
  event-window eligibility, and an audit of the legacy script's event times.
- Measured spectra, annotated C3/C4 traces, band-power changes, matched-task
  comparisons, and exploratory scalp maps.
- A separate PhysioNet benchmark with held-out-participant results, baseline
  performance, exclusions, source attribution, and participant-level variation.
- A descriptive comparison on shared electrodes. Agreement is not claimed as
  proof of model transfer, intention decoding, or repeatability.
- A task board with mouse/keyboard controls, optional scanning, generic
  selection/cancel/navigation, and a visibly labeled manual simulation.
  It is not connected to an EEG classifier. Live acquisition remains available
  in the separate expandable Live EEG connection section.

### Reproduce the research

Install the analysis dependencies in the project's virtual environment:

```sh
.venv/bin/python -m pip install -r research/requirements.txt
.venv/bin/python research/ant.py
.venv/bin/python research/external.py
.venv/bin/python research/compare.py
```

The external step downloads the predeclared **S001–S010** subjects, runs
**3, 4, 7, 8, 11, 12** of PhysioNet EEGMMIDB v1.0.0: 60 EDF files, approximately
155 MB, stored under `data/external/physionet/`. Raw external files are ignored
by Git. Cached downloads are reused. To explicitly disable downloading:

```sh
.venv/bin/python research/external.py --no-download
```

Generated JSON reports, Markdown summaries and SVG/PNG figures are in
`research/outputs/`. Run `compare.py` after either input report changes.
The external report records selection, file hashes, exclusions, exact methods,
software versions, held-out predictions and feature values. The ANT report
records source hashes and its generating script's hash.

Run validation after generating reports:

```sh
.venv/bin/python -m unittest discover -s research -p 'test_*.py'
.venv/bin/python -m unittest discover -s bridge -p 'test_*.py'
node --check research-ui.js
```

### Interpretation of the first run

ANT: 24 annotated movement/imagery intervals, 18 eligible fixed-window
comparisons, and only three matched left-hand task pairs. The unspecified-hand
imagery recording remains unspecified. Unlabelled pre-task periods are not
verified rest. No ANT classifier accuracy is reported.

PhysioNet: 864 balanced task epochs from ten participants. Fixed log-band-power
features with training-only standardization and logistic regression produced
**53.9% pooled leave-one-participant-out balanced accuracy**, compared with
**50.0%** for the dummy classifier. This simple model is near chance and is not
ready for software control. No threshold was optimized on the ANT participant.

The two datasets use the same ten common electrode names, reference definition,
160 Hz processing, 1–35 Hz passband, Welch estimator and task/baseline windows.
Tasks, participants, baseline certainty and quality-screening policies differ;
the comparison report discloses these differences. Hardware transfer has not
been validated. Physical-versus-imagined discrimination does not itself solve
intentional selection versus rest.

`data/analysis.py` is retained as legacy experimental source for the timing
audit. Its hardcoded timestamps, action mapper and authenticity-test claims
are not used by the new reports. Existing hand-direction demos are retained
as earlier simulations, not the primary interaction design.

### External data attribution

Dataset: Schalk, G. (2009), *EEG Motor Movement/Imagery Dataset*, version 1.0.0,
PhysioNet. DOI: https://doi.org/10.13026/C28G6P.

Original acquisition publication: Schalk et al. (2004), *BCI2000: A
General-Purpose Brain-Computer Interface (BCI) System*, IEEE Transactions on
Biomedical Engineering 51(6), 1034–1043.

Source, licensing and full required citation information:
https://physionet.org/content/eegmmidb/1.0.0/
The dataset lists the Open Data Commons Attribution License v1.0. Keep the
attribution with reused data/results. This project uses a declared subset;
it does not claim to analyze the entire public database.

PhysioNet platform citation listed by the source: Pollard et al. (2026),
*PhysioNet as a global platform for biomedical research*, Nature Health.
DOI: https://doi.org/10.1038/s44360-026-00096-z.

## Experimental live EEG tab clicks

Live samples can now drive **an amplitude-threshold interaction experiment**.
This does not use the public classifier and does not recognize thoughts.

1. Open the Live EEG connection section, find/select your eego stream, and connect.
2. Choose a signal channel using **Display channel**. This is also the trigger channel.
3. Click **Calibrate · 6 seconds** while keeping a steady baseline.
4. Choose **Click next website tab** or a specific website tab. Start with the
   default **1.8× baseline** threshold; the live meter shows the measured ratio.
5. Click **Arm tab clicks**. After the signal is near baseline for half a second,
   an amplitude rise held for half a second clicks the selected tab.
6. Return near baseline before another activation. A 2.5-second cooldown prevents
   repeated clicks from one sustained rise. Stop at any time with **Escape**,
   **Stop**, or the fixed **EEG tab clicks ARMED — STOP** button.

The detector uses centered RMS over the most recent 0.5 seconds, divided by
median RMS from calibration. It has no frequency-specific thought model:
blinks, muscle activity, movement, and noise can activate it. Its thresholds
are experimental controls, not validated confidence values. Only navigation
inside this website is affected; it does not click other applications or browser tabs.
The task board remains controlled conventionally or by its explicitly labeled
manual simulation; live EEG controls website navigation only.

Reconnection, a channel change, non-finite/flat data, or a sample/timestamp gap
requires calibration again. A hidden page stops automatic clicks; rearm after
returning. Changing the target or threshold also disarms clicks.

Detector regression tests:

```sh
node --test tests/eeg-trigger.test.cjs
```

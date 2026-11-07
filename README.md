# ANT_Team3

NEUROBIND website and EEG recording analysis.

## Local development

From this repository, run:

```sh
python3 -m http.server 8000 --bind 127.0.0.1
```

Open http://localhost:8000 in your browser. Edit `index.html`, save, and
refresh the browser to see changes. Stop the server with Ctrl+C.
No frontend dependencies or build step are required.

## EEG integration status

The website currently displays simulated waveforms and dispatches simulated
commands. It is not connected to a live EEG device or motion sensor.
Recorded EEG files are in `data/rotation_one` and `data/rotation_two`;
`data/analysis.py` contains a separate offline analysis pipeline.

For live integration, the next step is to identify the EEG device and its
streaming interface, plus any separate motion sensor. A local backend can
receive those streams and send timestamped samples and detected events to
the website. Recorded-data playback can be added first to develop the UI
without connected hardware.

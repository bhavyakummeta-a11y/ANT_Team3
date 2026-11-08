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

Open `http://localhost:8765` on Windows. Click **Find EEG streams**, select
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

The bridge serves only the two website pages and the live-panel JavaScript;
it does not expose recordings or other repository files. The old port-8000
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

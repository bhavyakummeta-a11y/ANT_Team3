"""Local LSL-to-WebSocket bridge. Run on the eego acquisition computer."""
import argparse
import asyncio
import json
import math
import logging
from pathlib import Path
import time

from aiohttp import web
from pylsl import StreamInlet, resolve_streams, local_clock, proc_clocksync
from p300 import P300Session

ROOT = Path(__file__).resolve().parents[1]


def summary(info):
    return dict(id=info.uid(), name=info.name(), type=info.type(),
                channels=info.channel_count(), sample_rate=info.nominal_srate(),
                hostname=info.hostname())


def channels(info):
    node = info.desc().child('channels').child('channel')
    result = []
    for i in range(info.channel_count()):
        result.append(dict(label=node.child_value('label') or f'Channel {i+1}',
                           unit=node.child_value('unit') or 'unspecified'))
        node = node.next_sibling()
    return result


def create_app(initial_timeout=30, stall_timeout=10):
    app = web.Application()
    available = {}
    discovery_lock = asyncio.Lock()

    async def discover(request):
        async with discovery_lock:
            found = await asyncio.to_thread(resolve_streams, wait_time=3.0)
            # Only numeric EEG streams; marker streams cannot be plotted as EEG.
            found = [s for s in found if s.type().lower() == 'eeg' and s.channel_format() in (1, 2, 3, 4, 5, 7)]
            available.clear()
            available.update({s.uid(): s for s in found})
            return web.json_response({'streams': [summary(s) for s in found]})

    async def stream(request):
        origin = request.headers.get('Origin')
        if origin and origin != f'{request.scheme}://{request.host}':
            raise web.HTTPForbidden(text='Open the website from this bridge address.')
        info = available.get(request.query.get('id', ''))
        if info is None:
            raise web.HTTPNotFound(text='Stream unavailable. Find EEG streams again.')
        ws = web.WebSocketResponse(heartbeat=15)
        await ws.prepare(request)
        inlet = None
        p300 = None
        use_p300 = request.query.get('p300') == '1'
        async def receive_close():
            async for message in ws:
                if not use_p300 or message.type != web.WSMsgType.TEXT:
                    continue
                payload = {}
                try:
                    payload = json.loads(message.data)
                    if not isinstance(payload, dict):
                        raise ValueError('Expected a JSON object.')
                    op = payload.get('op')
                    if op == 'clock':
                        result = {'clock': local_clock()}
                    elif p300 is None:
                        raise ValueError('Wait for EEG connection.')
                    elif op == 'begin':
                        if not p300.buffer or local_clock() - p300.buffer[-1][0] > 1:
                            raise ValueError('Fresh EEG samples are required.')
                        labels = payload.get('labels', [])
                        if not isinstance(labels, list) or len(labels) != 9 or any(not isinstance(x, str) or len(x) > 80 for x in labels):
                            raise ValueError('Nine grid labels are required.')
                        p300.begin(payload.get('target'))
                        p300.round['labels'] = labels
                        p300.round['purpose'] = str(payload.get('purpose', 'unknown'))[:40]
                        result = {'started': True}
                    elif op == 'marker':
                        p300.marker(payload['cell'], float(payload['onset']), float(payload['uncertainty']), local_clock())
                        result = {'marked': True}
                    elif op == 'finish':
                        result = p300.finish()
                    elif op == 'evaluate':
                        result = await asyncio.to_thread(p300.evaluate)
                    elif op == 'abort':
                        p300.round = None
                        result = {'aborted': True}
                    elif op == 'export':
                        result = p300.export()
                    else:
                        raise ValueError('Unknown P300 operation.')
                    await ws.send_json({'type': 'reply', 'id': payload.get('id'), 'result': result})
                except (ValueError, KeyError, TypeError) as exc:
                    if p300 is not None:
                        p300.round = None
                    await ws.send_json({'type': 'reply', 'id': payload.get('id') if isinstance(payload, dict) else None, 'error': str(exc)})
        receiver = asyncio.create_task(receive_close())
        try:
            inlet = StreamInlet(info, max_buflen=30, max_chunklen=1, recover=False,
                                processing_flags=proc_clocksync if use_p300 else 0)
            await asyncio.to_thread(inlet.open_stream, timeout=5.0)
            full = await asyncio.to_thread(inlet.info, timeout=5.0)
            if use_p300:
                p300 = P300Session(full.nominal_srate(), channels(full))
            await ws.send_json({'type': 'connected', 'stream': summary(full), 'channels': channels(full)})
            last_sample = time.monotonic()
            received = 0
            last_status = 0
            logging.info("Opened LSL stream %s on %s", full.name(), full.hostname())
            while not ws.closed:
                samples, stamps = await asyncio.to_thread(inlet.pull_chunk, timeout=0.1, max_samples=512)
                if len(stamps):
                    received += len(stamps)
                    last_sample = time.monotonic()
                    clean = [[float(x) if math.isfinite(float(x)) else None for x in row] for row in samples]
                    if p300 is not None:
                        p300.feed(clean, stamps)
                    await ws.send_str(json.dumps({'type': 'samples', 'samples': clean, 'timestamps': stamps}, allow_nan=False))
                else:
                    silence = time.monotonic() - last_sample
                    limit = stall_timeout if received else initial_timeout
                    if silence > limit:
                        message = (f'LSL stream opened, but no samples arrived in {limit} seconds.' if not received
                                   else f'EEG sample delivery stopped for {limit} seconds after {received} samples.')
                        logging.warning('%s Source: %s', message, full.name())
                        await ws.send_json({'type': 'error', 'message': message + ' The graph is waiting on LSL data, not a recording file. Find streams and reconnect.'})
                        break
                    if time.monotonic() - last_status >= 1:
                        last_status = time.monotonic()
                        await ws.send_json({'type': 'waiting', 'seconds': round(silence), 'received': received, 'timeout': limit})
                await asyncio.sleep(0.03)
        except Exception as exc:
            if not ws.closed:
                try:
                    await ws.send_json({'type': 'error', 'message': f'LSL connection failed: {exc}'})
                except ConnectionError:
                    pass
        finally:
            if inlet is not None:
                inlet.close_stream()
            await ws.close()
            await receiver
        return ws

    async def asset(request):
        name = request.match_info.get('name', 'speller.html')
        if name not in {'speller.html', 'speller.js', 'speller.css', 'index.html', 'index2.html', 'eeg-live.js', 'eeg-trigger.js', 'research.html', 'research-ui.js', 'research-ui.css'}:
            raise web.HTTPNotFound()
        return web.FileResponse(ROOT / name)

    async def research_output(request):
        name = request.match_info['name']
        # Only generated public-facing reports/figures, never recordings or model binaries.
        if (Path(name).name != name or not name.startswith(('ant_', 'external_', 'comparison_'))
                or Path(name).suffix not in {'.json', '.svg', '.png', '.md', '.csv'}):
            raise web.HTTPNotFound()
        path = ROOT / 'research' / 'outputs' / name
        if not path.is_file():
            raise web.HTTPNotFound()
        response = web.FileResponse(path)
        response.headers['Cache-Control'] = 'no-cache'
        return response

    app.router.add_get('/research/outputs/{name}', research_output)
    app.router.add_get('/api/streams', discover)
    app.router.add_get('/api/eeg', stream)
    app.router.add_get('/', asset)
    app.router.add_get('/{name}', asset)
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    web.run_app(create_app(), host=args.host, port=args.port)

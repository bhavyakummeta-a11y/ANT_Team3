"""Local LSL-to-WebSocket bridge. Run on the eego acquisition computer."""
import argparse
import asyncio
import json
import math
import logging
from pathlib import Path
import time

from aiohttp import web
from pylsl import StreamInlet, resolve_streams

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
        async def receive_close():
            async for _ in ws:
                pass
        receiver = asyncio.create_task(receive_close())
        try:
            inlet = StreamInlet(info, max_buflen=30, max_chunklen=1, recover=False)
            await asyncio.to_thread(inlet.open_stream, timeout=5.0)
            full = await asyncio.to_thread(inlet.info, timeout=5.0)
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
        name = request.match_info.get('name', 'index.html')
        if name not in {'index.html', 'index2.html', 'eeg-live.js'}:
            raise web.HTTPNotFound()
        return web.FileResponse(ROOT / name)

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

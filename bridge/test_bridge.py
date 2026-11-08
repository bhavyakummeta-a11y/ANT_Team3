import asyncio
import unittest
from unittest.mock import patch
from aiohttp.test_utils import TestClient, TestServer
from pylsl import StreamInfo, StreamOutlet
from server import create_app


class BridgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = TestClient(TestServer(create_app(initial_timeout=2, stall_timeout=2)))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    async def test_assets_and_missing_stream(self):
        for path in ['/', '/speller.html', '/speller.js', '/speller.css', '/index2.html', '/eeg-live.js', '/eeg-trigger.js', '/research.html', '/research-ui.js', '/research-ui.css', '/research/outputs/ant_report.json']:
            response = await self.client.get(path)
            self.assertEqual(response.status, 200)
        for path in ['/README.md', '/data/analysis.py', '/api/eeg?id=missing', '/research/ant.py', '/research/outputs/external_model.joblib']:
            response = await self.client.get(path)
            self.assertEqual(response.status, 404)

    async def test_empty_discovery(self):
        with patch('server.resolve_streams', return_value=[]):
            response = await self.client.get('/api/streams')
            self.assertEqual(await response.json(), {'streams': []})

    async def test_stream_without_samples(self):
        info = StreamInfo('NEUROBIND_EMPTY_TEST_ONLY', 'EEG', 1, 100, 'float32', 'neurobind-empty')
        outlet = StreamOutlet(info)
        response = await self.client.get('/api/streams')
        streams = (await response.json())['streams']
        test = next(s for s in streams if s['name'] == 'NEUROBIND_EMPTY_TEST_ONLY')
        ws = await self.client.ws_connect('/api/eeg?id=' + test['id'])
        self.assertEqual((await ws.receive_json(timeout=8))['type'], 'connected')
        waiting = await ws.receive_json(timeout=4)
        self.assertEqual(waiting['type'], 'waiting')
        self.assertEqual(waiting['received'], 0)
        while True:
            message = await ws.receive_json(timeout=4)
            if message['type'] == 'error':
                self.assertIn('no samples arrived', message['message'])
                break
        await ws.close()
        del outlet

    async def test_real_lsl_transport(self):
        info = StreamInfo('NEUROBIND_SYNTHETIC_TEST_ONLY', 'EEG', 2, 100, 'float32', 'neurobind-test')
        channels = info.desc().append_child('channels')
        for label in ['C3', 'C4']:
            node = channels.append_child('channel')
            node.append_child_value('label', label)
            node.append_child_value('unit', 'microvolts')
        outlet = StreamOutlet(info)
        response = await self.client.get('/api/streams')
        streams = (await response.json())['streams']
        test = next(s for s in streams if s['name'] == 'NEUROBIND_SYNTHETIC_TEST_ONLY')
        rejected = await self.client.get('/api/eeg?id=' + test['id'], headers={'Origin': 'http://unrelated.example'})
        self.assertEqual(rejected.status, 403)
        ws = await self.client.ws_connect('/api/eeg?id=' + test['id'])
        connected = await ws.receive_json(timeout=8)
        self.assertEqual(connected['type'], 'connected')
        self.assertEqual(connected['channels'][0], {'label':'C3', 'unit':'microvolts'})
        for _ in range(20):
            outlet.push_sample([12.5, -7.25])
            await asyncio.sleep(0.01)
        received = await ws.receive_json(timeout=5)
        while received['type'] == 'waiting':
            received = await ws.receive_json(timeout=5)
        self.assertEqual(received['type'], 'samples')
        self.assertEqual(received['samples'][0], [12.5, -7.25])
        self.assertEqual(len(received['samples']), len(received['timestamps']))
        while True:
            message = await ws.receive_json(timeout=7)
            if message['type'] == 'error':
                self.assertIn('sample delivery stopped', message['message'])
                break
        await ws.close()
        await asyncio.sleep(0.2)
        del outlet


if __name__ == '__main__':
    unittest.main()

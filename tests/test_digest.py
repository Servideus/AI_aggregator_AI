import asyncio
import datetime as dt
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
from unittest.mock import Mock

import aggregator_llm as app
import llm_adapter


class FakeClient:
    def __init__(self, *args):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class DigestTests(unittest.TestCase):
    def test_actual_sdk_json_configuration_without_network(self):
        client = Mock()
        client.models.generate_content.return_value = Mock(parsed=[{'id': 1}])
        with patch.object(llm_adapter, '_CLIENT', client), patch.object(llm_adapter, '_MODEL', 'test-model'):
            self.assertEqual(llm_adapter.complete_json([{'role': 'user', 'content': 'Example'}]), [{'id': 1}])
            config = client.models.generate_content.call_args.kwargs['config']
            self.assertEqual(config.response_mime_type, 'application/json')

    def test_generic_sources_and_disabled_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'sources.yml'
            path.write_text('sources:\n - username: example\n - username: disabled\n   enabled: false\n')
            self.assertEqual([c.id for c in app.load_sources(str(path))], ['example'])

    def test_full_digest_flow_saves_before_delivery(self):
        post = app.Post('example', 'Example', 1, dt.datetime.now(dt.UTC),
                        'https://t.me/example/1', 'Example news')
        async def verify_delivery(api_id, api_hash, text):
            files = list(Path('.').glob('digest_*.txt'))
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].read_text(encoding='utf-8'), text)
            self.assertIn('Example headline', text)

        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            os.chdir(folder)
            try:
                with patch.dict(os.environ, {'TG_API_ID': '123', 'TG_API_HASH': 'test'}), \
                     patch.object(app, 'load_sources', return_value=[app.Channel('example', 'Example')]), \
                     patch.object(app, 'TelegramClient', FakeClient), \
                     patch.object(app, 'fetch_posts', AsyncMock(return_value=[post])), \
                     patch.object(app, 'complete_json', return_value=[
                         {'id': 1, 'title': 'Example headline', 'summary': 'Summary'},
                         {'id': 1, 'title': 'Duplicate', 'summary': 'Duplicate'},
                         {'id': 999, 'title': 'Unknown', 'summary': 'Unknown'}]), \
                     patch.object(app, 'send_digest', side_effect=verify_delivery):
                    asyncio.run(app.run_digest())
            finally:
                os.chdir(previous)

    def test_empty_destination_does_not_connect(self):
        with patch.object(app, 'DIGEST_PEER', None), patch.object(app, 'TelegramClient') as client:
            asyncio.run(app.send_digest(123, 'test', 'Digest'))
            client.assert_not_called()


if __name__ == '__main__':
    unittest.main()

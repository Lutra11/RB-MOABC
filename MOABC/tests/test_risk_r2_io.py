import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class RiskR2IOTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_completed_run_requires_valid_marker_and_npz_hash(self):
        from MOABC.experiments.artifact_store import ArtifactStore

        store = ArtifactStore(self.root)
        manifest = store.freeze_manifest({'protocol_hash': 'p', 'expected_runs': 1})
        store.write_run_arrays('r1', schedule=np.ones((5, 3)))
        self.assertFalse(store.is_complete('r1', manifest))
        store.commit_run('r1', {'score': 1.0}, manifest)
        self.assertTrue(store.is_complete('r1', manifest))
        with (self.root / 'r1.npz').open('ab') as handle:
            handle.write(b'corrupt')
        self.assertFalse(store.is_complete('r1', manifest))

    def test_manifest_is_immutable_and_jsonl_is_append_only(self):
        from MOABC.experiments.artifact_store import ArtifactStore

        store = ArtifactStore(self.root)
        store.freeze_manifest({'protocol_hash': 'p'})
        with self.assertRaises(ValueError):
            store.freeze_manifest({'protocol_hash': 'different'})
        store.event('start', completed=0, total=2)
        store.event('done', completed=1, total=2)
        rows = [json.loads(line) for line in (self.root / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual([row['event'] for row in rows], ['start', 'done'])

    def test_latest_origin_ignores_partial_or_corrupt_checkpoint(self):
        from MOABC.experiments.artifact_store import ArtifactStore

        store = ArtifactStore(self.root)
        store.commit_origin('run', 0, {'remaining_budget': .04}, state=np.array([1., 2.]))
        folder = self.root / 'checkpoints' / 'run'
        (folder / 'origin_0001.json.partial').write_text('{}', encoding='utf-8')
        self.assertEqual(store.latest_valid_origin('run')['origin'], 0)
        (folder / 'origin_0000.npz').write_bytes(b'broken')
        self.assertIsNone(store.latest_valid_origin('run'))


if __name__ == '__main__':
    unittest.main()

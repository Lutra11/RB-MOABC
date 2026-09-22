import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class RiskR2SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _block(self, name='block', stage='ablation', protocol_hash='protocol'):
        from MOABC.experiments.risk_r2 import VERSION
        from MOABC.experiments.artifact_store import ArtifactStore

        folder = self.root / name
        store = ArtifactStore(folder)
        manifest = store.freeze_manifest({
            'protocol': {'version': VERSION, 'tail_days': 5}, 'protocol_hash': protocol_hash,
            'stage': stage, 'profile': 'screening', 'event_ids': ['P01'],
            'expected_runs': 1,
        })
        run_id = 'P01_matched_full_method_full_MOABC_s00'
        store.write_run_arrays(run_id, schedule=np.ones((5, 2)))
        record = dict(stage=stage, profile='screening', event_id='P01',
                      condition='matched', strategy='full_method', variant='full',
                      algorithm='MOABC', seed_index=0, factor=2.,
                      f1=.1, f2=2., f3=.01, tail_peak_ratio_cvar95=1.1,
                      exceedance_probability_any=.2, max_q_safe_ratio=1.1,
                      minimum_safety_margin_m3s=-10., peak_reduction_pct=5.,
                      plan_constraint_violation=0., operational_deviation=0.,
                      hard_physical_violation=0., seconds=1., evaluations=40,
                      execution_days=2, epsilon_sum=.05,
                      planning_all_feasible=True, feasible_front=[[.1, 2., .01]])
        store.commit_run(run_id, record, manifest)
        return folder, run_id

    def test_private_smoke_tree_is_excluded_from_publication_collection(self):
        from experiments.run_publication_risk_r2 import collect_records

        self._block('block', protocol_hash='publication')
        self._block('_smoke/block', protocol_hash='smoke')
        records, manifests = collect_records(self.root)
        self.assertEqual(len(records), 1)
        self.assertEqual(len(manifests), 1)

    def test_corrupt_artifact_is_rejected(self):
        from experiments.run_publication_risk_r2 import collect_records

        folder, run_id = self._block()
        with (folder / f'{run_id}.npz').open('ab') as handle:
            handle.write(b'corrupt')
        with self.assertRaises(ValueError):
            collect_records(self.root)

    def test_summary_writes_coverage_and_evidence_hashes(self):
        from MOABC.experiments.artifact_store import ArtifactStore
        from experiments.run_publication_risk_r2 import summarize_stage

        self._block()
        output = self.root / 'summary'
        output.mkdir()
        data_dir = self.root / 'data'
        data_dir.mkdir()
        (data_dir / 'publication_events_v3.csv').write_text(
            'event_id,start,end\nP01,2020-05-01,2020-05-02\n', encoding='utf-8')
        args = SimpleNamespace(results_root=self.root, output=output,
                               data_dir=data_dir)
        summarize_stage(args, ArtifactStore(output))
        coverage = json.loads((output / 'coverage.json').read_text(encoding='utf-8'))
        manifest = json.loads((output / 'artifact_manifest.json').read_text(encoding='utf-8'))
        self.assertTrue(coverage['blocks'][0]['complete'])
        self.assertIn('coverage.json', manifest['outputs'])
        self.assertTrue((output / 'ablation_results.csv').exists())


if __name__ == '__main__':
    unittest.main()

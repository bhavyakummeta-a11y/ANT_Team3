"""Validate source accounting and numerical consistency, not biological truth."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ant
import external


class ExternalNumerics(unittest.TestCase):
    def test_identical_estimator_on_shared_channels(self):
        t=np.arange(240)/160
        x=np.array([(i+1)*1e-6*np.sin(2*np.pi*10*t)+(11-i)*1e-6*np.sin(2*np.pi*20*t) for i in range(10)])
        a=ant.bandpower(x); b=external.bandpower(x)
        np.testing.assert_allclose(a['mu'],b[:,0],rtol=1e-12)
        np.testing.assert_allclose(a['beta'],b[:,1],rtol=1e-12)

    def test_prespecified_balancing_keeps_both_conditions_and_hands(self):
        epochs=[]
        for condition in ['executed','imagined']:
            for hand in ['left','right']:
                for i in range(4 if condition=='executed' else 3):
                    epochs.append({'subject':1,'run':3,'onset_seconds':i,'condition':condition,'hand':hand})
        kept,removed=external.balance_epochs(epochs)
        self.assertEqual(len(kept),12)
        self.assertEqual(len(removed),2)
        self.assertEqual(set(Counter((e['condition'],e['hand']) for e in kept).values()),{3})


class ExternalReportChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=json.loads((external.OUT/'external_report.json').read_text())

    def test_selection_and_accounting(self):
        d=self.report
        self.assertEqual(d['selection']['planned_subjects'],[f'S{i:03d}' for i in range(1,11)])
        self.assertEqual(len(d['recordings']),60)
        self.assertEqual(d['download_or_processing_failures'],[])
        self.assertEqual(d['counts']['epochs'],len(d['epoch_features']))
        self.assertEqual(len(d['held_out_predictions']),d['counts']['epochs'])
        for subject in range(1,11):
            sub=[e for e in d['epoch_features'] if e['subject']==subject]
            self.assertEqual(len(set(Counter((e['condition'],e['hand']) for e in sub).values())),1)
            self.assertEqual({e['condition'] for e in sub},{'executed','imagined'})
        keys=[(e['subject'],e['run'],e['onset_seconds']) for e in d['epoch_features']]
        self.assertEqual(len(keys),len(set(keys)))

    def test_metrics_match_held_out_predictions(self):
        d=self.report
        cm=np.zeros((2,2),dtype=int)
        for e in d['held_out_predictions']:
            cm[int(e['condition']=='imagined'),int(e['prediction']=='imagined')]+=1
        np.testing.assert_array_equal(cm,d['metrics']['confusion_matrix'])
        self.assertAlmostEqual(float(np.mean(cm.diagonal()/cm.sum(axis=1))),d['metrics']['balanced_accuracy'])
        self.assertEqual(d['metrics']['dummy_balanced_accuracy'],.5)
        self.assertTrue(np.isfinite(np.array([e['features'] for e in d['epoch_features']])).all())
        self.assertEqual(d['analysis_sha256'],hashlib.sha256(Path(external.__file__).read_bytes()).hexdigest())

    def test_comparison_has_matching_provenance(self):
        comparison=json.loads((external.OUT/'comparison_report.json').read_text())
        for name in ['ant','external']:
            self.assertEqual(comparison['provenance'][name+'_report_sha256'],hashlib.sha256((external.OUT/(name+'_report.json')).read_bytes()).hexdigest())
        self.assertEqual(len(comparison['channel_comparisons']),20)


if __name__=='__main__':unittest.main()

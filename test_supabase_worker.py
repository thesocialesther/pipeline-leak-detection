"""Offline checks; never access or modify a Supabase project."""
import unittest
from supabase_worker import ROOT, fetch_rows, infer_rows, read_config
from smart_flow_model import load


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.mapping = read_config(ROOT/'supabase_config.json')
        self.cfg, self.model = load(ROOT/'flow_artifacts/model.json')
        self.rows = [dict(node_id=node, timestamp=1000+cycle*20, flow_lpm=q)
                     for cycle in range(7)
                     for node,q in zip(('node_1','node_2','node_3'), (10,9,8.95))]

    def predict(self, rows, now=1120):
        return infer_rows(rows, self.mapping, self.cfg, self.model, now)

    def test_prediction_restarts_and_duplicates(self):
        result = self.predict(self.rows)
        self.assertEqual(result['segment_1_state'], 'minor')
        self.assertEqual(result['segment_2_state'], 'normal')
        self.assertEqual(result, self.predict(list(reversed(self.rows))+self.rows))

    def test_missing_stale_fault_and_conflict(self):
        self.assertEqual(self.predict(self.rows,1200)['data_quality'], 'STALE_DATA')
        self.assertEqual(self.predict([r for r in self.rows if r['node_id']!='node_3'])['data_quality'], 'MISSING_NODE_DATA')
        self.assertEqual(self.predict(self.rows+[dict(self.rows[-1],flow_lpm=7)])['data_quality'], 'CONFLICTING_DUPLICATE_READINGS')
        bad = self.rows.copy()
        bad[-1] = dict(bad[-1],flow_lpm=float('nan'))
        self.assertEqual(self.predict(bad)['data_quality'], 'INVALID_SOURCE_ROW')

    def test_warmup_cadence_future(self):
        self.assertEqual(self.predict(self.rows[:12],1060)['data_quality'], 'WARMING_UP')
        fast = [dict(r,timestamp=1000+(r['timestamp']-1000)/20) for r in self.rows]
        self.assertEqual(self.predict(fast,1006)['data_quality'], 'SAMPLING_INTERVAL_MISMATCH')
        self.assertEqual(self.predict(self.rows,1000)['data_quality'], 'FUTURE_DATA')

    def test_read_query_and_truncation(self):
        class Fake:
            def __init__(self): self.calls=[]
            def request(self,table,params):
                self.calls.append((table,params))
                return []
        rest=Fake()
        self.assertEqual(fetch_rows(rest,self.mapping,self.cfg,1120),[])
        self.assertEqual(len(rest.calls),3)
        self.assertEqual(rest.calls[0][1]['node_id'],'eq.node_1')
        rest.request=lambda *args: [{}]*500
        with self.assertRaises(ValueError):
            fetch_rows(rest,self.mapping,self.cfg,1120)


if __name__=='__main__':
    unittest.main()

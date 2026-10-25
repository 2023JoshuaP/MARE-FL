import unittest
from marefl.config import ExperimentConfig
from marefl.knowledge import KnowledgeBase
from marefl.mapek import Analyzer, Monitor, Planner


class TestMAPEK(unittest.TestCase):
    def setUp(self):
        self.cfg = ExperimentConfig()
        self.k = KnowledgeBase(4, 2, self.cfg)

    def test_knowledge_ewma_moves_reliability_down_after_failure(self):
        before = self.k.clients[0].reliability
        self.k.observe_client(0, "failed", 0.2)
        self.assertLess(self.k.clients[0].reliability, before)

    def test_analyzer_classifies_disconnect(self):
        t = Monitor().collect_client(0, "disconnected", 0.3, 1.0)
        a = Analyzer().analyze(t, 0.9)
        self.assertEqual(a.state, "Disconnected")

    def test_planner_never_uses_entropy(self):
        p = Planner(self.cfg)
        t = Monitor().collect_client(0, "failed", 0.3, 1.0)
        a = Analyzer().analyze(t, 0.9)
        d = p.client_decision(a, self.k.clients[0], 1.0, True)
        self.assertIn(d.action, {"REPLACE", "DROP"})
        self.assertFalse(hasattr(self.k.clients[0], "entropy"))

    def test_best_replacement_prefers_reliable_low_latency(self):
        self.k.clients[1].reliability = 0.5
        self.k.clients[2].reliability = 0.95
        self.k.clients[2].latency_s = 0.1
        self.assertEqual(self.k.best_replacement([1,2]), 2)


if __name__ == "__main__":
    unittest.main()
